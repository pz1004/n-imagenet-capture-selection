"""Paired evaluation of five frozen Histogram states; the training seed is not a sampling unit."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from acquisition_inputs import source_groups, write_json
from capture_policies import choose_vertical, fit_discordant

STATES = list(range(6))  # zero denotes the released classifier
PATHS = ("variant1","variant2","original","vertical","horizontal")
ARMS = {
    "V_H": ("variant1","variant2","variant1","variant2"),
    "V_S": ("variant1","original","variant1","original"),
    "H_V": ("variant2","variant1","variant2","variant1"),
    "S_V": ("original","variant1","original","variant1"),
    "transport_V_H": ("variant1","variant2","vertical","horizontal")
}
N, K = 36000, 10800
OUT = HERE/"results"


def data(path):
    captures = {}
    with np.load(path, allow_pickle=False) as table:
        sets = source_groups(table, calibration_split="policy")
        for arm in PATHS:
            for split, (_, labels) in sets.items():
                if arm in ("vertical", "horizontal") and split == "test":
                    continue
                classes = table[f"{split}_{arm}_class"]
                scores = table[f"{split}_{arm}_msp"]
                assert classes.shape == scores.shape == (len(STATES), len(labels))
                for s in STATES:
                    captures[arm, split, s] = classes[s], scores[s]
    return sets, captures

def tie_keys(ids):
    values=[int.from_bytes(hashlib.sha256(("adapt-quota/"+k).encode()).digest()[:8],"big") for k in ids]
    assert len(set(values))==len(values)
    return np.array(values,np.uint64)


@dataclass
class Arm:
    cf:np.ndarray
    cs:np.ndarray
    cx:np.ndarray
    tf:np.ndarray
    ts:np.ndarray
    tx:np.ndarray
    delta:np.ndarray
    mask_ix:np.ndarray
    fit:tuple


def prepare(sets,caps):
    ids,y=sets["test"]
    _,cy=sets["policy"]
    tie=tie_keys(ids)
    arms={}
    gates={}
    for s in STATES:
        for name,(first,second,cf,cs) in ARMS.items():
            fp,fs=caps[first,"test",s]
            sp,ss=caps[second,"test",s]
            cfp,cfs=caps[cf,"policy",s]
            csp,css=caps[cs,"policy",s]
            a,b=fp==y,sp==y
            d=b.astype(np.int8)-a.astype(np.int8)
            aa,bb=cfp==cy,csp==cy
            fit=fit_discordant(cfs-css,aa,bb)
            arms[s,name]=Arm(aa,bb,cfs-css,a,b,fs-ss,d,np.flatnonzero(d),fit)
            # Calibration for the transported rule uses reference first scores.
            gates[s,first,cf]=(np.lexsort((tie,fs)),fs,cfs)
    return arms,gates


def summary(a, mask, become_second):
    n=len(a.tf)
    dg=a.delta*become_second
    u=mask*dg
    req=int(mask.sum())
    disc=mask & (a.tf!=a.ts)
    cnt=int(disc.sum())
    first_on=int((disc & a.tf).sum())
    selected_on=int((disc & np.where(become_second,a.ts,a.tf)).sum())
    rescue=int((u==1).sum())
    loss=int((u==-1).sum())
    assert selected_on-first_on==rescue-loss
    pct=lambda x:100*float(x)/n
    return {"n":n,"requested":req,"first_correct":int(a.tf.sum()),"second_correct":int(a.ts.sum()),
            "oracle_correct":int((a.tf|a.ts).sum()),"always_correct":int(a.tf.sum()+dg.sum()),
            "policy_correct":int(a.tf.sum()+u.sum()),"request_pct":pct(req),"policy_pct":pct(a.tf.sum()+u.sum()),
            "random_pct":pct(a.tf.sum()+req/n*dg.sum()),"gain_pp":pct(u.sum()),"advantage_vs_random_pp":pct(u.sum()-req/n*dg.sum()),
            "rescues":rescue,"losses":loss,"discordant_requested":cnt,"first_correct_requested_discordant":first_on,
            "selected_correct_requested_discordant":selected_on,"discordance_pct_requested":100*cnt/req if req else None,
            "first_share_pct":100*first_on/cnt if cnt else None,"selector_success_pct":100*selected_on/cnt if cnt else None}


def quota_mask(order,k):
    mask=np.zeros(len(order),bool)
    mask[order[:k]]=True
    return mask


def estimate(arms,gates):
    estimates={}
    for s in STATES:
        per=estimates[s]={}
        for name,(first,second,cf,cs) in ARMS.items():
            a=arms[s,name]
            take=~choose_vertical(a.tx,a.fit)
            order,fs,cfs=gates[s,first,cf]
            released_order=gates[0,first,cf][0]
            truth=np.ones(N,bool)
            cutoff=float(np.quantile(cfs,.30,method="higher"))
            res=per[name]={"params":a.fit,"cutoff":cutoff,"all":summary(a,truth,take),
                "calibration_cutoff":summary(a,fs<cutoff,take),"quota_selected":{},"quota_accept":{},
                "released_mask_30":summary(a,quota_mask(released_order,K),take)}
            for pct in (0,10,30,50,100):
                mask=quota_mask(order,N*pct//100)
                res["quota_selected"][pct]=summary(a,mask,take)
                res["quota_accept"][pct]=summary(a,mask,truth)
    return estimates


def sweep_rows(estimates):
    for state,result in estimates.items():
        for arm,rs in result.items():
            for name in ("quota_selected","quota_accept"):
                for pct,r in rs[name].items():
                    yield {"seed":state,"order_fit":arm,"rule":name,"budget_pct":pct,**r}
            for name in ("calibration_cutoff","released_mask_30"):
                yield {"seed":state,"order_fit":arm,"rule":name,"budget_pct":30,**rs[name]}


def write_decisions(sets,caps,arms,gates):
    ids,y=sets["test"]
    for s in STATES:
        fields={"source_id":ids,"label":y}
        for name,(first,second,cf,cs) in ARMS.items():
            a=arms[s,name]
            order,fs,cfs=gates[s,first,cf]
            selected=np.where(choose_vertical(a.tx,a.fit),caps[first,"test",s][0],caps[second,"test",s][0])
            mask=quota_mask(order,K)
            output=np.where(mask,selected,caps[first,"test",s][0])
            fields.update({name+"_quota_request":mask.astype(np.int8),name+"_quota_class":output,
                           name+"_quota_change":(output==y).astype(np.int8)-a.tf.astype(np.int8),
                           name+"_cutoff_request":(fs<np.quantile(cfs,.30,method="higher")).astype(np.int8),
                           name+"_always_class":selected})
        with gzip.open(OUT/f"decisions_seed{s}.csv.gz","wt",newline="") as f:
            writer=csv.writer(f); writer.writerow(fields); writer.writerows(zip(*fields.values()))


def avg_metrics(metrics):
    # Metrics are means of five separately fitted policies, never a pooled classifier.
    out={}
    for name in metrics[1]:
        per=[metrics[s][name] for s in range(1,6)]
        out[name]={"seeds":per,"mean":float(np.mean(per)),"sample_sd":float(np.std(per,ddof=1)),
                   "min":min(per),"max":max(per),"released":metrics[0][name]}
    return out


def collect(state,arms,gain,fixed,common,direct,first_acc):
    alloc=lambda key:first_acc[key]+gain[key]
    values={
        "transport_H_gain_vs_vertical":gain["transport_V_H"],
        "local_H_gain_vs_vertical":gain["V_H"],"local_S_gain_vs_vertical":gain["V_S"],
        "local_H_policy":alloc("V_H"),"local_S_policy":alloc("V_S"),
        "S_first_V_policy":alloc("S_V"),"H_first_V_policy":alloc("H_V"),
        "local_path_effect":gain["V_S"]-gain["V_H"],
        "common_raw_gate_path_effect":common["V_S"]-common["V_H"],
        "take_second_path_effect":direct["V_S"]-direct["V_H"],
        "square_first_minus_vertical_first":alloc("S_V")-alloc("V_S"),
        "horizontal_first_minus_vertical_first":alloc("H_V")-alloc("V_H"),
        "transport_fixedcut_H_gain":fixed["transport_V_H"],
        "fixedcut_path_effect":fixed["V_S"]-fixed["V_H"]
    }
    return values


def points(est):
    measures={}
    for s,arm in est.items():
        gain={k:x["quota_selected"][30]["gain_pp"] for k,x in arm.items()}
        common={k:x["released_mask_30"]["gain_pp"] for k,x in arm.items()}
        direct={k:x["quota_accept"][30]["gain_pp"] for k,x in arm.items()}
        fixed={k:x["calibration_cutoff"]["gain_pp"] for k,x in arm.items()}
        base={k:100*x["all"]["first_correct"]/N for k,x in arm.items()}
        measures[s]=collect(s,arm,gain,fixed,common,direct,base)
    return avg_metrics(measures)


def sample(design,rng):
    classes=np.arange(900)[:,None]
    if design=="within":
        ci=(classes*5+rng.integers(5,size=(900,5))).ravel()
        ti=(classes*40+rng.integers(40,size=(900,40))).ravel()
    else:
        cs=rng.integers(900,size=(900,1))
        ci=(cs*5+np.arange(5)).ravel()
        ti=(cs*40+np.arange(40)).ravel()
    return ci,np.bincount(ti,minlength=N).astype(np.int32)


def weighted_quota(order,w):
    x=w[order]
    cum=x.cumsum()
    end=int(cum.searchsorted(K))
    used=np.zeros(N,np.int32)
    used[order[:end]]=x[:end]
    used[order[end]]=K-(cum[end-1] if end else 0)
    assert used.sum()==K
    return used


# Initialized before fork to avoid copying large test arrays between processes.
GL=None


def boot_chunk(arg):
    design,index,draws=arg
    rng=np.random.default_rng(np.random.SeedSequence([20261126,0 if design=="within" else 1,index]))
    arms,gates=GL
    stored=[]
    for _ in range(draws):
        ci,w=sample(design,rng)
        ranks={(s,f):weighted_quota(gates[s,f,cf][0],w) for s in STATES for f,cf in
               (("variant1","variant1"),("variant2","variant2"),("original","original"))}
        measures={}
        for s in STATES:
            gain,common,direct,fixed,base={},{},{},{},{}
            for k,(first,second,cf,cs) in ARMS.items():
                a=arms[s,k]
                fit=fit_discordant(a.cx[ci],a.cf[ci],a.cs[ci])
                ix=a.mask_ix
                deltas=a.delta[ix]
                d=deltas * ~choose_vertical(a.tx[ix],fit)
                weight=ranks[s,first][ix]
                gain[k]=100*np.dot(weight,d)/N
                common[k]=100*np.dot(ranks[0,first][ix],d)/N
                direct[k]=100*np.dot(weight,deltas)/N
                _,fs,cfs=gates[s,first,cf]
                threshold=np.quantile(cfs[ci],.30,method="higher")
                fixed[k]=100*np.dot(w[ix],d*(fs[ix]<threshold))/N
                base[k]=100*np.dot(w,a.tf)/N
            measures[s]=collect(s,arms,gain,fixed,common,direct,base)
        names=list(measures[0])
        raw=[measures[0][k] for k in names]
        avg=[np.mean([measures[s][k] for s in range(1,6)]) for k in names]
        stored.append(np.array([raw,avg,np.subtract(avg,raw)]))
    return design,names,stored


def set_bootstrap_inputs(arms, gates):
    global GL
    GL = (arms, gates)


def intervals(arms,gates,report,draws,workers):
    global GL
    GL=(arms,gates)
    block=50
    jobs=[(d,i//block,min(block,draws-i)) for d in ("within","class") for i in range(0,draws,block)]
    all_draws={"within":[],"class":[]}
    context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=set_bootstrap_inputs, initargs=(arms, gates)) as pool:
        for design,names,part in pool.map(boot_chunk,jobs):
            all_draws[design].extend(part)
            if len(all_draws[design])%500==0:
                print(design,len(all_draws[design]),flush=True)
    for d,vals in all_draws.items():
        arr=np.stack(vals)
        np.save(OUT/f"bootstrap_{d}.npy",arr)
        ci=np.percentile(arr,[2.5,97.5],axis=0)
        for j,name in enumerate(names):
            report[name][d+"_ci"]={item:ci[:,i,j].tolist() for i,item in enumerate(("released","mean_adapted","adapted_minus_released"))}
    return names


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=HERE / "inputs/scores.npz")
    ap.add_argument("--draws",type=int,default=2000)
    ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    group,caps=data(args.inputs)
    arms,gates=prepare(group,caps)
    estimates=estimate(arms,gates)
    write_json(OUT/"by_seed.json",estimates)
    write_decisions(group,caps,arms,gates)
    with (OUT/"policy_sweep.csv").open("w",newline="") as f:
        rows=list(sweep_rows(estimates))
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report=points(estimates)
    write_json(OUT/"point_estimates.json",report)
    names=intervals(arms,gates,report,args.draws,args.workers)
    write_json(OUT/"paired.json",{"bootstrap_names":names,"paired_metrics":report,"n_draws_each":args.draws})
    print(json.dumps({k:[round(x["mean"],3),round(x["released"],3)] for k,x in report.items()}),flush=True)


if __name__=="__main__":
    main()
