"""Source-paired analysis of fixed adapted states and released chronological replay."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gzip
import io
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from acquisition_inputs import source_groups, write_json as write
from capture_policies import fit_discordant, choose_vertical

OUT = HERE / "results"
STATES=range(6)
ARMS=("continue","horizontal","square")
GATES=("cutoff","released_mask","quota","complete")
N=36000

def load(path=HERE / "inputs/scores.npz"):
    """Join all six states to the frozen source labels and preserve the input precision."""
    with np.load(path, allow_pickle=False) as archive:
        groups = source_groups(archive, calibration_split="policy")
        ids, labels = groups["policy"]
        test_ids, test_labels = groups["test"]
        blocks = []
        for name, length in (("policy", len(ids)), ("test", len(test_ids))):
            classes = archive[name + "_class"]
            scores = archive[name + "_msp"]
            complete = archive[name + "_eligible"]
            if classes.shape != scores.shape or classes.shape != (6, 4, length) or complete.shape != (4, length):
                raise ValueError(f"Score dimensions differ from six states and four chronological views for {name}")
            if ((classes < 0) | (classes >= 1000)).any() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise ValueError(f"Invalid prediction or confidence for {name}")
            blocks.append((classes, scores, complete))
    return (ids + test_ids, np.r_[labels, test_labels],
            *(np.concatenate(pair, axis=-1) for pair in zip(*blocks)))

@dataclass
class Arm:
    cd: np.ndarray
    cf: np.ndarray
    cs: np.ndarray
    ce: np.ndarray
    td: np.ndarray
    diff: np.ndarray
    eligible: np.ndarray
    ix: np.ndarray

def arrays(y,pred,score,eligible):
    arms={}; ranks={}; first={}; cscore={};tscore={}
    for s in STATES:
        fc=pred[s,0]==y
        first[s]=fc[4500:]
        cscore[s]=score[s,0,:4500]; tscore[s]=score[s,0,4500:]
        ranks[s]=np.argsort(tscore[s],kind="stable")
        for a in range(1,4):
            sc=pred[s,a]==y
            sd=score[s,0]-score[s,a]
            diff=sc[4500:].astype(np.int8)-first[s]
            arms[s,a]=Arm(sd[:4500],fc[:4500],sc[:4500],eligible[a,:4500],sd[4500:],diff,eligible[a,4500:],
                          np.flatnonzero(eligible[a,4500:] & (diff!=0)))
    common=np.all(eligible[1:,4500:],axis=0)
    return arms,ranks,first,cscore,tscore,common

def fit(a,cal):
    use=cal[a.ce[cal]]
    return fit_discordant(a.cd[use],a.cf[use],a.cs[use])

def quota(order,weights):
    n=round(.3*int(weights.sum()))
    cum=np.cumsum(weights[order])
    end=cum.searchsorted(n)
    used=np.zeros_like(weights)
    used[order[:end]]=weights[order[:end]]
    used[order[end]]=n-(cum[end-1] if end else 0)
    return used

def point(ids,y,pred,score,eligible,inputs):
    arms,ranks,first,cscore,tscore,common=inputs
    ti=np.arange(N);ci=np.arange(4500)
    cut={s:float(np.quantile(cscore[s],.30,method="higher")) for s in STATES}
    pm={s:tscore[s]<cut[s] for s in STATES}
    out={}
    with gzip.GzipFile(OUT/"decisions.csv.gz", "wb",mtime=0) as raw, io.TextIOWrapper(raw,newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["seed","source_id","label","first_class","first_msp","request","quota_request","released_request","common_complete",
                         *[arm+"_"+c for arm in ARMS for c in ("eligible","class","msp","terminal","policy")]])
        for s in STATES:
            masks={"cutoff":pm[s],"released_mask":pm[0],"quota":quota(ranks[s],np.ones(N,int)).astype(bool),"complete":pm[s]&common}
            r={"cutoff":cut[s],"first_correct":int(first[s].sum()),"n":N,"fits":{},"gate":{}}
            cols=[np.full(N,s),ids[4500:],y[4500:],pred[s,0,4500:],tscore[s],pm[s].astype(int),masks["quota"].astype(int),pm[0].astype(int),common.astype(int)]
            terminals={}
            for a,name in enumerate(ARMS,1):
                ar=arms[s,a]
                fpars=fit(ar,ci)
                r["fits"][name]=list(map(float,fpars))
                switched=ar.eligible & ~choose_vertical(ar.td,fpars)
                terminal=np.where(switched,pred[s,a,4500:],pred[s,0,4500:])
                terminals[a]=terminal==y[4500:]
                chosen=np.where(pm[s],terminal,pred[s,0,4500:])
                cols += [ar.eligible.astype(int),pred[s,a,4500:],score[s,a,4500:],terminal,chosen]
            writer.writerows(zip(*cols))
            for gate,mask in masks.items():
                scope=common if gate=="complete" else np.ones(N,bool)
                n=int(scope.sum());ff=first[s]
                x={"requested":int(mask.sum()),"n":n,"first_correct":int((ff&scope).sum()),"arms":{}}
                for a,name in enumerate(ARMS,1):
                    ar=arms[s,a]
                    second=pred[s,a,4500:]==y[4500:]
                    selected=terminals[a]
                    avail=mask&ar.eligible
                    gain=int((mask&~ff&selected).sum());loss=int((mask&ff&~selected).sum());correct=x["first_correct"]+gain-loss
                    random=100*(x["first_correct"]/n+mask.sum()/n*((selected&scope).sum()-x["first_correct"])/n)
                    x["arms"][name]={"eligible":int(avail.sum()),"second_acc_complete_pct":100*float(second[scope&ar.eligible].mean()),
                        "attainable_rescues":int((avail&~ff&second).sum()),"first_only_successes":int((avail&ff&~second).sum()),
                        "direct_gain_pp":100*float((second.astype(int)-ff.astype(int))[avail].sum())/n,
                        "random_expected_pct":random,"rescues":gain,"losses":loss,"correct":correct,"accuracy_pct":100*correct/n}
                for name in ("horizontal","square"):
                    x[name+"_minus_continue"]=x["arms"][name]["accuracy_pct"]-x["arms"]["continue"]["accuracy_pct"]
                r["gate"][gate]=x
            out[s]=r
    write(OUT/"by_seed.json",out)
    return out

GL=None
def draw_metrics(cal,w):
    arms,ranks,first,cscore,tscore,common=GL
    cutoff={s:np.quantile(cscore[s][cal],.30,method="higher") for s in STATES}
    rm=tscore[0]<cutoff[0]
    common_n=int(w@common)
    result=[]
    for s in STATES:
        base=100*(w@first[s])/N
        base_complete=100*np.dot(w*common,first[s])/common_n
        wgate=w*(tscore[s]<cutoff[s])
        wquota=quota(ranks[s],w)
        wrm=w*rm
        values={"first":base}
        acc={}
        for a,name in enumerate(ARMS,1):
            ar=arms[s,a]; ix=ar.ix
            sel=~choose_vertical(ar.td[ix],fit(ar,cal))
            d=ar.diff[ix]*sel
            acc[name]=100*np.dot(wgate[ix],d)/N
            values["cutoff_"+name]=base+acc[name]
            values["released_mask_"+name]=100*np.dot(wrm[ix],d)/N
            values["quota_"+name]=100*np.dot(wquota[ix],d)/N
            values["complete_"+name]=base_complete+100*np.dot(wgate[ix]*common[ix],d)/common_n
            values["direct_all_"+name]=base+100*np.dot(w[ix],ar.diff[ix])/N
            values["standalone_complete_"+name]=base_complete+100*np.dot(w[ix]*common[ix],ar.diff[ix])/common_n
        for gate in (*GATES,"direct_all","standalone_complete"):
            for left,right in PAIRINGS:
                values[gate+"_"+left+"_minus_"+right]=values[gate+"_"+left]-values[gate+"_"+right]
        result.append([values[k] for k in METRICS])
    raw=np.array(result)
    mean=raw[1:].mean(0)
    return np.vstack([raw,mean,mean-raw[0]])

PAIRINGS=(("horizontal","continue"),("square","continue"),("square","horizontal"))
METRICS=["first",*[g+"_"+a for g in ("cutoff","complete","direct_all","standalone_complete") for a in ARMS],
         *[g+"_"+left+"_minus_"+right for g in (*GATES,"direct_all","standalone_complete") for left,right in PAIRINGS]]
def worker(job):
    design,start,n=job
    rng=np.random.default_rng(np.random.SeedSequence([20261128,int(design=="class"),start]))
    output=[]
    for _ in range(n):
        c=np.arange(900)[:,None]
        if design=="within":
            ci=(5*c+rng.integers(5,size=(900,5))).ravel()
            ti=(40*c+rng.integers(40,size=(900,40))).ravel()
            w=np.bincount(ti,minlength=N)
        else:
            cl=rng.integers(900,size=900)[:,None]
            ci=(5*cl+np.arange(5)).ravel()
            w=np.repeat(np.bincount(cl.ravel(),minlength=900),40)
        output.append(draw_metrics(ci,w))
    return design,np.stack(output)

def set_worker_inputs(inputs):
    global GL
    GL = inputs

def run(workers, draws, input_path):
    global GL
    OUT.mkdir(parents=True, exist_ok=True)
    ids,y,pred,score,eligible=load(input_path)
    GL=arrays(y,pred,score,eligible)
    by=point(ids,y,pred,score,eligible,GL)
    observed=draw_metrics(np.arange(4500),np.ones(N,int))
    for s,r in by.items():
        for g in GATES:
            for a in ("horizontal","square"):
                name=g+"_"+a+"_minus_continue"
                assert abs(observed[s,METRICS.index(name)]-r["gate"][g][a+"_minus_continue"])<1e-10
    print("primary points",observed[:,METRICS.index("cutoff_horizontal_minus_continue")],flush=True)
    jobs=[(design,start,min(25,draws-start)) for design in ("within","class") for start in range(0,draws,25)]
    summary={k:{"point":dict(zip(METRICS,row.tolist()))} for k,row in zip(["released",*[f"seed{i}" for i in range(1,6)],"mean","interaction"],observed)}
    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context(start_method),
                             initializer=set_worker_inputs, initargs=(GL,)) as pool:
        results={d:[] for d in ("within","class")}
        for name,block in pool.map(worker,jobs):
            results[name].append(block)
            if len(results[name])%20==0:
                print(name,len(results[name])*25,flush=True)
    for name,blocks in results.items():
        samples=np.concatenate(blocks)
        assert samples.shape==(draws,8,len(METRICS))
        np.save(OUT/f"bootstrap_{name}.npy",samples)
        bounds=np.quantile(samples,[.025,.975],axis=0)
        for i,k in enumerate(summary):
            summary[k][name]={key:bounds[:,i,j].tolist() for j,key in enumerate(METRICS)}
    write(OUT/"paired.json",{"draws_each":draws,"metrics":METRICS,"states":list(summary),"estimates":summary})
    print("analyzed",flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs/scores.npz")
    parser.add_argument("--output", type=Path, required=True, help="New destination for replay outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1 or args.draws < 1:
        parser.error("Use a new output directory and positive draw/worker counts")
    OUT = args.output
    run(args.workers, args.draws, args.inputs)
