"""Replay DiST confidence ranking and chronological acquisition from per-source measurements."""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capture_policies import read, sorted_ids, fit_discordant, choose_vertical
from acquisition_inputs import write_json as write

ARMS=("continue","horizontal","square")
NAMES={"v":("vertical","hash"),"h":("horizontal","hash"),"first":("variant1","first"),
       "continue":("variant1","second"),"horizontal":("variant2","first"),"square":("original","first")}
DESIGNS=("within","class")

def load(required, inputs):
    manifest = read(ROOT / "data/source_index.csv", "cohort", "full")
    ids = sorted_ids(manifest, "calibration") + sorted_ids(manifest, "test")
    y = np.array([int(manifest[k]["label"]) for k in ids])
    for labels, n in ((y[:9000],10),(y[9000:],40)):
        expected = np.repeat(np.arange(100,1000),n)
        np.testing.assert_array_equal(labels, expected)
    tables = {}
    data = {"label": y}
    for name in required:
        path, crop = NAMES[name]
        if path not in tables:
            with gzip.open(inputs / (path+".csv.gz"), "rt", newline="") as stream:
                values = list(csv.DictReader(stream))
            table = {row["source_id"]: row for row in values}
            if len(table) != len(values) or len(table) != (50000 if path == "original" else 45000):
                raise ValueError(f"Unexpected number of distinct source IDs for {path}")
            ordered = [table[k] for k in ids]
            if any(r["split"] != manifest[k]["split"] or int(r["label"]) != label
                   for r, k, label in zip(ordered, ids, y)):
                raise ValueError(f"Source split or class mismatch for {path}")
            tables[path] = ordered
        ordered = tables[path]
        prediction = np.array([int(r[crop+"_class"]) for r in ordered])
        msp = np.array([float(r[crop+"_msp"]) for r in ordered])
        if np.any((prediction<0) | (prediction>=1000)) or not np.all((msp>0) & (msp<=1)):
            raise ValueError(f"Invalid class prediction or confidence for {path}/{crop}")
        data[name+"_p"] = prediction
        data[name+"_s"] = msp
        data[name+"_ok"] = prediction == y
        data[name+"_full"] = np.array([int(r[crop+"_stop"])-int(r[crop+"_start"])==30000 for r in ordered])
    return ids, data

def tidy(series):
    return {k:v.item() if isinstance(v,np.generic) else v for k,v in series.items()}

def resampling(kind,rng,k):
    if kind=="within":
        te=(40*np.arange(900)[:,None]+rng.integers(40,size=(900,40))).ravel()
        ca=(k*np.arange(900)[:,None]+rng.integers(k,size=(900,k))).ravel()
    else:
        classes=rng.integers(900,size=900)[:,None]
        te=(classes*40+np.arange(40)).ravel()
        ca=(classes*k+np.arange(k)).ravel()
    return te,ca

def rank(d,i,counts=False):
    vs=d["v_s"][i]
    hs=d["h_s"][i]
    v=d["v_ok"][i]
    h=d["h_ok"][i]
    only_v=v & ~h
    only_h=h & ~v
    ties=vs==hs
    r_v=100*((vs>hs)+.5*ties)[only_v].mean()
    r_h=100*((hs>vs)+.5*ties)[only_h].mean()
    choose_first=vs>=hs
    selected=np.where(choose_first,v,h)
    values={"v_only_rank":r_v,"h_only_rank":r_h,"balanced_rank":.5*(r_v+r_h),
            "first_pct":100*v.mean(),"horizontal_pct":100*h.mean(),"raw_accuracy_pct":100*selected.mean(),
            "gain_pp":100*(selected.sum()-v.sum())/len(v),"d_pct":100*(only_v|only_h).mean(),
            "first_share_pct":100*only_v.sum()/(only_v|only_h).sum(),
            "pooled_selector_pct":100*selected[only_v|only_h].mean()}
    if counts:
        values.update(only_vertical_n=only_v.sum(),only_horizontal_n=only_h.sum(),
                      vertical_wins=((vs>hs)&only_v).sum(),horizontal_wins=((hs>vs)&only_h).sum(),
                      vertical_ties=(ties&only_v).sum(),horizontal_ties=(ties&only_h).sum(),
                      both_correct=(v&h).sum(),both_wrong=(~v&~h).sum(),
                      rescues=(selected&~v).sum(),losses=(~selected&v).sum())
    return tidy(values)

def subtract(a,b):
    return (a.astype(np.int8)-b.astype(np.int8))

def quota(idx,score):
    cutoff=np.partition(score,int(round(.3*len(score)))-1)[int(round(.3*len(score)))-1]
    q=score<cutoff
    ties=np.flatnonzero(score==cutoff)
    num=round(.3*len(q))-q.sum()
    q[ties[np.argsort(idx[ties],kind="stable")[:num]]]=True
    return q

def chrono(d,test,cal,counts=False):
    first_score=d["first_s"]
    first_ok=d["first_ok"]
    cut=np.quantile(first_score[cal],.30,method="higher")
    fit={}
    elig={}
    select={}
    terminal={}
    for arm in ARMS:
        eligible=d[arm+"_full"]
        sel=cal[eligible[cal]]
        diff=first_score-d[arm+"_s"]
        fit[arm]=fit_discordant(diff[sel],first_ok[sel],d[arm+"_ok"][sel])
        use=eligible[test] & ~choose_vertical(diff[test],fit[arm])
        terminal[arm]=np.where(use,d[arm+"_p"][test],d["first_p"][test])
        elig[arm]=eligible[test]
        select[arm]=terminal[arm]==d["label"][test]
    masks={"cutoff":first_score[test]<cut,"quota":quota(test,first_score[test])}
    common=elig["continue"]&elig["horizontal"]&elig["square"]
    first=first_ok[test]
    out={}
    detail={}
    for a,b in (("horizontal","continue"),("square","continue"),("square","horizontal")):
        a_ok=d[a+"_ok"][test]
        b_ok=d[b+"_ok"][test]
        out["standalone_common."+a+"-"+b]=100*subtract(a_ok,b_ok)[common].mean()
        out["always_accept_with_fallback."+a+"-"+b]=100*subtract(np.where(elig[a],a_ok,first),np.where(elig[b],b_ok,first)).mean()
    for gate,mask in masks.items():
        paired={}
        if counts:
            detail[gate]={"requests":int(mask.sum()),"complete_common":int(common.sum()),
                          "complete_common_requests":int((common&mask).sum()),"arms":{}}
        for arm in ARMS:
            second_ok=d[arm+"_ok"][test]
            ok=select[arm]
            acc=np.where(mask,ok,first)
            direct=np.where(mask & elig[arm],second_ok,first)
            paired[arm]=acc
            out[gate+"."+arm+"_accuracy"]=100*acc.mean()
            out[gate+"."+arm+"_gain"]=100*subtract(acc,first).mean()
            out[gate+"."+arm+"_direct_accuracy"]=100*direct.mean()
            if counts:
                sub=detail[gate]["arms"][arm]={}
                sub["completed"]=int((mask&elig[arm]).sum())
                sub["available_rescues"]=int((mask & elig[arm] & ~first & second_ok).sum())
                sub["rescues"]=int((~first&acc).sum())
                sub["losses"]=int((first&~acc).sum())
                sub["correct"]=int(acc.sum())
                sub["direct_rescues"]=int((~first&direct).sum())
                sub["direct_losses"]=int((first&~direct).sum())
                sub["candidate_standalone_all_available_pct"]=100*second_ok[elig[arm]].mean()
                sub["candidate_standalone_common_pct"]=100*second_ok[common].mean()
                sub["candidate_standalone_common_correct"]=int(second_ok[common].sum())
                sub["all_sources_direct_with_fallback_pct"]=100*np.where(elig[arm],second_ok,first).mean()
                sub["all_sources_direct_with_fallback_correct"]=int(np.where(elig[arm],second_ok,first).sum())
                sub["cohort_policy_on_common_pct"]=100*acc[common].mean()
                sub["cohort_policy_on_common_correct"]=int(acc[common].sum())
                sub["expected_random_pct"]=100*(first.mean()+mask.mean()*subtract(ok,first).mean())
        for a,b in (("horizontal","continue"),("square","continue"),("square","horizontal")):
            delta=subtract(paired[a],paired[b])
            out[gate+"."+a+"-"+b]=100*delta.mean()
            out[gate+"."+a+"-"+b+"_common"]=100*delta[common].mean()
    out["first_accuracy"]=100*first.mean()
    out["requested_pct"]=100*masks["cutoff"].mean()
    if counts:
        return (tidy(out),dict(numerical_cutoff=float(cut),fits=fit,first_correct=int(first.sum()),**detail),masks,terminal)
    return out

def intervals(rows):
    return {k:np.quantile([r[k] for r in rows],[.025,.975]).tolist() for k in rows[0]}

def run_ranking(inputs, out, draws):
    ids,d=load(["v","h"], inputs)
    test=np.arange(9000,45000)
    point=rank(d,test,True)
    write(out/"ranking.json",point)
    print("RANK:",point,flush=True)
    with gzip.open(out/"ranking_decisions.csv.gz","wt",newline="") as f:
        w=csv.writer(f)
        w.writerow(["source_id","label","vertical_class","vertical_msp","horizontal_class","horizontal_msp","raw_class"])
        for i in test:
            w.writerow([ids[i],d["label"][i],d["v_p"][i],d["v_s"][i],d["h_p"][i],d["h_s"][i],
                        d["v_p"][i] if d["v_s"][i]>=d["h_s"][i] else d["h_p"][i]])
    for j,design in enumerate(DESIGNS if draws else ()):
        rng=np.random.default_rng(2026112801+j)
        rows=[rank(d,test[resampling(design,rng,10)[0]]) for k in range(draws)]
        write(out/("ranking_"+design+".json"),intervals(rows))

def run_chrono(inputs, out, draws):
    ids,d=load(["first",*ARMS], inputs)
    test=np.arange(9000,45000)
    for k in (10,5):
        cal=np.arange(9000).reshape(900,10)[:,-k:].ravel()
        suffix=str(len(cal))
        point,detail,masks,terminal=chrono(d,test,cal,True)
        write(out/("chrono_"+suffix+".json"),dict(point=point,detail=detail))
        print("CHRONO",suffix,point,detail["cutoff"]["requests"],flush=True)
        with gzip.open(out/("chrono_decisions_"+suffix+".csv.gz"),"wt",newline="") as f:
            w=csv.writer(f)
            w.writerow(["source_id","label","first_class","first_msp","cutoff_request","quota_request", *[a+"_complete" for a in ARMS],*[a+"_terminal" for a in ARMS]])
            for n,i in enumerate(test):
                w.writerow([ids[i],d["label"][i],d["first_p"][i],d["first_s"][i],int(masks["cutoff"][n]),int(masks["quota"][n]),
                            *[int(d[a+"_full"][i]) for a in ARMS],*[terminal[a][n] for a in ARMS]])
    for k in (10,5):
        cal=np.arange(9000).reshape(900,10)[:,-k:].ravel()
        suffix=str(len(cal))
        for j,design in enumerate(DESIGNS if draws else ()):
            rng=np.random.default_rng(2026112803+j)
            rows=[]
            for _ in range(draws):
                te,ca=resampling(design,rng,k)
                rows.append(chrono(d,test[te],cal[ca]))
            write(out/("chrono_"+suffix+"_"+design+".json"),intervals(rows))
            print("finished",suffix,design,flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE/"inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000, help="Resamples per design; 0 computes only point estimates and per-source decisions.")
    parser.add_argument("--analysis", choices=("all","ranking","chrono"), default="all")
    args = parser.parse_args()
    if args.draws < 0 or args.output.resolve() in [(HERE/"inputs").resolve(), (HERE/"results").resolve()]:
        parser.error("Provide a nonnegative draw count and a separate output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.analysis in ("all","ranking"):
        run_ranking(args.inputs, args.output, args.draws)
    if args.analysis in ("all","chrono"):
        run_chrono(args.inputs, args.output, args.draws)
