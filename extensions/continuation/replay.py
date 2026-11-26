"""Paired calibrated replay, numeric cutoff and fixed quota, with calibration-refit intervals."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from acquisition_inputs import source_groups, write_json
from capture_policies import choose_vertical, fit_discordant

N = 30000
PAIRS = ("second", "horizontal", "square")
ARMS = ("second", "horizontal", "square", "pooled")
RULES = ("accept", "select", "average")
DESIGNS = ("within", "class")
CONTRASTS = (
    ("square.select","second.select"), ("horizontal.select","second.select"),
    ("square.select","pooled.accept"), ("pooled.accept","second.select"),
    ("square.select","horizontal.select"), ("square.accept","second.accept"),
    ("square.average","second.average"), ("horizontal.average","second.average"),
)
OUT = HERE/"results"

def load(path):
    with np.load(path, allow_pickle=False) as table:
        groups = source_groups(table)
        cohorts = {}
        for split, (ids, labels) in groups.items():
            data = {"label": labels}
            for arm in ("first", *ARMS):
                for field in ("class", "msp", "n", "us", "eligible"):
                    name = f"{arm}.{field}"
                    data[name] = table[f"{split}_{name}"]
                data[f"{arm}.ok"] = data[f"{arm}.class"] == labels
            for arm in PAIRS:
                name = arm + ".fused"
                data[name] = table[f"{split}_{name}"]
                data[name + "_ok"] = data[name] == labels
            data["common"] = np.logical_and.reduce([data[k+".eligible"] for k in PAIRS])
            cohorts[split] = data
    return groups["test"][0], cohorts["calibration"], cohorts["test"]

def fit_models(c,ix):
    cut=float(np.quantile(c["first.msp"][ix],.30,method="higher"))
    fits={}
    for arm in PAIRS:
        selected=ix[c[arm+".eligible"][ix]]
        delta=c["first.msp"][selected]-c[arm+".msp"][selected]
        fits[arm]=fit_discordant(delta,c["first.ok"][selected],c[arm+".ok"][selected])
    return cut,fits

def quota(s, ix):
    scores=s[ix]
    k=round(.30*len(ix))
    boundary=np.partition(scores,k-1)[k-1]
    request=scores<boundary
    ties=np.flatnonzero(scores==boundary)
    request[ties[np.argsort(ix[ties],kind="stable")[:k-int(request.sum())]]]=True
    assert request.sum()==k
    return request

def choices(d, fits, which):
    first=d["first."+which]
    arms={}
    for arm in ARMS:
        ok=d[arm+".eligible"]
        follow=d[arm+"."+which]
        arms[arm+".accept"]=np.where(ok,follow,first)
        if arm in PAIRS:
            pick_first=choose_vertical(d["first.msp"]-d[arm+".msp"],fits[arm])
            arms[arm+".select"]=np.where(ok & ~pick_first,follow,first)
            arms[arm+".average"]=np.where(ok,d[arm+(".fused_ok" if which=="ok" else ".fused")],first)
    return arms

def metrics(d,ix,req,selected):
    counts={}
    for scope,allowed in (("all",np.ones(len(ix),bool)),("common",d["common"][ix])):
        base=d["first.ok"][ix][allowed]
        selected_ix=ix[req & allowed]
        count=len(base)
        acc=100*base.mean()
        request_rate=len(selected_ix)/count
        first_correct=d["first.ok"][selected_ix].astype(int)
        values={"first":acc}
        counts[scope+".requests_pct"]=100*request_rate
        counts[scope+".first.acc"]=acc
        for name,current in selected.items():
            gain=100*(current[selected_ix].sum()-first_correct.sum())/count
            values[name]=acc+gain
            counts[scope+"."+name+".acc"]=acc+gain
            counts[scope+"."+name+".gain"]=gain
        for left,right in CONTRASTS:
            counts[scope+"."+left+"-"+right]=values[left]-values[right]
    return counts

def outputs(cal,test,ids):
    ci=np.arange(len(cal["label"])); ti=np.arange(len(test["label"]))
    cut,fits=fit_models(cal,ci)
    test_choices=choices(test,fits,"class")
    correct_choices={k:v==test["label"] for k,v in test_choices.items()}
    result={"msp_cutoff":cut,"fits":fits,"calibration":len(ci),"test":len(ti),
            "complete_n":{sp:{arm:int(d[arm+".eligible"].sum()) for arm in ARMS} | {"common":int(d["common"].sum())}
                          for sp,d in (("calibration",cal),("test",test))},
            "raw_first_correct":int(test["first.ok"].sum())}
    first_us=test["first.us"]
    gates={"cutoff":test["first.msp"]<cut,"quota":quota(test["first.msp"],ti)}
    for name,req in gates.items():
        row={"requests":int(req.sum()),"eligible_request":{}, "cost":{}, "arms":{}, "scope":{}}
        for scope,eligible in (("all",np.ones(len(ti),bool)),("common",test["common"])):
            row["scope"][scope]={"n":int(eligible.sum()),"requested":int((eligible & req).sum())}
        for arm in ARMS:
            eligible=test[arm+".eligible"]
            row["eligible_request"][arm]=int((req & eligible).sum())
            extra=test[arm+".us"]
            mean_first=first_us.mean()/1000
            mean_extra=extra[req].mean()/1000
            mean_used=(first_us+req*extra).mean()/1000
            switches=arm in ("horizontal","square")
            row["cost"][arm]={"mean_first_ms":mean_first,"mean_extra_per_request_ms":mean_extra,
                "mean_observed_ms":mean_used,"extra_span_ms_q25_q50_q75":(np.quantile(extra[req],[.25,.5,.75])/1000).tolist(),
                "mean_recordings_started":1+int(switches)*req.mean(),
                "mean_inferences":1+(req & eligible).mean(),
                "mean_observed_events":(test["first.n"]+req*test[arm+".n"]).mean(),
                "mean_time_ms_overhead_scenarios":{str(h):mean_used+int(switches)*req.mean()*h for h in (0,5,10,20)}}
        for arm in ("horizontal","square"):
            row["cost"][arm]["switch_overhead_ms_equal_to_continuation"]=(
                row["cost"]["second"]["mean_observed_ms"]-row["cost"][arm]["mean_observed_ms"])/req.mean()
        for rule,ok in correct_choices.items():
            s=test["first.ok"]
            gain=int((req & ok & ~s).sum())
            loss=int((req & ~ok & s).sum())
            if_all=int(ok.sum())
            random=(s.sum()+req.mean()*(if_all-s.sum()))/len(s)*100
            row["arms"][rule]={"rescues":gain,"losses":loss,"correct":int(s.sum())+gain-loss,
                "all_requests_correct":if_all,"net_per_100_requests":100*(gain-loss)/req.sum(),
                "expected_random_accuracy_pct":random}
        row["statistics"]=metrics(test,ti,req,correct_choices)
        result[name]=row
    with gzip.GzipFile(OUT/"heldout_decisions.csv.gz", "wb", mtime=0) as raw, io.TextIOWrapper(raw, newline="") as f:
        writer=csv.writer(f)
        names=list(test_choices)
        columns=["source_id","label","first_class","first_msp","calibrated_request","quota_request","common_complete",
                 *[a+"_complete" for a in ARMS], *[a+"_us" for a in ("first",*ARMS)],*names]
        writer.writerow(columns)
        for i,key in enumerate(ids):
            writer.writerow([key,int(test["label"][i]),int(test["first.class"][i]),test["first.msp"][i],
                             int(gates["cutoff"][i]),int(gates["quota"][i]),int(test["common"][i]),
                             *[int(test[a+".eligible"][i]) for a in ARMS],
                             *[int(test[a+".us"][i]) for a in ("first",*ARMS)],
                             *[int(test_choices[n][i]) for n in names]])
    return result

def write_figure_data(summary):
    with (OUT/"time_accuracy.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["gate","arm","requested","completed","accuracy_pct","mean_recordings","mean_inferences","mean_event_span_ms","serial_ms_switch_5","serial_ms_switch_10","serial_ms_switch_20"])
        for gate in ("cutoff","quota"):
            row=summary[gate]
            base=row["cost"]["second"]["mean_first_ms"]
            writer.writerow([gate,"vertical only",0,0,100*summary["raw_first_correct"]/summary["test"],1,1,base,base,base,base])
            for arm in ARMS:
                rule=arm+(".accept" if arm=="pooled" else ".select")
                t=row["cost"][arm]
                writer.writerow([gate,rule,row["requests"],row["eligible_request"][arm],row["statistics"]["all."+rule+".acc"],
                            t["mean_recordings_started"],t["mean_inferences"],t["mean_observed_ms"],
                            *[t["mean_time_ms_overhead_scenarios"][str(h)] for h in (5,10,20)]])

def replicate(cal,test, design, draws):
    rng=np.random.default_rng(202611266 if design=="within" else 202611267)
    tc=40*np.arange(900)[:,None];cc=10*np.arange(900)[:,None]
    table={}
    for iteration in range(draws):
        if design=="within":
            ci=(cc+rng.integers(10,size=(900,10))).reshape(-1)
            ti=(tc+rng.integers(40,size=(900,40))).reshape(-1)
        else:
            classes=rng.integers(900,size=900)[:,None]
            ci=(10*classes+np.arange(10)).ravel()
            ti=(40*classes+np.arange(40)).ravel()
        cut,fits=fit_models(cal,ci)
        selected=choices(test,fits,"ok")
        req1=test["first.msp"][ti]<cut
        req2=quota(test["first.msp"],ti)
        row={}
        for name,req in (("cutoff",req1),("quota",req2)):
            row.update({name+"."+k:v for k,v in metrics(test,ti,req,selected).items()})
        for key,value in row.items():
            table.setdefault(key,[]).append(value)
    return {key:np.quantile(values,[.025,.975]).tolist() for key,values in table.items()}

def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs/scores.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    ids, cal, test = load(args.inputs)
    summary = outputs(cal, test, ids)
    write_json(OUT/"point.json", summary)

    write_figure_data(summary)
    print("point",summary["cutoff"]["requests"],summary["cutoff"]["statistics"]["all.square.select-second.select"],flush=True)
    for design in DESIGNS:
        write_json(OUT/(design+"_intervals.json"),replicate(cal,test,design,2000))
        print("complete",design,flush=True)


if __name__=="__main__":
    main()
