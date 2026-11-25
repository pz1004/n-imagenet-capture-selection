"""Refit per-acquisition temperatures and recompute probability fusion from raw logits."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from trajectory_data import ids_for


def load(folder, keys, model):
    with (folder/"predictions.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    row_index={r["source_id"]:i for i,r in enumerate(rows)}
    if len(row_index)!=len(rows) or not set(keys)<=row_index.keys():
        raise ValueError(f"Source ID mismatch in {folder}")
    where=[row_index[key] for key in keys]
    labels=np.array([int(rows[i]["label"]) for i in where])
    scores=np.array([float(rows[i][model+"_msp"]) for i in where])
    logits=np.load(folder/(model+".npy"),mmap_mode="r")
    if logits.shape != (len(rows),1000):
        raise ValueError(f"Unexpected logit shape in {folder}: {logits.shape}")
    return logits[where],labels,scores


def fit_temperature(x,y):
    z=x.astype(np.float64)
    z-=z.max(axis=1,keepdims=True)
    true=z[np.arange(len(y)),y]
    def slope(beta):
        p=np.exp(beta*z)
        return ((p*z).sum(axis=1)/p.sum(axis=1)-true).mean()
    lo,hi=0.,1.
    while hi<128 and slope(hi)<0:
        hi*=2
    if slope(lo)>=0 or slope(hi)<0:
        raise ValueError("A finite positive temperature optimum could not be bracketed")
    for _ in range(42):
        middle=(lo+hi)/2
        if slope(middle)<0:
            lo=middle
        else:
            hi=middle
    return 2/(lo+hi)


def probability(x,temperature):
    z=x.astype(np.float64)/temperature
    z-=z.max(axis=1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(axis=1,keepdims=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("first-cal","second-cal","first-test","second-test"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--model",choices=("histogram","timestamp"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    cal=ids_for("calibration")
    test=ids_for("test")
    temps=[]
    for folder in (args.first_cal,args.second_cal):
        x,y,_=load(folder,cal,args.model)
        temps.append(fit_temperature(x,y))
        del x
    first,labels,score_first=load(args.first_test,test,args.model)
    second,labels_h,score_second=load(args.second_test,test,args.model)
    assert np.array_equal(labels,labels_h)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    losses=np.zeros((2,2))
    with args.output.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("source_id","label","first","second","confidence_first","confidence_second",
                         "scaled_confidence_first","scaled_confidence_second","raw_choice",
                         "scaled_choice","probability_average"))
        for i in range(0,len(test),512):
            sl=slice(i,i+512)
            ax,bx=first[sl],second[sl]
            a,b=probability(ax,1),probability(bx,1)
            pred_a,pred_b=a.argmax(1),b.argmax(1)
            calibrated_a=probability(ax,temps[0]).max(1)
            calibrated_b=probability(bx,temps[1]).max(1)
            ys=labels[sl]
            for direction,z in enumerate((ax,bx)):
                for version,temperature in enumerate((1.,temps[direction])):
                    u=z.astype(np.float64)/temperature
                    u-=u.max(axis=1,keepdims=True)
                    losses[direction,version]+=(np.log(np.exp(u).sum(axis=1))-u[np.arange(len(ys)),ys]).sum()
            raw=np.where(score_first[sl]>=score_second[sl],pred_a,pred_b)
            scaled=np.where(calibrated_a>=calibrated_b,pred_a,pred_b)
            writer.writerows(zip(test[sl],labels[sl],pred_a,pred_b,score_first[sl],score_second[sl],
                                 calibrated_a,calibrated_b,raw,scaled,(a+b).argmax(1)))
    diagnostics={arm:{"temperature":temps[i],"raw_nll_nats":losses[i,0]/len(test),
                      "scaled_nll_nats":losses[i,1]/len(test)}
                 for i,arm in enumerate(("first","second"))}
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    print(f"Temperatures {temps[0]:.9f} and {temps[1]:.9f}. Wrote {len(test)} rows.")


if __name__=="__main__":
    main()
