"""Independent refits and replay of held-out masks, decision counts and paired intervals."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
from itertools import groupby
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import replay as analyzed

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
ARMS=analyzed.ARMS

def refit(delta,first,second,allowed):
    use=allowed & (first!=second)
    xx=delta[use]; yy=first[use].astype(float)
    mean,sd=xx.mean(),xx.std()
    z=(xx-mean)/sd
    def objective(b):
        eta=b[0]+b[1]*z
        err=expit(eta)-yy
        return (np.logaddexp(0.,eta)-yy*eta).sum()+.5*b[1]**2, np.array([err.sum(),err@z+b[1]])
    opt=minimize(objective,[0,0],jac=True,method="BFGS",options={"gtol":1e-8})
    assert np.linalg.norm(objective(opt.x)[1])<1e-4
    return lambda d:opt.x[0]+opt.x[1]*((d-mean)/sd)>=0

def recon(pred,score,eligible,y,ci,ti):
    select={}
    v=pred[:,0]
    vs=score[:,0]
    primary=[]
    interactions=[]
    for s in range(6):
        cv=v[s,ci]==y[ci]
        cut=np.sort(vs[s,ci])[math.ceil(.3*(len(ci)-1))]
        ask=vs[s,ti]<cut
        first=v[s,ti]
        gates={"cutoff":ask}
        if s==0: release=ask.copy()
        gates["released_mask"]=release
        seq=np.lexsort((ti,vs[s,ti]))
        quota=np.zeros(len(ti),bool);quota[seq[:10800]]=True;gates["quota"]=quota
        common=np.all(eligible[1:,ti],axis=0)
        gates["complete"]=ask&common
        state={"first":100*np.mean(first==y[ti])}
        for a,name in enumerate(ARMS,1):
            delta=vs[s]-score[s,a]
            keep=refit(delta[ci],cv,pred[s,a,ci]==y[ci],eligible[a,ci])
            terminal=np.where(eligible[a,ti]&~keep(delta[ti]),pred[s,a,ti],first)
            select[s,name]=(terminal,ask,quota,common)
            for gate,request in gates.items():
                final=(np.where(request,terminal,first)==y[ti])
                state[gate+"_"+name]=100*final[common if gate=="complete" else slice(None)].mean()
            replaced=np.where(eligible[a,ti],pred[s,a,ti],first)==y[ti]
            state["direct_all_"+name]=100*replaced.mean()
            state["standalone_complete_"+name]=100*(pred[s,a,ti][common]==y[ti][common]).mean()
        for g in (*gates,"direct_all","standalone_complete"):
            for left,right in analyzed.PAIRINGS:
                state[g+"_"+left+"_minus_"+right]=state[g+"_"+left]-state[g+"_"+right]
        primary.append([state[k] for k in analyzed.METRICS])
    raw=np.array(primary)
    mean=raw[1:].mean(axis=0)
    return select,np.vstack((raw,mean,mean-raw[0]))

def main():
    ids,y,pred,score,eligible=analyzed.load()
    report=json.loads((OUT/"paired.json").read_text())
    by=json.loads((OUT/"by_seed.json").read_text())
    assert report["draws_each"]==2000
    ti=np.arange(4500,int(len(ids)))
    ci=np.arange(4500)
    decisions,point=recon(pred,score,eligible,y,ci,ti)
    with gzip.open(OUT/"decisions.csv.gz","rt") as f:
        for ss,group in groupby(csv.DictReader(f),key=lambda r:int(r["seed"])):
            rows=list(group)
            assert [r["source_id"] for r in rows]==ids[4500:]
            for a,name in enumerate(ARMS,1):
                terminal,gate,quota,common=decisions[ss,name]
                first=pred[ss,0,ti]; final=np.where(gate,terminal,first)
                for k,expect in ((name+"_terminal",terminal),(name+"_policy",final),("request",gate),("quota_request",quota),("common_complete",common),(name+"_eligible",eligible[a,ti])):
                    np.testing.assert_array_equal([int(r[k]) for r in rows],expect)
                s=by[str(ss)]["gate"]["cutoff"]["arms"][name]
                assert (s["rescues"],s["losses"])==(np.sum((first!=y[ti])&(final==y[ti])),np.sum((first==y[ti])&(final!=y[ti])))
    for i,state in enumerate(report["states"]):
        expected=report["estimates"][state]["point"]
        np.testing.assert_allclose([expected[k] for k in analyzed.METRICS],point[i],atol=1e-10)
    for design in ("within","class"):
        matrix=np.load(OUT/f"bootstrap_{design}.npy")
        for seq in (0,250,1975):
            rng=np.random.default_rng(np.random.SeedSequence([20261128,int(design=="class"),seq]))
            c=np.arange(900)[:,None]
            if design=="within":
                ci=(c*5+rng.integers(5,size=(900,5))).ravel();t=(c*40+rng.integers(40,size=(900,40))).ravel()
            else:
                cl=rng.integers(900,size=900)[:,None];ci=(cl*5+np.arange(5)).ravel();t=(cl*40+np.arange(40)).ravel()
            _,ref=recon(pred,score,eligible,y,ci,t+4500)
            np.testing.assert_allclose(ref,matrix[seq],atol=2e-11)
        ci=np.quantile(matrix,[.025,.975],axis=0)
        for i,state in enumerate(report["states"]):
            for j,k in enumerate(report["metrics"]):
                np.testing.assert_allclose(ci[:,i,j],report["estimates"][state][design][k],atol=1e-12)
    print("Verified all 216,000 source-by-state decisions, six independent refits, and both 2,000-draw interval sets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    OUT = args.results
    main()
