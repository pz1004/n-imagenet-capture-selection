"""Independent full-class checks of fusion fits, test decisions and bootstrap counts."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/"results"
METHODS=["first","local_selector","arithmetic_average","product","stack"]
BASES={"within":20269127,"classes":20269128}
from inputs import capture, cohort


def read_inputs(logit_dir, split):
    ids, labels = cohort()[split]
    inputs = []
    for view in ("variant1", "variant2"):
        z, order, _, _, _, _ = capture(logit_dir, split, view, ids, labels)
        x = z[order].astype(np.float64)
        x -= logsumexp(x, axis=1, keepdims=True)
        inputs.append(x)
    return ids, labels, inputs


def coefficients(theta,xs,y,counts):
    a,b=xs
    m=np.minimum(a,b)
    weights=counts/counts.sum()
    z=theta[0]*a+theta[1]*b+theta[2]*m
    unorm=z-logsumexp(z,axis=1,keepdims=True)
    p=np.exp(unorm)
    idx=np.arange(len(y))
    grad=[]
    for val in (a,b,m):
        grad.append(weights @ ((p*val).sum(axis=1)-val[idx,y]))
    grad=np.array(grad)+.01*theta[:3]
    gt=[]
    for c,val in zip(theta[3:],(a,b)):
        zz=c*val
        pr=np.exp(zz-logsumexp(zz,axis=1,keepdims=True))
        gt.append(weights @ ((pr*val).sum(axis=1)-val[idx,y]))
    return np.array([*grad,*gt])


def local(xs,y,ci):
    a,b=xs
    sca=np.exp(a.max(axis=1))-np.exp(b.max(axis=1))
    ok1=a.argmax(axis=1)==y
    ok2=b.argmax(axis=1)==y
    chosen=ci[ok1[ci]!=ok2[ci]]
    d=sca[chosen]
    center,scale=d.mean(),d.std()
    z=(d-center)/scale
    y0=ok1[chosen].astype(float)
    def cost(ab):
        u=ab[0]+ab[1]*z
        err=expit(u)-y0
        return (np.logaddexp(0.,u).sum()-y0@u+.5*ab[1]**2,
                np.array([err.sum(),err@z+ab[1]]))
    sol=minimize(cost,[0.,0.],jac=True,method="BFGS",options={"gtol":1e-7})
    assert np.max(np.abs(sol.jac))<1e-4
    return center,scale,*sol.x


def get_pred(xs,theta,selector):
    a,b=xs
    p=a.argmax(axis=1)
    h=b.argmax(axis=1)
    centers,scale,intercept,slope=selector
    confidence=np.exp(a.max(axis=1))
    take=intercept+slope*((confidence-np.exp(b.max(axis=1))-centers)/scale)>=0
    out=np.empty((len(a),5),dtype=int)
    out[:,0]=p
    out[:,1]=np.where(take,p,h)
    for i in range(0,len(a),384):
        sl=slice(i,i+384)
        out[sl,2]=(np.exp(a[sl])+np.exp(b[sl])).argmax(axis=1)
        out[sl,3]=(theta[3]*a[sl]+theta[4]*b[sl]).argmax(axis=1)
        out[sl,4]=(theta[0]*a[sl]+theta[1]*b[sl]+theta[2]*np.minimum(a[sl],b[sl])).argmax(axis=1)
    return out,confidence


def quota(scores,ti):
    # Explicit sort, compared with production's partition and tie handling.
    priority=np.lexsort((np.arange(len(ti)),ti,scores[ti]))
    g=np.zeros(len(ti),dtype=bool)
    g[priority[:10800]]=True
    return g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=OUT)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    point = json.loads((args.results / "point.json").read_text())
    ids,yt,xt=read_inputs(args.logits, "test")
    cal_ids,yc,xc=read_inputs(args.logits, "calibration")
    assert len(ids)==36000 and len(cal_ids)==9000 and not set(ids)&set(cal_ids)
    theta=np.r_[point["stack_weights"],1/np.array(point["temperatures"])]
    theta_grad=np.max(np.abs(coefficients(theta,xc,yc,np.ones(len(yc)))))
    assert theta_grad<1e-8
    pred,score=get_pred(xt,theta,local(xc,yc,np.arange(len(yc))))
    g=quota(score,np.arange(len(yt)))
    with gzip.open(ROOT/"data/trajectory_baseline_decisions.csv.gz", "rt") as fh:
        by_id = {r["source_id"]:r for r in csv.DictReader(fh)
                 if (r["pair"],r["window"],r["model"])==("new","events30k","histogram")}
    with gzip.open(ROOT/"extensions/scan_order/results/VH_test.csv.gz", "rt") as fh:
        existing={r["source_id"]:r for r in csv.DictReader(fh)}
    with gzip.open(args.results/"per_source.csv.gz","rt") as fh:
        per=list(csv.DictReader(fh))
    assert [r["source_id"] for r in per]==ids and len(by_id)==len(existing)==len(ids)
    for i,r in enumerate(per):
        assert int(r["label"])==yt[i] and int(r["request_30"])==int(g[i])==int(existing[ids[i]]["msp_request"])
        assert np.array_equal([int(r[k]) for k in ("vertical",*METHODS[1:])],pred[i])
        assert int(r["arithmetic_average"])==int(by_id[ids[i]]["probability_average"])
        assert int(r["local_selector"])==int(existing[ids[i]]["terminal_class"])
    h=xt[1].argmax(axis=1)
    for b,mask in (("100",np.ones(len(yt),dtype=bool)),("30",g)):
        for j,name in enumerate(METHODS):
            use=mask if name!="first" else np.zeros(len(yt),dtype=bool)
            final=np.where(use,pred[:,j],pred[:,0])==yt
            fv=pred[:,0]==yt
            either=fv|(h==yt)
            s=point["results"][b][name]
            for key,val in dict(correct=final.sum(),requests=use.sum(),rescues=(final&~fv).sum(),
                                losses=(~final&fv).sum(),both_wrong_rescued=(final&~either).sum(),
                                either_correct_missed=(~final&either).sum(),
                                either_correct_missed_on_requests=(~final&either&use).sum()).items():
                assert s[key]==val
    checked={}
    budget=json.loads((args.results/"budget_sensitivity.json").read_text())["methods"]
    for j,name in enumerate(METHODS[1:],start=1):
        final=pred[:,j]==yt
        first=pred[:,0]==yt
        missed_gate=~g
        assert missed_gate.sum()==25200
        local_gains=(missed_gate&~first&final).sum()
        local_losses=(missed_gate&first&~final).sum()
        observed=budget[name]
        assert observed["rescues_if_rescanned_on_other_sources"]==local_gains
        assert observed["losses_if_rescanned_on_other_sources"]==local_losses
        assert observed["net_correct_30_minus_100"]==local_losses-local_gains
        assert observed["difference_pp"]==(local_losses-local_gains)/360.
    for design,seed in BASES.items():
        arr=np.load(args.results/f"draws_{design}.npy")
        assert arr.shape==(2000,17) and np.isfinite(arr).all() and arr[:,5].max()<1e-8
        intervals=json.loads((args.results/f"intervals_{design}.json").read_text())["intervals"]
        acc=arr[:,7:].reshape(-1,2,5)/360.
        for j,name in enumerate(METHODS[1:],start=1):
            difference=np.sort(acc[:,1,j]-acc[:,0,j])
            end=np.interp([.025*(len(difference)-1),.975*(len(difference)-1)],np.arange(len(difference)),difference)
            np.testing.assert_allclose(budget[name]["paired_ci95"][design],end,atol=1e-12,rtol=0)
        for bi,b in enumerate(("100","30")):
            for term,ci0 in intervals[b].items():
                a,*r=term.split("_minus_")
                v=acc[:,bi,METHODS.index(a)]
                if r:
                    v=v-acc[:,bi,METHODS.index(r[0])]
                np.testing.assert_allclose(np.percentile(v,[2.5,97.5]),ci0,rtol=0,atol=1e-12)
        checked[design]={}
        for ii in (0,991,1999):
            rng=np.random.default_rng([seed,ii])
            if design=="within":
                ci=(10*np.arange(900)[:,None]+rng.integers(10,size=(900,10))).ravel()
                ti=(40*np.arange(900)[:,None]+rng.integers(40,size=(900,40))).ravel()
            else:
                cl=rng.integers(900,size=(900,1))
                ci=(10*cl+np.arange(10)).ravel()
                ti=(40*cl+np.arange(40)).ravel()
            th=arr[ii,:5]
            residual=np.max(np.abs(coefficients(th,xc,yc,np.bincount(ci,minlength=len(yc)))))
            assert residual<1.01e-8
            pr,sc=get_pred(xt,th,local(xc,yc,ci))
            ok=(pr[ti]==yt[ti,None])
            requested=quota(sc,ti)
            numbers=np.r_[ok.sum(axis=0),np.where(requested[:,None],ok,ok[:,:1]).sum(axis=0)]
            assert np.array_equal(numbers,arr[ii,7:])
            checked[design][ii]=float(residual)
    receipt={"independent_point_check_sources":len(ids),"full_class_logit_check":1000,
             "archive_baseline_comparisons":["local selector","arithmetic average","request_30"],
             "point_gradient_max":float(theta_grad),"full_evaluation_brute_force_bootstrap_checks":checked,
             "verified_post_hoc_budget_sensitivity":True,
             "intervals_recomputed_draws_per_design":2000,"bootstrap_optimality_tolerance":1e-8,
             "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
             "compiler":subprocess.check_output(["g++","--version"],text=True).splitlines()[0],
             "protocol_sha256":point["protocol_sha256"]}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    print("Verified all independent point and interval checks.",flush=True)


if __name__=="__main__":
    main()
