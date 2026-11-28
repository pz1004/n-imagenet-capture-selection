"""Temperature-scaled product and class-shared stacking on frozen transfer logits."""
import argparse
import csv
import ctypes
import gzip
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
from inputs import capture, cohort, digest
sys.path.insert(0, str(ROOT / "analysis"))
from acquisition_inputs import write_json as json_write
from capture_policies import choose_vertical, fit_discordant

spec = importlib.util.spec_from_file_location("scan_order", ROOT / "extensions/scan_order/replay.py")
order = importlib.util.module_from_spec(spec)
spec.loader.exec_module(order)
quota = order.quota

PENALTY = .01
DESIGN_SEEDS = {"within": 20269127, "classes": 20269128}
NAMES = ("first", "local_selector", "arithmetic_average", "product", "stack")
BUDGETS = ("100", "30")
PAIRS = [(right,left) for right in (3,4) for left in (0,1,2)] + [(4,3)]
CAL = TEST = CENTER = CENTER_DERIV = SURFACES = None
KERNEL = None


def load_kernel():
    library = HERE / "likelihood.so"
    if not library.is_file():
        raise FileNotFoundError("Compile likelihood.cpp as described in extensions/fusion/README.md.")
    handle = ctypes.CDLL(str(library))
    double = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    kernel = handle.row_derivatives
    kernel.argtypes = [double, ctypes.c_int, ctypes.c_int, integer, integer, double, double,
                       ctypes.c_bool, ctypes.c_bool]
    kernel.restype = None
    return kernel


def load_block(logit_dir, split, ids, y):
    y = np.ascontiguousarray(y, dtype=np.int32)
    features = np.empty((len(y),3,1000), dtype=np.float64)
    scores = []
    tops = []
    hashes = {}
    for j, view in enumerate(("variant1","variant2")):
        z, row_order, predicted, msp, key, checksum = capture(logit_dir, split, view, ids, y)
        hashes[key] = checksum
        for start in range(0,len(y),512):
            sl = slice(start,start+512)
            raw = z[row_order[sl]].astype(np.float64)
            assert np.isfinite(raw).all()
            logp = raw-logsumexp(raw,axis=1)[:,None]
            assert np.array_equal(logp.argmax(axis=1),predicted[sl])
            assert np.allclose(np.exp(logp.max(axis=1)),msp[sl],rtol=1e-11,atol=1e-14)
            features[sl,j] = logp
        scores.append(msp)
        tops.append(predicted)
    np.minimum(features[:,0],features[:,1],out=features[:,2])
    return {"ids":ids,"y":y,"f":features,"tops":tops,"scores":scores,"hashes":hashes,
            "difference":scores[0]-scores[1]}


def derivatives(theta, counts, aggregate=True):
    n,_,k = CAL["f"].shape
    assert k==1000 and np.max(np.abs(theta))<10
    out = np.empty(13 if aggregate else (n,13))
    KERNEL(CAL["f"],n,k,CAL["y"],counts,theta,out,aggregate,True)
    assert np.isfinite(out).all()
    return out


def steps(theta, ave):
    grad = ave[:5].copy()
    grad[:3] += PENALTY*theta[:3]
    m = ave[[5,6,7,6,8,9,7,9,10]].reshape(3,3) + PENALTY*np.eye(3)
    return np.r_[np.linalg.solve(m,grad[:3]), grad[3:] / ave[11:]], np.max(np.abs(grad))


def fit(counts, warm=False):
    n = counts.sum()
    theta = np.array([.5,.5,0.,.7,.7]) if not warm else CENTER.copy()
    if warm:
        theta -= steps(theta,counts @ CENTER_DERIV/n)[0]
    for it in range(14):
        delta, residual = steps(theta,derivatives(theta,counts)/n)
        if residual < 1e-8:
            return theta, residual, it+1
        theta -= delta
    raise RuntimeError(f"Calibration Newton fit did not converge: gradient {residual}")


def local(cal_indices):
    a,b=CAL["tops"]
    return fit_discordant(CAL["difference"][cal_indices],
                          a[cal_indices]==CAL["y"][cal_indices],
                          b[cal_indices]==CAL["y"][cal_indices])


def full_predictions(theta):
    f=TEST["f"]
    v,h=TEST["tops"]
    result=np.empty((len(v),5),dtype=np.int32)
    result[:,0]=v
    result[:,1]=np.where(choose_vertical(TEST["difference"],local(np.arange(len(CAL["y"])))),v,h)
    for i in range(0,len(v),256):
        sl=slice(i,i+256)
        a,b,m=f[sl,0],f[sl,1],f[sl,2]
        result[sl,2]=(np.exp(a)+np.exp(b)).argmax(axis=1)
        result[sl,3]=(theta[3]*a+theta[4]*b).argmax(axis=1)
        result[sl,4]=(theta[0]*a+theta[1]*b+theta[2]*m).argmax(axis=1)
    return result


def correctness_screen(center, dimensions, radius):
    # A competitor can beat the true class in the coefficient box only when
    # its maximum possible score difference reaches zero. Every retained rival
    # is evaluated exactly in every bootstrap draw using the fitted weights.
    f=TEST["f"]
    y=TEST["y"]
    hard=[]; diffs=[];ties=[]
    always=np.ones(len(y),dtype=bool)
    for i in range(0,len(y),256):
        j=min(i+256,len(y))
        block=f[i:j][:,dimensions,:].copy()
        true=block[np.arange(j-i),:,y[i:j]]
        diff=block-true[:,:,None]
        d0=np.einsum("rck,c->rk",diff,center)
        spread=radius*np.abs(diff).sum(axis=1)
        worst=d0-spread
        best=d0+spread
        pred_column=np.arange(1000)[None,:]
        live=(best>=-1e-11)&(pred_column!=y[i:j,None])
        wrong=(worst>1e-11).any(axis=1)
        always[i:j]=~wrong
        candidates=live.any(axis=1)&~wrong
        for n in np.flatnonzero(candidates):
            pos=np.flatnonzero(live[n])
            hard.append(n+i)
            diffs.append(diff[n][:,pos].T)
            ties.extend(pos<y[n+i])
    sizes=[len(x) for x in diffs]
    return (always,np.asarray(hard),np.vstack(diffs).copy(order="C"),
            np.r_[0,np.cumsum(sizes)[:-1]],np.asarray(ties),center,radius)


def screened_correct(theta, surface):
    always,ids,features,starts,tie,center,radius=surface
    assert np.max(np.abs(theta-center))<=radius
    relative = features @ theta
    wins = (relative>0) | ((relative==0)&tie)
    correct=always.copy()
    correct[ids]=~np.logical_or.reduceat(wins,starts)
    return correct


def summarize(pred, mask):
    first,second=TEST["tops"]
    y=TEST["y"]
    out=np.where(mask,pred,first)
    ok=out==y
    f,h=first==y,second==y
    important={
        "n":len(y),"requests":mask.sum(),"correct":ok.sum(),"accuracy_percent":100*ok.mean(),
        "rescues":(~f&ok).sum(),"losses":(f&~ok).sum(),"both_wrong_rescued":(~f&~h&ok).sum(),
        "either_correct_missed":((f|h)&~ok).sum(),
        "either_correct_missed_on_requests":(mask&(f|h)&~ok).sum(),
        "either_correct_on_requests":(mask&(f|h)).sum(),
        "both_wrong_on_requests":(mask&~f&~h).sum(),
        "third_class_on_requests":(mask&(out!=first)&(out!=second)).sum()}
    return {k:float(v) if k=="accuracy_percent" else int(v) for k,v in important.items()}


def point():
    pred=full_predictions(CENTER)
    TEST["pred"]=pred
    y=TEST["y"]
    gate=quota(-TEST["scores"][0],.3)
    assert gate.sum()==10800
    gates=(np.ones(len(y),dtype=bool),gate)
    no_requests = np.zeros(len(y),dtype=bool)
    summaries={budget:{name:summarize(pred[:,j],no_requests if name=="first" else g) for j,name in enumerate(NAMES)}
               for budget,g in zip(BUDGETS,gates)}
    assert summaries["100"]["local_selector"]["correct"]==15923
    assert summaries["100"]["arithmetic_average"]["correct"]==15833
    assert summaries["30"]["local_selector"]["correct"]==15783
    with gzip.open(OUT/"per_source.csv.gz","wt",newline="") as f:
        wr=csv.writer(f)
        wr.writerow(["source_id","label","vertical","horizontal","first_confidence","second_confidence",
                     "request_30","local_selector","arithmetic_average","product","stack"])
        wr.writerows(zip(TEST["ids"],y,*TEST["tops"],*TEST["scores"],gate.astype(int),
                         *(pred[:,i] for i in range(1,5))))
    result={"model":"Event Histogram","window":"up to 30000 events","ridge":PENALTY,
            "cal_n":len(CAL["y"]),"test_n":len(y),"stack_weights":CENTER[:3].tolist(),
            "temperatures":(1/CENTER[3:]).tolist(),"logit_sha256":{**CAL["hashes"],**TEST["hashes"]},
            "source_manifest_sha256":digest(ROOT/"data/source_index.csv"),
            "protocol_sha256":digest(HERE/"PROTOCOL.md"),"results":summaries}
    json_write(OUT/"point.json",result)
    print("Point estimates",json.dumps(summaries), flush=True)


def bootstrap_draw(task):
    design, i=task
    rng=np.random.default_rng([DESIGN_SEEDS[design],i])
    classes=np.arange(900)[:,None]
    if design=="within":
        ci=(10*classes+rng.integers(10,size=(900,10))).ravel()
        ti=(40*classes+rng.integers(40,size=(900,40))).ravel()
    else:
        cl=rng.integers(900,size=(900,1))
        ci=(10*cl+np.arange(10)).ravel()
        ti=(40*cl+np.arange(40)).ravel()
    counts=np.bincount(ci,minlength=len(CAL["y"])).astype(np.int32)
    theta,res,iterations=fit(counts,warm=True)
    y=TEST["y"]
    v,h=TEST["tops"]
    ok=(TEST["pred"]==y[:,None]).copy()
    ok[:,1]=np.where(choose_vertical(TEST["difference"],local(ci)),v==y,h==y)
    for j,t in enumerate((theta[3:],theta[:3])):
        ok[:,j+3]=screened_correct(t,SURFACES[j])
    first=ok[ti,0]
    gate=quota(-TEST["scores"][0][ti],.3,ti)
    resampled=ok[ti]
    correct=np.array([resampled.sum(axis=0),
                      np.where(gate[:,None],resampled,first[:,None]).sum(axis=0)])
    return i, np.r_[theta,res,iterations,correct.ravel()]


def run(args):
    global CAL, TEST, CENTER, CENTER_DERIV, SURFACES
    global OUT, KERNEL
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    KERNEL = load_kernel()
    groups=cohort()
    CAL=load_block(args.logits, "calibration",*groups["calibration"])
    counts=np.ones(len(CAL["y"]),dtype=np.int32)
    CENTER,_,_=fit(counts)
    TEST=load_block(args.logits, "test",*groups["test"])
    point()
    if args.point_only:
        return
    CENTER_DERIV=derivatives(CENTER,counts,aggregate=False)
    SURFACES=[correctness_screen(CENTER[3:],(0,1),.25),
              correctness_screen(CENTER[:3],(0,1,2),.35)]
    for j,t in enumerate((CENTER[3:],CENTER[:3])):
        assert np.array_equal(screened_correct(t,SURFACES[j]),TEST["pred"][:,j+3]==TEST["y"])
    for design in DESIGN_SEEDS:
        output=OUT/f"draws_{design}.npy"
        if output.exists():
            saved=np.load(output)
            assert saved.shape==(args.draws,17)
        else:
            saved=np.full((args.draws,17),np.nan)
        tasks=[(design,i) for i,row in enumerate(saved) if np.isnan(row).any()]
        start=time.monotonic()
        with mp.get_context("fork").Pool(args.workers) as pool:
            for n,(i,row) in enumerate(pool.imap_unordered(bootstrap_draw,tasks,chunksize=1),1):
                saved[i]=row
                if n%50==0 or n==len(tasks):
                    tmp=output.with_suffix(".tmp.npy")
                    np.save(tmp,saved)
                    tmp.replace(output)
                    print(design,n,len(tasks),round(time.monotonic()-start,1),"s",flush=True)
        assert np.isfinite(saved).all()
        acc=saved[:,7:].reshape(-1,2,5)*100/len(TEST["y"])
        intervals={}
        for j,b in enumerate(BUDGETS):
            cols={name:acc[:,j,k] for k,name in enumerate(NAMES)}
            for a,k in PAIRS:
                cols[f"{NAMES[a]}_minus_{NAMES[k]}"]=acc[:,j,a]-acc[:,j,k]
            intervals[b]={name:np.quantile(x,[.025,.975]).tolist() for name,x in cols.items()}
        json_write(OUT/f"intervals_{design}.json",{"intervals":intervals,"max_gradient":float(saved[:,5].max()),
                                                  "replicates":len(saved),"seed":DESIGN_SEEDS[design]})


if __name__=="__main__":
    par=argparse.ArgumentParser()
    par.add_argument("--logits", type=Path, required=True, help="Extracted fusion-transfer-logits.zip directory")
    par.add_argument("--output", type=Path, required=True)
    par.add_argument("--workers",type=int,default=6)
    par.add_argument("--draws",type=int,default=2000)
    par.add_argument("--point-only",action="store_true")
    run(par.parse_args())
