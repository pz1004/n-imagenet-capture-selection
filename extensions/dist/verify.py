"""Independent source joins, crop comparison, SciPy terminal fits, and decision counts."""
import csv
import gzip
import hashlib
import json
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/"results"
INPUTS=HERE/"inputs"

def rows(path,key="source_id"):
    op=gzip.open if path.suffix==".gz" else open
    with op(path,"rt",newline="") as f:
        data=list(csv.DictReader(f))
    records={r[key]:r for r in data}
    assert len(data)==len(records)
    return records

def filehash(file):
    h=hashlib.sha256()
    with file.open("rb") as fh:
        while b:=fh.read(8*1024**2):
            h.update(b)
    return h.hexdigest()

def solve(x,truth,w):
    use=(truth!=w)
    x=x[use]
    yy=truth[use]
    center=x.mean()
    scale=x.std()
    z=(x-center)/scale
    def objective(theta):
        a,b=theta
        eta=a+b*z
        residual=expit(eta)-yy
        return np.logaddexp(0,eta).sum()-eta@yy+.5*b*b, np.array([residual.sum(),residual@z+b])
    res=minimize(objective,(0.,0.),method="BFGS",jac=True,options={"gtol":1e-7})
    assert np.linalg.norm(res.jac) < 2e-4
    return res.x, center,scale

def verify_sources():
    provenance=json.loads((HERE/"provenance.json").read_text())
    manifest={k:r for k,r in rows(ROOT/"data/source_index.csv").items() if r["cohort"]=="full"}
    assert filehash(ROOT/"data/source_index.csv")==provenance["source_index_sha256"]
    assert filehash(ROOT/"data/class_map.csv")==provenance["class_map_sha256"]
    mapp={k:int(v["label"]) for k,v in manifest.items()}
    test=sorted(k for k in manifest if manifest[k]["split"]=="test")
    cal=sorted((k for k in manifest if manifest[k]["split"]=="calibration"),
               key=lambda k:(mapp[k],hashlib.sha256(("split/"+k).encode()).digest()[:8],k))
    source={}
    for path in ("original","vertical","horizontal","variant1","variant2"):
        path_to_table=INPUTS/(path+".csv.gz")
        saved=rows(path_to_table)
        source[path]=saved
        assert len(saved)==(50000 if path=="original" else 45000)
        assert all(saved[k]["split"]==manifest[k]["split"] and int(saved[k]["label"])==mapp[k] for k in manifest)
        with gzip.open(path_to_table,"rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==provenance["inference"][path]["sha256"]["sources.csv"]
    cases=[r for v in provenance["calibration_parity"].values() for r in v]
    assert len(cases)==21 and all(r["input_max_abs"]==0 and r["cpu_logit_max_abs"]==0 for r in cases)
    rk=rows(OUT/"ranking_decisions.csv.gz")
    p=json.loads((OUT/"ranking.json").read_text())
    c=Counter()
    y=np.array([mapp[k] for k in test])
    for k in test:
        v=source["vertical"][k]
        h=source["horizontal"][k]
        vy=int(v["hash_class"])
        hy=int(h["hash_class"])
        sv,sh=float(v["hash_msp"]),float(h["hash_msp"])
        selected=vy if sv>=sh else hy
        assert int(rk[k]["raw_class"])==selected
        true=mapp[k]
        c["rescues"]+=selected==true and vy!=true
        c["losses"]+=selected!=true and vy==true
        for n,w,a,b,ok,other in (("vertical","vertical_wins",sv,sh,vy==true,hy==true),
                                  ("horizontal","horizontal_wins",sh,sv,hy==true,vy==true)):
            if ok and not other:
                c["only_"+n+"_n"]+=1
                c[w]+=a>b
                c[n+"_ties"]+=a==b
    assert all(int(p[n])==v for n,v in c.items())
    for n in ("vertical","horizontal"):
        assert np.isclose(p[n[0]+"_only_rank"],100*(c[n+"_wins"]+.5*c[n+"_ties"])/c["only_"+n+"_n"])
    assert np.isclose(p["gain_pp"],100*(c["rescues"]-c["losses"])/len(test))
    original_correct=sum(int(r["hash_class"])==int(r["label"]) for r in source["original"].values())
    assert original_correct==provenance["inference"]["original"]["gate"]["correct"]
    def get(ids,path,crop):
        d=source[path]
        pred=np.array([int(d[k][crop+"_class"]) for k in ids])
        msp=np.array([float(d[k][crop+"_msp"]) for k in ids])
        complete=np.array([int(d[k][crop+"_stop"])-int(d[k][crop+"_start"])==30000 for k in ids])
        return pred,msp,complete
    for n in (4500,9000):
        policy=cal if n==9000 else [k for i,k in enumerate(cal) if i%10>=5]
        ys=np.array([mapp[k] for k in policy])
        p1,s1,_=get(policy,"variant1","first")
        t1,t1s,_=get(test,"variant1","first")
        cutoff=np.sort(s1)[int(np.ceil(.30*(n-1)))]
        gate=t1s<cutoff
        point=json.loads((OUT/("chrono_"+str(n)+".json")).read_text())
        assert int(gate.sum())==point["detail"]["cutoff"]["requests"]
        decisions=rows(OUT/("chrono_decisions_"+str(n)+".csv.gz"))
        accuracy={}
        for key,path,crop in (("continue","variant1","second"),("horizontal","variant2","first"),("square","original","first")):
            p2,s2,e2=get(policy,path,crop)
            (a,b),center,scale=solve((s1-s2)[e2],(p1==ys)[e2],(p2==ys)[e2])
            t2,t2s,el=get(test,path,crop)
            terminal=np.where(el & (a+b*((t1s-t2s)-center)/scale<0),t2,t1)
            assert np.array_equal(terminal,[int(decisions[k][key+"_terminal"]) for k in test])
            picked=np.where(gate,terminal,t1)
            actual=point["detail"]["cutoff"]["arms"][key]
            assert int(((t1!=y)&(picked==y)).sum())==actual["rescues"]
            assert int(((t1==y)&(picked!=y)).sum())==actual["losses"]
            assert int((picked==y).sum())==actual["correct"]
            accuracy[key]=100*(picked==y).mean()
        for a,b in (("horizontal","continue"),("square","continue"),("square","horizontal")):
            assert np.isclose(accuracy[a]-accuracy[b],point["point"]["cutoff."+a+"-"+b])
    print(f"{original_correct}/50,000 original recordings, {len(test):,} paired test sources, "
          "and all 216,000 terminal decisions agree with the release.")


def compare_json(expected, actual):
    if isinstance(expected,dict):
        if set(expected)!=set(actual):
            raise ValueError("Metric keys differ")
        for k in expected:
            compare_json(expected[k], actual[k])
    else:
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)


def verify_replay(directory, points_only):
    for path in sorted(OUT.iterdir()):
        if points_only and (path.stem.endswith("_within") or path.stem.endswith("_class")):
            continue
        actual=directory/path.name
        if path.suffix==".json":
            compare_json(json.loads(path.read_text()),json.loads(actual.read_text()))
        else:
            with gzip.open(path,"rb") as a, gzip.open(actual,"rb") as b:
                assert hashlib.file_digest(a,"sha256").digest()==hashlib.file_digest(b,"sha256").digest(),path.name
    print("Replayed decision rows and numerical estimates agree with published results.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replayed",type=Path)
    parser.add_argument("--points-only",action="store_true",help="Compare a --draws 0 replay.")
    args=parser.parse_args()
    verify_sources()
    if args.replayed:
        verify_replay(args.replayed,args.points_only)
