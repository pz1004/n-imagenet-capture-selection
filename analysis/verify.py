"""Verify the released measurements, study estimates and bootstrap intervals."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables"


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def column(rows, key, dtype=float):
    return np.asarray([r[key] for r in rows], dtype=dtype)


def close(observed, expected, tolerance=1e-10):
    np.testing.assert_allclose(observed, np.asarray(expected, dtype=float),
                               rtol=0, atol=tolerance)


def check_manifest():
    for row in read(ROOT / "data_manifest.csv"):
        path = ROOT / row["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row["sha256"] or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"File integrity check failed for {path.name}")


def check_orientation(rows, split, cv, ch, windows, features, crop, model, summary):
    ids = [r["source_id"] for r in rows]
    dur = lambda w, d: (int(w["last_used_us_"+d]) - int(w["first_used_us_"+d])) / 1000
    x = np.asarray([
        [float(r["msp_v"]) - float(r["msp_h"]),
         np.log(int(windows[s]["raw_events_v"])/int(windows[s]["raw_events_h"])),
         dur(windows[s], "v") - dur(windows[s], "h"),
         float(features[s]["orientation"])] for s, r in zip(ids, rows)
    ])
    discordant = cv != ch
    cal = (split == "calibration") & discordant
    test = (split == "test") & discordant
    y = cv.astype(float)
    losses = []
    for cols in (3, 4):
        center = x[cal, :cols].mean(axis=0)
        sd = x[cal, :cols].std(axis=0)
        sd[sd <= 1e-12] = 1
        z = np.column_stack((np.ones(len(y)), (x[:, :cols]-center)/sd))
        a = z[cal]
        penalty = np.eye(cols+1)
        penalty[0, 0] = 0
        coef = np.zeros(cols+1)
        for _ in range(100):
            p = 1 / (1+np.exp(-(a@coef)))
            gradient = a.T@(p-y[cal]) + penalty@coef
            hessian = a.T@((p*(1-p))[:, None]*a) + penalty
            step = np.linalg.solve(hessian, gradient)
            coef -= step
            if np.max(np.abs(step)) < 1e-11:
                break
        else:
            raise RuntimeError("Orientation fit did not converge")
        eta = z[test]@coef
        losses.append(np.logaddexp(0, eta)-eta*y[test])
    test_outcomes = cv[test]
    for prefix, mask in [("",np.ones(test.sum(),bool)),("v_only_",test_outcomes),("h_only_",~test_outcomes)]:
        for metric, values in zip(("loss_baseline","loss_augmented","loss_reduction"),
                                  (losses[0],losses[1],losses[0]-losses[1])):
            key = (crop, model, prefix+metric)
            close(values[mask].mean(), summary[key]["estimate"], 2e-8)
    return int(test.sum())


def rank_values(v, h, a, b):
    v_only, h_only = v & ~h, h & ~v
    tie = a == b
    res = []
    for mask, wins in ((v_only,a>b),(h_only,b>a)):
        n, win, ties = int(mask.sum()), int((mask&wins).sum()), int((mask&tie).sum())
        res.append((n, win, ties, (win+0.5*ties)/n))
    return res


def check_transfer(a, b, cv, ch, split, expected):
    cal, test = split=="calibration", split=="test"
    cut = np.quantile(a[cal], 1-0.70, method="higher")
    av, bv, source_ok, target_ok = a[test]>=cut, b[test]>=cut, cv[test], ch[test]
    initial = av & source_ok
    new = initial & ~target_ok
    stats = {"source_coverage":(av.sum(), test.sum()), "target_coverage":(bv.sum(), test.sum()),
             "source_risk":((av & ~source_ok).sum(), av.sum()),
             "target_risk":((bv & ~target_ok).sum(), bv.sum()),
             "newly_wrong_fraction":(new.sum(),initial.sum()),
             "still_accepted_fraction":((new & bv).sum(),new.sum())}
    close(cut, expected["threshold"]["estimate"])
    assert int((a[cal]>=cut).sum()) == int(expected["threshold"]["calibration_accepted"])
    for metric,(n,d) in stats.items():
        row = expected[metric]
        assert (int(n),int(d)) == (int(row["numerator"]),int(row["denominator"]))
        close(n/d, row["estimate"])


def primary_bootstrap(rows, fusion_rows, expected_ranks, selection, fusion):
    def order(row):
        return int(row["label"]), hashlib.sha256(("split/"+row["source_id"]).encode()).digest()
    rows = sorted((r for r in rows if r["split"]=="test"), key=order)
    raw = {r["source_id"]:r for r in fusion_rows}
    y = column(rows,"label",int)
    v, h = column(rows,"correct_v",int).astype(bool), column(rows,"correct_h",int).astype(bool)
    a, b = column(rows,"msp_v"), column(rows,"msp_h")
    choice = np.where(a>=b, v, h)
    fpred = np.array([int(raw[r["source_id"]]["averaged_probability_class"]) for r in rows])
    dif = (fpred==y).astype(int)-choice.astype(int)
    draws = np.empty((2000,5))
    rng = np.random.default_rng(20261121)
    starts = np.arange(900)[:,None]*40
    for i in range(len(draws)):
        ix = (starts+rng.integers(40,size=(900,40))).ravel()
        out = rank_values(v[ix],h[ix],a[ix],b[ix])
        rv,rh=out[0][3],out[1][3]
        draws[i] = (rv, rh, (rv+rh)/2, choice[ix].mean()-v[ix].mean(), dif[ix].mean()*100)
    endpoints = np.quantile(draws,[.025,.975],axis=0)
    for j,group in enumerate(("vertical_only","horizontal_only","balanced")):
        row=expected_ranks[("full","events30k","histogram","msp",group)]
        close(endpoints[:,j],[row["ci95_lower"],row["ci95_upper"]])
    row=selection[("events30k","histogram","difference_vs_vertical")]
    close(endpoints[:,3],[row["ci95_lower"],row["ci95_upper"]])
    close(endpoints[:,4],fusion["ci95_pp"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",action="store_true",help="also replay 2,000 primary full-cohort and fusion bootstrap draws")
    args=parser.parse_args()
    check_manifest()
    def make_index(table, cols):
        return {tuple(r[c] for c in cols):r for r in read(TABLES/(table+".csv"))}
    ranking=make_index("ranking",("cohort","crop","model","score","subgroup"))
    selection=make_index("selection",("crop","model","policy"))
    transfer=make_index("transfer",("cohort","crop","model","score","direction","metric"))
    orientation=make_index("orientation",("crop","model","metric"))
    joint=make_index("joint_correctness",("cohort","crop","model"))
    index={r["source_id"]:r for r in read(DATA/"source_index.csv")}
    features={r["source_id"]:r for r in read(DATA/"whole_source_features.csv")}
    assert len(index)==50000 and len(features)==45000
    original=read(DATA/"original_sanity_predictions.csv")
    for r in read(TABLES/"original_sanity.csv"):
        subset=[t for t in original if t["model_set"]==r["model_set"]]
        n=sum(t[r["model"]+"_pred"]==t["label"] for t in subset)
        assert len(subset)==int(r["n"]) and n==int(r["correct"])
    for cohort in ("mini","full"):
        for crop in ("events30k","time20"):
            windows={r["source_id"]:r for r in read(DATA/f"{cohort}_{crop}_windows.csv")}
            for model in ("histogram","timestamp"):
                rows=read(DATA/f"{cohort}_{crop}_{model}_pairs.csv")
                ids={r["source_id"] for r in rows}
                assert len(ids)==len(rows) and ids==windows.keys()
                for r in rows:
                    ix=index[r["source_id"]]
                    assert (ix["cohort"],ix["label"],ix["split"])==(cohort,r["label"],r["split"])
                split=column(rows,"split",str)
                test=split=="test"
                assert set(Counter(column(rows,"label",int)[test]).values())=={40}
                assert set(Counter(column(rows,"label",int)[~test]).values())=={10}
                label=column(rows,"label",int)
                cv,ch=column(rows,"pred_v",int)==label,column(rows,"pred_h",int)==label
                close(cv,column(rows,"correct_v",int)); close(ch,column(rows,"correct_h",int))
                v,h=cv[test],ch[test]
                counts=joint[(cohort,crop,model)]
                for key,mask in (("both_right",v&h),("vertical_only",v&~h),("horizontal_only",h&~v),("both_wrong",~v&~h)):
                    assert int(mask.sum())==int(counts[key])
                for score in ("msp","margin"):
                    a,b=column(rows,score+"_v"),column(rows,score+"_h")
                    rates=rank_values(v,h,a[test],b[test])
                    for j,key in enumerate(("vertical_only","horizontal_only","balanced")):
                        r=ranking[(cohort,crop,model,score,key)]
                        if j<2:
                            n,w,t,value=rates[j]
                            assert (n,w,t)==tuple(int(r[k]) for k in ("n","wins","ties"))
                        else:
                            value=(rates[0][3]+rates[1][3])/2
                        close(value,r["estimate"])
                    for name,x,z,u,w in (("V_to_H",a,b,cv,ch),("H_to_V",b,a,ch,cv)):
                        expected={k[-1]:r for k,r in transfer.items() if k[:-1]==(cohort,crop,model,score,name)}
                        check_transfer(x,z,u,w,split,expected)
                if cohort=="full":
                    selected=np.where(column(rows,"msp_v")[test]>=column(rows,"msp_h")[test],v,h)
                    saved=selection[(crop,model,"higher_msp")]
                    assert int((selected&~v).sum())==int(saved["rescues"])
                    assert int((~selected&v).sum())==int(saved["harms"])
                    for key,arr in (("vertical",v),("horizontal",h),("higher_msp",selected),("oracle",v|h),
                                    ("difference_vs_vertical",selected.astype(int)-v.astype(int))):
                        r=selection[(crop,model,key)]
                        assert arr.sum()==int(r["correct_or_net_count"])
                        close(arr.mean(),r["estimate"])
                    check_orientation(rows,split,cv,ch,windows,features,crop,model,orientation)
                    if crop=="events30k" and model=="histogram":
                        primary=rows
    draws=read(TABLES/"orientation_bootstrap_draws.csv")
    for (crop,model,metric),row in orientation.items():
        values=[float(r[metric]) for r in draws if r["crop"]==crop and r["model"]==model]
        assert len(values)==2000
        close(np.quantile(values,[.025,.975]),[row["ci95_lower"],row["ci95_upper"]])
    fusion_rows=read(DATA/"fusion.csv")
    fusion=json.loads((TABLES/"fusion.json").read_text())
    y=column(fusion_rows,"label",int)
    base=column(fusion_rows,"msp_selected_class",int)==y
    averaged=column(fusion_rows,"averaged_probability_class",int)==y
    assert (int(base.sum()),int(averaged.sum()),int((~base&averaged).sum()),int((base&~averaged).sum())) == (
        fusion["selector_correct"],fusion["fusion_correct"],fusion["gained"],fusion["lost"])
    if args.bootstrap:
        primary_bootstrap(primary,fusion_rows,ranking,selection,fusion)
    print("Verified derived-data hashes, original sanity checks, eight model/crop/cohort settings, "
          "all ranking and transfer counts/rates, four full-cohort selection policies and differences "
          "in each setting, four orientation model pairs, and exploratory fusion switch counts.")
    if args.bootstrap:
        print("Verified 2,000-draw primary subgroup/balanced ranking, policy gain and fusion intervals.")


if __name__=="__main__":
    main()

