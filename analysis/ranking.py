"""Recompute full-cohort bootstrap ranking, transfer and MSP policies from paired predictions.
Writes one machine-readable JSON result to stdout. NumPy 2.x and Python 3.11+.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

NBOOT = 2000
SEED = 20261121
RANK = ("vertical_only", "horizontal_only", "balanced")
POLICIES = ("vertical", "horizontal", "higher_msp", "oracle", "difference_vs_vertical")
TRANSFER = ("threshold", "source_coverage", "target_coverage", "source_risk", "target_risk",
            "newly_wrong_fraction", "still_accepted_fraction")



def rate(n, d):
    return n / d if d else np.nan


def interval(x):
    finite = np.asarray(x)[np.isfinite(x)]
    return np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) >= .99 * NBOOT else None


def shuffle_mean(label, mask, av, bv):
    expected, singleton = 0., 0
    for lab in np.unique(label[mask]):
        take = mask & (label == lab)
        a, b = av[take], bv[take]
        diff = a[:, None] - b[None, :]
        expected += ((diff > 0).sum() + .5*(diff == 0).sum()) / len(a)
        singleton += int(len(a) == 1)
    return {"expected_rate": rate(expected, mask.sum()), "singleton_pairs": singleton}


def trans(cs, ct, a, b, cal_source):
    cut = np.quantile(cal_source, 1-.70, method="higher")
    src, dst = a >= cut, b >= cut
    initial = src & cs
    new = initial & ~ct
    counts = (len(a), int(src.sum()), int(dst.sum()), int((src & ~cs).sum()), int((dst & ~ct).sum()),
              int(initial.sum()), int(new.sum()), int((new & dst).sum()))
    n, ss, tt, se, te, ini, nn, still = counts
    return (cut, ss/n, tt/n, rate(se, ss), rate(te, tt), rate(nn, ini), rate(still, nn)), counts


def frac3(a):
    n1, n2, win1, win2 = a[:4]
    r1, r2 = rate(win1, n1), rate(win2, n2)
    return r1, r2, (r1+r2)/2


def run_score(pairs, key):
    cal = pairs["split"] == "calibration"
    test = ~cal
    yy = pairs["label"]
    assert (np.unique(yy[test], return_counts=True)[1] == 40).all()
    assert (np.unique(yy[cal], return_counts=True)[1] == 10).all()
    assert cal.sum() == 9000 and test.sum() == 36000
    a, b = pairs[key+"_v"][test], pairs[key+"_h"][test]
    ac, bc = pairs[key+"_v"][cal], pairs[key+"_h"][cal]
    cv = pairs["pred_v"][test] == yy[test]
    ch = pairs["pred_h"][test] == yy[test]
    vm, hm = cv & ~ch, ch & ~cv
    wv, wh = (a > b) + .5*(a == b), (b > a) + .5*(a == b)
    chosen_v = pairs["msp_v"][test] >= pairs["msp_h"][test]
    chooser = np.where(chosen_v, cv, ch)
    rescue, harm = (~cv & chooser), (cv & ~chooser)
    policy = np.column_stack([cv, ch, chooser, cv | ch, rescue.astype(float)-harm])
    sufficient = np.column_stack([vm, hm, vm*wv, hm*wh])
    point = frac3(sufficient.sum(axis=0))
    counts = {"both_right": int((cv & ch).sum()), "vertical_only": int(vm.sum()),
              "horizontal_only": int(hm.sum()), "both_wrong": int((~cv & ~ch).sum())}
    ranking = {}
    for name, mask, higher in (("vertical_only", vm, a>b), ("horizontal_only", hm, b>a)):
        ranking[name] = {"wins": int((mask & higher).sum()), "ties": int((mask & (a==b)).sum()), "n": int(mask.sum())}
    ranking["balanced"] = {}
    nullv = shuffle_mean(yy[test], vm, a, b)
    nullh = shuffle_mean(yy[test], hm, b, a)
    control = {"vertical_only": nullv, "horizontal_only": nullh,
               "balanced": (nullv["expected_rate"]+nullh["expected_rate"])/2}
    controls = ((cv, ch, a, b, ac), (ch, cv, b, a, bc))
    rs = np.random.default_rng(SEED)
    rc = np.random.default_rng(SEED+1)
    rg = np.random.default_rng(SEED)
    starts = np.arange(900)[:, None]
    rank_draws = np.empty((NBOOT,3))
    cl_draws = np.empty_like(rank_draws)
    policy_draws = np.empty((NBOOT,5))
    tr_draws = np.empty((NBOOT,2,7))
    class_sums = sufficient.reshape(900,40,4).sum(1)
    for i in range(NBOOT):
        ti = (40*starts + rs.integers(40,size=(900,40))).ravel()
        ci = (10*starts + rc.integers(10,size=(900,10))).ravel()
        rank_draws[i] = frac3(sufficient[ti].sum(0))
        cl_draws[i] = frac3(class_sums[rg.integers(900,size=900)].sum(0))
        if key == "msp":
            policy_draws[i] = policy[ti].mean(0)
        for j,(su,tu,aa,bb,cc) in enumerate(controls):
            tr_draws[i,j], _ = trans(su[ti],tu[ti],aa[ti],bb[ti],cc[ci])
    for j,name in enumerate(RANK):
        ranking[name].update(rate=point[j], ci95=interval(rank_draws[:,j]), class_ci95=interval(cl_draws[:,j]))
    selection = {}
    if key == "msp":
        for j,name in enumerate(POLICIES):
            selection[name] = {"correct_or_net_count": int(policy[:,j].sum()), "rate": policy[:,j].mean(),
                               "ci95": interval(policy_draws[:,j])}
        selection.update(rescues=int(rescue.sum()), harms=int(harm.sum()), tie_to_vertical=int((a==b).sum()))
    transfer = {}
    for j,name in enumerate(("V_to_H","H_to_V")):
        values, ns = trans(*controls[j])
        entry = {k: {"value": val, "ci95": interval(tr_draws[:,j,t])} for t,(k,val) in enumerate(zip(TRANSFER,values))}
        entry["counts"] = dict(zip(("n","source_accepted","target_accepted","source_errors","target_errors",
                                    "initial_correct_accepted","newly_wrong","still_accepted"),ns))
        entry["calibration_accepted"] = int((controls[j][-1] >= values[0]).sum())
        transfer[name] = entry
    return {"ranking": ranking, "shuffle": control, "transfer": transfer}, selection, counts


def defined_values(value):
    if isinstance(value, dict):
        return {k: defined_values(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [defined_values(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", choices=("events30k", "time20"))
    parser.add_argument("model", choices=("histogram", "timestamp"))
    parser.add_argument("score", choices=("msp", "margin"))
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "data" / f"full_{args.crop}_{args.model}_pairs.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: (int(r["label"]), hashlib.sha256(("split/"+r["source_id"]).encode()).digest()))
    arrays = {}
    for key in ("split", "label", "pred_v", "pred_h", "msp_v", "msp_h", "margin_v", "margin_h"):
        dtype = str if key == "split" else int if key in ("label", "pred_v", "pred_h") else float
        arrays[key] = np.array([r[key] for r in rows], dtype=dtype)
    score, selection, correctness = run_score(arrays, args.score)
    result = dict(score=score, selection=selection, joint_correctness=correctness)
    print(json.dumps(defined_values(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
