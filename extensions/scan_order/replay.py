"""Matched first-path comparison and cross-fitted gain requests on the archived split."""
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
from acquisition_inputs import source_groups, write_json
from capture_policies import choose_vertical, fit_discordant

SOURCES = {"V": "variant1", "H": "variant2", "S": "original"}
ORDERS = ("SV", "VS", "VH", "HV")
FRACS = (0, .1, .3, .5, 1)
OUT = HERE / "results"
DRAWS = 2000


def load(path):
    with np.load(path, allow_pickle=False) as table:
        groups = source_groups(table)
        records = {
            split: {code: tuple(table[f"{split}_{code}_{k}"] for k in ("msp", "class", "duration_ms"))
                    for code in SOURCES}
            for split in groups
        }
    return groups, records

def block(data, ids, y, order):
    s1, a, span = data[order[0]]
    s2, b, _ = data[order[1]]
    return {"ids": ids, "x": s1 - s2, "a": a == y, "b": b == y, "p1": a, "p2": b,
            "X": np.column_stack([s1, np.log1p(span)]), "msp": s1}


def fit_terminal(row, idx):
    return fit_discordant(row["x"][idx], row["a"][idx], row["b"][idx])


def crossfit(row, idx):
    # Calibration sources are sorted by class and original split hash.
    # The original fold travels with every bootstrap copy.
    folds = idx % 5
    gain = np.empty(len(idx), dtype=np.int8)
    for f in range(5):
        hold = folds == f
        j = idx[hold]
        fit = fit_terminal(row, idx[~hold])
        take1 = choose_vertical(row["x"][j], fit)
        gain[hold] = np.where(take1, row["a"][j], row["b"][j]).astype(np.int8) - row["a"][j]
    return gain


def ridge(x, gain):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    z = (x-mean)/std
    square = z*z
    sqmean, sqstd = square.mean(axis=0), square.std(axis=0)
    design = np.column_stack([z,(square-sqmean)/sqstd])
    center = gain.mean()
    slope = np.linalg.solve(design.T @ design + np.eye(4), design.T @ (gain-center))
    return mean, std, sqmean, sqstd, center, slope


def predict(x, model):
    mean, std, sqmean, sqstd, center, slope = model
    z = (x-mean)/std
    return center + np.column_stack([z,(z*z-sqmean)/sqstd]) @ slope


def quota(score, frac, indices=None):
    """Higher request score gets priority; ties by original row then draw order."""
    k = round(frac*len(score))
    if k == 0 or k == len(score):
        return np.full(len(score), bool(k))
    boundary = np.partition(score, -k)[-k]
    mask = score > boundary
    tied = np.flatnonzero(score == boundary)
    if indices is not None and len(tied) > 1:
        tied = tied[np.argsort(indices[tied], kind="stable")]
    mask[tied[:k-mask.sum()]] = True
    return mask


def summary(base, delta, mask):
    n = len(base)
    applied = delta * mask
    expected = base.mean() + mask.mean()*delta.mean()
    return {"requested": int(mask.sum()), "rescued": int((applied == 1).sum()),
            "lost": int((applied == -1).sum()), "correct": int(base.sum()+applied.sum()),
            "accuracy": 100*(base.sum()+applied.sum())/n,
            "gain": 100*applied.mean(), "random_accuracy": 100*expected}


def point(order, c, t):
    ci = np.arange(len(c["a"]))
    f = fit_terminal(c, ci)
    test_class = np.where(choose_vertical(t["x"], f), t["p1"], t["p2"])
    terminal = np.where(choose_vertical(t["x"], f), t["a"], t["b"])
    delta = terminal.astype(np.int8) - t["a"]
    accept = t["b"].astype(np.int8) - t["a"]
    train_gain = crossfit(c, ci)
    gain_fit = ridge(c["X"], train_gain)
    scores = {"msp": -t["msp"], "expected": predict(t["X"], gain_fit)}
    cal_scores = {"msp": -c["msp"], "expected": predict(c["X"], gain_fit)}
    row = {"fit": list(f), "ridge": [np.asarray(v).tolist() for v in gain_fit],
           "first": 100*t["a"].mean(), "second": 100*t["b"].mean(),
           "all": 100*terminal.mean(), "oracle": 100*(t["a"] | t["b"]).mean(),
           "cal_cv_accuracy": 100*(c["a"].sum()+train_gain[quota(-c["msp"], .3)].sum())/len(ci),
           "curves": {}, "fixed": {}}
    cols = {"source_id":t["ids"], "first_class":t["p1"], "second_class":t["p2"],
            "terminal_class":test_class, "first_correct":t["a"].astype(int), "delta_if_requested":delta}
    for key, score in scores.items():
        mask = quota(score,.3)
        threshold = np.min(cal_scores[key][quota(cal_scores[key],.3)])
        cols[key+"_score"] = score
        cols[key+"_request"] = mask.astype(int)
        row["fixed"][key] = {"threshold": threshold, **summary(t["a"],delta,score >= threshold)}
        for frac in FRACS:
            g = quota(score,frac)
            row["curves"][key+"_"+str(frac)] = summary(t["a"],delta,g)
            if key == "msp":
                row["curves"]["accept_"+str(frac)] = summary(t["a"],accept,g)
    with gzip.open(OUT/(order+"_test.csv.gz"), "wt",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(cols)
        writer.writerows(zip(*cols.values()))
    return row


def bootstrap_sample(pairs, ci, ti):
    acc, gains, fixed = {}, {}, {}
    # Reversing the pair negates the standardized difference and exchanges
    # the two outcomes. Fit ordered decisions explicitly to preserve tie rules.
    for order, (c, t) in pairs.items():
        f = fit_terminal(c, ci)
        a, b = t["a"][ti], t["b"][ti]
        terminal = np.where(choose_vertical(t["x"][ti], f), a, b)
        delta = terminal.astype(np.int8) - a
        first_score = -t["msp"][ti]
        gate = quota(first_score,.3,ti)
        gains[order] = 100*(delta*gate).mean()
        acc[order] = 100*a.mean()+gains[order]
        t0 = np.partition(-c["msp"][ci], -2700)[-2700]
        fixed[order] = 100*(a.mean()+(delta*(first_score >= t0)).mean())
        if order == "VH":
            gfit = ridge(c["X"][ci], crossfit(c,ci))
            score = predict(t["X"][ti],gfit)
            cg = predict(c["X"][ci],gfit)
            cutoff = np.partition(cg,-2700)[-2700]
            gain_gate = quota(score,.3,ti)
            expected_diff = 100*(delta*(gain_gate.astype(np.int8)-gate)).mean()
            expected_fixed = 100*(a.mean() + (delta*(score>=cutoff)).mean())
    return {**{k+"_gain":v for k,v in gains.items()}, "SV_minus_VS":acc["SV"]-acc["VS"],
            "SV_minus_VS_fixed":fixed["SV"]-fixed["VS"],
            "VH_expected_minus_msp":expected_diff, "VH_expected_minus_msp_fixed":expected_fixed-fixed["VH"]}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs/scores.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    groups, data = load(args.inputs)
    pairs = {o:tuple(block(data[s],*groups[s],o) for s in ("calibration","test")) for o in ORDERS}
    rows={o:point(o,*records) for o,records in pairs.items()}
    best = max(ORDERS, key=lambda o: rows[o]["cal_cv_accuracy"])
    if args.inputs.resolve() == (HERE / "inputs/scores.npz").resolve():
        assert rows["VS"]["all"] == 100*17883/36000 or abs(rows["VS"]["all"]-49.675)<1e-12
        assert abs(rows["VS"]["oracle"] - 100*20481/36000) < 1e-12
    result = {"orders": rows, "calibration_chosen_order": best}
    write_json(OUT/"orders.json", result)

    print("Calibration-chosen:",best,flush=True)
    for design,seed in (("within",68422),("classes",68423)):
        rng = np.random.default_rng(seed)
        draws=[]
        cl=np.arange(900)[:,None]
        for i in range(DRAWS):
            if design=="within":
                ci=(10*cl+rng.integers(10,size=(900,10))).ravel()
                ti=(40*cl+rng.integers(40,size=(900,40))).ravel()
            else:
                classes=rng.integers(900,size=(900,1))
                ci=(10*classes+np.arange(10)).ravel()
                ti=(40*classes+np.arange(40)).ravel()
            draws.append(bootstrap_sample(pairs,ci,ti))
        result[design] = {k: np.percentile([r[k] for r in draws],[2.5,97.5]).tolist() for k in draws[0]}
        write_json(OUT/"orders.json",result)
        print(design,result[design],flush=True)


if __name__=="__main__":
    main()
