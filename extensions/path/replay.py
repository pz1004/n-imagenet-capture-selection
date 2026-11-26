"""Compare horizontal and square follow-ups at identical first-scan request decisions."""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capture_policies import choose_vertical, fit_discordant, read
from trajectory_data import DATA, MODELS, ids_for, scan

WINDOW = "events30k"
PAIRS = {"transfer": ("variant1", "variant2"), "reference": ("vertical", "horizontal")}
PATHS = ("horizontal", "square")
RULES = ("take_second", "local_selector", "reference_selector", "probability_average")
DRAWS = 2000
SEEDS = {"within": 20261211, "class": 20261212, "refit_cal": 20261213,
         "refit_test": 20261214, "refit_class": 20261215}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compressed_rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def cohort():
    manifest = read(DATA / "source_index.csv", "cohort", "full")
    groups = {}
    for split in ("calibration", "test"):
        keys = ids_for(split)
        groups[split] = keys, np.array([int(manifest[key]["label"]) for key in keys])
    return groups


def load_captures(groups, input_dir):
    square = compressed_rows(input_dir / "square_confidence.csv.gz")
    score = {row["source_id"]:row for row in square}
    original = read(DATA / "original_sanity_predictions.csv", "model_set", "full")
    all_ids = {key for keys, _ in groups.values() for key in keys}
    assert len(score) == len(square) and set(score) == all_ids
    captures = {}
    for split, (ids, labels) in groups.items():
        scans = {}
        for pair, names in PAIRS.items():
            for direction, name in zip(("vertical","horizontal"),names):
                cap = SimpleNamespace(msp={}, pred={})
                for m in MODELS:
                    prob, pred, y = scan("old" if pair == "reference" else "new",
                                         split, direction, WINDOW, m, ids)
                    assert np.array_equal(y,labels)
                    cap.pred[m],cap.msp[m] = pred,prob
                scans[name] = cap
        assert all(score[k]["split"] == split and int(score[k]["label"]) == y
                   and int(original[k]["label"]) == y for k,y in zip(ids,labels))
        cap = SimpleNamespace(msp={}, pred={})
        for m in MODELS:
            cap.pred[m] = np.array([int(original[k][m+"_pred"]) for k in ids])
            cap.msp[m] = np.array([float(score[k][m+"_msp"]) for k in ids])
            assert np.isfinite(cap.msp[m]).all() and ((cap.msp[m]>0) & (cap.msp[m]<=1)).all()
        scans["original"] = cap
        captures[split] = scans
    return captures


def fusion_choices(ids, labels, input_dir):
    sq_rows = compressed_rows(input_dir / "square_fusion.csv.gz")
    sq = {r["source_id"]:r for r in sq_rows}
    assert len(sq_rows)==len(sq)==len(ids) and set(sq)==set(ids)
    assert np.array_equal(labels,[int(sq[k]["label"]) for k in ids])
    choices = {m:{} for m in MODELS}
    for pair in PAIRS:
        for m in MODELS:
            choices[m][pair,"square"] = np.array([int(sq[k][pair+"_"+m]) for k in ids])
    rows = { (p,m):{} for p in PAIRS for m in MODELS }
    with gzip.open(DATA/"trajectory_baseline_decisions.csv.gz","rt",newline="") as stream:
        for r in csv.DictReader(stream):
            if r["window"]!=WINDOW or r["pair"] not in ("old","new"):
                continue
            pair = "reference" if r["pair"]=="old" else "transfer"
            group = rows[pair,r["model"]]
            key = r["source_id"]
            assert key not in group
            group[key] = (int(r["probability_average"]),int(r["label"]))
    for (pair,m), group in rows.items():
        assert set(group)==set(ids) and np.array_equal(labels,[group[k][1] for k in ids])
        choices[m][pair,"horizontal"] = np.array([group[k][0] for k in ids])
    return choices


def fit_for(first, second, model, y, ix=None):
    if ix is None:
        ix = slice(None)
    return fit_discordant((first.msp[model] - second.msp[model])[ix],
                          (first.pred[model] == y)[ix], (second.pred[model] == y)[ix])


def fit_summary(p):
    center, scale, intercept, slope = p
    assert slope > 0
    return {"center": center, "scale": scale, "intercept": intercept, "slope": slope,
            "switch_if_second_minus_first_gt": scale * intercept / slope - center}


def paired_draws(matrix, design):
    # Raw sufficient statistics are additive over subjects. Reweighting once per
    # draw ensures common bootstrap indices across every policy/output column.
    rng = np.random.default_rng(SEEDS[design])
    n, k = matrix.shape
    all_draws = np.empty((DRAWS, k))
    if design == "class":
        totals = matrix.reshape(900, 40, k).sum(axis=1)
        for start in range(0, DRAWS, 100):
            end = min(DRAWS, start + 100)
            ix = rng.integers(900, size=(end - start, 900))
            all_draws[start:end] = totals[ix].sum(axis=1)
    else:
        starts = 40 * np.arange(900)[:, None]
        # Using np.bincount on integer indices is faster than allocating
        # the bootstrap sample itself for every policy column.
        for start in range(0, DRAWS, 16):
            end = min(DRAWS, start + 16)
            weights = np.empty((end - start, n))
            for i in range(end - start):
                ix = (starts + rng.integers(40, size=(900, 40))).ravel()
                weights[i] = np.bincount(ix, minlength=n)
            all_draws[start:end] = weights @ matrix
    return all_draws


def metrics(t, names, arms, n):
    s = {name: t[..., i] for i, name in enumerate(names)}
    result = {}
    first, req = s["vertical"], s["requested"]
    for name in ("vertical", "horizontal", "square"):
        result[f"{name}_alone/accuracy_pct"] = 100 * s[name] / n
    result["one_scan/square_minus_vertical_pp"] = 100 * (s["square"] - first) / n
    result["one_scan/square_minus_horizontal_pp"] = 100 * (s["square"] - s["horizontal"]) / n
    result["requests/percent"] = 100 * req / n
    for arm in arms:
        rescue, loss = s[arm + "/rescue"], s[arm + "/loss"]
        gain, all_gain = rescue - loss, s[arm + "/ungated_gain"]
        result[arm + "/accuracy_pct"] = 100 * (first + gain) / n
        result[arm + "/gain_vs_vertical_pp"] = 100 * gain / n
        result[arm + "/net_per_100_requests"] = 100 * gain / req
        result[arm + "/all_requested_accuracy_pct"] = 100 * (first + all_gain) / n
        rnd = req / n * all_gain
        result[arm + "/matched_random_accuracy_pct"] = 100 * (first + rnd) / n
        result[arm + "/gain_vs_random_pp"] = 100 * (gain - rnd) / n
        result[arm + "/gain_vs_square_alone_pp"] = 100 * (first + gain - s["square"]) / n
    for rule in RULES:
        result["path_contrast/" + rule + "_square_minus_horizontal_pp"] = (
            result["square_" + rule + "/accuracy_pct"] -
            result["horizontal_" + rule + "/accuracy_pct"])
    for path in PATHS:
        result["rule_contrast/" + path + "_selector_minus_take_second_pp"] = (
            result[path + "_local_selector/accuracy_pct"] -
            result[path + "_take_second/accuracy_pct"])
    result["interaction/selection_minus_take_second_path_effect_pp"] = (
        result["path_contrast/local_selector_square_minus_horizontal_pp"] -
        result["path_contrast/take_second_square_minus_horizontal_pp"])
    return result


def interval(draws):
    return np.percentile(draws, [2.5, 97.5], axis=0).tolist()


def analyze(pair, model, scans, groups, fused, cutoff, output):
    ids, y = groups["test"]
    yc = groups["calibration"][1]
    v_name, h_name = PAIRS[pair]
    ct, cc = scans["test"], scans["calibration"]
    v = ct[v_name]
    G = v.msp[model] < cutoff
    # The gate uses the vertical scores only.
    vp = v.pred[model]
    vc = vp == y
    decisions = {"source_id": ids, "label": y, "requested": G.astype(np.int8),
                 "vertical_class": vp, "vertical_msp": v.msp[model]}
    total = {"requested": G, "vertical": vc, "horizontal": ct[h_name].pred[model] == y,
             "square": ct["original"].pred[model] == y}
    arms, counts, fits = [], {}, {}
    for path, scan_name in (("horizontal", h_name), ("square", "original")):
        second = ct[scan_name]
        sp = second.pred[model]
        scores = v.msp[model] - second.msp[model]
        local = fit_for(cc[v_name], cc[scan_name], model, yc)
        ref = fit_for(cc["vertical"], cc["horizontal" if path == "horizontal" else "original"], model, yc)
        fits[path] = {"local": fit_summary(local), "reference": fit_summary(ref)}
        decisions.update({f"{path}_class": sp, f"{path}_msp": second.msp[model]})
        terminal_classes = {"take_second": sp,
                            "local_selector": np.where(choose_vertical(scores, local), vp, sp),
                            "reference_selector": np.where(choose_vertical(scores, ref), vp, sp),
                            "probability_average": fused[(pair, path)]}
        for rule, always in terminal_classes.items():
            arm = path + "_" + rule
            arms.append(arm)
            final = np.where(G, always, vp)
            directed = (final == y).astype(np.int8) - vc.astype(np.int8)
            full = always == y
            total[arm + "/rescue"] = directed == 1
            total[arm + "/loss"] = directed == -1
            total[arm + "/ungated_gain"] = full.astype(np.int8) - vc.astype(np.int8)
            decisions[arm + "_directed_class"] = final
            decisions[arm + "_if_requested_class"] = always
            decisions[arm + "_change_in_correctness"] = directed
            counts[arm] = {"requested": int(G.sum()), "rescues": int((directed == 1).sum()),
                           "losses": int((directed == -1).sum()), "correct": int((final == y).sum()),
                           "correct_if_all_requested": int(full.sum())}
    for rule in RULES:
        a = decisions["square_" + rule + "_directed_class"] == y
        b = decisions["horizontal_" + rule + "_directed_class"] == y
        counts["paired_" + rule] = {
            "square_only_correct": int((a & ~b).sum()),
            "horizontal_only_correct": int((b & ~a).sum())}
        assert ((a == b) | G).all()
    matrix = np.column_stack(list(total.values())).astype(np.float64)
    names = list(total)
    values = metrics(matrix.sum(axis=0), names, arms, len(y))
    stats = {key: {"estimate": float(x)} for key, x in values.items()}
    for design in ("within", "class"):
        draws = metrics(paired_draws(matrix, design), names, arms, len(y))
        for key in stats:
            stats[key][design + "_ci95"] = interval(draws[key])
    stem = pair + "_" + model
    with gzip.open(output / (stem + "_decisions.csv.gz"), "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(decisions.keys())
        writer.writerows(zip(*decisions.values()))
    res = {"pair": pair, "model": model, "n": len(ids), "classes": 900,
           "request_cutoff": float(cutoff), "requests": int(G.sum()),
           "mask_hash": hashlib.sha256(("\n".join(x for x, g in zip(ids, G) if g)+"\n").encode()).hexdigest(),
           "fit": fits, "counts": counts, "stats": stats}
    write_json(output / (stem + ".json"), res)
    print(stem, "requests", int(G.sum()), flush=True)
    for name in ("take_second", "local_selector"):
        k = "path_contrast/" + name + "_square_minus_horizontal_pp"
        print(k, stats[k], flush=True)
    return res


def refit_sensitivity(scans, groups, cutoff):
    ct = scans["test"]
    cc = scans["calibration"]
    yc, yt = groups["calibration"][1], groups["test"][1]
    v_c, v_t = cc["variant1"], ct["variant1"]
    model = "histogram"
    calx = [v_c.msp[model] - cc[s].msp[model] for s in ("variant2", "original")]
    testx = [v_t.msp[model] - ct[s].msp[model] for s in ("variant2", "original")]
    cv, tv = v_c.pred[model] == yc, v_t.pred[model] == yt
    cs = [cc[s].pred[model] == yc for s in ("variant2", "original")]
    ts = [ct[s].pred[model] == yt for s in ("variant2", "original")]
    G = v_t.msp[model] < cutoff
    rngc, rngt, rngcl = [np.random.default_rng(SEEDS[key])
                            for key in ("refit_cal", "refit_test", "refit_class")]
    cstart = np.arange(900)[:, None] * 10
    tstart = np.arange(900)[:, None] * 40
    out = {}
    for design in ("within", "class"):
        diffs = []
        for _ in range(DRAWS):
            if design == "within":
                ci = (cstart + rngc.integers(10, size=(900,10))).ravel()
                ti = (tstart + rngt.integers(40, size=(900,40))).ravel()
            else:
                classes = rngcl.integers(900, size=(900,1))
                ci = (10*classes + np.arange(10)).ravel()
                ti = (40*classes + np.arange(40)).ravel()
            num = []
            for x, xt, second_cal, second_test in zip(calx, testx, cs, ts):
                p = fit_discordant(x[ci], cv[ci], second_cal[ci])
                use = G[ti] & ~choose_vertical(xt[ti], p)
                num.append(np.where(use, second_test[ti], tv[ti]).sum())
            diffs.append(100 * (num[1] - num[0]) / len(ti))
        out[design + "_ci95"] = interval(diffs)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for recalculated source decisions and intervals")
    parser.add_argument("--inputs", type=Path, default=HERE/"inputs", help="Square confidence and probability-averaging inputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = cohort()
    scans = load_captures(groups, args.inputs)
    fused = fusion_choices(*groups["test"], args.inputs)
    results = {}
    for model in MODELS:
        cutoff = np.quantile(scans["calibration"]["vertical"].msp[model], 0.30, method="higher")
        for pair in PAIRS:
            results[pair, model] = analyze(pair, model, scans, groups, fused[model], cutoff, args.output)
        if model == "histogram":
            write_json(args.output/"transfer_histogram_refit_sensitivity.json", refit_sensitivity(scans, groups, cutoff))

    with (args.output/"all_contrasts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["first_pair","model","estimand","estimate","within_ci_lo","within_ci_hi",
                         "class_ci_lo","class_ci_hi"])
        for pair in PAIRS:
            for model in MODELS:
                for name, estimate in results[pair, model]["stats"].items():
                    writer.writerow([pair,model,name,estimate["estimate"],
                                     *estimate["within_ci95"],*estimate["class_ci95"]])


if __name__ == "__main__":
    main()
