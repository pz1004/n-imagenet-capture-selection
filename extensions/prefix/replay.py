"""Refit and evaluate first-20-ms request and switching rules on the study split."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from capture_policies import choose_vertical, fit_discordant, read
from trajectory_data import DATA, MODELS, ids_for, scan, threshold

BUDGET = 0.30
REPS = 2000
GATES = {"confidence": (0,), "rate": (1,), "confidence_plus_rate": (0, 1)}
SUMMARY_FIELDS = ("requests_pct", "accuracy_pct", "gain_vs_first_pp",
                  "directed_minus_random_pp", "net_per_100_requests")


def load_pair(pair, split, ids, labels):
    """Join first-window counts and directional model scores in frozen source order."""
    if pair == "old":
        windows = read(DATA / "full_time20_windows.csv")
        counts = {direction: np.array([int(windows[k][f"used_events_{direction[0]}"])
                                       for k in ids]) for direction in ("vertical", "horizontal")}
    else:
        counts = {}
        for direction, variant in (("vertical","variant1"), ("horizontal","variant2")):
            with gzip.open(DATA / f"{variant}_time20.csv.gz", "rt", newline="") as stream:
                rows = [row for row in csv.DictReader(stream) if row["split"] == split]
            n_events = {row["source_id"]: int(row["used_events"]) for row in rows}
            assert len(n_events) == len(rows) and set(n_events) == set(ids)
            counts[direction] = np.array([n_events[k] for k in ids])

    captures = []
    for direction in ("vertical", "horizontal"):
        data = {"used_events": counts[direction]}
        assert (data["used_events"] >= 0).all()
        for model in MODELS:
            scores, predictions, observed = scan(pair, split, direction, "time20", model, ids)
            assert np.array_equal(observed, labels)
            assert np.isfinite(scores).all() and ((scores > 0) & (scores <= 1)).all()
            data[model] = (scores, predictions)
        captures.append(data)
    return captures


def tie_keys(ids):
    keys = np.array([int.from_bytes(hashlib.sha256(("prefix-request/" + k).encode()).digest()[:8], "big")
                     for k in ids], dtype=np.uint64)
    assert len(np.unique(keys)) == len(ids)
    return keys


def features(v, model):
    return np.column_stack((v[model][0], np.log1p(v["used_events"] / 0.020)))


def basis(X, fit):
    z = (X - fit["mean"]) / fit["scale"]
    return np.column_stack((z, (z*z - fit["sqmean"]) / fit["sqscale"]))


def order(score, tie):
    return np.lexsort((tie, -score))


def learn(v, h, y, model, tie):
    sv, pv = v[model]
    sh, ph = h[model]
    fitted = fit_discordant(sv-sh, pv == y, ph == y)
    a, b, intercept, slope = fitted
    assert slope > 0
    window_cutoff = b * intercept / slope - a
    terminal = np.where(sh-sv > threshold(model), ph, pv)
    gain = (terminal == y).astype(np.int8) - (pv == y).astype(np.int8)
    full = features(v, model)
    regressions = {}
    for name, cols in GATES.items():
        X = full[:, cols]
        fit = {"mean": X.mean(axis=0).tolist(), "scale": X.std(axis=0).tolist()}
        z = (X - fit["mean"]) / fit["scale"]
        fit.update(sqmean=(z*z).mean(axis=0).tolist(), sqscale=(z*z).std(axis=0).tolist())
        design = basis(X, fit)
        fit["intercept"] = float(gain.mean())
        fit["slopes"] = np.linalg.solve(design.T @ design + np.eye(design.shape[1]),
                                        design.T @ (gain-fit["intercept"])).tolist()
        score = design @ fit["slopes"] + fit["intercept"]
        k = int(round(len(y) * BUDGET))
        boundary = order(score, tie)[k-1]
        fit["cutoff"] = float(score[boundary])
        fit["max_tie_key_at_cutoff"] = int(tie[boundary])
        mask = calibration_mask(score, tie, fit)
        assert mask.sum() == k
        regressions[name] = fit
    return {"ref20_threshold_h_minus_v": window_cutoff, "ref20_logistic_fit": fitted,
            "ref30_threshold_h_minus_v": threshold(model),
            "calibration_discordant_n": int(((pv == y) != (ph == y)).sum()),
            "gates": regressions}


def calibration_mask(score, tie, fit):
    return ((score > fit["cutoff"]) | ((score == fit["cutoff"]) &
                                     (tie <= np.uint64(fit["max_tie_key_at_cutoff"]))))


def metrics(n, first_correct, always_net, req, net):
    return [100*req/n, 100*(first_correct+net)/n, 100*net/n,
            100*(net/n - req/n*(always_net/n)), 100*net/req if req else 0.]


def estimate(v, h, y, model, ids, tie, fit, pair, output):
    sv, pv = v[model]
    sh, ph = h[model]
    base = (pv == y).astype(np.int8)
    term_pred = {"ref30": np.where(sh-sv > fit["ref30_threshold_h_minus_v"], ph, pv),
                 "ref20": np.where(sh-sv > fit["ref20_threshold_h_minus_v"], ph, pv),
                 "raw": np.where(sh > sv, ph, pv)}
    v_choice = choose_vertical(sv-sh, fit["ref20_logistic_fit"])
    assert np.array_equal(np.where(v_choice, pv, ph), term_pred["ref20"])
    correct = {k:(z == y).astype(np.int8) for k,z in term_pred.items()}
    gain = correct["ref30"] - base
    X = features(v, model)
    n, k = len(y), int(round(len(y)*BUDGET))
    gates, masks, scores = {}, {}, {}
    rate = v["used_events"] / 0.020
    base_matrix = np.column_stack((base, correct["ref20"], correct["ref30"], correct["raw"]))
    choices = {}
    names = []
    base_correct = int(base.sum())
    for name, cols in GATES.items():
        fitted = fit["gates"][name]
        sc = basis(X[:, cols], fitted) @ fitted["slopes"] + fitted["intercept"]
        ords = order(sc, tie)
        fixed = calibration_mask(sc, tie, fitted)
        exact = np.zeros(n, dtype=bool)
        exact[ords[:k]] = True
        scores[name] = sc
        gates[name] = {"order": ords, "sorted_gain": gain[ords]}
        for mode, reqs in (("quota", exact), ("calibrated", fixed)):
            key = name+"_"+mode
            masks[key] = reqs
            net = reqs * gain
            pred = np.where(reqs, term_pred["ref30"], pv)
            assert np.array_equal(base + net, pred == y)
            choices[key] = {"n_requested": int(reqs.sum()), "rescued": int((net > 0).sum()),
                             "lost": int((net < 0).sum()), "net_correct": int(net.sum()),
                             "correct": int((pred==y).sum())}
            choices[key].update(dict(zip(SUMMARY_FIELDS, metrics(n, base_correct, gain.sum(),
                                                                reqs.sum(), net.sum()))))
            names.append(key)
    fields = ["source_id","label","vertical_prefix_events","vertical_events_per_s","vertical_msp",
              "horizontal_msp","vertical_class","horizontal_class","ref30_selected_class",
              "ref20_selected_class","raw_selected_class","ref30_gain_if_requested"]
    fields.extend(name+"_score" for name in GATES)
    for name in masks:
        fields.extend([name+"_requested",name+"_selected_class",name+"_change_in_correctness"])
    with gzip.open(output / f"{pair}_{model}_decisions.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, source in enumerate(ids):
            row = [source, y[i], v["used_events"][i], rate[i], sv[i], sh[i], pv[i], ph[i],
                   term_pred["ref30"][i], term_pred["ref20"][i], term_pred["raw"][i], gain[i]]
            row.extend(scores[name][i] for name in GATES)
            for name, mask in masks.items():
                row.extend([int(mask[i]), term_pred["ref30"][i] if mask[i] else pv[i], int(mask[i])*gain[i]])
            w.writerow(row)
    fixed_matrix = np.column_stack([x for name in GATES for x in
                                    (masks[name+"_calibrated"], masks[name+"_calibrated"] * gain)])
    sufficient = np.column_stack((base_matrix, fixed_matrix)).astype(np.int32)
    net_gains = {k: correct[k]-base for k in term_pred}
    def counts(diff):
        return {"rescued": int((diff==1).sum()), "lost": int((diff==-1).sum()),
                "net": int(diff.sum()), "pp": float(100*diff.mean())}
    terminal = {"vertical_accuracy_pct": float(100*base.mean()),
                "correct_n": {x:int(correct[x].sum()) for x in correct},
                "accuracy_pct": {x:float(100*correct[x].mean()) for x in correct},
                "vs_first": {x:counts(net_gains[x]) for x in correct},
                "ref20_vs_ref30":counts(correct["ref20"]-correct["ref30"]),
                "ref20_vs_raw":counts(correct["ref20"]-correct["raw"])}
    results = {"terminal": terminal, "gates": choices, "gate_contrasts": {}}
    contrast_keys = [(plus, other, mode) for plus, other in
                     (("confidence_plus_rate","confidence"),("rate","confidence"))
                     for mode in ("quota","calibrated")]
    for plus, other, mode in contrast_keys:
        delta = (masks[plus+"_"+mode].astype(np.int8)-masks[other+"_"+mode].astype(np.int8)) * gain
        results["gate_contrasts"][plus+"_minus_"+other+"_"+mode] = counts(delta)
    for design, seed in (("within",20261121),("class",20261125)):
        rng = np.random.default_rng(seed)
        starts = 40*np.arange(900)[:,None]
        boot = np.empty((REPS,len(names),len(SUMMARY_FIELDS)))
        terminal_boot = np.empty((REPS,2))
        for it in range(REPS):
            if design == "within":
                ix = (starts+rng.integers(40, size=(900,40))).ravel()
                w = np.bincount(ix, minlength=n).astype(np.int32)
            else:
                classes = rng.integers(900,size=900)
                w = np.repeat(np.bincount(classes,minlength=900).astype(np.int32),40)
            vals = w @ sufficient
            base_b, ref20_b, ref30_b, raw_b = vals[:4]
            full_net = ref30_b-base_b
            terminal_boot[it] = 100*np.array((ref20_b-ref30_b,ref20_b-raw_b))/n
            for j,name in enumerate(GATES):
                info = gates[name]
                weights = w[info["order"]]
                cumulative = np.cumsum(weights, dtype=np.int32)
                boundary = int(np.searchsorted(cumulative,k))
                rem = k - (cumulative[boundary-1] if boundary else 0)
                profit = info["sorted_gain"]
                net = weights[:boundary] @ profit[:boundary] + rem*profit[boundary]
                boot[it,2*j] = metrics(n,base_b,full_net,k,net)
                req, fn = vals[4+2*j:6+2*j]
                boot[it,2*j+1] = metrics(n,base_b,full_net,req,fn)
        # Paired uncertainty conditional on the trained score and terminal-rule coefficients.
        for j,key in enumerate(names):
            choices[key][design+"_ci95"] = dict(zip(SUMMARY_FIELDS,
                np.percentile(boot[:,j,:], [2.5,97.5], axis=0).T.tolist()))
        for plus,other,mode in contrast_keys:
            diff = boot[:,names.index(plus+"_"+mode),1] - boot[:,names.index(other+"_"+mode),1]
            results["gate_contrasts"][plus+"_minus_"+other+"_"+mode][design+"_ci95_pp"] = np.percentile(diff,[2.5,97.5]).tolist()
        for j, key in enumerate(("ref20_vs_ref30","ref20_vs_raw")):
            terminal[key][design+"_ci95_pp"] = np.percentile(terminal_boot[:,j],[2.5,97.5]).tolist()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Directory for recalculated fits, decisions and intervals")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(DATA / "source_index.csv", "cohort", "full")
    cal, test = ids_for("calibration"), ids_for("test")
    cal_y, test_y = (np.array([int(manifest[k]["label"]) for k in ids]) for ids in (cal, test))
    cal_ties, test_ties = tie_keys(cal), tie_keys(test)
    vcal, hcal = load_pair("old", "calibration", cal, cal_y)
    fits = {model: learn(vcal, hcal, cal_y, model, cal_ties) for model in MODELS}
    (output/"reference_calibration_fits.json").write_text(json.dumps(fits, indent=2, allow_nan=False)+"\n")
    for pair in ("old", "new"):
        v, h = load_pair(pair, "test", test, test_y)
        for model in MODELS:
            results = estimate(v, h, test_y, model, test, test_ties, fits[model], pair, output)
            (output/f"{pair}_{model}.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
            print(pair, model, results["gate_contrasts"]["confidence_plus_rate_minus_confidence_quota"])


if __name__ == "__main__":
    main()
