#!/usr/bin/env python3
"""Fit the prespecified RGB orientation increment to frozen camera-direction predictions."""
import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Thousands of small, independent 4-parameter Newton problems do not need threaded BLAS.
for thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[thread_var] = "1"

import numpy as np


DATA = Path(__file__).resolve().parents[1] / "data"
BASE = ("msp_difference", "log_full_event_count_ratio", "duration_difference_ms")
TERMS = ("intercept", *BASE, "orientation")
NBOOT = 2000
SEED_CAL = 20261122
SEED_TEST = 20261121
METRICS = ("loss_baseline", "loss_augmented", "loss_reduction",
           "v_only_loss_baseline", "v_only_loss_augmented", "v_only_loss_reduction",
           "h_only_loss_baseline", "h_only_loss_augmented", "h_only_loss_reduction",
           "orientation_coefficient_standardized", "orientation_coefficient_raw")


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def keyed(data, expected, name):
    result = {r["source_id"]: r for r in data}
    if len(result) != len(data):
        raise ValueError(f"Duplicate source IDs in {name}")
    if set(result) != expected:
        raise ValueError(f"{name}: {len(expected-result.keys())} missing and {len(result.keys()-expected)} unexpected IDs")
    return result


@dataclass
class Cohort:
    ids: np.ndarray
    labels: np.ndarray
    X: np.ndarray
    v_correct: np.ndarray
    h_correct: np.ndarray

    @property
    def discord(self):
        return self.v_correct != self.h_correct

    def subset(self, mask):
        return Cohort(self.ids[mask], self.labels[mask], self.X[mask], self.v_correct[mask], self.h_correct[mask])


def loss(eta, y):
    # y is float, with 1 for vertical-only and 0 for horizontal-only.
    return np.logaddexp(0.0, (1.0-2.0*y)*eta)


def sigmoid(eta):
    return np.exp(-np.logaddexp(0.0, -eta))


@dataclass
class Logistic:
    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    iterations: int
    constant_features: np.ndarray

    def linear(self, x):
        raw_slopes = self.coef[1:] / self.scale
        return self.coef[0] + (x-self.center) @ raw_slopes

    def report(self):
        slopes = self.coef[1:] / self.scale
        return {"terms": list(TERMS[:len(self.coef)]), "calibration_mean": self.center.tolist(),
                "calibration_sd_or_unit_scale_if_constant": self.scale.tolist(), "constant_feature_indices": self.constant_features.tolist(),
                "standardized_coefficients": self.coef.tolist(), "raw_coefficients": [float(self.coef[0]-self.center@slopes), *slopes.tolist()],
                "newton_iterations": self.iterations}


def fit(x, y, w):
    # Equivalent to expanding bootstrap weights; the intercept has no penalty.
    total = w.sum()
    positives = w@y
    if not (total > positives > 0):
        raise ValueError("Calibration sample needs both directions of discordance")
    center = w@x/total
    delta = x-center
    sd = np.sqrt(w@(delta*delta)/total)
    constant = np.flatnonzero(sd <= 1e-12)
    sd[constant] = 1.0
    design = np.column_stack((np.ones(len(x)), delta/sd))
    ridge = np.ones(design.shape[1])
    ridge[0] = 0.
    beta = np.zeros(design.shape[1])
    beta[0] = np.log(positives/(total-positives))
    for iteration in range(1, 101):
        eta = design@beta
        p = sigmoid(eta)
        gradient = design.T@(w*(p-y)) + ridge*beta
        if np.max(abs(gradient))/total < 1e-10:
            return Logistic(center, sd, beta, iteration, constant)
        hessian = design.T@((w*p*(1-p))[:, None]*design) + np.diag(ridge)
        step = np.linalg.solve(hessian, gradient)
        old = w@loss(eta, y) + .5*np.dot(beta[1:], beta[1:])
        decrement = gradient @ step
        step_size = 1.
        for _ in range(35):
            candidate = beta-step_size*step
            value = w@loss(design@candidate, y) + .5*np.dot(candidate[1:], candidate[1:])
            if value <= old - 1e-4*step_size*decrement + 1e-12*(1+old):
                beta = candidate
                break
            step_size *= .5
        else:
            raise ArithmeticError("Newton line search failed")
    raise ArithmeticError("Newton fit failed to converge")


@dataclass
class ModelPair:
    baseline: Logistic
    augmented: Logistic

    def predict(self, x):
        return self.baseline.linear(x[:, :3]), self.augmented.linear(x)


def fit_pair(x, y, w):
    return ModelPair(fit(x[:, :3], y, w), fit(x, y, w))


def weighted(v, w):
    return float(w@v/w.sum()) if w.sum() else float("nan")


def measures(pair, x, y, w):
    e0, e1 = pair.predict(x)
    l0, l1 = loss(e0,y), loss(e1,y)
    quantities = []
    for weights in (w, w*y, w*(1-y)):
        quantities.extend((weighted(l0, weights), weighted(l1, weights), weighted(l0-l1, weights)))
    quantities.extend((pair.augmented.coef[-1], pair.augmented.coef[-1]/pair.augmented.scale[-1]))
    return np.array(quantities)


class Stratified:
    def __init__(self, labels):
        # Each block contains all source IDs for classes with an equal number of observations.
        groups = {}
        for label in np.unique(labels):
            group = np.flatnonzero(labels == label)
            groups.setdefault(len(group), []).append(group)
        self.blocks = [np.stack(v) for v in groups.values()]
        self.n = len(labels)

    def draw(self, rng):
        indices = [np.take_along_axis(g, rng.integers(g.shape[1], size=g.shape), axis=1).ravel() for g in self.blocks]
        return np.bincount(np.concatenate(indices), minlength=self.n).astype(float)


def assignment_control(labels, x, y, pair):
    # Each receiver keeps its own label and three baseline values. Only the standardized
    # orientation coordinate changes; all within-synset donors (including self) are equally likely.
    base, aug = pair.predict(x)
    slope = pair.augmented.coef[-1]/pair.augmented.scale[-1]
    shuffled = np.empty(len(y))
    singleton = 0
    crossed = 0
    for lab in np.unique(labels):
        ix = np.flatnonzero(labels == lab)
        z = x[ix,-1]
        eta = aug[ix, None] + slope*(z[None, :]-z[:, None])
        shuffled[ix] = loss(eta, y[ix, None]).mean(axis=1)
        singleton += int(len(ix)==1)
        crossed += len(ix)**2
    actual, ref = loss(aug, y), loss(base, y)
    return {"n":len(y),"occupied_synsets":len(np.unique(labels)), "singleton_receivers":singleton,
            "receiver_donor_evaluations":crossed, "observed_augmented_loss":float(actual.mean()),
            "expected_reassigned_augmented_loss":float(shuffled.mean()),
            "expected_reassigned_gain":float((ref-shuffled).mean()),
            "matched_gain_minus_expected_reassigned_gain":float((shuffled-actual).mean())}


def bootstrap(train, test, nboot):
    cal_source, held_source = Stratified(train.labels), Stratified(test.labels)
    gc, gt = np.random.default_rng(SEED_CAL), np.random.default_rng(SEED_TEST)
    cd, td = train.discord, test.discord
    cx, cy = train.X[cd], train.v_correct[cd].astype(float)
    tx, ty = test.X[td], test.v_correct[td].astype(float)
    values = np.empty((nboot, len(METRICS)))
    for i in range(nboot):
        wc, wt = cal_source.draw(gc)[cd], held_source.draw(gt)[td]
        pair = fit_pair(cx, cy, wc)
        values[i] = measures(pair, tx, ty, wt)
    if not np.isfinite(values).all():
        raise ValueError("Undefined bootstrap statistic; inspect counts before interpretation")
    return values


def orient_bins(train, test):
    cuts = np.quantile(train.X[train.discord,-1], [.2,.4,.6,.8], method="linear")
    return cuts, np.searchsorted(cuts, test.X[:,-1], side="right")+1


def binned_report(test, bin_id, pair):
    result = []
    discord = test.discord
    y = test.v_correct[discord].astype(float)
    e0, e1 = pair.predict(test.X[discord])
    row_metrics = {"mean_p_v_baseline_discordant": sigmoid(e0), "mean_p_v_augmented_discordant": sigmoid(e1),
                   "mean_loss_baseline_discordant": loss(e0, y), "mean_loss_augmented_discordant": loss(e1, y)}
    # MSP selection follows the vertical tie convention. Ranking credit gives ties half.
    for b in range(1,6):
        take = bin_id == b
        vo = take & test.v_correct & ~test.h_correct
        ho = take & test.h_correct & ~test.v_correct
        msdiff = test.X[:,0]
        v_lost, h_lost = (vo & (msdiff < 0)).sum(), (ho & (msdiff > 0)).sum()
        v_tied, h_tied = (vo & (msdiff == 0)).sum(), (ho & (msdiff == 0)).sum()
        n, nv, nh = int(take.sum()), int(vo.sum()), int(ho.sum())
        def frac(k,m):
            return float(k/m) if m else None
        result.append({"bin":b,"n_all":n,"n_both_correct":int((take & test.v_correct & test.h_correct).sum()),
                       "n_both_wrong":int((take & ~test.v_correct & ~test.h_correct).sum()),
                       "n_discordant":nv+nh,"n_v_only":nv,"n_h_only":nh,"p_discordant":frac(nv+nh,n),
                       "p_v_only_all":frac(nv,n),"p_h_only_all":frac(nh,n),"p_v_only_given_discordant":frac(nv,nv+nh),
                       "n_v_only_correct_ranked_lower":int(v_lost),"n_h_only_correct_ranked_lower":int(h_lost),
                       "n_v_only_ties":int(v_tied),"n_h_only_ties":int(h_tied),
                       "n_msp_chooser_wrong_given_discordance":int(v_lost+h_lost+h_tied),
                       "p_msp_chooser_wrong_given_discordance":frac(v_lost+h_lost+h_tied,nv+nh),
                       "p_v_only_msp_failure_half_tie":frac(v_lost+.5*v_tied,nv),
                       "p_h_only_msp_failure_half_tie":frac(h_lost+.5*h_tied,nh)})
        active = bin_id[discord] == b
        result[-1].update({k:float(values[active].mean()) if active.any() else None for k,values in row_metrics.items()})
    return result


def read_cohort(crop, model):
    sources = [r for r in rows(DATA / "source_index.csv") if r["cohort"] == "full"]
    sources.sort(key=lambda r: (int(r["label"]), r["source_id"]))
    ids = np.array([r["source_id"] for r in sources])
    labels = np.array([int(r["label"]) for r in sources])
    splits = np.array([r["split"] for r in sources])
    expected = set(ids)
    if len(expected) != 45000 or len(ids) != len(expected):
        raise ValueError("Expected 45,000 distinct confirmation images")
    def load(name):
        return keyed(rows(DATA / name), expected, name)
    pairs = load(f"full_{crop}_{model}_pairs.csv")
    windows = load(f"full_{crop}_windows.csv")
    features = load("whole_source_features.csv")
    def column(table, name, dtype):
        return np.array([table[key][name] for key in ids], dtype=dtype)
    for part, n in (("calibration", 10), ("test", 40)):
        values, counts = np.unique(labels[splits == part], return_counts=True)
        if not np.array_equal(values, np.arange(100, 1000)) or not np.all(counts == n):
            raise ValueError(f"Invalid {part} split")
    for table in (pairs, features):
        if not np.array_equal(column(table, "label", int), labels) or not np.array_equal(column(table, "split", str), splits):
            raise ValueError("Source assignments disagree")
    def duration(direction):
        return (column(windows, "last_used_us_" + direction, int) - column(windows, "first_used_us_" + direction, int)) / 1000
    features_x = np.column_stack((
        column(pairs, "msp_v", float) - column(pairs, "msp_h", float),
        np.log(column(windows, "raw_events_v", int) / column(windows, "raw_events_h", int)),
        duration("v") - duration("h"), column(features, "orientation", float)))
    if not np.isfinite(features_x).all():
        raise ValueError("Invalid orientation-model inputs")
    cohort = Cohort(ids, labels, features_x, column(pairs, "pred_v", int) == labels, column(pairs, "pred_h", int) == labels)
    return cohort.subset(splits == "calibration"), cohort.subset(splits == "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", choices=("events30k", "time20"))
    parser.add_argument("model", choices=("histogram", "timestamp"))
    parser.add_argument("--output", type=Path, help="JSON output; defaults to stdout")
    args = parser.parse_args()
    train, test = read_cohort(args.crop, args.model)
    calibration, heldout = train.subset(train.discord), test.subset(test.discord)
    pair = fit_pair(calibration.X, calibration.v_correct.astype(float), np.ones(len(calibration.ids)))
    y = heldout.v_correct.astype(float)
    point = measures(pair, heldout.X, y, np.ones(len(y)))
    draws = bootstrap(train, test, NBOOT)
    bounds = np.percentile(draws, (2.5, 97.5), axis=0)
    edges, bins = orient_bins(train, test)
    result = {"crop": args.crop, "model": args.model, "bootstrap_draws": NBOOT,
              "calibration_discordant": len(calibration.ids), "test_discordant": len(heldout.ids),
              "models": {"baseline": pair.baseline.report(), "augmented": pair.augmented.report()},
              "estimands": {metric: {"value": float(v), "ci95": ci.tolist()}
                            for metric, v, ci in zip(METRICS, point, bounds.T)},
              "assignment_control": assignment_control(heldout.labels, heldout.X, y, pair),
              "quintile_edges": edges.tolist(), "quintiles": binned_report(test, bins, pair)}
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
