"""Fit exploratory capture selectors and compare policies on the held-out source images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPLICATES = 2000
TEST_SEED = 20261121
CAL_SEED = 20261122


def read(path, field=None, value=None):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if field is None or r[field] == value]
    by_id = {r["source_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate source identifier in {path}")
    return by_id


def sorted_ids(manifest, split):
    def order(source):
        prefix = hashlib.sha256(("split/" + source).encode()).digest()[:8]
        return int(manifest[source]["label"]), prefix
    return sorted((s for s, row in manifest.items() if row["split"] == split), key=order)


def inputs(root):
    manifest = read(root / "data/source_index.csv", "cohort", "full")
    original = read(root / "data/original_sanity_predictions.csv", "model_set", "full")
    fusion = read(root / "data/fusion.csv")
    calibration, test = (sorted_ids(manifest, s) for s in ("calibration", "test"))
    assert len(manifest) == 45000 and len(original) == 50000 and len(test) == 36000
    assert len(calibration) == 9000 and fusion.keys() == set(test)
    assert manifest.keys() <= original.keys()
    assert all(original[k]["label"] == row["label"] for k, row in manifest.items())
    labels = set()
    for keys, size in ((calibration, 10), (test, 40)):
        grouped = np.array([int(manifest[k]["label"]) for k in keys]).reshape(900, size)
        assert (grouped == grouped[:, :1]).all() and np.all(np.diff(grouped[:, 0]) > 0)
        assert not labels or labels == set(grouped[:, 0])
        labels = set(grouped[:, 0])
    assert all(fusion[k]["label"] == manifest[k]["label"] for k in test)
    return manifest, original, fusion, calibration, test


def features(pairs, keys):
    v = np.array([pairs[k]["pred_v"] == pairs[k]["label"] for k in keys])
    h = np.array([pairs[k]["pred_h"] == pairs[k]["label"] for k in keys])
    x = np.array([float(pairs[k]["msp_v"]) - float(pairs[k]["msp_h"]) for k in keys])
    assert np.isfinite(x).all()
    return x, v, h


def fit(x, y):
    """Penalized binomial likelihood with unpenalized intercept and unit ridge slope."""
    center, scale = x.mean(), x.std()
    if not (0 < y.sum() < len(y)) or scale <= 0:
        raise ValueError("Both directional outcomes and variable scores are required")
    z = (x - center) / scale
    intercept, slope = np.log(y.mean() / (1 - y.mean())), 0.
    for _ in range(50):
        eta = intercept + slope * z
        probability = np.exp(-np.logaddexp(0, -eta))
        error = y - probability
        weight = probability * (1 - probability)
        g0, g1 = error.sum(), np.dot(error, z) - slope
        if max(abs(g0), abs(g1)) < 1e-10 * len(y):
            return float(center), float(scale), float(intercept), float(slope)
        a, b, d = weight.sum(), np.dot(weight, z), 1 + np.dot(weight, z * z)
        determinant = a * d - b * b
        intercept += (d * g0 - b * g1) / determinant
        slope += (a * g1 - b * g0) / determinant
    raise ArithmeticError("Penalized logistic fit did not converge")


def fit_discordant(x, v, h):
    mask = v != h
    return fit(x[mask], v[mask].astype(float))


def choose_vertical(x, fitted):
    center, scale, intercept, slope = fitted
    return intercept + slope * ((x - center) / scale) >= 0


def compare(model, root, manifest, original, fusion, cal, test):
    filename = f"events30k_{model}_pairs.csv"
    pairs = read(root / "data" / ("full_" + filename))
    assert pairs.keys() == manifest.keys()
    assert all(row["label"] == manifest[k]["label"] and row["split"] == manifest[k]["split"]
               for k, row in pairs.items())
    xc, vc, hc = features(pairs, cal)
    xt, vt, ht = features(pairs, test)
    fitted = fit_discordant(xc, vc, hc)
    center, scale, intercept, slope = fitted
    assert slope > 0
    square_col = model + "_pred"
    square = np.array([original[k][square_col] == manifest[k]["label"] for k in test])
    raw = np.where(xt >= 0, vt, ht)
    selected = choose_vertical(xt, fitted)
    adjusted = np.where(selected, vt, ht)
    policies = {"vertical": vt, "horizontal": ht, "raw_msp": raw, "adjusted": adjusted,
                "square": square, "oracle": vt | ht}
    if model == "histogram":
        assert all(pairs[k]["pred_v" if xt[i] >= 0 else "pred_h"] == fusion[k]["msp_selected_class"]
                   for i, k in enumerate(test))
        policies["fusion"] = np.array([fusion[k]["averaged_probability_class"] == manifest[k]["label"]
                                       for k in test])
    comparisons = [("raw_msp", "vertical"), ("adjusted", "vertical"), ("adjusted", "raw_msp"),
                   ("square", "vertical"), ("square", "raw_msp"), ("square", "adjusted"), ("square", "oracle")]
    if "fusion" in policies:
        comparisons += [("fusion", "raw_msp"), ("adjusted", "fusion"), ("square", "fusion")]
    rng_test, rng_cal = (np.random.default_rng(s) for s in (TEST_SEED, CAL_SEED))
    test_starts, cal_starts = (np.arange(900)[:, None] * size for size in (40, 10))
    draws = {key: np.empty(REPLICATES) for key in policies}
    for i in range(REPLICATES):
        ti = (test_starts + rng_test.integers(40, size=(900, 40))).ravel()
        ci = (cal_starts + rng_cal.integers(10, size=(900, 10))).ravel()
        refit = fit_discordant(xc[ci], vc[ci], hc[ci])
        for name, outcomes in policies.items():
            if name == "adjusted":
                draws[name][i] = np.count_nonzero(np.where(choose_vertical(xt[ti], refit), vt[ti], ht[ti]))
            else:
                draws[name][i] = np.count_nonzero(outcomes[ti])
    n = len(test)
    accuracy = {key: {"correct": int(val.sum()), "fraction": float(val.mean()),
                      "ci95_fraction": np.percentile(draws[key] / n, [2.5, 97.5]).tolist()}
                for key, val in policies.items()}
    differences = {}
    for which, ref in comparisons:
        a, b = policies[which], policies[ref]
        gain, loss = int(np.count_nonzero(a & ~b)), int(np.count_nonzero(~a & b))
        differences[which + "_minus_" + ref] = {
            "gain_count": gain, "loss_count": loss, "net_count": gain - loss,
            "delta_pp": 100 * (gain - loss) / n,
            "ci95_pp": np.percentile(100 * (draws[which] - draws[ref]) / n, [2.5, 97.5]).tolist()}
    return {"calibration_discordant": int((vc != hc).sum()), "calibration_v_only": int((vc & ~hc).sum()),
            "calibration_h_only": int((hc & ~vc).sum()), "fit_center": center, "fit_scale": scale,
            "fit_intercept": intercept, "fit_slope": slope,
            "switch_to_H_if_MSP_H_minus_MSP_V_exceeds": float(scale * intercept / slope - center),
            "adjusted_horizontal_choices": int((~selected).sum()), "accuracy": accuracy,
            "paired_contrasts": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, original, fusion, cal, test = inputs(root)
    result = {"analysis": "exploratory; 30,000 events; square is a training-condition contextual reference",
              "calibration_n": len(cal), "heldout_n": len(test), "classes": 900,
              "calibration_discordant_outcome": "vertical-only correctness; predictor MSP_V minus MSP_H",
              "fit": "standardized signed difference; summed logistic loss + 0.5 * slope^2; unpenalized intercept; predicted p>=0.5 selects vertical",
              "test_order_sha256": hashlib.sha256(("\n".join(test) + "\n").encode()).hexdigest(),
              "bootstrap": {"replicates": REPLICATES, "test_seed": TEST_SEED, "calibration_seed": CAL_SEED,
                            "method": "independent within-class source-pair calibration and test draws; select discordances and refit standardization and logistic coefficients in every calibration draw; paired percentile intervals"},
              "models": {}}
    for model in ("histogram", "timestamp"):
        result["models"][model] = compare(model, root, manifest, original, fusion, cal, test)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
