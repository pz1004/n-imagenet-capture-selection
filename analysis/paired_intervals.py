"""Calculate paired and whole-class intervals and the within-group randomization test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


IMAGE_SEED = 20261121
CALIBRATION_SEED = 20261122
CLASS_SEED = 20261121
PERMUTATION_SEED = 20261123
N_IMAGE = 2000
N_CLASS = 20000
N_PERMUTATIONS = 9999
DIRECTIONS = ("V_to_H", "H_to_V")


def read_pairs(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 45000 or len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Expected 45,000 distinct source IDs")
    def split_key(row):
        digest = hashlib.sha256(("split/" + row["source_id"]).encode()).digest()[:8]
        return int(row["label"]), digest
    rows.sort(key=split_key)
    if {row["split"] for row in rows} != {"calibration", "test"}:
        raise ValueError("Unrecognized split")
    classes = sorted({int(row["label"]) for row in rows})
    if len(classes) != 900:
        raise ValueError("Expected 900 classes")
    parts = []
    for part, n in (("calibration", 10), ("test", 40)):
        subset = [r for r in rows if r["split"] == part]
        labels = np.array([int(r["label"]) for r in subset])
        if not np.array_equal(labels, np.repeat(classes, n)):
            raise ValueError(f"Invalid per-class size in {part}")
        scores = np.array([[float(r["msp_v"]), float(r["msp_h"])] for r in subset])
        correct = np.array([[int(r["pred_v"]) == y, int(r["pred_h"]) == y]
                            for r, y in zip(subset, labels)])
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite score")
        parts.append((scores, correct))
    return parts[0][0], *parts[1]


def ci(draws):
    return np.quantile(draws, [0.025, 0.975]).tolist()


def ranking_by_class(scores, correct):
    # Each class contributes its two subgroup sizes and its observed and expected
    # win totals. Build both totals from the same sources. Singleton strata have
    # identical expected and matched counts and zero permutation variance.
    classes = scores.reshape(900, 40, 2)
    outcomes = correct.reshape(900, 40, 2)
    totals = np.zeros((900, 2, 3))
    matrices = [[], []]
    for i, (class_scores, class_correct) in enumerate(zip(classes, outcomes)):
        for d in range(2):
            mask = class_correct[:, d] & ~class_correct[:, 1-d]
            v, h = class_scores[mask].T
            if not len(v):
                continue
            # Keep V fixed and assign H donors to V positions, in either subgroup.
            gap = v[:, None] - h[None, :]
            if d == 1:
                gap = -gap
            credit = (gap > 0).astype(float) + 0.5 * (gap == 0)
            totals[i, d] = [len(v), np.trace(credit), credit.sum() / len(v)]
            matrices[d].append(credit)
    return totals, matrices


def balanced_ranking(totals):
    # Last two axes index direction and [group size, matched wins, null wins].
    by_direction = totals[..., 1:] / totals[..., [0]]
    return by_direction.mean(axis=-2) * 100.0


def class_intervals(totals, class_gain):
    rng = np.random.default_rng(CLASS_SEED)
    gain_draws, excess_draws = [], []
    for _ in range(N_CLASS // 100):
        classes = rng.integers(900, size=(100, 900))
        # Resample complete original classes. Repeated class draws replicate their
        # matched pairs and original pool of within-class donors as one cluster.
        matched, expected = balanced_ranking(totals[classes].sum(axis=1)).T
        excess_draws.extend(matched - expected)
        gain_draws.extend(100.0 * class_gain[classes].mean(axis=1) / 40)
    return ci(excess_draws), ci(gain_draws)


def randomization(matrices, n_directional, observed, expectation):
    # Conditional null: within each (class, directional correctness) group, all
    # assignments of the fixed H scores to the fixed V scores are equiprobable.
    # Independent uniform permutations; self-assignments allowed by design.
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.zeros(N_PERMUTATIONS)
    variance = 0.0
    for d, strata in enumerate(matrices):
        w = 50.0 / n_directional[d]
        directional_wins = np.zeros(N_PERMUTATIONS)
        for credit in strata:
            n = len(credit)
            if n == 1:
                directional_wins += credit[0, 0]
                continue
            rows = np.arange(n)
            donors = rng.permuted(np.broadcast_to(rows, (N_PERMUTATIONS, n)), axis=1)
            directional_wins += credit[rows, donors].sum(axis=1)
            # For a uniform permutation, the variance of summed assignment
            # scores is the sum of squared double-centered entries divided by n-1.
            residual = credit - credit.mean(0) - credit.mean(1)[:, None] + credit.mean()
            variance += w**2 * np.square(residual).sum() / (n - 1)
        null += w * directional_wins
    exceed = int((np.abs(null - expectation) >= abs(observed - expectation) - 1e-12).sum())
    return {
        "null": "exchangeable assignments of horizontal scores within class and directional correctness group",
        "draws": N_PERMUTATIONS, "seed": PERMUTATION_SEED,
        "two_sided_extreme_draws": exceed, "p_two_sided_plus_one": (1 + exceed) / (1 + N_PERMUTATIONS),
        "exact_null_sd_pp": float(np.sqrt(variance)), "simulated_null_sd_pp": float(null.std(ddof=1)),
        "simulated_null_mean_pct": float(null.mean()), "null_reference_95pct": ci(null),
    }


def transferred(cal_scores, test_scores, correct):
    results = []
    for source in range(2):
        threshold = np.quantile(cal_scores[:, source], 1 - .70, method="higher")
        accepted = test_scores >= threshold
        n_accepted = accepted.sum(axis=0)
        if not n_accepted.all():
            raise ValueError("Coverage zero; selective error undefined")
        cov = 100.0 * n_accepted / len(accepted)
        risk = 100.0 * (accepted & ~correct).sum(axis=0) / n_accepted
        target = 1 - source
        results.append([cov[target] - cov[source], risk[target] - risk[source]])
    return np.array(results)


def paired_image_intervals(cal_scores, test_scores, correct, gain):
    test_rng = np.random.default_rng(IMAGE_SEED)
    cal_rng = np.random.default_rng(CALIBRATION_SEED)
    gains = np.empty(N_IMAGE)
    changes = np.empty((N_IMAGE, 2, 2))
    starts = np.arange(900)[:, None]
    for i in range(N_IMAGE):
        test = (40 * starts + test_rng.integers(40, size=(900, 40))).ravel()
        cal = (10 * starts + cal_rng.integers(10, size=(900, 10))).ravel()
        gains[i] = 100 * gain[test].mean()
        changes[i] = transferred(cal_scores[cal], test_scores[test], correct[test])
    return ci(gains), changes


def calculate(path):
    cal_scores, test_scores, correct = read_pairs(path)
    gain = np.where(test_scores[:, 0] >= test_scores[:, 1], correct[:, 0], correct[:, 1]).astype(int)
    gain -= correct[:, 0]
    class_gain = gain.reshape(900, 40).sum(1)
    totals, matrices = ranking_by_class(test_scores, correct)
    overall = totals.sum(0)
    matched, re_paired = balanced_ranking(overall)
    excess_ci, class_gain_ci = class_intervals(totals, class_gain)
    gain_ci, change_draws = paired_image_intervals(cal_scores, test_scores, correct, gain)
    point = transferred(cal_scores, test_scores, correct)
    transfer = {}
    for d, direction in enumerate(DIRECTIONS):
        transfer[direction] = {}
        for j, measure in enumerate(("coverage", "selective_error")):
            transfer[direction][measure] = {
                "target_minus_source_pp": float(point[d, j]),
                "paired_refit_ci95_pp": ci(change_draws[:, d, j]),
            }
    return {
        "scope": "Post hoc uncertainty analyses, frozen 30,000-event Histogram / MSP confirmation",
        "methods": {
            "ordering": "ascending integer label, then first 64 bits of SHA-256('split/' + source_id)",
            "within_fixed_classes": {"draws": N_IMAGE, "test_seed": IMAGE_SEED, "calibration_seed": CALIBRATION_SEED,
                "method": "sample paired sources within each class; refit each cutoff on independent calibration resamples; percentile interval for target minus source on common source draws"},
            "whole_classes": {"draws": N_CLASS, "seed": CLASS_SEED,
                "method": "sample 900 whole classes with replacement, retaining their 40 test pairs; compare matched and exact expected ranking and both policies within each draw; percentile intervals"},
        },
        "pairing": {
            "direction_order": ["vertical_only", "horizontal_only"], "group_sizes": overall[:, 0].astype(int).tolist(),
            "singleton_counts": [sum(len(c) == 1 for c in direction) for direction in matrices],
            "matched_ranking_pct": float(matched), "expected_repaired_ranking_pct": float(re_paired),
            "matched_minus_expected_pp": float(matched - re_paired),
            "paired_class_ci95_pp": excess_ci,
            "randomization_test": randomization(matrices, overall[:, 0], matched, re_paired),
        },
        "selection": {"policy": "higher raw MSP versus always vertical, ties select vertical",
            "net_correct": int(gain.sum()), "gain_pp": float(100 * gain.mean()),
            "paired_within_class_ci95_pp": gain_ci, "paired_whole_class_ci95_pp": class_gain_ci,
            "classes_improved": int((class_gain > 0).sum()), "classes_tied": int((class_gain == 0).sum()),
            "classes_worsened": int((class_gain < 0).sum())},
        "cutoff_transfer": transfer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON file; writes to stdout otherwise")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "data/full_events30k_histogram_pairs.csv"
    result = json.dumps(calculate(path), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
