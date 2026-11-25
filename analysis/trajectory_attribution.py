"""Attribute the transfer change in discordant accuracy and estimate gain per rescan."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from trajectory_data import B, MODELS, TABLES, ids_for, samples, scan, threshold


PAIRS = ("old", "new")
COUNTS = ("first_only_n", "second_only_n", "first_only_chosen_correct",
          "second_only_chosen_correct", "requested_n", "rescued_n", "lost_n")
COMPONENTS = ("change_discordant_accuracy_pp", "mix_pp",
              "first_retention_pp", "second_recovery_pp")
RATES = ("first_only_share_pct", "first_retention_pct", "second_recovery_pct",
         "discordant_accuracy_pct", "net_correct_per_100_extra_directed",
         "net_correct_per_100_extra_uniform_expected", "additional_net_per_100_vs_uniform",
         "share_of_always_two_incremental_gain_pct")


def paired_indicators(model, window, cal_ids, test_ids):
    """Align both acquisition pairs by source, with a common frozen request cutoff."""
    terminal = threshold(model)
    cal_scores = scan("old", "calibration", "vertical", window, model, cal_ids)[0]
    gate = float(np.quantile(cal_scores, 0.30, method="higher"))
    indicators = []
    reference_labels = None
    for pair in PAIRS:
        vscore, vpred, labels = scan(pair, "test", "vertical", window, model, test_ids)
        hscore, hpred, hlabels = scan(pair, "test", "horizontal", window, model, test_ids)
        if not np.array_equal(labels, hlabels):
            raise ValueError("Inconsistent ground truth across directions")
        if reference_labels is not None and not np.array_equal(labels, reference_labels):
            raise ValueError("Inconsistent ground truth across trajectory pairs")
        reference_labels = labels
        first_only = (vpred == labels) & (hpred != labels)
        second_only = (vpred != labels) & (hpred == labels)
        take_second = hscore - vscore > terminal
        requested = vscore < gate
        frozen_correct = np.where(take_second, hpred, vpred) == labels
        rescues = second_only & frozen_correct
        losses = first_only & ~frozen_correct
        indicators.append(np.column_stack((first_only, second_only, first_only & frozen_correct,
                                           rescues, requested, requested & rescues,
                                           requested & losses)))
    return gate, terminal, np.column_stack(indicators)


def metrics(counts, n):
    """The leading dimension can hold any number of bootstrap samples."""
    first, second, retained, recovered, requests, rescues, losses = counts.T
    share = first / (first + second)
    retention = retained / first
    recovery = recovered / second
    accuracy = (recovered + retained) / (first + second)
    directed_net = rescues - losses
    full_net = recovered - (first - retained)
    directed_yield = directed_net / requests
    uniform_yield = full_net / n
    # The share has an interpretation only when the all-image rule produces a gain.
    captured = np.divide(directed_net, full_net, out=np.full_like(share, np.nan),
                         where=full_net > 0)
    return 100 * np.stack((share, retention, recovery, accuracy, directed_yield,
                           uniform_yield, directed_yield - uniform_yield, captured), axis=-1)


def results(counts, n):
    """Midpoint decomposition of a = p*r1 + (1-p)*r2 across the two pairs."""
    old = metrics(counts[..., :len(COUNTS)], n)
    new = metrics(counts[..., len(COUNTS):], n)
    midpoint = (old + new) / 2
    change = new - old
    mix = change[..., 0] * (midpoint[..., 1] - midpoint[..., 2]) / 100
    first = midpoint[..., 0] * change[..., 1] / 100
    second = (1 - midpoint[..., 0] / 100) * change[..., 2]
    np.testing.assert_allclose(mix + first + second, change[..., 3], rtol=0, atol=1e-12)
    decomposition = np.stack((change[..., 3], mix, first, second), axis=-1)
    return np.concatenate((decomposition, old, new), axis=-1)


def evaluate(model, window, cal_ids, test_ids, expected_counts, expected_requests):
    gate, terminal, rows = paired_indicators(model, window, cal_ids, test_ids)
    n = len(test_ids)
    counts = rows.sum(axis=0)
    arms = {}
    for index, pair in enumerate(PAIRS):
        arm = counts[index * len(COUNTS):(index + 1) * len(COUNTS)]
        data = dict(zip(COUNTS, map(int, arm)))
        prior = expected_counts[f"{pair}_subgroups"]
        assert all(data[k] == prior[k] for k in COUNTS[:4])
        budget = expected_requests[(pair, model, window)]
        assert gate == float(budget["cutoff_msp"])
        assert terminal == float(budget["terminal_h_minus_v_threshold"])
        assert n == int(budget["n"])
        assert all(data[k] == int(budget[k]) for k in COUNTS[4:])
        full_net = data["second_only_chosen_correct"] - (data["first_only_n"] - data["first_only_chosen_correct"])
        assert full_net == int(budget["always_two_correct"]) - int(budget["first_correct"])
        assert data["rescued_n"] - data["lost_n"] == int(budget["net_correct_n"])
        arms[pair] = {**data, "all_two_net_correct_n": full_net}
    point = results(counts, n)
    for index, pair in enumerate(PAIRS):
        rates = point[len(COMPONENTS) + index * len(RATES):][:len(RATES)]
        allocation_gain = arms[pair]["requested_n"] / n * rates[6]
        np.testing.assert_allclose(allocation_gain,
                                   float(expected_requests[(pair, model, window)]["gain_vs_random_pp"]),
                                   rtol=0, atol=1e-12)

    # Each unique row is a joint outcome on the exact same source across both pairs.
    # Counting repeated outcomes avoids materializing a 36,000 by 14 matrix per draw.
    patterns, indices = np.unique(rows, axis=0, return_inverse=True)
    indices = indices.ravel()
    output_keys = list(COMPONENTS) + [f"{pair}_{rate}" for pair in PAIRS for rate in RATES]
    output = {key: {"estimate": float(value)} for key, value in zip(output_keys, point)}
    for design in ("within", "class"):
        totals = np.empty((B, rows.shape[1]), dtype=np.int64)
        for i, draw in enumerate(samples(design, train=False)):
            multiplicity = np.bincount(indices[draw], minlength=len(patterns))
            totals[i] = multiplicity @ patterns
        replications = results(totals, n)
        for key, values in zip(output_keys, replications.T):
            valid = np.isfinite(values)
            output[key][design + "_95"] = np.percentile(values[valid], [2.5, 97.5]).tolist()
            output[key][design + "_valid_draws"] = int(valid.sum())
    return {"n_sources": n, "first_scan_request_cutoff": gate, "terminal_threshold": terminal,
            "counts": arms, "results": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this file (default: stdout).")
    args = parser.parse_args()
    cal_ids, test_ids = (ids_for(split) for split in ("calibration", "test"))
    prior = json.loads((TABLES / "trajectory_effect_changes.json").read_text())["results"]
    request_rows = {}
    for pair in PAIRS:
        with (TABLES / f"trajectory_rescans_{pair}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["request_gate"] == "old_frozen" and int(row["budget_pct"]) == 30:
                    request_rows[(pair, row["checkpoint"], row["window"])] = row
    report = {
        "methods": {
            "decomposition": "Transfer minus reference, midpoint weights, in percentage points of discordant accuracy.",
            "yield": "Net additional correct choices vs fixed vertical per 100 second scans. Uniform requests have the same expected workload and terminal selector.",
            "fraction": "Directed net gain divided by always-two net gain. Replicates with nonpositive denominator excluded from this fraction alone.",
            "intervals": f"Pointwise 95% percentile intervals. {B} paired resamples within the 900 classes or of whole classes. Cutoffs are fixed.",
        },
        "model_windows": {},
    }
    for window in ("events30k", "time20"):
        for model in MODELS:
            key = window + "_" + model
            report["model_windows"][key] = evaluate(model, window, cal_ids, test_ids, prior[key], request_rows)
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
