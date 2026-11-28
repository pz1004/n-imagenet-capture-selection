"""Recompute the fusion table and confidence intervals from published decisions and draws."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from inputs import cohort

HERE = Path(__file__).resolve().parent
METHODS = ("first", "local_selector", "arithmetic_average", "product", "stack")


def check(folder):
    point = json.loads((folder / "point.json").read_text())["results"]
    sensitivity = json.loads((folder / "budget_sensitivity.json").read_text())["methods"]
    with gzip.open(folder / "per_source.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids, labels = cohort()["test"]
    assert [r["source_id"] for r in rows] == ids

    def column(key, dtype):
        return np.array([r[key] for r in rows], dtype=dtype)

    np.testing.assert_array_equal(column("label", int), labels)
    vertical = column("vertical", int)
    horizontal = column("horizontal", int)
    first_correct = vertical == labels
    either_correct = first_correct | (horizontal == labels)
    gate = column("request_30", int).astype(bool)
    priority = np.argsort(column("first_confidence", float), kind="stable")
    expected_gate = np.zeros(len(ids), dtype=bool)
    expected_gate[priority[:10800]] = True
    np.testing.assert_array_equal(gate, expected_gate)

    for budget, requests in (("100", np.ones(len(ids), dtype=bool)), ("30", gate)):
        for method in METHODS:
            predictions = vertical if method == "first" else column(method, int)
            issued = np.zeros(len(ids), dtype=bool) if method == "first" else requests
            chosen = np.where(issued, predictions, vertical)
            correct = chosen == labels
            values = {
                "n": len(ids),
                "requests": issued.sum(),
                "correct": correct.sum(),
                "rescues": ((~first_correct) & correct).sum(),
                "losses": (first_correct & (~correct)).sum(),
                "both_wrong_rescued": ((~either_correct) & correct).sum(),
                "either_correct_missed": (either_correct & (~correct)).sum(),
                "either_correct_missed_on_requests": (issued & either_correct & (~correct)).sum(),
                "either_correct_on_requests": (issued & either_correct).sum(),
                "both_wrong_on_requests": (issued & (~either_correct)).sum(),
                "third_class_on_requests": (issued & (chosen != vertical) & (chosen != horizontal)).sum(),
            }
            saved = point[budget][method]
            for key, value in values.items():
                assert saved[key] == value, (method, budget, key)
            np.testing.assert_allclose(saved["accuracy_percent"], 100 * correct.mean())
            if budget == "30" and method != "first":
                other = ~gate
                full = predictions == labels
                rescues = (other & (~first_correct) & full).sum()
                losses = (other & first_correct & (~full)).sum()
                report = sensitivity[method]
                assert report["rescues_if_rescanned_on_other_sources"] == rescues
                assert report["losses_if_rescanned_on_other_sources"] == losses
                assert report["net_correct_30_minus_100"] == losses - rescues

    for design in ("within", "classes"):
        draws = np.load(folder / f"draws_{design}.npy", allow_pickle=False)
        assert draws.shape == (2000, 17) and np.isfinite(draws).all()
        fits = json.loads((folder / f"intervals_{design}.json").read_text())
        assert fits["replicates"] == len(draws) and draws[:, 5].max() < 1e-8
        accuracy = draws[:, 7:].reshape(-1, 2, 5) / (len(ids) / 100)
        for i, budget in enumerate(("100", "30")):
            for key, interval in fits["intervals"][budget].items():
                method, _, baseline = key.partition("_minus_")
                values = accuracy[:, i, METHODS.index(method)]
                if baseline:
                    values = values - accuracy[:, i, METHODS.index(baseline)]
                np.testing.assert_allclose(np.quantile(values, [.025, .975]), interval, atol=1e-12, rtol=0)
        for j, method in enumerate(METHODS[1:], start=1):
            difference = accuracy[:, 1, j] - accuracy[:, 0, j]
            expected = sensitivity[method]["paired_ci95"][design]
            np.testing.assert_allclose(np.quantile(difference, [.025, .975]), expected, atol=1e-12, rtol=0)
    print("Verified 36,000 paired decisions, both workloads and 4,000 bootstrap rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    check(parser.parse_args().results)
