"""Check acquisition results against paired source labels and optionally a fresh replay."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from acquisition_inputs import ROOT, source_groups

EXT = ROOT / "extensions"
NAMES = ("scan_order", "variant5", "continuation", "multiseed")


def read_rows(path):
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows, key):
    return np.array([int(r[key]) for r in rows])


def paired_counts(base, outcome, gate, reference, rescued, lost):
    delta = gate * (outcome.astype(int) - base)
    assert int((delta == 1).sum()) == reference[rescued]
    assert int((delta == -1).sum()) == reference[lost]
    return int(base.sum() + delta.sum())


def verify_order(ext, ids, labels):
    result = json.loads((ext / "results/orders.json").read_text())
    with (ext / "results/requested_discordance.csv").open(newline="") as f:
        groups = {r["order"]: r for r in csv.DictReader(f)}
    for order, info in result["orders"].items():
        rows = read_rows(ext / "results" / (order + "_test.csv.gz"))
        assert [r["source_id"] for r in rows] == ids
        first = column(rows, "first_class") == labels
        second = column(rows, "second_class") == labels
        terminal = column(rows, "terminal_class") == labels
        for gate_name in ("msp", "expected"):
            gate = column(rows, gate_name + "_request").astype(bool)
            ref = info["curves"][gate_name + "_0.3"]
            assert int(gate.sum()) == ref["requested"] == 10800
            assert paired_counts(first, terminal, gate, ref, "rescued", "lost") == ref["correct"]
            if gate_name == "msp":
                discordant = gate & (first != second)
                rec = groups[order]
                for k, v in (("discordant", discordant), ("first_correct_discordant", discordant & first),
                             ("selected_correct_discordant", discordant & terminal)):
                    assert int(v.sum()) == int(rec[k])


def verify_variant(ext, ids, labels):
    receipt = json.loads((ext / "chronology.json").read_text())
    stamps = receipt["chronology_utc"]
    assert stamps["protocol"] < stamps["calibration_freeze"] < stamps["test_start"] < stamps["test_end"]
    for filename, sha in receipt["sha256"].items():
        with (ext / filename).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == sha
    cal = json.loads((ext / "calibration_inference.json").read_text())
    test = json.loads((ext / "test_inference.json").read_text())
    assert cal["completed_utc"] < stamps["calibration_freeze"] < test["started_utc"] == stamps["test_start"]
    assert test["completed_utc"] == stamps["test_end"]
    assert test["inputs"]["freeze_sha256"] == receipt["original_calibration_freeze_sha256"]
    assert cal["inputs"]["protocol_sha256"] == test["inputs"]["protocol_sha256"] == receipt["sha256"]["PROTOCOL.md"]
    assert test["outputs_sha256"]["predictions.csv"] == receipt["original_test_predictions_sha256"]
    assert cal["outputs_sha256"]["predictions.csv"] == receipt["original_calibration_predictions_sha256"]
    for model in ("histogram", "timestamp"):
        rows = read_rows(ext / "results" / (model + "_decisions.csv.gz"))
        assert [r["source_id"] for r in rows] == ids and np.array_equal(column(rows, "label"), labels)
        result = json.loads((ext / "results" / (model + ".json")).read_text())
        first = column(rows, "vertical") == labels
        for arm in ("variant5", "horizontal", "original"):
            for gate_name in ("quota", "cutoff"):
                gate = column(rows, gate_name + "_req").astype(bool)
                for rule, key in (("accept", arm), ("local", arm + "_local_if_requested")):
                    ref = result["counts"][f"{gate_name}/{rule}/{arm}"]
                    assert int(gate.sum()) == ref["requests"]
                    assert paired_counts(first, column(rows, key) == labels, gate, ref, "rescues", "losses") == ref["correct"]


def verify_continuation(ext, ids, labels):
    rows = read_rows(ext / "results/heldout_decisions.csv.gz")
    assert [r["source_id"] for r in rows] == ids and np.array_equal(column(rows, "label"), labels)
    point = json.loads((ext / "results/point.json").read_text())
    first = column(rows, "first_class") == labels
    for name, field in (("cutoff", "calibrated_request"), ("quota", "quota_request")):
        gate = column(rows, field).astype(bool)
        ref = point[name]
        assert int(gate.sum()) == ref["requests"]
        for arm, n in ref["eligible_request"].items():
            assert (gate & column(rows, arm + "_complete").astype(bool)).sum() == n
        for arm, counts in ref["arms"].items():
            assert paired_counts(first, column(rows, arm) == labels, gate, counts, "rescues", "losses") == counts["correct"]


def verify_multiseed(ext, ids, labels, table):
    result = json.loads((ext / "results/by_seed.json").read_text())
    first_path = {"V_H": "variant1", "V_S": "variant1", "H_V": "variant2", "S_V": "original", "transport_V_H": "variant1"}
    for seed, orders in result.items():
        rows = read_rows(ext / "results" / f"decisions_seed{seed}.csv.gz")
        assert [r["source_id"] for r in rows] == ids and np.array_equal(column(rows, "label"), labels)
        for order, info in orders.items():
            first = table["test_" + first_path[order] + "_class"][int(seed)] == labels
            gate = column(rows, order + "_quota_request").astype(bool)
            predicted = column(rows, order + "_quota_class") == labels
            ref = info["quota_selected"]["30"]
            assert gate.sum() == ref["requested"] == 10800
            assert paired_counts(first, predicted, gate, ref, "rescues", "losses") == ref["policy_correct"]


def compare_json(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            compare_json(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for ai, bi in zip(a, b):
            compare_json(ai, bi)
    elif isinstance(a, (int, float)):
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-12)
    else:
        assert a == b


def compare_files(reference, new):
    if reference.suffix == ".json":
        compare_json(json.loads(reference.read_text()), json.loads(new.read_text()))
    elif reference.suffix == ".npy":
        np.testing.assert_allclose(np.load(reference), np.load(new), rtol=1e-12, atol=1e-10)
    else:
        opener = gzip.open if reference.suffix == ".gz" else open
        with opener(reference, "rt", newline="") as a, opener(new, "rt", newline="") as b:
            for x, y in itertools.zip_longest(csv.reader(a), csv.reader(b)):
                assert x is not None and y is not None and len(x) == len(y)
                if x != y:
                    for left, right in zip(x, y):
                        if left != right:
                            np.testing.assert_allclose(float(left), float(right), atol=1e-10, rtol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replayed", type=Path, help="Parent directory containing the four freshly replayed result folders")
    args = parser.parse_args()
    for name in NAMES:
        ext = EXT / name
        with np.load(ext / "inputs/scores.npz", allow_pickle=False) as table:
            split = "policy" if name == "multiseed" else "calibration"
            ids, labels = source_groups(table, calibration_split=split)["test"]
            if name == "multiseed":
                verify_multiseed(ext, ids, labels, table)
            else:
                {"scan_order": verify_order, "variant5": verify_variant, "continuation": verify_continuation}[name](ext, ids, labels)
        if args.replayed:
            fresh = args.replayed / name
            for source in (ext / "results").iterdir():
                if source.name not in ("verification.json", "terminal_tick_sensitivity.json", "requested_discordance.csv"):
                    compare_files(source, fresh / source.name)
        print(name + ": verified")


if __name__ == "__main__":
    main()
