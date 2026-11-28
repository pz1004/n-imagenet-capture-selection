"""Read transfer logits and verify their source, class and published digest."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from capture_policies import read, sorted_ids


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cohort():
    manifest = read(ROOT / "data/source_index.csv", "cohort", "full")
    groups = {}
    for split, per_class in (("calibration", 10), ("test", 40)):
        ids = sorted_ids(manifest, split)
        labels = np.array([int(manifest[source]["label"]) for source in ids])
        blocks = labels.reshape(900, per_class)
        assert np.all(blocks == blocks[:, :1]) and np.all(np.diff(blocks[:, 0]) > 0)
        groups[split] = (ids, labels)
    return groups


def capture(logit_dir, split, view, ids, labels):
    with gzip.open(ROOT / "data" / f"{view}_events30k.csv.gz", "rt", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == split]
    indices = {r["source_id"]: i for i, r in enumerate(rows)}
    if len(indices) != len(rows) or set(indices) != set(ids):
        raise ValueError(f"Source IDs differ for {split}/{view}")
    order = np.array([indices[source] for source in ids])
    labels_in = np.array([int(r["label"]) for r in rows])
    np.testing.assert_array_equal(labels_in[order], labels)
    key = f"{split}/{view}/histogram.npy"
    path = logit_dir / key
    receipt = json.loads((ROOT / "provenance" / f"{view}_{split}.json").read_text())
    actual = digest(path)
    if actual != receipt["outputs_sha256"]["events30k/histogram.npy"]:
        raise ValueError(f"Incorrect SHA-256 for {key}")
    logits = np.load(path, mmap_mode="r", allow_pickle=False)
    if logits.shape != (len(labels), 1000) or logits.dtype != np.float32:
        raise ValueError(f"Unexpected score array in {path}")
    tops = np.array([int(r["histogram_prediction"]) for r in rows])[order]
    scores = np.array([float(r["histogram_msp"]) for r in rows])[order]
    return logits, order, tops, scores, key, actual
