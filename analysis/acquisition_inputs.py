"""Validate compact score tables against the published N-ImageNet source split."""
import json
from pathlib import Path

import numpy as np

from capture_policies import read, sorted_ids

ROOT = Path(__file__).resolve().parents[1]


def source_groups(table, calibration_split="calibration"):
    """Return source IDs and class labels in the original within-class hash order."""
    manifest = read(ROOT / "data/source_index.csv", "cohort", "full")
    groups = {}
    for split in (calibration_split, "test"):
        full_split = "calibration" if split == "policy" else split
        expected_ids = sorted_ids(manifest, full_split)
        if split == "policy":
            expected_ids = [k for i, k in enumerate(expected_ids) if i % 10 >= 5]
        labels = np.array([int(manifest[k]["label"]) for k in expected_ids])
        if table[f"{split}_ids"].tolist() != expected_ids or not np.array_equal(table[f"{split}_labels"], labels):
            raise ValueError(f"Score rows disagree with the source manifest for {split}")
        groups[split] = (expected_ids, labels)
    return groups


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
