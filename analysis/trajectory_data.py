"""Read the paired N-ImageNet trajectory measurements in canonical source order."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from capture_policies import read, sorted_ids


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables"
MODELS = ("histogram", "timestamp")
B = 2000


def ids_for(split):
    manifest = read(DATA / "source_index.csv", "cohort", "full")
    keys = sorted_ids(manifest, split)
    groups = np.asarray([int(manifest[k]["label"]) for k in keys]).reshape(900, -1)
    assert groups.shape[1] == {"calibration":10, "test":40}[split]
    assert np.all(groups == groups[:, :1]) and np.all(np.diff(groups[:, 0]) > 0)
    return keys


def scan(pair, split, direction, window, model, keys):
    """Return unscaled maximum softmax probabilities, predicted classes and labels."""
    if pair == "old":
        path = DATA / f"full_{window}_{model}_pairs.csv"
        chosen = read(path, "split", split)
        pred_col, score_col = ("pred_v","msp_v") if direction == "vertical" else ("pred_h","msp_h")
    else:
        arm = {"vertical":"variant1", "horizontal":"variant2"}[direction]
        path = DATA / f"{arm}_{window}.csv.gz"
        with gzip.open(path, "rt", newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["split"] == split]
        chosen = {r["source_id"]:r for r in rows}
        assert len(chosen) == len(rows)
        pred_col, score_col = model + "_prediction", model + "_msp"
    assert set(chosen) == set(keys)
    labels = np.array([int(chosen[k]["label"]) for k in keys])
    scores = np.array([float(chosen[k][score_col]) for k in keys])
    preds = np.array([int(chosen[k][pred_col]) for k in keys])
    return scores, preds, labels


def pair_data(pair, split, model, keys, window="events30k"):
    v, vp, labels = scan(pair, split, "vertical", window, model, keys)
    h, hp, labels_h = scan(pair, split, "horizontal", window, model, keys)
    assert np.array_equal(labels, labels_h)
    return v - h, vp == labels, hp == labels


def threshold(model):
    stored = json.loads((TABLES / "capture_policies.json").read_text())
    return stored["models"][model]["switch_to_H_if_MSP_H_minus_MSP_V_exceeds"]


def samples(design, train):
    size = 10 if train else 40
    if design == "within":
        rng = np.random.default_rng(20261122 if train else 20261121)
        offsets = np.arange(900)[:, None] * size
        for _ in range(B):
            yield (offsets + rng.integers(size, size=(900, size))).ravel()
    else:
        rng = np.random.default_rng(20261125)
        for _ in range(B):
            yield (rng.integers(900,size=(900,1))*size + np.arange(size)).ravel()
