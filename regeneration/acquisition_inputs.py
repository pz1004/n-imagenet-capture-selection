"""Build the compact two-split replay inputs from regenerated prediction tables and logits."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capture_policies import read, sorted_ids


DATA = ROOT / "data"
SPLITS = ("calibration", "test")
BLOCK = 30_000
CAPTURE_SETS = {
    "variant5": (("vertical", "vertical"), ("horizontal", "horizontal"),
                 ("original", "original"), ("variant5", "variant5")),
    "scan_order": (("variant1", "V"), ("variant2", "H"), ("original", "S")),
}


def index(path):
    return {key: (i, r) for i, (key, r) in enumerate(read(path).items())}


def split_keys():
    rows = read(DATA / "source_index.csv", "cohort", "full")
    result = {}
    for split in SPLITS:
        ids = sorted_ids(rows, split)
        if len(ids) != {"calibration": 9000, "test": 36000}[split]:
            raise ValueError(f"Unexpected {split} population")
        labels = np.asarray([int(rows[k]["label"]) for k in ids], dtype=np.int16)
        result[split] = (ids, labels)
    return result


def align(rows, ids, labels):
    entries = [rows[k] for k in ids]
    order = np.asarray([i for i, _ in entries])
    aligned = [r for _, r in entries]
    if not np.array_equal([int(r["label"]) for r in aligned], labels):
        raise ValueError("Label mismatch after joining prediction rows")
    return order, aligned


def capture_scores(runs, splits, analysis):
    out = {}
    models = ("histogram", "timestamp") if analysis == "variant5" else ("histogram",)
    for split, (ids, labels) in splits.items():
        for condition, name in CAPTURE_SETS[analysis]:
            directory = runs / condition / split
            predictions = index(directory / "predictions.csv")
            _, rows = align(predictions, ids, labels)
            if analysis == "scan_order":
                out[f"{split}_{name}_duration_ms"] = np.asarray(
                    [(int(r["last_used_us"]) - int(r["first_used_us"])) / 1000 for r in rows])
            for model in models:
                prefix = f"{split}_{name}_{model}_" if analysis == "variant5" else f"{split}_{name}_"
                out[prefix + "class"] = np.array([r[model + "_pred"] for r in rows], dtype=np.int16)
                out[prefix + "msp"] = np.array([r[model + "_msp"] for r in rows], dtype=np.float64)
    return out


def probabilities(logits):
    a = logits.astype(np.float64)
    p = np.exp(a - a.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def continuation(runs, splits):
    out = {}
    views = {"variant1": (("first", "first"), ("second", "second"), ("pooled", "pooled")),
             "variant2": (("horizontal", "first"),), "original": (("square", "first"),)}
    for split, (ids, labels) in splits.items():
        logits = {}
        for condition, names in views.items():
            directory = runs / condition
            rows_by_id = index(directory / "sources.csv")
            order, rows = align(rows_by_id, ids, labels)
            if any(r["split"] != split for r in rows):
                raise ValueError("Split mismatch after joining chronological views")
            num = np.array([r["raw_events"] for r in rows], dtype=np.int32)
            start, first_end, end = (np.array([r[col] for r in rows], dtype=np.int32)
                                     for col in ("start_us", "first_end_us", "end_us"))
            for name, file in names:
                prefix = f"{split}_{name}."
                continued = name in ("second", "pooled")
                current = np.load(directory / (file + ".npy"), mmap_mode="r")
                if current.shape != (len(rows_by_id), 1000):
                    raise ValueError(f"Unexpected {file} logit shape in {directory}")
                out[prefix + "n"] = np.clip(num - (BLOCK if continued else 0), 0, BLOCK)
                out[prefix + "us"] = (end - first_end) if continued else (first_end - start)
                out[prefix + "eligible"] = num >= (2 * BLOCK if continued else BLOCK)
                out[prefix + "class"] = np.array([r[file + "_class"] for r in rows], dtype=np.int32)
                out[prefix + "msp"] = np.array([r[file + "_msp"] for r in rows], dtype=np.float64)
                logits[name] = current, order
        for name in ("second", "horizontal", "square"):
            fused = np.empty(len(ids), dtype=np.int32)
            for begin in range(0, len(ids), 256):
                s = slice(begin, begin + 256)
                pa, pb = (probabilities(matrix[keys[s]]) for matrix, keys in (logits["first"], logits[name]))
                fused[s] = (pa + pb).argmax(axis=1)
            out[f"{split}_{name}.fused"] = fused
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=("variant5", "scan_order", "continuation"))
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Specify a new output file")
    splits = split_keys()
    if args.analysis == "continuation":
        out = continuation(args.runs, splits)
    else:
        out = capture_scores(args.runs, splits, args.analysis)
    for split, (ids, labels) in splits.items():
        out[split + "_ids"] = np.asarray(ids)
        out[split + "_labels"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        packed = Path(scratch) / "replay.npz"
        np.savez_compressed(packed, **out)
        os.rename(packed, args.output)


if __name__ == "__main__":
    main()
