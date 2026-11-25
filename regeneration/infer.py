"""Regenerate event-window predictions and logits from an official N-ImageNet archive."""
import argparse
import csv
import hashlib
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torchvision.models import resnet34


DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = ("histogram", "timestamp")
METADATA = ("raw_events", "used_events", "first_used_us", "last_used_us", "zero_span")


def hashed(seed_text):
    return int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recording_key(name):
    p = PurePosixPath(name.replace("\\", "/"))
    if "__MACOSX" in p.parts or p.name.startswith("._"):
        return None
    if p.suffix.lower() != ".npz":
        return None
    match = re.search(r"(?<![a-zA-Z0-9])(n\d{8})(?!\d)", str(p))
    if not match:
        raise ValueError(f"Event file has no synset: {p}")
    return match.group(1) + "/" + p.stem


def inputs(blob, key, crop):
    with np.load(io.BytesIO(blob), allow_pickle=False) as sample:
        ev = sample["event_data"].reshape(-1)
    n = len(ev)
    if not n:
        raise ValueError(f"Empty raw recording: {key}")
    raw_t = ev["t"].astype(np.int64)
    # One verified full vertical recording has an out-of-order final timestamp.
    # Both planned windows end before that event. Keep its original length for the crop hash.
    drops = np.flatnonzero(raw_t[1:] < raw_t[:-1]) + 1
    if drops.size:
        known = (key == "n04458633/ILSVRC2012_val_00049132" and n == 142449
                 and drops.tolist() == [n-1] and ev["t"].dtype == np.dtype("<u2")
                 and tuple(raw_t[-2:]) == (46703, 436))
        if not known:
            raise ValueError(f"Unexpected timestamp discontinuity: {key}")
    if crop == "events30k":
        size = min(30000, n)
        # Same source image gets the same fractional position in both event sequences.
        start = (hashed("crop/" + key) * (n - size + 1)) >> 64
        stop = start + size
    else:
        start = 0
        stop = int(np.searchsorted(raw_t, raw_t[0] + 20000, side="left"))
    if drops.size:
        boundary = np.flatnonzero(raw_t >= raw_t[0] + 20000)
        if stop > drops[0] or not len(boundary) or np.searchsorted(raw_t, raw_t[0]+20000) != boundary[0]:
            raise ValueError(f"Discontinuity intersects the selected window: {key}")
    ev = ev[start:stop]
    t = raw_t[start:stop]
    span = int(t[-1] - t[0])
    if np.any(ev["x"] >= 640) or np.any(ev["y"] >= 480) or np.any(ev["x"] < 0) or np.any(ev["y"] < 0):
        raise ValueError(f"Sensor coordinates outside 640x480: {key}")
    p = np.asarray(ev["p"])
    if not np.all(np.isin(p, (-1, 0, 1))):
        raise ValueError(f"Unexpected polarity values: {key}")
    x = (ev["x"].astype(np.float64) * (224 / 640)).astype(np.int64)
    y = (ev["y"].astype(np.float64) * (224 / 480)).astype(np.int64)
    # Channel 0 is ON, channel 1 is OFF, exactly as in the authors' loader.
    idx = x + 224 * y + (p <= 0).astype(np.int64) * (224 * 224)
    length = 2 * 224 * 224
    count = np.bincount(idx, minlength=length).astype(np.float32)
    surface = np.zeros(length, dtype=np.float32)
    if span:
        np.maximum.at(surface, idx, (t - t[0]) / span)
    info = (n, len(ev), int(t[0]), int(t[-1]), int(span == 0))
    return {"histogram": count.reshape(2, 224, 224),
            "timestamp": surface.reshape(2, 224, 224)}, info


def scores(logits):
    x = logits.astype(np.float64)
    ordered = np.partition(x, -2, axis=1)
    mx = ordered[:, -1]
    return {"msp": 1 / np.exp(x - mx[:, None]).sum(axis=1),
            "margin": mx - ordered[:, -2]}


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_file(path, release):
    if path.stat().st_size != int(release["size_bytes"]) or sha256(path) != release["sha256"]:
        raise ValueError(f"File does not match the pinned release for {release['condition_or_model']}")


def classifier(path, kind, release, n_classes, device):
    verify_file(path, release)
    model = resnet34(weights=None, num_classes=n_classes)
    model.conv1 = torch.nn.Conv2d(2, 64, kernel_size=14, stride=2, padding=3, bias=False)
    state = torch.load(path, weights_only=True, map_location="cpu")["state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def infer(args, directory):
    releases = {(r["kind"], r["condition_or_model"]): r for r in csv_rows(DATA / "input_releases.csv") if r["model_set"] == args.cohort}
    verify_file(args.archive, releases["event_archive", args.condition])
    families = {args.cohort}
    if args.cohort == "full" and args.condition == "original":
        families.add("mini")
    study_sources = csv_rows(DATA / "source_index.csv")
    labels = {r["source_id"]: int(r["label"]) for r in study_sources
              if r["cohort"] in families and (args.split is None or r["split"] == args.split)}
    n_classes = 1000 if args.cohort == "full" else 100
    models = {k: classifier(getattr(args, k), k, releases["checkpoint", k], n_classes, args.device) for k in MODELS}
    with zipfile.ZipFile(args.archive) as archive:
        entries = [(recording_key(m.filename), m) for m in archive.infolist()]
        names = [key for key, _ in entries if key is not None]
        expected = 50000 if args.cohort == "full" else 5000
        if len(names) != expected or len(set(names)) != expected or not labels.keys() <= set(names):
            raise ValueError("Archive source IDs differ from the pinned study population")
        selected = [(key, m) for key, m in entries if key in labels and (args.source_id is None or key == args.source_id)]
        if not selected:
            raise ValueError("No matching source in the requested cohort and condition")
        arrays = {k: np.lib.format.open_memmap(directory/(k+".npy"), "w+", np.float32,
                                                shape=(len(selected), n_classes)) for k in MODELS}
        fields = ["source_id", "label", *METADATA, *(k+"_"+v for k in MODELS for v in ("pred", "msp", "margin"))]
        with (directory/"predictions.csv").open("w", newline="") as f, torch.inference_mode():
            writer = csv.writer(f)
            writer.writerow(fields)
            for start in range(0, len(selected), args.batch_size):
                entries_batch = selected[start:start + args.batch_size]
                batch = [inputs(archive.read(m), key, args.crop) for key, m in entries_batch]
                values = []
                for k, net in models.items():
                    x = torch.from_numpy(np.stack([encoded[k] for encoded, _ in batch])).to(args.device)
                    logits = net(x).cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise ValueError("Nonfinite network output")
                    arrays[k][start:start+len(batch)] = logits
                    summaries = scores(logits)
                    values.extend((logits.argmax(axis=1), summaries["msp"], summaries["margin"]))
                for i, (key, _) in enumerate(entries_batch):
                    writer.writerow([key, labels[key], *batch[i][1], *(v[i] for v in values)])
        for arr in arrays.values():
            arr.flush()
    (directory/"settings.json").write_text(json.dumps({"cohort": args.cohort, "condition": args.condition,
        "crop": args.crop, "count": len(selected), "batch_size": args.batch_size, "device": str(args.device),
        "torch": str(torch.__version__), "numpy": np.__version__, "source_id": args.source_id, "split":args.split}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("mini", "full"))
    parser.add_argument("condition", choices=("original", "vertical", "horizontal", "variant1", "variant2"))
    parser.add_argument("crop", choices=("events30k", "time20"))
    parser.add_argument("--archive", type=Path, required=True)
    for model in MODELS:
        parser.add_argument("--"+model, type=Path, required=True, help=f"Official {model} checkpoint for this cohort")
    parser.add_argument("--output", type=Path, required=True, help="A new directory for the two logit arrays and their row mapping")
    parser.add_argument("--source-id", help="Optionally evaluate one synset/filename-stem")
    parser.add_argument("--split", choices=("calibration", "test"), help="Process one frozen split in archive order")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.condition in ("variant1", "variant2") and args.cohort != "full":
        parser.error("The published variant 1 and 2 measurements use the full checkpoints")
    if args.output.exists() or args.batch_size < 1:
        parser.error("Specify a new output directory and a positive batch size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        stage = Path(scratch) / "inference"
        stage.mkdir()
        infer(args, stage)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
