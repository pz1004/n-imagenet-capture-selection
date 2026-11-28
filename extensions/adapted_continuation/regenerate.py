"""Score chronological blocks with the released Histogram and five frozen per-trajectory states."""
import argparse
import csv
import json
import sys
import tempfile
import zipfile
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "regeneration"))
from continuation import windows
from infer import DATA, classifier, csv_rows, recording_key, scores, sha256, verify_file

CHANNELS = {"variant1": (("first", 0), ("second", 1)), "variant2": (("first", 2),), "original": (("first", 3),)}


def reference():
    groups = {}
    with np.load(HERE / "inputs/scores.npz", allow_pickle=False) as table:
        for split in ("policy","test"):
            ids = table[split + "_ids"].tolist()
            groups[split] = (dict(zip(ids, range(len(ids)))), table[split + "_labels"],
                             table[split + "_class"], table[split + "_msp"], table[split + "_eligible"])
    return groups


def load_states(args, weights, archive):
    receipt = json.loads((args.states / "receipt.json").read_text())
    if (receipt["trajectory"], receipt["archive_sha256"], receipt["checkpoint_sha256"]) != (
            args.trajectory, archive["sha256"], weights["sha256"]):
        raise ValueError("The adaptation receipt does not match this trajectory and base checkpoint")
    models = {}
    for seed in [0, *args.seeds]:
        model = classifier(args.histogram, "histogram", weights, 1000, args.device)
        if seed:
            path = args.states / f"seed{seed}_bn.pt"
            if receipt["sha256"].get(path.name) != sha256(path):
                raise ValueError(f"State digest mismatch for {path.name}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            bn = {name for name, layer in model.named_modules() if isinstance(layer, torch.nn.BatchNorm2d)}
            if set(state) != {name for name in model.state_dict() if name.rsplit(".", 1)[0] in bn}:
                raise ValueError(f"State {seed} does not cover every batch-normalization parameter and buffer")
            model.load_state_dict(state, strict=False)
        models[seed] = model.requires_grad_(False).eval()
    return models


@torch.inference_mode()
def score(args, destination):
    inventory = {(r["kind"], r["condition_or_model"]): r for r in csv_rows(DATA / "input_releases.csv")
                 if r["model_set"] == "full"}
    official = inventory["event_archive", args.trajectory]
    weights = inventory["checkpoint", "histogram"]
    verify_file(args.archive, official)
    groups = reference()
    manifest = {r["source_id"]: r for r in csv_rows(DATA / "source_index.csv")}
    expected = {k: split for split, (index, *_) in groups.items() for k in index}
    models = load_states(args, weights, official)
    channels = CHANNELS[args.trajectory]
    with ExitStack() as stack:
        archive = stack.enter_context(zipfile.ZipFile(args.archive))
        items = [(recording_key(item.filename), item) for item in archive.infolist() if not item.is_dir()]
        members = {k: item for k, item in items if k is not None}
        if len(members) != 50000 or len(items) != len(members) or set(members) != set(manifest):
            raise ValueError("Source identities differ from the N-ImageNet archive inventory")
        # Match each classifier's original inference batch shape even for a single-photo probe.
        ordinal = {}
        for state_group in ("released", "adapted"):
            keys = [k for k, _ in items if (manifest[k]["cohort"] == "full" if state_group == "released" else k in expected)]
            ordinal[state_group] = {k: (min(args.batch_size, len(keys) - (i // args.batch_size) * args.batch_size))
                                    for i, k in enumerate(keys)}
        rows = [(k, item) for k, item in items if k in expected and (not args.source_id or args.source_id == k)]
        if not rows or len(rows) != (1 if args.source_id else 40500):
            raise ValueError("The archive does not contain the requested evaluation or policy-calibration sources")
        logits = {(view, seed): np.lib.format.open_memmap(destination / f"{view}_seed{seed}.npy", "w+",
                                                         np.float32, (len(rows), 1000))
                  for view, _ in channels for seed in models}
        out = csv.writer(stack.enter_context((destination / "sources.csv").open("w", newline="")))
        out.writerow(["source_id", "split", "label", "raw_events", "start_us", "first_end_us", "end_us"])
        pool = stack.enter_context(ThreadPoolExecutor(4))
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            inputs = [(key, archive.read(member)) for key, member in batch]
            encoded = list(pool.map(lambda x: windows(x, args.trajectory == "variant1"), inputs))
            for i, (key, _) in enumerate(batch):
                group = groups[expected[key]]
                label = int(group[1][group[0][key]])
                if label != int(manifest[key]["label"]):
                    raise ValueError(f"Source label mismatch for {key}")
                out.writerow([key, expected[key], label, *encoded[i][1]])
            for name, channel in channels:
                batch_inputs = torch.from_numpy(np.stack([views[name] for views, _ in encoded])).to(args.device)
                for seed, model in models.items():
                    sizes = [ordinal["released" if seed == 0 else "adapted"][key] for key, _ in batch]
                    if all(size == len(batch) for size in sizes):
                        z = model(batch_inputs).cpu().numpy()
                    else:
                        z = np.empty((len(batch), 1000), dtype=np.float32)
                        for size in sorted(set(sizes)):
                            ix = np.flatnonzero(np.asarray(sizes) == size)
                            for start in range(0, len(ix), size):
                                group = ix[start:start + size]
                                x = batch_inputs[group]
                                if len(group) < size:
                                    x = torch.cat((x, x[:1].expand(size - len(group), -1, -1, -1)))
                                z[group] = model(x)[:len(group)].cpu().numpy()
                    if not np.isfinite(z).all():
                        raise ValueError(f"Nonfinite logit for {name}, seed {seed}")
                    logits[name, seed][offset:offset+len(batch)] = z
                    predictions = z.argmax(axis=1)
                    confidence = scores(z)["msp"]
                    for i, (key, _) in enumerate(batch):
                        index, _, cp, cm, ce = groups[expected[key]]
                        j = index[key]
                        if (int(cp[seed, channel, j]) != predictions[i] or
                                abs(cm[seed, channel, j] - confidence[i]) > 1e-5):
                            raise ValueError(f"Published class or confidence differs for {key}, {name}, seed {seed}: {predictions[i]} vs {cp[seed,channel,j]}, MSP {confidence[i]} vs {cm[seed,channel,j]}")
                        if bool(ce[channel, j]) != (encoded[i][1][0] >= (60000 if channel == 1 else 30000)):
                            raise ValueError(f"Published completion differs for {key}, {name}")
        for array in logits.values():
            array.flush()
    check = {"trajectory": args.trajectory, "source_rows": len(rows), "states": list(models),
             "device": args.device, "torch": str(torch.__version__), "cuda": torch.version.cuda,
             "archive_sha256": official["sha256"], "histogram_sha256": weights["sha256"],
             "sha256": {f.name: sha256(f) for f in sorted(destination.iterdir())}}
    (destination / "receipt.json").write_text(json.dumps(check, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", choices=CHANNELS)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--histogram", type=Path, required=True, help="Official full 1,000-class checkpoint")
    parser.add_argument("--states", type=Path, required=True, help="Trajectory output from extensions/multiseed/regenerate_histogram.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", choices=range(1,6), type=int, default=list(range(1,6)))
    parser.add_argument("--source-id", help="Restrict to one policy or evaluation photograph")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or len(args.seeds) != len(set(args.seeds)) or args.batch_size < 1:
        parser.error("Supply a new output directory, distinct seeds and a positive batch size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        dest = Path(scratch) / "run"
        dest.mkdir()
        score(args, dest)
        dest.rename(args.output)


if __name__ == "__main__":
    main()
