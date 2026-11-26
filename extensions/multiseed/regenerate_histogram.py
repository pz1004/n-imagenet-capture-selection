"""Regenerate one trajectory's frozen Histogram Ev-TTA states and single-crop scores."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extensions/adaptation"))
from regenerate import UPSTREAM_COMMIT, ev_loss, load_split, loss_components, tensor
from infer import DATA, channel_indices, classifier, csv_rows, hashed, inputs, recording_key, scores, sha256

REFERENCE = {"vertical", "horizontal"}
PATHS = ("vertical", "horizontal", "variant1", "variant2", "original")
PIXELS = 224 * 224


def crop(indices, key, prefix):
    size = min(len(indices), 30000)
    start = hashed(prefix + key) * (len(indices) - size + 1) >> 64
    return np.bincount(indices[start:start + size], minlength=2 * PIXELS).reshape(2, 224, 224).astype(np.float32)


def decode(key_blob):
    key, blob = key_blob
    with np.load(io.BytesIO(blob), allow_pickle=False) as sample:
        events = sample["event_data"].reshape(-1)
        if not len(events):
            raise ValueError(f"Empty recording: {key}")
        return channel_indices(events, key).astype(np.int32)


def cache_adaptation(archive, members, adapt, pool):
    cached = {}
    for start in range(0, len(adapt), 64):
        batch = adapt[start:start + 64]
        cached.update(zip(batch, pool.map(decode, [(k, archive.read(members[k])) for k in batch])))
    return cached


def fit(net, cached, seed, pool, components):
    prefix = "" if seed == 1 else f"seed-{seed}/"
    ordered = sorted(cached, key=lambda key: hashed(prefix + "adapt-order/" + key))
    net.requires_grad_(False).train()
    bn_names = set()
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.reset_running_stats()
            layer.requires_grad_(True)
            bn_names.add(name)
    optimizer = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=0.00025, weight_decay=0.0001)
    def views(key):
        indices = cached[key]
        return [crop(indices, key, "crop/")] + [
            crop(indices, key, prefix + f"adaptation-view-{view}/") for view in range(1, 4)
        ]

    batches = []
    for start in range(0, len(ordered), 64):
        keys = ordered[start:start + 64]
        encoded = list(pool.map(views, keys))
        data = tensor([v[0] for v in encoded] + [a for v in encoded for a in v[1:]])
        optimizer.zero_grad(set_to_none=True)
        loss, n_agree = ev_loss(net(data), len(keys), components)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite adaptation loss")
        loss.backward()
        optimizer.step()
        batches.append((len(keys), float(loss.detach()), n_agree))
        if start % 1280 == 0:
            print(f"seed {seed}, adaptation {start}/{len(ordered)}", flush=True)
    net.requires_grad_(False).eval()
    state = {key: value.detach().cpu() for key, value in net.state_dict().items()
             if key.rsplit(".", 1)[0] in bn_names}
    return state, {"order_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
                   "batch_losses_and_counts": batches}


def checked_archive(stack, path, record, cache):
    if path.stat().st_size != int(record["size_bytes"]):
        raise ValueError("Wrong archive size")
    source = stack.enter_context(path.open("rb"))
    copy = stack.enter_context(tempfile.TemporaryFile(dir=cache)) if cache else None
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(16 * 1024**2), b""):
        digest.update(block)
        if copy is not None:
            copy.write(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError("Archive SHA-256 differs from the pinned release")
    stream = copy if copy is not None else source
    stream.seek(0)
    return stack.enter_context(zipfile.ZipFile(stream))


def check_manifest(archive, assign):
    valid = [(recording_key(item.filename), item) for item in archive.infolist() if not item.is_dir()]
    members = {key: item for key, item in valid if key is not None}
    manifest = csv_rows(DATA / "source_index.csv")
    class_labels = {r["synset"]: int(r["label"]) for r in csv_rows(DATA / "class_map.csv")}
    if any(r["source_id"].split("/", 1)[0] != r["synset"] or
           class_labels.get(r["synset"]) != int(r["label"]) for r in manifest):
        raise ValueError("A frozen source ID, synset and class label do not agree")
    expected = {r["source_id"] for r in manifest}
    if len(valid) != 50000 or len(members) != 50000 or set(members) != expected or not assign.keys() <= expected:
        raise ValueError("Archive inventory does not match the 50,000 frozen source IDs")
    return members


def reference_scores(path, condition):
    reference = {}
    with np.load(path, allow_pickle=False) as data:
        for split in (("policy",) if condition in REFERENCE else ("policy", "test")):
            ids = data[f"{split}_ids"].tolist()
            reference[split] = (
                {key: i for i, key in enumerate(ids)}, data[f"{split}_labels"],
                data[f"{split}_{condition}_class"], data[f"{split}_{condition}_msp"]
            )
    return reference


def check_scores(keys, splits, labels, predictions, confidence, reference, seed):
    maxerr = 0.0
    for split in set(splits):
        rows = [j for j, s in enumerate(splits) if s == split]
        index, ref_labels, ref_predictions, ref_confidence = reference[split]
        idx = [index[keys[j]] for j in rows]
        np.testing.assert_array_equal(labels[rows], ref_labels[idx])
        np.testing.assert_array_equal(predictions[rows], ref_predictions[seed, idx])
        expected = ref_confidence[seed, idx]
        np.testing.assert_allclose(confidence[rows], expected, rtol=0, atol=1e-5)
        maxerr = max(maxerr, float(np.max(np.abs(confidence[rows] - expected))))
    return maxerr


@torch.inference_mode()
def run_inference(models, archive, members, ids, assign, labels, pool, output, reference):
    arrays = {seed: np.lib.format.open_memmap(output / f"logits_seed{seed}.npy", mode="w+",
              dtype=np.float32, shape=(len(ids), 1000)) for seed in models}
    errors = dict.fromkeys(models, 0.0)
    with gzip.open(output / "predictions.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "split", "label"] +
                        [f"seed{seed}_{field}" for seed in models for field in ("class", "msp")])
        for start in range(0, len(ids), 32):
            keys = ids[start:start + 32]
            splits = [assign[k] for k in keys]
            y = np.array([labels[k] for k in keys])
            indices = pool.map(decode, [(k, archive.read(members[k])) for k in keys])
            x = tensor([crop(events, k, "crop/") for events, k in zip(indices, keys)])
            columns = []
            for seed, net in models.items():
                z = net(x).cpu().numpy()
                if z.shape != (len(keys), 1000) or not np.isfinite(z).all():
                    raise ValueError("Nonfinite or incomplete logits")
                arrays[seed][start:start + len(keys)] = z
                pred, conf = z.argmax(1), scores(z)["msp"]
                columns.extend((pred, conf))
                err = check_scores(keys, splits, y, pred, conf, reference, seed)
                errors[seed] = max(errors[seed], err)
            for j, key in enumerate(keys):
                writer.writerow([key, splits[j], y[j]] + [column[j] for column in columns])
            if start % 6400 == 0:
                print(f"inference {start}/{len(ids)}", flush=True)
    for arr in arrays.values():
        arr.flush()
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=PATHS)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--histogram", type=Path, required=True, help="Official full Histogram checkpoint")
    parser.add_argument("--ev-tta", type=Path, required=True, help="Checkout of the upstream Ev-TTA repository")
    parser.add_argument("--output", type=Path, required=True, help="A new result directory")
    parser.add_argument("--seeds", nargs="+", choices=range(1, 6), type=int, default=list(range(1, 6)))
    parser.add_argument("--cache-directory", type=Path, help="Optional space for a temporary copy of the ZIP")
    args = parser.parse_args()
    if args.output.exists() or len(args.seeds) != len(set(args.seeds)):
        parser.error("Supply a new output directory and distinct seeds")
    if not torch.cuda.is_available():
        parser.error("Adaptation requires CUDA PyTorch")
    torch.set_num_threads(2)
    releases = {(r["kind"], r["condition_or_model"]): r for r in csv_rows(DATA / "input_releases.csv")
                if r["model_set"] == "full"}
    components = loss_components(args.ev_tta)
    record = releases["event_archive", args.condition]
    weights = releases["checkpoint", "histogram"]
    def load_model():
        return classifier(args.histogram, "histogram", weights, 1000, "cuda")

    adapt, assign, labels, _ = load_split()
    reference = reference_scores(HERE / "inputs/scores.npz", args.condition)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        archive = checked_archive(stack, args.archive, record, args.cache_directory)
        pool = stack.enter_context(ThreadPoolExecutor(6))
        members = check_manifest(archive, assign)
        ids = [k for k in members if assign.get(k) in reference]
        for split, (index, _, _, _) in reference.items():
            if {k for k in ids if assign[k] == split} != set(index):
                raise ValueError(f"Published {split} score IDs differ from the frozen assignment")
        stage = Path(stack.enter_context(tempfile.TemporaryDirectory(dir=args.output.parent))) / "results"
        stage.mkdir()
        cached = cache_adaptation(archive, members, adapt, pool)
        check_ids = [k for k in ids if assign[k] == "policy"][:32]
        raw = []
        for key in check_ids:
            blob = archive.read(members[key])
            actual = crop(decode((key, blob)), key, "crop/")
            np.testing.assert_array_equal(actual, inputs(blob, key, "events30k")[0]["histogram"])
            raw.append(actual)
        with torch.inference_mode():
            z = load_model()(tensor(raw)).cpu().numpy()
        baseline = check_scores(check_ids, ["policy"] * len(check_ids), np.array([labels[k] for k in check_ids]),
                                z.argmax(1), scores(z)["msp"], reference, 0)
        print("Archive, split, checkpoint and 32 unadapted probes verified.", flush=True)
        models = {}
        records = {}
        for seed in sorted(args.seeds):
            torch.manual_seed(seed)
            net = load_model()
            state, records[seed] = fit(net, cached, seed, pool, components)
            torch.save(state, stage / f"seed{seed}_bn.pt")
            models[seed] = net
        del cached
        errors = run_inference(models, archive, members, ids, assign, labels, pool, stage, reference)
        receipt = {"trajectory": args.condition, "seeds": sorted(args.seeds), "evaluation_rows": len(ids),
                   "adaptation_sources": len(adapt), "archive_sha256": record["sha256"],
                   "checkpoint_sha256": weights["sha256"], "evtta_commit": UPSTREAM_COMMIT,
                   "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                   "raw_max_abs_msp_error": baseline, "max_abs_msp_errors": errors, "training": records,
                   "sha256": {f.name: sha256(f) for f in stage.iterdir() if f.is_file()}}
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        stage.rename(args.output)


if __name__ == "__main__":
    main()
