"""Adapt both released classifiers on unlabeled calibration recordings, then score one trajectory."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "regeneration"))
from infer import (
    DATA,
    classifier,
    csv_rows,
    hashed,
    inputs,
    recording_key,
    scores,
    sha256,
    verify_file,
)
from replay import assignments
from trajectory_data import scan

KINDS = ("histogram", "timestamp")
STATES = ("bn", "evtta")
CONDITIONS = ("vertical", "horizontal", "variant1", "variant2")
LOSS_SHA = "e4e79eb73f4c9dadda59c531ead348dc2060502d1c516d03f37f17c52f3a1191"
UPSTREAM_COMMIT = "130fbd35132a25c609f3d32f3e32a1cc426078e2"


def loss_components(upstream):
    path = upstream / "real_cnn_model/train/loss.py"
    if sha256(path) != LOSS_SHA:
        raise ValueError(f"Expected the Ev-TTA loss from upstream commit {UPSTREAM_COMMIT}")
    spec = importlib.util.spec_from_file_location("evtta_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SoftCrossEntropy(1.0), module.KLDivLogits()


def load_split():
    manifest, calibration, policy, test = assignments()
    adapt = [key for i, key in enumerate(calibration) if i % 10 < 5]
    assignment = {
        key: group
        for group, keys in (("adapt", adapt), ("policy", policy), ("test", test))
        for key in keys
    }
    return adapt, assignment, {key: int(manifest[key]["label"]) for key in assignment}, calibration


def encode_views(item):
    key, blob = item
    main, _ = inputs(blob, key, "events30k")
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        events = data["event_data"].reshape(-1)
    out = [main]
    size = min(len(events), 30000)
    for view in range(1, 4):
        begin = (hashed(f"adaptation-view-{view}/" + key) * (len(events) - size + 1)) >> 64
        ev = events[begin : begin + size]
        t = ev["t"].astype(np.int64)
        if (np.diff(t) < 0).any():
            raise ValueError(f"Disordered auxiliary window for {key}, view {view}")
        x = (ev["x"].astype(np.float64) * (224 / 640)).astype(np.int64)
        y = (ev["y"].astype(np.float64) * (224 / 480)).astype(np.int64)
        idx = x + 224 * y + (ev["p"] <= 0).astype(np.int64) * 224**2
        count = np.bincount(idx, minlength=2 * 224**2).astype(np.float32)
        surf = np.zeros(2 * 224**2, dtype=np.float32)
        if t[-1] > t[0]:
            np.maximum.at(surf, idx, (t - t[0]) / (t[-1] - t[0]))
        out.append(dict(histogram=count.reshape(2, 224, 224), timestamp=surf.reshape(2, 224, 224)))
    return out


def anchor(item):
    key, blob = item
    return inputs(blob, key, "events30k")[0]


def tensor(array):
    return torch.from_numpy(np.stack(array)).to("cuda")


def make_models(load_model):
    states = {state: {k: load_model(k) for k in KINDS} for state in STATES}
    opts = {}
    for state, models in states.items():
        for k, net in models.items():
            net.requires_grad_(False).train()
            for layer in net.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.reset_running_stats()
                    if state == "evtta":
                        layer.requires_grad_(True)
            if state == "evtta":
                opts[k] = torch.optim.Adam(
                    (p for p in net.parameters() if p.requires_grad),
                    lr=0.00025,
                    weight_decay=0.0001,
                )
    return states, opts


def ev_loss(logits, b, components):
    """Same four-view objective as upstream EvTTACNNTrainer.train for linear motion modes."""
    entropy, kl = components
    main = logits[:b]
    aux = logits[b:].reshape(b, 3, -1)
    agreed = (aux.argmax(-1) == main.argmax(-1, keepdim=True)).sum(-1) >= 2

    def selected_entropy(x):
        return entropy(x, x, return_vector=True)[agreed].mean() if agreed.any() else x.sum() * 0.0

    loss = selected_entropy(main)
    for other in aux.unbind(1):
        loss = loss + 0.5 * (kl(main, other) + kl(other, main)) + selected_entropy(other)
    return loss, int(agreed.sum())


def adapt_batch(states, opts, encoded, start, components):
    b = len(encoded)
    metrics = {}
    for k in KINDS:
        main = tensor([views[0][k] for views in encoded])
        with torch.no_grad():
            bn_model = states["bn"][k]
            for layer in bn_model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.momentum = b / (start + b)
            bn_model(main)
        aux = tensor([a[k] for views in encoded for a in views[1:]])
        model = states["evtta"][k]
        opts[k].zero_grad(set_to_none=True)
        loss, n_agree = ev_loss(model(torch.cat((main, aux))), b, components)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite Ev-TTA loss")
        loss.backward()
        opts[k].step()
        metrics[k] = (float(loss.detach()), n_agree)
    return metrics


def check_raw(archive, members, adapt_ids, calibration, condition, pool, load_model):
    keys = adapt_ids[:32]
    encoded = list(pool.map(anchor, [(k, archive.read(members[k])) for k in keys]))
    pair = "old" if condition in ("vertical", "horizontal") else "new"
    direction = "vertical" if condition in ("vertical", "variant1") else "horizontal"
    offsets = np.array([calibration.index(k) for k in keys])
    with torch.inference_mode():
        for k in KINDS:
            reference_msp, reference_pred, _ = scan(
                pair, "calibration", direction, "events30k", k, calibration
            )
            logits = load_model(k)(tensor([item[k] for item in encoded])).cpu().numpy()
            np.testing.assert_array_equal(logits.argmax(1), reference_pred[offsets])
            np.testing.assert_allclose(
                scores(logits)["msp"], reference_msp[offsets], rtol=0, atol=1e-5
            )


def adapt(archive, members, adapt_ids, folder, pool, components, load_model):
    batch = 64
    states, opts = make_models(load_model)
    order = sorted(adapt_ids, key=lambda k: hashed("adapt-order/" + k))
    progress = []
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        encoded = list(pool.map(encode_views, [(k, archive.read(members[k])) for k in chunk]))
        metrics = adapt_batch(states, opts, encoded, start, components)
        progress.append({"n": len(chunk), **metrics})
        if start % (batch * 20) == 0:
            print("adapt", folder.name, start, len(order), flush=True)
    for state, models in states.items():
        (folder / state).mkdir(exist_ok=True)
        for k, net in models.items():
            net.eval().requires_grad_(False)
            bn_names = set(
                n for n, child in net.named_modules() if isinstance(child, torch.nn.BatchNorm2d)
            )
            bn_state = {
                name: value.detach().cpu()
                for name, value in net.state_dict().items()
                if name.rsplit(".", 1)[0] in bn_names
            }
            torch.save(bn_state, folder / state / (k + "_bn.pt"))
    return states, progress


@torch.inference_mode()
def infer(states, archive, members, assign, labels, folder, pool):
    ids = [k for k in members if k in assign and assign[k] != "adapt"]
    assert len(ids) == 40500
    arrays = {
        state: {
            k: np.lib.format.open_memmap(
                folder / state / (k + ".npy"), "w+", np.float32, shape=(len(ids), 1000)
            )
            for k in KINDS
        }
        for state in STATES
    }
    with (folder / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["source_id", "split", "label"]
            + [
                f"{state}_{k}_{field}"
                for state in STATES
                for k in KINDS
                for field in ("prediction", "msp")
            ]
        )
        for start in range(0, len(ids), 32):
            chunk = ids[start : start + 32]
            encoded = list(pool.map(anchor, [(k, archive.read(members[k])) for k in chunk]))
            tensors = {k: tensor([d[k] for d in encoded]) for k in KINDS}
            cols = []
            for state in STATES:
                for k in KINDS:
                    logits = states[state][k](tensors[k]).cpu().numpy()
                    assert logits.shape == (len(chunk), 1000) and np.isfinite(logits).all()
                    arrays[state][k][start : start + len(chunk)] = logits
                    cols.extend((logits.argmax(1), scores(logits)["msp"]))
            for i, key in enumerate(chunk):
                writer.writerow([key, assign[key], labels[key]] + [c[i] for c in cols])
            if start % 6400 == 0 or start + len(chunk) == len(ids):
                print("inference", folder.name, start + len(chunk), len(ids), flush=True)
    for models in arrays.values():
        for arr in models.values():
            arr.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--histogram", required=True, type=Path)
    parser.add_argument("--timestamp", required=True, type=Path)
    parser.add_argument(
        "--ev-tta", required=True, type=Path, help="Checkout of the official Ev-TTA repository"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for one condition"
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        help="Optionally stage the compressed archive on fast local storage",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify inputs and raw forward passes without fitting or writing results",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.check_only:
        parser.error("Use a new output directory")
    if not torch.cuda.is_available():
        parser.error("GPU regeneration requires CUDA PyTorch")
    torch.set_num_threads(2)
    torch.manual_seed(1)
    components = loss_components(args.ev_tta)
    releases = {
        (r["kind"], r["condition_or_model"]): r
        for r in csv_rows(DATA / "input_releases.csv")
        if r["model_set"] == "full"
    }

    def load_model(kind):
        return classifier(getattr(args, kind), kind, releases["checkpoint", kind], 1000, "cuda")

    adapt_ids, assign, labels, calibration = load_split()
    verify_file(args.archive, releases["event_archive", args.condition])
    with ExitStack() as stack:
        source = stack.enter_context(args.archive.open("rb"))
        if args.cache_directory:
            cached = stack.enter_context(tempfile.TemporaryFile(dir=args.cache_directory))
            shutil.copyfileobj(source, cached, 16 * 1024**2)
            cached.seek(0)
            source = cached
        archive = stack.enter_context(zipfile.ZipFile(source))
        pool = stack.enter_context(ThreadPoolExecutor(6))
        entries = [(recording_key(item.filename), item) for item in archive.infolist()]
        valid = [(k, m) for k, m in entries if k is not None]
        members = dict(valid)
        if len(valid) != 50000 or len(members) != 50000 or not assign.keys() <= members.keys():
            raise ValueError("Invalid or duplicate source identifiers in archive")
        check_raw(archive, members, adapt_ids, calibration, args.condition, pool, load_model)
        print("Archive, checkpoints and raw forward passes verified.", flush=True)
        if args.check_only:
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as sandbox:
            folder = Path(sandbox) / "scores"
            folder.mkdir()
            states, progress = adapt(
                archive, members, adapt_ids, folder, pool, components, load_model
            )
            infer(states, archive, members, assign, labels, folder, pool)
            receipt = {
                "condition": args.condition,
                "archive_sha256": releases["event_archive", args.condition]["sha256"],
                "evtta_commit": UPSTREAM_COMMIT,
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "training_batches": progress,
            }
            (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            folder.rename(args.output)


if __name__ == "__main__":
    main()
