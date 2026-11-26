"""Score chronological 30,000-event views in the N-ImageNet continuation experiment."""
import argparse
import csv
import io
import json
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from infer import DATA, channel_indices, classifier, csv_rows, recording_key, scores, verify_file

BLOCK = 30_000


def windows(item, continued):
    key, blob = item
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        events = z["event_data"].reshape(-1)
    n = len(events)
    if not n:
        raise ValueError(f"Empty recording: {key}")
    prefix = events[:2 * BLOCK if continued else BLOCK]
    t = prefix["t"].astype(np.int64)
    if np.any(t[1:] < t[:-1]):
        raise ValueError(f"Unordered timestamps in chronological prefix: {key}")
    index = channel_indices(prefix, key)

    def histogram(ix):
        return np.bincount(ix, minlength=2 * 224**2).astype(np.float32).reshape(2, 224, 224)

    first = histogram(index[:BLOCK])
    views = {"first": first}
    if continued:
        second = histogram(index[BLOCK:])
        views.update(second=second, pooled=first + second)
    return views, (n, int(t[0]), int(t[min(BLOCK, n) - 1]), int(t[-1]))


@torch.inference_mode()
def run(args, output):
    releases = {(r["kind"], r["condition_or_model"]): r for r in csv_rows(DATA / "input_releases.csv")
                if r["model_set"] == "full"}
    verify_file(args.archive, releases["event_archive", args.condition])
    model = classifier(args.histogram, "histogram", releases["checkpoint", "histogram"], 1000, args.device)
    sources = {r["source_id"]: r for r in csv_rows(DATA / "source_index.csv") if r["cohort"] == "full"}
    continued = args.condition == "variant1"
    view_names = ("first", "second", "pooled") if continued else ("first",)
    columns = ["source_id", "split", "label", "raw_events", "start_us", "first_end_us", "end_us"]
    columns.extend(f"{view}_{field}" for view in view_names for field in ("class", "msp"))
    with args.archive.open("rb", buffering=8 * 1024**2) as stream, zipfile.ZipFile(stream) as archive:
        members = [(recording_key(m.filename), m) for m in archive.infolist()]
        ids = [k for k, _ in members if k is not None]
        if len(ids) != 50000 or len(set(ids)) != 50000 or len(sources) != 45000 or not sources.keys() <= set(ids):
            raise ValueError("Archive source IDs differ from the 50,000-source release or frozen split")
        rows = [(k, m) for k, m in members if k in sources and (args.source_id is None or k == args.source_id)]
        if not rows:
            raise ValueError("No matching study source in this archive")
        arrays = {name: np.lib.format.open_memmap(output / (name + ".npy"), "w+", np.float32,
                                                   shape=(len(rows), 1000)) for name in view_names}
        with (output / "sources.csv").open("w", newline="") as file, ThreadPoolExecutor(4) as pool:
            writer = csv.writer(file)
            writer.writerow(columns)
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                raw = [(key, archive.read(m)) for key, m in batch]
                encoded = list(pool.map(lambda item: windows(item, continued), raw))
                values = []
                for name, array in arrays.items():
                    x = torch.from_numpy(np.stack([view[name] for view, _ in encoded])).to(args.device)
                    logits = model(x).cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise ValueError(f"Nonfinite network output for {name}")
                    array[start:start + len(batch)] = logits
                    values.extend((logits.argmax(axis=1), scores(logits)["msp"]))
                for i, (key, _) in enumerate(batch):
                    r = sources[key]
                    writer.writerow([key, r["split"], r["label"], *encoded[i][1], *(v[i] for v in values)])
        for array in arrays.values():
            array.flush()
    settings = dict(condition=args.condition, count=len(rows), batch_size=args.batch_size, device=args.device,
                    torch=str(torch.__version__), numpy=np.__version__, source_id=args.source_id,
                    archive_sha256=releases["event_archive", args.condition]["sha256"],
                    histogram_sha256=releases["checkpoint", "histogram"]["sha256"])
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=("variant1", "variant2", "original"))
    parser.add_argument("--archive", type=Path, required=True, help="Official event archive for this condition")
    parser.add_argument("--histogram", type=Path, required=True, help="Official full Event Histogram checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    parser.add_argument("--source-id", help="Optionally score one synset/filename-stem")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("Specify a new output directory and a positive batch size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        stage = Path(scratch) / "inference"
        stage.mkdir()
        run(args, stage)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
