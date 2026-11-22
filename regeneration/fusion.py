"""Reproduce the exploratory two-capture probability average from saved Histogram logits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vertical", type=Path, help="Full vertical, events30k inference directory")
    p.add_argument("horizontal", type=Path, help="Full horizontal, events30k inference directory")
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = [r for r in read(root/"data/source_index.csv") if r["cohort"] == "full" and r["split"] == "test"]
    manifest.sort(key=lambda r:(int(r["label"]),hashlib.sha256(("split/"+r["source_id"]).encode()).digest()))
    keys = [r["source_id"] for r in manifest]
    labels = np.array([int(r["label"]) for r in manifest])
    logits, orders = [], []
    for direction, folder in (("vertical", args.vertical), ("horizontal", args.horizontal)):
        settings = folder/"settings.json"
        if settings.is_file():
            spec = json.loads(settings.read_text())
            if (spec["cohort"], spec["condition"], spec["crop"]) != ("full", direction, "events30k"):
                raise ValueError(f"Wrong inference settings for {direction}")
        rows = read(folder/"predictions.csv")
        pos = {row["source_id"]: i for i,row in enumerate(rows)}
        indices = [pos[k] for k in keys]
        assert len(pos)==45000 and all(int(rows[i]["label"])==y for i,y in zip(indices,labels))
        orders.append(indices)
        logits.append(np.load(folder/"histogram.npy", mmap_mode="r"))
    assert len(set(keys))==36000 and len(np.unique(labels))==900
    records = []
    for start in range(0,len(keys),128):
        scores = []
        for arr, ix in zip(logits,orders):
            z = arr[ix[start:start+128]].astype(float)
            expz = np.exp(z-z.max(axis=1,keepdims=True))
            scores.append(expz/expz.sum(axis=1,keepdims=True))
        v,h = scores
        fusion=(v+h).argmax(axis=1)
        chooser=np.where(v.max(axis=1)>=h.max(axis=1),v.argmax(axis=1),h.argmax(axis=1))
        for j,(base,other) in enumerate(zip(chooser,fusion)):
            records.append((keys[start+j],labels[start+j],int(base),int(other)))
    y,base,other=np.array([r[1:] for r in records]).T
    delta=(other==y).astype(int)-(base==y)
    rng=np.random.default_rng(20261121)
    blocks=delta.reshape(900,40)
    draws=[np.take_along_axis(blocks,rng.integers(40,size=(900,40)),axis=1).mean() for _ in range(2000)]
    args.destination.mkdir(parents=True,exist_ok=False)
    with (args.destination/"fusion.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(("source_id","label","msp_selected_class","averaged_probability_class"));w.writerows(records)
    result=dict(analysis="post hoc; held-out Event Histogram, 30000 events", n=len(y),
                selector_correct=int((base==y).sum()),fusion_correct=int((other==y).sum()),
                gained=int((delta==1).sum()),lost=int((delta==-1).sum()),delta_pp=100*delta.mean(),
                ci95_pp=(100*np.quantile(draws,[.025,.975])).tolist(),seed=20261121)
    (args.destination/"fusion.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    main()
