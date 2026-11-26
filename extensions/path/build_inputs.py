"""Build square confidence and averaged-probability classes from three 30k inference runs."""
import argparse
import csv
import gzip
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from capture_policies import read
from trajectory_data import DATA, MODELS, ids_for


class Logits:
    def __init__(self, folder, model):
        with (folder/"predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.index = {r["source_id"]: i for i,r in enumerate(rows)}
        if len(rows) != len(self.index):
            raise ValueError(f"Duplicate source in {folder}")
        self.labels = np.array([int(r["label"]) for r in rows])
        self.msp = np.array([float(r[f"{model}_msp"]) for r in rows])
        pred = f"{model}_pred" if f"{model}_pred" in rows[0] else f"{model}_prediction"
        self.classes = np.array([int(r[pred]) for r in rows])
        self.logits = np.load(folder/f"{model}.npy", mmap_mode="r")
        if self.logits.shape != (len(rows),1000) or self.logits.dtype != np.float32:
            raise ValueError(f"Unexpected logits for {model} in {folder}")

    def probabilities(self, ids, labels):
        ix = [self.index[k] for k in ids]
        z = self.logits[ix].astype(np.float64)
        if not np.array_equal(self.labels[ix], labels):
            raise ValueError("Labels differ from the public source manifest")
        z -= z.max(axis=1, keepdims=True)
        weights = np.exp(z)
        probs = weights / weights.sum(axis=1, keepdims=True)
        if not np.array_equal(probs.argmax(1),self.classes[ix]) or not np.allclose(
                probs.max(1), self.msp[ix], atol=1.e-14, rtol=1.e-11):
            raise ValueError("Logits and saved classes or confidence do not agree")
        return probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("square","reference-vertical","transfer-vertical","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cal,test=ids_for("calibration"),ids_for("test")
    keys=cal+test
    manifest=read(DATA/"source_index.csv","cohort","full")
    labels=np.asarray([int(manifest[k]["label"]) for k in keys])
    sanity=read(DATA/"original_sanity_predictions.csv","model_set","full")
    scores={}
    fused={}
    for model in MODELS:
        square=Logits(args.square,model)
        first={pair:Logits(folder,model) for pair,folder in
               (("reference",args.reference_vertical),("transfer",args.transfer_vertical))}
        scores[model]=np.empty(len(keys))
        for pair in first:
            fused[pair,model]=np.empty(len(test),dtype=np.int32)
        for start in range(0,len(cal),256):
            sl=slice(start,min(start+256,len(cal)))
            prob=square.probabilities(keys[sl],labels[sl])
            assert np.array_equal(prob.argmax(1),[int(sanity[k][model+"_pred"]) for k in keys[sl]])
            scores[model][sl]=prob.max(1)
        for start in range(0,len(test),256):
            stop=min(start+256,len(test))
            idx=slice(start,stop)
            all_idx=slice(len(cal)+start,len(cal)+stop)
            y=labels[all_idx]
            prob=square.probabilities(test[idx],y)
            assert np.array_equal(prob.argmax(1),[int(sanity[k][model+"_pred"]) for k in test[idx]])
            scores[model][all_idx]=prob.max(1)
            for pair,vertical in first.items():
                fused[pair,model][idx]=(prob+vertical.probabilities(test[idx],y)).argmax(1)

    with gzip.open(args.output/"square_confidence.csv.gz","wt",newline="") as stream:
        w=csv.writer(stream); w.writerow(["source_id","label","split",*(m+"_msp" for m in MODELS)])
        for i,key in enumerate(keys):
            w.writerow([key,labels[i],manifest[key]["split"],*(scores[m][i] for m in MODELS)])
    with gzip.open(args.output/"square_fusion.csv.gz","wt",newline="") as stream:
        w=csv.writer(stream); w.writerow(["source_id","label",*(p+"_"+m for p,m in fused)])
        for i,key in enumerate(test):
            w.writerow([key,labels[len(cal)+i],*(z[i] for z in fused.values())])


if __name__=="__main__":
    main()
