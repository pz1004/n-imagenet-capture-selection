"""Check source joins, gate choices, and rescue/loss totals in the first-window analysis."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "analysis"))
from capture_policies import read
from trajectory_data import DATA, MODELS, ids_for


def verify_one(folder, pair, model, ids, labels, fits):
    stem = f"{pair}_{model}"
    summary = json.loads((folder / f"{stem}.json").read_text())
    with gzip.open(folder / f"{stem}_decisions.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert ids == [r["source_id"] for r in rows]

    def column(name, cast):
        return np.array([cast(r[name]) for r in rows])

    assert np.array_equal(column("label",int), labels)
    v, h = column("vertical_class",int), column("horizontal_class",int)
    sv, sh = column("vertical_msp",float), column("horizontal_msp",float)
    rate = column("vertical_prefix_events",int) / 0.020
    assert np.array_equal(rate, column("vertical_events_per_s",float))
    v_correct = (v == labels).astype(int)
    for arm, cutoff in (("ref30",fits["ref30_threshold_h_minus_v"]),
                        ("ref20",fits["ref20_threshold_h_minus_v"]),("raw",0.0)):
        pred = np.where(sh - sv > cutoff,h,v)
        assert np.array_equal(pred,column(arm+"_selected_class",int))
        assert int((pred == labels).sum()) == summary["terminal"]["correct_n"][arm]
    second = column("ref30_selected_class",int)
    full_delta = (second == labels).astype(int) - v_correct
    tie = np.array([int.from_bytes(hashlib.sha256(("prefix-request/"+key).encode()).digest()[:8],"big")
                    for key in ids],dtype=np.uint64)
    for name in ("confidence","rate","confidence_plus_rate"):
        gate = fits["gates"][name]
        variables = {"confidence":[sv],"rate":[np.log1p(rate)],
                     "confidence_plus_rate":[sv,np.log1p(rate)]}[name]
        normalized = (np.column_stack(variables) - gate["mean"]) / gate["scale"]
        basis = np.column_stack((normalized, (normalized**2 - gate["sqmean"])/gate["sqscale"]))
        scores = column(name+"_score",float)
        np.testing.assert_allclose(scores, basis @ gate["slopes"] + gate["intercept"], atol=1e-11, rtol=1e-11)
        quota = np.zeros(len(ids),dtype=bool)
        quota[np.lexsort((tie,-scores))[:10800]]=True
        cutoff = (scores > gate["cutoff"]) | ((scores == gate["cutoff"]) &
                                              (tie <= np.uint64(gate["max_tie_key_at_cutoff"])))
        for mode, expected in (("quota",quota),("calibrated",cutoff)):
            arm = name+"_"+mode
            assert np.array_equal(expected,column(arm+"_requested",int))
            pred = np.where(expected,second,v)
            assert np.array_equal(pred,column(arm+"_selected_class",int))
            delta = (pred==labels).astype(int)-v_correct
            assert np.array_equal(delta,column(arm+"_change_in_correctness",int))
            published=summary["gates"][arm]
            counts=[expected.sum(), (delta==1).sum(), (delta==-1).sum(), delta.sum(),(pred==labels).sum()]
            assert counts == [published[key] for key in ("n_requested","rescued","lost","net_correct","correct")]
            random_gain = expected.mean() * full_delta.mean()
            np.testing.assert_allclose(100*(delta.mean()-random_gain),published["directed_minus_random_pp"],atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results")
    folder=parser.parse_args().results
    manifest=read(DATA/"source_index.csv","cohort","full")
    ids=ids_for("test")
    labels=np.array([int(manifest[k]["label"]) for k in ids])
    fits=json.loads((folder/"reference_calibration_fits.json").read_text())
    for pair in ("old","new"):
        for model in MODELS:
            verify_one(folder,pair,model,ids,labels,fits[model])
    print("Verified 36,000 source decisions per classifier and trajectory pair.")


if __name__ == "__main__":
    main()
