"""Replay first-scan-confidence requests and workload-matched random requests."""
import argparse
import csv
from pathlib import Path

import numpy as np

from trajectory_data import B, MODELS, ids_for, scan, threshold


BUDGETS = (0, 10, 30, 50, 100)
INNER = np.array(BUDGETS[1:-1]) / 100
FIELDS = ("request_fraction", "scans_per_source", "rescans_per_1000", "targeted_accuracy_pct",
          "random_accuracy_pct", "gain_vs_first_pp", "gain_vs_random_pp",
          "net_correct_vs_first_per_1000", "net_correct_vs_random_per_1000")



def thresholds(cal_scores):
    return np.quantile(cal_scores, INNER, method="higher")



def counts(first, gain, scores, cutoffs):
    """The last column is the number corrected minus the number lost."""
    matrix = np.empty((len(BUDGETS), 4), dtype=np.int64)
    n = len(first)
    first_correct = int(first.sum())
    positive = scores[gain == 1]
    negative = scores[gain == -1]
    for b, t in enumerate(cutoffs, 1):
        req = np.count_nonzero(scores < t)
        rescues = np.count_nonzero(positive < t)
        losses = np.count_nonzero(negative < t)
        matrix[b] = (req, rescues, losses, rescues-losses)
    matrix[0] = (0, 0, 0, 0)
    matrix[-1] = (n,len(positive),len(negative),len(positive)-len(negative))
    return matrix, first_correct


def measures(matrix, first_correct, n):
    req = matrix[:, 0]/n
    gain = matrix[:, 3]/n
    random_gain = req * gain[-1]
    values = np.column_stack((req, 1+req, 1000*req, 100*(first_correct/n+gain),
                             100*(first_correct/n+random_gain), 100*gain, 100*(gain-random_gain),
                             1000*gain, 1000*(gain-random_gain)))
    assert values[0, 6] == 0 and values[-1, 6] == 0
    return values


def bootstrap(design, replicates, source_cutoffs, local_cal, first, gain, observed_scores):
    cal_rng = np.random.default_rng(20261122)
    test_rng = np.random.default_rng(20261121)
    class_rng = np.random.default_rng(20261125)
    test_start = np.arange(900)[:, None] * 40
    cal_start = np.arange(900)[:, None] * 10
    tpos = np.arange(40)
    cpos = np.arange(10)
    out = np.empty((replicates, 2, len(BUDGETS), len(FIELDS)))
    for i in range(replicates):
        if design == "within":
            test = (test_start + test_rng.integers(40, size=(900,40))).ravel()
            cal = (cal_start + cal_rng.integers(10, size=(900,10))).ravel()
        else:
            sampled = class_rng.integers(900, size=900)
            test = (test_start[sampled] + tpos).ravel()
            cal = (cal_start[sampled] + cpos).ravel()
        f, d, s = first[test], gain[test], observed_scores[test]
        for c, cuts in enumerate((source_cutoffs, thresholds(local_cal[cal]))):
            outcomes, correct = counts(f,d,s,cuts)
            out[i,c] = measures(outcomes, correct, len(f))
    return np.percentile(out, [2.5,97.5], axis=0)


def evaluate(pair, window, checkpoint, ids, threshold, draws):
    old_cal = scan("old","calibration","vertical",window,checkpoint,ids["calibration"])[0]
    target_cal = old_cal if pair == "old" else scan(
        "new","calibration","vertical",window,checkpoint,ids["calibration"])[0]
    first_scores, first_pred, first_labels = scan(pair,"test","vertical",window,checkpoint,ids["test"])
    second_scores, second_pred, second_labels = scan(pair,"test","horizontal",window,checkpoint,ids["test"])
    assert np.array_equal(first_labels,second_labels)
    first_correct = first_pred == first_labels
    select_second = second_scores - first_scores > threshold
    chosen = np.where(select_second, second_pred, first_pred)
    gain = (chosen == first_labels).astype(np.int8) - first_correct.astype(np.int8)
    locked, local = thresholds(old_cal), thresholds(target_cal)
    cis = {key: bootstrap(key, draws, locked, target_cal,first_correct,gain,first_scores)
           for key in ("within","class")}
    result=[]
    for c, (which, cuts) in enumerate((("old_frozen",locked),("target_local",local))):
        matrix, base = counts(first_correct,gain,first_scores,cuts)
        vals = measures(matrix,base,len(gain))
        for b,budget in enumerate(BUDGETS):
            req,rescue,loss,net=map(int,matrix[b])
            if b in (0,len(BUDGETS)-1):
                gate = np.full(len(gain),b!=0)
                cutoff = ""
            else:
                cutoff=float(cuts[b-1])
                gate=first_scores < cutoff
            via_preds=np.where(gate & select_second, second_pred, first_pred)
            correct=int(np.sum(via_preds==first_labels))
            assert int(gate.sum())==req and correct==base+net
            assert np.count_nonzero(~first_correct & (via_preds==first_labels))==rescue
            assert np.count_nonzero(first_correct & (via_preds!=first_labels))==loss
            row={"pair":pair,"window":window,"checkpoint":checkpoint,"request_gate":which,
                 "budget_pct":budget,"cutoff_msp":cutoff,"terminal_h_minus_v_threshold":threshold,
                 "n":len(gain),"first_correct":base,"always_two_correct":int((chosen==first_labels).sum()),
                 "requested_n":req,"rescued_n":rescue,"lost_n":loss,"net_correct_n":net,
                 "targeted_correct_n":correct}
            for j, field in enumerate(FIELDS):
                row[field] = float(vals[b,j])
                for design, ci in cis.items():
                    row[field+"_"+design+"_lo"],row[field+"_"+design+"_hi"]=map(float,ci[:,c,b,j])
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=("old","new"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = {split: ids_for(split) for split in ("calibration","test")}
    rows = []
    for window in ("events30k","time20"):
        for model in MODELS:
            rows.extend(evaluate(args.pair,window,model,ids,threshold(model),B))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
