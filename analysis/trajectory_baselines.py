"""Verify all probability-averaging, temperature-selection and same-axis outcomes."""
import csv
import gzip
import itertools
import json
from collections import Counter

import numpy as np

from trajectory_data import DATA, TABLES


def main():
    results=json.loads((TABLES/"trajectory_baselines.json").read_text())
    seen=set()
    with gzip.open(DATA/"trajectory_baseline_decisions.csv.gz","rt",newline="") as stream:
        for config,group in itertools.groupby(csv.DictReader(stream),lambda r:(r["pair"],r["window"],r["model"])):
            tag="_".join(config)
            assert tag not in seen
            seen.add(tag)
            expected=results[tag]
            counts=Counter()
            ids=set()
            bins=np.zeros((2,2,15,3))
            for row in group:
                key=row["source_id"]
                assert key not in ids
                ids.add(key)
                label=int(row["label"])
                a,b=int(row["first"]),int(row["second"])
                hfirst,hsecond=a==label,b==label
                score=np.array([[float(row["confidence_first"]),float(row["scaled_confidence_first"])],
                                [float(row["confidence_second"]),float(row["scaled_confidence_second"])]])
                raw=a if score[0,0]>=score[1,0] else b
                scaled=a if score[0,1]>=score[1,1] else b
                assert raw==int(row["raw_choice"]) and scaled==int(row["scaled_choice"])
                for name,value in (("first",hfirst),("second",hsecond),("raw",raw==label),
                                   ("scaled",scaled==label),("fusion",int(row["probability_average"])==label),
                                   ("oracle",hfirst or hsecond),("rescues",(raw==label) and not hfirst),
                                   ("losses",hfirst and raw!=label)):
                    counts[name]+=value
                for view,hit in enumerate((hfirst,hsecond)):
                    for version,conf in enumerate(score[view]):
                        slot=min(int(conf*15),14)
                        bins[view,version,slot]+=(1,conf,hit)
            assert len(ids)==expected["n"]==36000
            assert all(counts[k]==value for k,value in expected["correct"].items())
            if "same_axis_raw" in expected:
                assert all(counts[k]==expected["same_axis_raw"][k] for k in ("rescues","losses"))
            for view,arm in enumerate(expected["pair"]):
                for version,score_kind in enumerate(("raw","scaled")):
                    reliability=expected["directional_calibration"][arm][score_kind]
                    ece=0
                    for item in reliability["bins"]:
                        n,tot_conf,correct=bins[view,version,item["bin"]]
                        np.testing.assert_allclose([n,tot_conf/n,correct/n],
                            [item["n"],item["mean_confidence"],item["accuracy"]],rtol=0,atol=1e-12)
                        ece+=abs(tot_conf-correct)/len(ids)
                    np.testing.assert_allclose(ece,reliability["ece"],rtol=0,atol=1e-12)
    assert seen==set(results) and len(seen)==12
    print("Verified all 12 two-capture configurations, 432,000 paired decisions and reliability bins.")


if __name__=="__main__":
    main()
