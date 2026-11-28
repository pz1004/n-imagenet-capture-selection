"""Post hoc paired change when a fixed terminal rule is gated instead of used for all."""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ("first","local_selector","arithmetic_average","product","stack")


def main(out):
    point=json.loads((out/"point.json").read_text())["results"]
    draws={k:np.load(out/f"draws_{k}.npy")[:,7:].reshape(2000,2,5) for k in ("within","classes")}
    data={"status":"post hoc descriptive sensitivity","denominator":36000,
          "low_confidence_requests":10800,"higher_confidence_sources":25200,
          "difference":"accuracy at 30% minus accuracy at 100%, percentage points","methods":{}}
    for j,method in enumerate(METHODS[1:],start=1):
        all_sources=point["100"][method]
        gated=point["30"][method]
        net=gated["correct"]-all_sources["correct"]
        rescues=all_sources["rescues"]-gated["rescues"]
        losses=all_sources["losses"]-gated["losses"]
        assert net==losses-rescues
        intervals={}
        for design,a in draws.items():
            assert a.shape==(2000,2,5) and np.isfinite(a).all()
            intervals[design]=np.percentile((a[:,1,j]-a[:,0,j])/360.,[2.5,97.5]).tolist()
        data["methods"][method]={"net_correct_30_minus_100":net,"difference_pp":net/360.,
                                 "rescues_if_rescanned_on_other_sources":rescues,
                                 "losses_if_rescanned_on_other_sources":losses,"paired_ci95":intervals}
    (out/"budget_sensitivity.json").write_text(json.dumps(data,indent=2)+"\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    main(parser.parse_args().results)
