"""Decompose the previously measured hash-crop order effect by request overlap."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "multiseed/results"

def main(dest):
    old=json.loads((SRC/"by_seed.json").read_text())
    out=[]
    for seed in range(6):
        with gzip.open(SRC/f"decisions_seed{seed}.csv.gz","rt",newline="") as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==len({r["source_id"] for r in rows})==36000
        y=np.array([int(row["label"]) for row in rows])
        for left,right in (("S_V","V_S"),("H_V","V_H")):
            def numbers(k):
                return np.array([int(r[k]) for r in rows])
            lreq,rreq=(numbers(arm+"_quota_request").astype(bool) for arm in (left,right))
            l,r=(numbers(arm+"_quota_class")==y for arm in (left,right))
            for arm,p in ((left,l),(right,r)):
                assert p.sum()==old[str(seed)][arm]["quota_selected"]["30"]["policy_correct"]
            for status,subset in (("neither",~lreq&~rreq),("left_only",lreq&~rreq),("right_only",~lreq&rreq),("both",lreq&rreq)):
                out.append({"seed":seed,"left":left,"right":right,"request_status":status,"n":int(subset.sum()),
                            "left_only_correct":int((l & ~r & subset).sum()),"right_only_correct":int((~l & r & subset).sum()),
                            "net_correct":int(l[subset].sum()-r[subset].sum())})
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"hash_order_breakdown.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(out[0]));wr.writeheader();wr.writerows(out)
    for pair in ("S_V","H_V"):
        for setname in ("neither","left_only","right_only","both"):
            use=[r for r in out if r["left"]==pair and r["request_status"]==setname]
            print(pair,setname,use[0]["net_correct"],round(np.mean([r["net_correct"] for r in use[1:]]),1),
                  round(np.mean([r["n"] for r in use[1:]]),1))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output directory for hash_order_breakdown.csv")
    main(parser.parse_args().output)
