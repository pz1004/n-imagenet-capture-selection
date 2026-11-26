"""Verify saved decisions and archived results from source IDs and scalar counts."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
MANIFEST = PROJECT / "data" / "source_index.csv"


def read_csv(path):
    f = gzip.open(path, "rt") if path.suffix == ".gz" else path.open()
    with f:
        return list(csv.DictReader(f))


def load_json(path):
    return json.loads(path.read_text())


def check_arms(record, sources, archived):
    n = len(sources)
    y = [int(row["label"]) for row in sources]
    v = [int(row["vertical_class"]) for row in sources]
    g = [bool(int(row["requested"])) for row in sources]
    ok_v = [a == b for a,b in zip(v,y)]
    assert g == [float(row["vertical_msp"]) < record["request_cutoff"] for row in sources]
    assert sum(g) == record["requests"]
    ids = [row["source_id"] for row, take in zip(sources,g) if take]
    assert hashlib.sha256(("\n".join(ids)+"\n").encode()).hexdigest() == record["mask_hash"]
    for arm, target in record["counts"].items():
        if arm.startswith("paired_"):
            rule = arm.removeprefix("paired_")
            sq = [int(row["square_" + rule + "_directed_class"]) == y[i] for i, row in enumerate(sources)]
            hz = [int(row["horizontal_" + rule + "_directed_class"]) == y[i] for i, row in enumerate(sources)]
            assert sum(a and not b for a,b in zip(sq,hz)) == target["square_only_correct"]
            assert sum(b and not a for a,b in zip(sq,hz)) == target["horizontal_only_correct"]
            continue
        path, rule = arm.split("_",1)
        selected, hypothetical = [], []
        for i, row in enumerate(sources):
            candidate = int(row[path + "_class"])
            alt = int(row[arm + "_if_requested_class"])
            if rule == "take_second":
                assert alt == candidate
            if rule in ("local_selector","reference_selector"):
                which = rule.removesuffix("_selector")
                threshold = record["fit"][path][which]["switch_if_second_minus_first_gt"]
                gap = float(row[path + "_msp"]) - float(row["vertical_msp"])
                assert alt == (candidate if gap > threshold else v[i])
            direct = alt if g[i] else v[i]
            assert direct == int(row[arm + "_directed_class"])
            selected.append(direct == y[i])
            hypothetical.append(alt == y[i])
            assert int(row[arm + "_change_in_correctness"]) == selected[-1] - ok_v[i]
        count = sum(selected)
        assert target == {"requested": sum(g), "rescues": sum(c and not a for a,c in zip(ok_v,selected)),
                          "losses": sum(a and not c for a,c in zip(ok_v,selected)), "correct": count,
                          "correct_if_all_requested": sum(hypothetical)}
        expected_random = (1 - mean(g))*mean(ok_v) + mean(g)*mean(hypothetical)
        assert abs(100*expected_random - record["stats"][arm+"/matched_random_accuracy_pct"]["estimate"]) < 1e-10
        if arm == "horizontal_reference_selector":
            for local, pub in (("requested","requested_n"),("rescues","rescued_n"),("losses","lost_n"),
                               ("correct","targeted_correct_n")):
                assert target[local] == int(archived[pub])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    folder = parser.parse_args().results
    test = {r["source_id"]: r for r in read_csv(MANIFEST) if r["split"] == "test" and r["cohort"] == "full"}
    assert len(test) == 36000
    table = []
    for pair, old in (("transfer","new"),("reference","old")):
        pub = [r for r in read_csv(PROJECT / "tables" / ("trajectory_rescans_" + old + ".csv"))
               if r["window"] == "events30k" and r["request_gate"] == "old_frozen" and r["budget_pct"] == "30"]
        for model in ("histogram","timestamp"):
            name = pair + "_" + model
            record = load_json(folder / (name + ".json"))
            rows = read_csv(folder / (name + "_decisions.csv.gz"))
            assert len(rows) == len(test) and {r["source_id"] for r in rows} == set(test)
            assert all(r["label"] == test[r["source_id"]]["label"] for r in rows)
            check_arms(record, rows, next(r for r in pub if r["checkpoint"] == model))
            for term, val in record["stats"].items():
                row = [pair, model, term, val["estimate"], *val["within_ci95"], *val["class_ci95"]]
                table.append(row)
    listed = read_csv(folder / "all_contrasts.csv")
    assert len(listed) == len(table)
    for record, expected in zip(listed,table):
        fields = list(record.values())
        assert fields[:3] == expected[:3]
        assert [float(value) for value in fields[3:]] == expected[3:]
    print(f"Verified source decisions for {len(test):,} test sources per pair/classifier and {len(table)} contrasts.")


if __name__ == "__main__":
    main()
