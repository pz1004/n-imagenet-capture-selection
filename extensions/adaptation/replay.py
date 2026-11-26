"""Refit the reference selectors and replay the calibration-only adaptation comparisons."""

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from capture_policies import choose_vertical, fit_discordant, read, sorted_ids
from trajectory_data import B, MODELS, samples, scan

STATES = ("raw", "bn", "evtta")
PAIRS = {"reference": ("vertical", "horizontal"), "transfer": ("variant1", "variant2")}
QUOTA = 10800
GATE_NAMES = (
    "vertical_acc",
    "horizontal_acc",
    "always_acc",
    "always_gain",
    "request_pct",
    "directed_acc",
    "random_acc",
    "directed_gain",
    "allocation_advantage",
    "net_per_100_requests",
    "random_net_per_100_requests",
)
BREAK_NAMES = (
    "discordant_pct",
    "v_share_pct",
    "selector_success_pct",
    "success_minus_v_share_pp",
    "all_gain_pp",
)
QUOTA_NAMES = (
    "vertical_accuracy_pct",
    "directed_accuracy_pct",
    "gain_vs_own_vertical_pp",
    "random_accuracy_pct",
    "directed_minus_random_pp",
    "net_per_100_requests",
)
BREAK_SCOPE = (
    "Retrospective description of all-rescan terminal rules. "
    "The identities of discordant sources may change with model state."
)


def assignments():
    manifest = read(ROOT / "data/source_index.csv", "cohort", "full")
    calibration, test = (sorted_ids(manifest, split) for split in ("calibration", "test"))
    for keys, n_per_class in ((calibration, 10), (test, 40)):
        groups = np.array([int(manifest[k]["label"]) for k in keys]).reshape(900, n_per_class)
        if not (np.all(groups == groups[:, :1]) and np.all(np.diff(groups[:, 0]) > 0)):
            raise ValueError("Unexpected class structure in source_index.csv")
    policy = [key for i, key in enumerate(calibration) if i % 10 >= 5]
    return manifest, calibration, policy, test


def calibration_budget(keys):
    """Count distinct event indices in the four calibration crops, including temporal overlaps."""

    def hash64(text):
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")

    prefixes = ("crop/", "adaptation-view-1/", "adaptation-view-2/", "adaptation-view-3/")
    targets = set(keys)
    totals = {}
    for condition in ("vertical", "horizontal", "variant1", "variant2"):
        if condition in ("vertical", "horizontal"):
            path = ROOT / "data/full_events30k_windows.csv"
            column = "raw_events_v" if condition == "vertical" else "raw_events_h"
            stream = path.open(newline="")
        else:
            path = ROOT / "data" / f"{condition}_events30k.csv.gz"
            column = "raw_events"
            stream = gzip.open(path, "rt", newline="")
        seen = set()
        anchors, distinct = [], []
        with stream:
            for row in csv.DictReader(stream):
                key = row["source_id"]
                if key not in targets:
                    continue
                if key in seen:
                    raise ValueError(f"Duplicate calibration source {key} in {condition}")
                seen.add(key)
                n = int(row[column])
                length = min(n, 30000)
                starts = sorted(
                    hash64(prefix + key) * (n - length + 1) >> 64 for prefix in prefixes
                )
                count = length + sum(min(b - a, length) for a, b in zip(starts, starts[1:]))
                if not length <= count <= min(n, 4 * length):
                    raise ValueError(f"Invalid crop union for {key} in {condition}")
                anchors.append(length)
                distinct.append(count)
        if seen != targets:
            raise ValueError(f"Incomplete calibration observations for {condition}")
        totals[condition] = dict(
            n=len(seen),
            mean_distinct_events=float(np.mean(distinct)),
            median_distinct_events=float(np.median(distinct)),
            mean_anchor_events=float(np.mean(anchors)),
            maximum_distinct_events=max(distinct),
            sources_with_only_anchor_events=sum(a == d for a, d in zip(anchors, distinct)),
        )
    return totals


def adapted_rows(condition, directory, manifest, policy, test):
    path = directory / (condition + ".csv.gz")
    # Optional GPU output contains the same columns as the packaged CSV.
    if path.exists():
        stream = gzip.open(path, "rt", newline="")
    else:
        stream = (directory / condition / "predictions.csv").open(newline="")
    with stream:
        rows = list(csv.DictReader(stream))
    source = {r["source_id"]: r for r in rows}
    expected = {k: "policy" for k in policy}
    expected.update(dict.fromkeys(test, "test"))
    if len(rows) != len(source) or source.keys() != expected.keys():
        raise ValueError(f"Duplicate or incomplete source IDs in {condition}")
    for key, group in expected.items():
        row = source[key]
        if group != row["split"] or int(row["label"]) != int(manifest[key]["label"]):
            raise ValueError(f"Split/class mismatch in {condition} for {key}")
    return source


def scores(pair, split, keys, model, sources, labels):
    data = {}
    for state in STATES:
        observations = []
        for condition, direction in zip(PAIRS[pair], ("vertical", "horizontal")):
            if state == "raw":
                score, pred, y = scan(
                    "old" if pair == "reference" else "new",
                    split,
                    direction,
                    "events30k",
                    model,
                    keys,
                )
                np.testing.assert_array_equal(labels, y)
            else:
                # Four archived adapted forward passes are absent for the unlabeled update set.
                subset = [sources[condition].get(k) for k in keys]
                prefix = state + "_" + model
                pred = np.array([int(r[prefix + "_prediction"]) if r else -1 for r in subset])
                score = np.array([float(r[prefix + "_msp"]) if r else np.nan for r in subset])
                selected = np.array([r is not None for r in subset])
                if not (
                    np.isfinite(score[selected]).all()
                    and np.all((score[selected] > 0) & (score[selected] <= 1))
                    and np.all((pred[selected] >= 0) & (pred[selected] < 1000))
                ):
                    raise ValueError(f"Invalid predictions in {condition} for {prefix}")
            observations.extend((pred, score))
        data[state] = tuple(observations)
    return data


def fit_policies(observations, labels):
    keep = np.arange(len(labels)) % 10 >= 5
    fits = {}
    for state, (pv, sv, ph, sh) in observations.items():
        first, second = sv[keep], sh[keep]
        v_correct, h_correct = pv[keep] == labels[keep], ph[keep] == labels[keep]
        fit = fit_discordant(first - second, v_correct, h_correct)
        center, scale, intercept, slope = fit
        assert slope > 0
        fits[state] = {
            "fit": list(fit),
            "switch_if_h_minus_v_exceeds": scale * intercept / slope - center,
            "request_cutoff": float(np.quantile(first, 0.30, method="higher")),
            "discordant": int(np.count_nonzero(v_correct != h_correct)),
        }
    return fits


def pack(observations, fits, ids, labels):
    """Pack signed source contributions and save each state's realized decisions."""
    tie = np.array(
        [
            int.from_bytes(hashlib.sha256(("adapt-quota/" + k).encode()).digest()[:8], "big")
            for k in ids
        ],
        dtype=np.uint64,
    )
    assert len(np.unique(tie)) == len(tie)
    packed, orders, decisions = [], [], []
    fixed_counts, break_counts, quota_counts = {}, {}, {}
    for state in STATES:
        pv, sv, ph, sh = observations[state]
        chosen = np.where(choose_vertical(sv - sh, fits[state]["fit"]), pv, ph)
        req = sv < fits[state]["request_cutoff"]
        returned = np.where(req, chosen, pv)
        order = np.lexsort((tie, sv))
        qreq = np.zeros(len(ids), dtype=bool)
        qreq[order[:QUOTA]] = True
        qreturned = np.where(qreq, chosen, pv)
        v, h, c = pv == labels, ph == labels, chosen == labels
        delta = c.astype(np.int8) - v.astype(np.int8)
        gate_gain, quota_gain = delta * req, delta * qreq
        fixed_counts[state] = dict(
            n=len(ids),
            vertical_correct=int(v.sum()),
            horizontal_correct=int(h.sum()),
            always_correct=int(c.sum()),
            always_rescued=int((delta == 1).sum()),
            always_lost=int((delta == -1).sum()),
            requests=int(req.sum()),
            rescued=int((gate_gain == 1).sum()),
            lost=int((gate_gain == -1).sum()),
            directed_correct=int((returned == labels).sum()),
        )
        break_counts[state] = {
            "v_only": int((v & ~h).sum()),
            "h_only": int((h & ~v).sum()),
            "selected_correct_on_discordant": int((c & (v != h)).sum()),
        }
        quota_counts[state] = dict(
            requested=QUOTA,
            correct=int((qreturned == labels).sum()),
            rescued=int((quota_gain == 1).sum()),
            lost=int((quota_gain == -1).sum()),
            net=int(quota_gain.sum()),
            vertical_correct=int(v.sum()),
            full_net=int(delta.sum()),
        )
        packed.extend((v, h, delta, req, gate_gain, v & ~h, h & ~v, c & (v != h)))
        orders.append((order, delta[order]))
        decisions.extend((chosen, req.astype(int), returned, qreq.astype(int), qreturned))
    return (
        np.ascontiguousarray(np.column_stack(packed), dtype=np.int8),
        orders,
        decisions,
        fixed_counts,
        break_counts,
        quota_counts,
    )


def quota_gain(weights, delta):
    """Select exactly QUOTA observations, including bootstrap multiplicities at the boundary."""
    rank = np.cumsum(weights, dtype=np.int32)
    stop = int(np.searchsorted(rank, QUOTA))
    remaining = QUOTA - (rank[stop - 1] if stop else 0)
    return int(weights[:stop] @ delta[:stop] + remaining * delta[stop])


def measures(totals, quota, n):
    vertical, horizontal, gain, rate, directed, v_only, h_only, selected = (totals / n).T
    discordance = v_only + h_only
    a, pi = selected / discordance, v_only / discordance
    random = rate * gain
    gated = np.stack(
        (
            vertical,
            horizontal,
            vertical + gain,
            gain,
            rate,
            vertical + directed,
            vertical + random,
            directed,
            directed - random,
            directed / rate,
            gain,
        ),
        axis=1,
    )
    broken = np.stack((discordance, pi, a, a - pi, selected - v_only), axis=1)
    target_rate = QUOTA / n
    qgain = np.array(quota) / n
    at_quota = np.stack(
        (
            vertical,
            vertical + qgain,
            qgain,
            vertical + target_rate * gain,
            qgain - target_rate * gain,
            qgain / target_rate,
        ),
        axis=1,
    )
    return 100 * np.concatenate((gated, broken, at_quota), axis=1)


def summarize(values, draws, names):
    base = {name: {"estimate": float(x)} for name, x in zip(names, values)}
    for design, sample in draws.items():
        limits = np.percentile(sample, [2.5, 97.5], axis=0).T
        for name, ci in zip(names, limits):
            base[name][design] = ci.tolist()
    return base


def quota_summary(stats):
    out = {"points": {k: v["estimate"] for k, v in stats.items()}}
    for kind in ("within", "class"):
        if kind in next(iter(stats.values())):
            out[kind + "_ci95"] = {k: v[kind] for k, v in stats.items()}
    return out


def evaluate(pair, model, observation, fits, ids, y, output, bootstrap):
    packed, orders, decisions, counts, bcounts, qcounts = pack(observation, fits, ids, y)
    n = len(ids)
    qgains = [qcounts[s]["net"] for s in STATES]
    est = measures(packed.sum(0).reshape(3, 8), qgains, n)
    draws = {}
    if bootstrap:
        for design in ("within", "class"):
            ensemble = []
            for indices in samples(design, train=False):
                weights = np.bincount(indices, minlength=n).astype(np.int32)
                totals = (weights @ packed).reshape(3, 8)
                qgains = [quota_gain(weights[order], delta) for order, delta in orders]
                ensemble.append(measures(totals, qgains, n))
            draws[design] = np.array(ensemble)
    by_state, delta_state, breakdown, quota_states, quota_contrasts = {}, {}, {}, {}, {}
    for index, other in ((0, None), (1, None), (2, None), (1, 0), (2, 0), (2, 1)):
        key = STATES[index] if other is None else STATES[index] + "_minus_" + STATES[other]
        v = est[index] if other is None else est[index] - est[other]
        b = {
            k: a[:, index] if other is None else a[:, index] - a[:, other] for k, a in draws.items()
        }

        def get(section, names):
            return summarize(v[section], {k: a[:, section] for k, a in b.items()}, names)

        (by_state if other is None else delta_state)[key] = get(slice(0, 11), GATE_NAMES)
        if other in (None, 0):
            breakdown[key] = get(slice(11, 16), BREAK_NAMES)
            q = quota_summary(get(slice(16, None), QUOTA_NAMES))
            if other is None:
                q["counts"] = qcounts[key]
                quota_states[key] = q
            else:
                quota_contrasts[key] = q
    gated = dict(
        pair=pair, model=model, counts=counts, results=by_state, differences=delta_state, fits=fits
    )
    quota = dict(n=n, quota=QUOTA, states=quota_states, contrasts=quota_contrasts)
    if bootstrap:
        gated["draws"] = B
        quota["draws_each"] = B
    name = pair + "_" + model
    columns = ["source_id", "label"] + [
        s + "_" + c
        for s in STATES
        for c in (
            "all_rescan_class",
            "request",
            "selected_class",
            "quota_request",
            "quota_selected_class",
        )
    ]
    with gzip.open(output / (name + "_decisions.csv.gz"), "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(ids, y, *decisions))
    return {
        "gated": gated,
        "discordance": {"scope": BREAK_SCOPE, "counts": bcounts, "estimates": breakdown},
        "quota": quota,
    }


def verify_values(actual, expected):
    if isinstance(actual, dict):
        for key, val in actual.items():
            verify_values(val, expected[key])
    elif isinstance(actual, (int, float, list)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for the replay outputs"
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=HERE / "data",
        help="Released score directory or GPU regeneration output root",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Recompute 2,000 draws under each resampling design",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare all reconstructed values and per-source decisions with the release",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must name a new directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest, calibration, policy, test = assignments()
    sources = {
        c: adapted_rows(c, args.scores, manifest, policy, test)
        for pair in PAIRS.values()
        for c in pair
    }
    cy = np.array([int(manifest[k]["label"]) for k in calibration])
    ty = np.array([int(manifest[k]["label"]) for k in test])
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        output = Path(temp) / "results"
        output.mkdir()
        event_budget = calibration_budget(set(calibration) - set(policy))
        if args.verify:
            verify_values(
                event_budget, json.loads((HERE / "tables/calibration_budget.json").read_text())
            )
        (output / "calibration_budget.json").write_text(json.dumps(event_budget, indent=2) + "\n")
        all_fits = {}
        for model in MODELS:
            cal = scores("reference", "calibration", calibration, model, sources, cy)
            fits = fit_policies(cal, cy)
            all_fits[model] = fits
            for pair in PAIRS:
                obs = scores(pair, "test", test, model, sources, ty)
                result = evaluate(pair, model, obs, fits, test, ty, output, args.bootstrap)
                name = pair + "_" + model
                if args.verify:
                    verify_values(
                        result, json.loads((HERE / "tables" / (name + ".json")).read_text())
                    )
                    name_csv = name + "_decisions.csv.gz"
                    with (
                        gzip.open(output / name_csv, "rb") as a,
                        gzip.open(HERE / "data" / name_csv, "rb") as b,
                    ):
                        for left, right in zip(a, b, strict=True):
                            assert left == right, f"Decision mismatch: {name}"
                (output / (name + ".json")).write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                print(f"{name}: {len(test)} test sources", flush=True)
        if args.verify:
            verify_values(all_fits, json.loads((HERE / "tables/reference_fits.json").read_text()))
        (output / "reference_fits.json").write_text(json.dumps(all_fits, indent=2) + "\n")
        output.rename(args.output)


if __name__ == "__main__":
    main()
