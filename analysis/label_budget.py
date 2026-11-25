"""Retrospective target-label budgets for a fixed capture selector, at 30,000 events."""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

from trajectory_data import B, MODELS, TABLES, ids_for, scan, threshold


SEED_STRATIFIED = 20261126
SEED_ANNOTATION = 20261127
PER_CLASS = (2, 3, 4, 6, 8, 10)
ANNOTATION_BUDGETS = (100, 250, 500, 900)
Z95 = NormalDist().inv_cdf(.975)


def outcomes(split, keys):
    """Form eligibility using observable predictions before looking up correctness."""
    changes, eligible = [], []
    for model in MODELS:
        vs, vp, labels = scan("new", split, "vertical", "events30k", model, keys)
        hs, hp, hlabels = scan("new", split, "horizontal", "events30k", model, keys)
        np.testing.assert_array_equal(labels, hlabels)
        selected = np.where(hs - vs > threshold(model), hp, vp)
        changed_class = selected != vp
        delta = (selected == labels).astype(int) - (vp == labels).astype(int)
        assert np.all(delta[~changed_class] == 0)
        changes.append(delta)
        eligible.append(changed_class)
    return np.stack(changes), np.stack(eligible)


def stratified(estimates, test):
    """Independent superpopulation variances for the two class-stratified means."""
    models, n = estimates.shape
    classes = n // 10
    by_class = estimates.reshape(models, classes, 10)
    test_by_class = test.reshape(models, classes, 40)
    test_var = test_by_class.var(axis=2, ddof=1).sum(axis=1) / (40 * classes**2)
    rng = np.random.default_rng(SEED_STRATIFIED)
    frame = np.broadcast_to(np.arange(10), (classes, 10))
    point = np.empty((B, models, len(PER_CLASS)))
    half_width = np.empty_like(point)
    for rep in range(B):
        order = rng.permuted(frame, axis=1)
        draws = np.take_along_axis(by_class, order[None], axis=2)
        totals = draws.cumsum(axis=2)
        squares = (draws**2).cumsum(axis=2)
        for col, k in enumerate(PER_CLASS):
            counts = totals[:, :, k - 1]
            point[rep, :, col] = 100 * counts.sum(axis=1) / (classes * k)
            sample_var = (squares[:, :, k - 1] - counts**2 / k) / (k - 1)
            cal_var = sample_var.sum(axis=1) / (k * classes**2)
            half_width[rep, :, col] = 100 * Z95 * np.sqrt(cal_var + test_var)
    return point, half_width


def annotation(estimates, eligible):
    """Nest both labeling policies in a common uniform permutation of source IDs."""
    models, n = estimates.shape
    rng = np.random.default_rng(SEED_ANNOTATION)
    points = np.empty((B, models, 2, len(ANNOTATION_BUDGETS)))
    budgets = np.asarray(ANNOTATION_BUDGETS)
    assert np.all(eligible.sum(axis=1) >= max(budgets))
    for rep in range(B):
        order = rng.permutation(n)
        for i in range(models):
            changed = order[eligible[i, order]]
            for policy, ids in enumerate((order, changed)):
                weight = len(ids) / n
                cumulative = estimates[i, ids[:max(budgets)]].cumsum()
                points[rep, i, policy] = 100 * weight * cumulative[budgets - 1] / budgets
    return points


def summarize(points, full_point, test_point):
    error = points - test_point
    annotation_error = points - full_point
    lo, median, hi = np.percentile(error, [2.5, 50, 97.5])
    return dict(
        forecast_test_error_q025_pp=lo, forecast_test_error_median_pp=median,
        forecast_test_error_q975_pp=hi, median_abs_forecast_test_error_pp=np.median(abs(error)),
        median_abs_annotation_error_pp=np.median(abs(annotation_error)),
        rmse_annotation_pp=np.sqrt(np.mean(annotation_error**2)),
    )


def check_draws(values, expected, design_variance):
    """Check simulation against the exact mean and variance for finite-pool sampling."""
    if design_variance == 0:
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
    else:
        assert abs(values.mean() - expected) < 5 * np.sqrt(design_variance / B)
        np.testing.assert_allclose(values.var(ddof=1), design_variance, rtol=.15)


def run():
    cal_ids, test_ids = (ids_for(split) for split in ("calibration", "test"))
    assert set(cal_ids).isdisjoint(test_ids)
    cal, eligible = outcomes("calibration", cal_ids)
    test, _ = outcomes("test", test_ids)
    forecast = json.loads((TABLES / "trajectory_forecast.json").read_text())["models"]
    gains = json.loads((TABLES / "trajectory_gains.json").read_text())["model_windows"]
    balanced, half_width = stratified(cal, test)
    ann = annotation(cal, eligible)
    n = len(cal_ids)
    classes = n // 10
    rows = []
    for i, model in enumerate(MODELS):
        full, observed = 100 * cal[i].mean(), 100 * test[i].mean()
        historical = gains["events30k_" + model]["forecast"]["direct"]
        eligible_n = int(eligible[i].sum())
        np.testing.assert_allclose([full, observed], [forecast[model]["direct_pp"], historical["observed_pp"]])
        assert np.sum(cal[i] == 1) == forecast[model]["target_rescues"]
        assert np.sum(cal[i] == -1) == forecast[model]["target_losses"]
        np.testing.assert_allclose(100 * cal[i, eligible[i]].sum() / n, full)
        common = dict(checkpoint=model, repetitions=B, calibration_n=n, test_n=len(test_ids),
                      eligible_n=eligible_n, eligible_shared_n=int(eligible.all(axis=0).sum()),
                      eligible_union_n=int(eligible.any(axis=0).sum()),
                      full_cal_gain_pp=full, test_gain_pp=observed)
        class_variances = cal[i].reshape(classes, 10).var(axis=1, ddof=1)
        for j, k in enumerate(PER_CLASS):
            point = balanced[:, i, j]
            error = point - observed
            width = half_width[:, i, j]
            lo, hi = error - width, error + width
            # The subset spread is conditional on this pool. It vanishes at k=10.
            conditional_variance = 100**2 * (1 - k/10) * class_variances.sum() / (classes**2 * k)
            check_draws(point, full, conditional_variance)
            row = dict(**common, policy="within_class", labels=k*classes, labels_per_class=k,
                       **summarize(point, full, observed),
                       median_interval_low_pp=np.median(lo), median_interval_high_pp=np.median(hi),
                       interval_containment_fraction=np.mean((lo >= -1) & (hi <= 1)))
            if k == 10:
                row["archived_interval_low_pp"], row["archived_interval_high_pp"] = historical["within_error_95_pp"]
            rows.append(row)
        for policy_idx, policy in enumerate(("uniform", "changed_prediction")):
            pool = cal[i] if policy_idx == 0 else cal[i, eligible[i]]
            weight = len(pool) / n
            for col, budget in enumerate(ANNOTATION_BUDGETS):
                point = ann[:, i, policy_idx, col]
                variance = (100*weight)**2 * (1 - budget/len(pool)) * pool.var(ddof=1) / budget
                check_draws(point, full, variance)
                rows.append(dict(**common, policy=policy, labels=budget,
                                 **summarize(point, full, observed)))
    return rows


def write_table(rows, stream):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: f"{v:.9f}" if isinstance(v, (np.floating, float)) else v
                         for k, v in row.items()})


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "axes.linewidth": .7, "legend.frameon": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(160 / 25.4, 68 / 25.4))
    fig.subplots_adjust(left=.106, right=.982, bottom=.235, top=.845, wspace=.43)
    blue, muted = "#115C73", "#485457"
    block = [r for r in rows if r["checkpoint"] == "histogram" and r["policy"] == "within_class"]
    x = np.array([r["labels"] for r in block]) / 1000
    y, low, high = (np.array([r["forecast_test_error_" + key + "_pp"] for r in block])
                    for key in ("median", "q025", "q975"))
    left.axhline(0, color=muted, lw=.7, ls="--")
    left.errorbar(x, y, yerr=[y-low, high-y], fmt="o-", ms=4, lw=1.0,
                  capsize=2.4, elinewidth=.9, color=blue)
    left.set_xticks(x, ["1.8", "2.7", "3.6", "5.4", "7.2", "9.0"])
    left.set(xlabel="Labeled sources (thousands)", ylabel="Forecast − test gain (points)",
             ylim=(-1.05, 1.25))
    left.set_title("a   Class-balanced labels", loc="left", fontweight="bold", fontsize=9.5)
    for policy, name, style, color in (("uniform", "Uniform", "--s", muted),
                                       ("changed_prediction", "Changed prediction", "-o", blue)):
        block = [r for r in rows if r["checkpoint"] == "histogram" and r["policy"] == policy]
        right.plot([r["labels"] for r in block], [r["median_abs_annotation_error_pp"] for r in block],
                   style, color=color, lw=1.0, ms=4, mfc="white" if policy == "uniform" else color, label=name)
    right.set(xlabel="Labeled sources", ylabel="Median absolute error (points)", xticks=ANNOTATION_BUDGETS)
    right.set_ylim(bottom=0)
    right.set_title("b   Equal labeling budgets", loc="left", fontweight="bold", fontsize=9.5)
    right.legend(loc="upper right", fontsize=8, handlelength=1.6, handletextpad=.45)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D2D9D9", linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, width=.7, length=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the numerical table to this CSV file")
    parser.add_argument("--figure", type=Path, help="Write the Histogram PDF and PNG preview (requires Matplotlib)")
    args = parser.parse_args()
    rows = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            write_table(rows, stream)
    else:
        write_table(rows, sys.stdout)
    if args.figure:
        plot(rows, args.figure)


if __name__ == "__main__":
    main()
