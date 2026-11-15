"""Summarize by graph, budget, and nominal sampling rate. One complete release is the unit for pairing and bootstrapping."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ACCURACY = ("EMD", "MSE", "KS")
TIMES = ("collection_seconds", "end_to_end_seconds")
MEASURES = ("visits", "candidates", "realized_fraction", "active_fraction",
            *TIMES, "postprocessing_seconds", "collection_peak_rss_bytes", "peak_rss_bytes", *ACCURACY)
KEY = ("dataset", "epsilon", "seed", "rate")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def valid(row, field):
    value = row.get(field)
    return value is not None and math.isfinite(value)


def estimate(values, *, ratio=False):
    """Estimate a ratio as the ratio of two paired sample means. Each resample with replacement preserves entire rows."""
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=float)
    def statistic(sample):
        means = sample.mean(axis=-2)
        return means[..., 0] / means[..., 1] if ratio else means[..., 0]
    point = float(statistic(arr))
    if len(arr) < 2:
        return point, None, None
    indices = np.random.default_rng(731).integers(len(arr), size=(2000, len(arr)))
    low, high = np.percentile(statistic(arr[indices]), [2.5, 97.5])
    return point, float(low), float(high)


def add_stat(row, name, values, *, ratio=False):
    value, low, high = estimate(values, ratio=ratio)
    row.update({name: value, name + "_lo95": low, name + "_hi95": high, name + "_n": len(values)})


def publication_rows(root):
    plan = read(root / "plan.json")
    controls = read(root / "controls.json")
    samples = []
    for job in plan["jobs"]:
        folder = root / "jobs" / job["id"]
        row = {key: job[key] for key in KEY}
        row.update(id=job["id"], status="pending")
        if (folder / "result.json").is_file():
            saved = read(folder / "result.json")
            if any(saved.get(key) != row[key] for key in KEY):
                raise ValueError(f"Sampling result does not match its plan: {folder}")
            row.update(saved)
            if row["status"] == "complete":
                marker = folder / "complete.json"
                if not marker.is_file():
                    # The scheduler may stop between writing results and committing the completion marker; retain any completed S1 phase.
                    row["status"] = "awaiting_completion_marker"
                elif read(marker)["result_sha256"] != hashlib.sha256((folder / "result.json").read_bytes()).hexdigest():
                    raise ValueError(f"Changed completed result: {folder}")
        elif (folder / "status.json").is_file():
            state = read(folder / "status.json")
            row["status"] = state["status"]
        samples.append(row)
    rows = controls + samples
    if len({tuple(r[f] for f in KEY) for r in rows}) != len(rows):
        raise ValueError("Duplicate published dataset, budget, seed, and rate")
    for row in rows:
        if valid(row, "visits") and valid(row, "candidates") and row["candidates"] > 0:
            row["realized_fraction"] = row["visits"] / row["candidates"]
    return rows


def summarize(root):
    root = Path(root)
    rows = publication_rows(root)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["dataset"], r["epsilon"], r["rate"], r["seed"]))
    save_csv(out / "runs.csv", rows)
    groups = defaultdict(list)
    full = {}
    for row in rows:
        groups[row["dataset"], row["epsilon"], row["rate"]].append(row)
        if row["rate"] == 1:
            full[row["dataset"], row["epsilon"], row["seed"]] = row
    summaries, contrasts, paired_runs = [], [], []
    for (dataset, epsilon, rate), group in groups.items():
        heading = dict(dataset=dataset, epsilon=epsilon, rate=rate)
        summary = dict(heading, n_planned=len(group), n_completed=sum(r["status"] == "complete" for r in group),
                       n_s1=sum(valid(r, "visits") for r in group))
        for metric in MEASURES:
            values = [[r[metric]] for r in group if valid(r, metric)
                      and (metric not in ACCURACY or r["status"] == "complete")]
            add_stat(summary, metric, values)
        summaries.append(summary)
        if rate == 1:
            continue
        pairs = [(r, full[dataset, epsilon, r["seed"]]) for r in group]
        contrast = dict(heading, n_planned=len(group),
                        n_both_complete=sum(r["status"] == f["status"] == "complete" for r, f in pairs))
        for sample, reference in pairs:
            entry = dict(heading, seed=sample["seed"], sample_status=sample["status"],
                         full_status=reference["status"])
            for metric in (*ACCURACY, *TIMES, "visits"):
                entry["sample_" + metric] = sample.get(metric)
                entry["full_" + metric] = reference.get(metric)
            for metric in ACCURACY:
                if (sample["status"] == reference["status"] == "complete"
                        and valid(sample, metric) and valid(reference, metric)):
                    entry["delta_" + metric] = sample[metric] - reference[metric]
            paired_runs.append(entry)
        for metric in ACCURACY:
            differences = [[r[metric] - f[metric]] for r, f in pairs if r["status"] == f["status"] == "complete"
                           and valid(r, metric) and valid(f, metric)]
            add_stat(contrast, "sample_minus_full_" + metric, differences)
        proportions = [[r["visits"], f["visits"]] for r, f in pairs if valid(r, "visits")
                       and valid(f, "visits") and f["visits"] > 0]
        add_stat(contrast, "sample_over_full_visits", proportions, ratio=True)
        for metric in TIMES:
            observed = [(r, f) for r, f in pairs if valid(r, metric) and valid(f, metric)
                        and r[metric] > 0 and f[metric] > 0]
            # Raw runtime ratios may be described; report speedups and intervals only when both sets of timing conditions have been verified.
            contrast["historical_full_over_sample_" + metric] = (
                sum(f[metric] for r, f in observed) / sum(r[metric] for r, f in observed)
                if observed else None)
            contrast["historical_full_over_sample_" + metric + "_n"] = len(observed)
            compatible = [[f[metric], r[metric]] for r, f in observed
                          if r.get("timing_comparable") is True and f.get("timing_comparable") is True]
            add_stat(contrast, "speedup_" + metric, compatible, ratio=True)
        contrasts.append(contrast)
    save_csv(out / "summary.csv", summaries)
    save_csv(out / "paired_runs.csv", paired_runs)
    save_csv(out / "paired.csv", contrasts)
    save_csv(out / "contrast_10pct.csv", [r for r in contrasts if math.isclose(r["rate"], .1)])
    if any(r["n_s1"] for r in summaries):
        plot_summary(out, summaries, rows)
    return out


def plot_summary(out, summaries, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    labels = {"visits": "Evaluated neighbor pairs", "collection_seconds": "Report generation (s)",
              "end_to_end_seconds": "End to end (s)", "EMD": "EMD", "MSE": "MSE", "KS": "KS"}
    for dataset in sorted({r["dataset"] for r in summaries}):
        records = [r for r in summaries if r["dataset"] == dataset]
        rates = sorted({r["rate"] for r in records})
        budgets = sorted({r["epsilon"] for r in records})
        figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
        for ax, metric in zip(axes.flat, labels):
            for ix, budget in enumerate(budgets):
                color = f"C{ix % 10}"
                values = sorted((r for r in records if r["epsilon"] == budget and valid(r, metric)),
                                key=lambda r: r["rate"])
                if not values:
                    continue
                # Show archived timings as separate hollow markers to avoid connecting different timing environments into one performance curve.
                separate_full = metric in TIMES and not all(
                    r.get("timing_comparable") is True for r in rows if r["dataset"] == dataset
                    and r["epsilon"] == budget and valid(r, metric))
                connected = [r for r in values if not (separate_full and r["rate"] == 1)]
                ax.plot([r["rate"] for r in connected], [r[metric] for r in connected], "o-",
                        color=color, lw=1.2, ms=3.5, label=f"ε = {budget:g}")
                for r in values:
                    if separate_full and r["rate"] == 1:
                        ax.plot(1, r[metric], "o", color=color, markerfacecolor="white", ms=5)
                    lo, hi = r[metric + "_lo95"], r[metric + "_hi95"]
                    if lo is not None:
                        ax.vlines(r["rate"], lo, hi, color=color, lw=.8)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(rates))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{100*x:g}%"))
            ax.minorticks_off()
            ax.tick_params(axis="x", labelrotation=35, labelsize=8)
            ax.set_ylabel(labels[metric])
            ax.grid(axis="y", alpha=.25)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(dataset)
        figure.supxlabel("Target fraction of projected neighbor pairs")
        figure.savefig(out / f"{dataset}.pdf")
        plt.close(figure)
