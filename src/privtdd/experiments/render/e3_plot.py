"""Draw the single-column E3 bar chart from raw mean errors."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from .e3_statistics import BUDGETS, METRICS, PAIRS
from pathlib import Path


PAIR_LABELS = ("80/90", "90/95 (default)", "95/99")
STYLES = (
    dict(facecolor="#BCD3DF", edgecolor="#5A7688"),
    dict(facecolor="#286789", edgecolor="#286789"),
    dict(facecolor="#F1D8B9", edgecolor="#86613B", hatch="///"),
)
# Each bar starts at zero. These ticks bracket the observed means in original units.
SCALE_END = {
    ("DBLP", "MSE"): .3,   ("Orkut", "MSE"): .004,
    ("DBLP", "EMD"): 10,   ("Orkut", "EMD"): 200,
    ("DBLP", "KS"): .3,    ("Orkut", "KS"): .08,
    ("Higgs social", "MSE"): .03, ("Higgs social", "EMD"): 550,
    ("Higgs social", "KS"): .15,
}
DIGITS = {"MSE": 5, "EMD": 2, "KS": 4}
# At this setting the original precision makes distinct means look identical.
EXTRA_DIGITS = {("Higgs social", 2, "MSE"): 7, ("Higgs social", 2, "KS"): 6}


def load_means(networks, data_dir):
    DATA = Path(data_dir)
    keys = ["dataset", "epsilon", "metric", "quantiles"]
    means = pd.read_csv(DATA / "means.csv", float_precision="round_trip")
    means = means[means.dataset.isin(networks)].set_index(keys).sort_index()
    runs = pd.read_csv(DATA / "releases.csv", float_precision="round_trip")
    runs = runs[runs.dataset.isin(networks)]
    sets = runs.groupby(["dataset", "epsilon", "seed"])
    if (sets.size().ne(len(PAIRS)).any() or sets.report_sha256.nunique().ne(1).any()
            or runs.duplicated(["dataset", "epsilon", "seed", "quantiles"]).any()):
        raise ValueError("The three boundary pairs must use the same report for each seed.")
    expected = pd.MultiIndex.from_product([networks, BUDGETS, METRICS, PAIRS], names=keys).sort_values()
    long = runs.melt(id_vars=["dataset", "epsilon", "quantiles"], value_vars=list(METRICS),
                     var_name="metric", value_name="error")
    observed = long.groupby(keys).error.agg(["mean", "count"]).sort_index()
    if not means.index.equals(expected) or not observed.index.equals(expected):
        raise ValueError("Each network and budget must have means for all three metrics and pairs.")
    expected_counts = [8 if dataset == "Higgs social" and budget == 0.5 else 10
                       for dataset, budget, _, _ in expected]
    np.testing.assert_array_equal(observed["count"], expected_counts)
    np.testing.assert_array_equal(means["n"], expected_counts)
    np.testing.assert_allclose(observed["mean"], means["mean"], rtol=1e-12, atol=0)
    return means["mean"]


def main(data_dir, output_dir, left_network="Higgs social"):
    networks = (left_network, "Orkut")
    means = load_means(networks, data_dir)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8.4, "hatch.linewidth": .45, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(3.5, 3.38))
    fig.subplots_adjust(left=.174, right=.992, bottom=.050, top=.869, hspace=.32, wspace=.09)
    group_starts = np.array([0., 3.6])

    for r, metric in enumerate(METRICS):
        for c, dataset in enumerate(networks):
            ax = axes[r, c]
            values = means.xs((dataset, metric), level=("dataset", "metric"))
            scale_end = max(SCALE_END[dataset, metric], values.max())
            # Extra space to the right of the scale is for the labels at the bar ends.
            ax.set_xlim(0, scale_end * 1.56)
            ax.set_xticks([0, scale_end])
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.set_ylim(group_starts[-1] + 2.67, -.55)
            ax.set_yticks(group_starts + 1, [f"ε={e:g}" for e in BUDGETS])
            ax.tick_params(axis="y", labelleft=c == 0, length=0, pad=3)
            ax.tick_params(axis="x", length=2, width=.55, pad=1)
            ax.spines[["top", "right"]].set_visible(False)
            for edge in ("left", "bottom"):
                ax.spines[edge].set_color("#778087")
                ax.spines[edge].set_linewidth(.55)
            ax.spines["bottom"].set_bounds(0, scale_end)
            for i, (pair, style) in enumerate(zip(PAIRS, STYLES)):
                y = group_starts + i
                widths = [values.at[(budget, pair)] for budget in BUDGETS]
                label = PAIR_LABELS[i] if r == c == 0 else None
                ax.barh(y, widths, height=.70, linewidth=.5, label=label, **style)
                for ypos, value, budget in zip(y, widths, BUDGETS):
                    digits = EXTRA_DIGITS.get((dataset, budget, metric), DIGITS[metric])
                    ax.annotate(f"{value:.{digits}f}", (value, ypos), xytext=(2, 0),
                                textcoords="offset points", ha="left", va="center", fontsize=8.2)
            if c == 0:
                ax.set_ylabel(metric, fontsize=9, fontweight="bold", labelpad=3)
            if r == 0:
                ax.set_title(dataset, fontsize=9.3, fontweight="bold", pad=3)
    fig.legend(loc="upper center", bbox_to_anchor=(.50, .999), ncol=3, frameon=False,
               handlelength=1.15, handletextpad=.35, columnspacing=1.15, borderaxespad=.1)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "e3_sensitivity_means"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--left-network", choices=["DBLP", "Higgs social"], default="Higgs social")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.left_network)
