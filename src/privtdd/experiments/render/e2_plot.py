"""Plot E2 at IEEE double-column width or as four single-column network strips."""
import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NETWORKS = ("Bio-DMELA", "DBLP", "Higgs social", "Orkut")
SLUGS = {"Bio-DMELA": "bio_dmela", "DBLP": "dblp", "Higgs social": "higgs", "Orkut": "orkut"}
METRICS = ("MSE", "EMD", "KS")
BUDGETS = np.arange(0.5, 4.5, 0.5)
# Legend order, colour, marker, line pattern.
STYLES = {
    "PrivTDD": ("#005E88", "o", "-"),
    "SCRec": ("#41494E", "s", "--"),
    "Fixed-0.25": ("#BF5413", "^", "-."),
    "Fixed-0.50": ("#267B65", "D", (0, (4, 1.5, 1, 1.5))),
    "Fixed-0.75": ("#794D8D", "v", ":"),
}


def read_data(path):
    data = pd.read_csv(path, float_precision="round_trip")
    if "suite" in data:
        data = data.loc[data.suite == "E2"]
    keys = ["metric", "dataset", "method", "epsilon"]
    expected = pd.MultiIndex.from_product([METRICS, NETWORKS, list(STYLES), BUDGETS], names=keys)
    data = data.set_index(keys, verify_integrity=True)
    if len(expected.difference(data.index)) or len(data.index.difference(expected)):
        raise ValueError("Expected three metrics, four networks, five methods and eight budgets.")
    data = data.sort_index()
    if not np.isfinite(data[["mean", "lower95", "upper95", "n"]]).all().all():
        raise ValueError("Found a missing or non-finite estimate.")
    if not ((data.lower95 <= data["mean"]) & (data["mean"] <= data.upper95)).all():
        raise ValueError("Check the interval endpoints.")
    if data.groupby(level=["dataset", "epsilon"]).n.nunique().max() != 1 or data.n.min() < 2:
        raise ValueError("The five curves need estimates on matched releases at each setting.")
    return data


def set_panel_axis(axis, metric, values, limits=None, tick_size=8):
    """By default, give all confidence intervals in the panel a small margin."""
    lower, upper = values.lower95.min(), values.upper95.max()
    if metric == "KS":
        margin = max(.055 * (upper - lower), .002)
        low, high = limits or (max(0, lower - margin), min(1, upper + margin))
        locator = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
        ticks = locator.tick_values(low, high)
        ticks = ticks[(low <= ticks) & (ticks <= high)]
        axis.set_yticks(ticks[::2] if len(ticks) > 4 else ticks)
        axis.set_ylim(low, high)
        return
    if lower <= 0:
        raise ValueError(f"The {metric} interval includes a nonpositive value on the log axis.")
    log_min, log_max = np.log10([lower, upper])
    margin = max(.055 * (log_max - log_min), .04)
    low, high = limits or (10 ** (log_min - margin), 10 ** (log_max + margin))
    decades = 10.0 ** np.arange(np.floor(np.log10(low)), np.floor(np.log10(high)) + 1)
    candidates = np.outer(decades, (1, 2, 5)).ravel()
    candidates = candidates[(low <= candidates) & (candidates <= high)]
    if len(candidates) < 2:
        candidates = np.geomspace(low, high, 3)
    axis_height_pt = axis.get_position().height * axis.figure.get_figheight() * 72
    for count in range(min(len(candidates), 5), 1, -1):
        indices = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        ticks = candidates[indices]
        gaps_pt = np.diff(np.log(ticks)) / np.log(high / low) * axis_height_pt
        if gaps_pt.min() >= 1.2 * tick_size:
            break
    axis.set_yscale("log")
    axis.set_yticks(ticks)
    axis.set_ylim(low, high)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    axis.yaxis.set_minor_locator(NullLocator())


def configure_font(font, size):
    try:
        face = font_manager.findfont(font_manager.FontProperties(family=font), fallback_to_default=False)
    except ValueError:
        face = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        warnings.warn(
            f"Requested font {font!r} is unavailable; using Matplotlib's bundled DejaVu Sans.",
            UserWarning,
            stacklevel=2,
        )
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font_manager.FontProperties(fname=face).get_name()],
        "font.size": size, "text.usetex": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": .52, "axes.edgecolor": "#7B8B91", "xtick.major.size": 2.2,
        "ytick.major.size": 2.2, "xtick.major.width": .5, "ytick.major.width": .5,
    })

def draw_panel(axis, data, network, metric, args, *, narrow=False):
    panel = data.xs((metric, network), level=("metric", "dataset"))
    for name in reversed(STYLES):
        group = panel.loc[name].reindex(BUDGETS)
        color, marker, pattern = STYLES[name]
        ours = name == "PrivTDD"
        axis.fill_between(BUDGETS, group.lower95.to_numpy(), group.upper95.to_numpy(),
                          color=color, alpha=.12, linewidth=0, zorder=1)
        axis.plot(BUDGETS, group["mean"], color=color, ls=pattern, marker=marker, markersize=2.65 if narrow else 3,
                  lw=1.12 if ours else .83, zorder=4 if ours else 3,
                  markerfacecolor=color if ours else "white", markeredgewidth=.64)
    axis.grid(axis="y", color="#D9E0E2", linewidth=.4, zorder=0)
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_xlim(.32, 4.17)
    major = [.5, 2, 4] if narrow else [.5, 2, 3, 4]
    axis.set_xticks(major, [f"{value:g}" for value in major])
    axis.set_xticks([x for x in BUDGETS if x not in major], minor=True)
    axis.tick_params(axis="x", which="minor", length=1.3, width=.4)
    tick_size = args.font_size - .2
    axis.tick_params(labelsize=tick_size, pad=1.65)
    set_panel_axis(axis, metric, panel, args.panel_limits.get((network, metric)), tick_size)


def add_legend(fig, *, narrow=False):
    handles = [Line2D([], [], color=color, marker=marker, ls=pattern, markersize=3.2, lw=1,
                      markerfacecolor=color if name == "PrivTDD" else "white", markeredgewidth=.68)
               for name, (color, marker, pattern) in STYLES.items()]
    options = dict(loc="center" if narrow else "upper center", ncol=5,
                   bbox_to_anchor=(.5, .5 if narrow else .985), frameon=False, borderaxespad=0, borderpad=0,
                   handlelength=1.03 if narrow else 1.85, columnspacing=.48 if narrow else 1.25,
                   handletextpad=.30 if narrow else .42)
    legend = fig.legend(handles, list(STYLES), **options)
    if narrow:
        fig.canvas.draw()
        width = legend.get_window_extent().width
        if width > fig.bbox.width:
            # Font metrics vary by platform; keep the complete legend inside the column.
            size = legend.get_texts()[0].get_fontsize() * .98 * fig.bbox.width / width
            legend.remove()
            fig.legend(handles, list(STYLES), fontsize=size, **options)


def save(fig, stem, args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        dest = args.out_dir / f"{stem}.{extension}"
        fig.savefig(dest, dpi=args.dpi, facecolor="white")
        print(dest)
    plt.close(fig)


def draw(data, args):
    configure_font(args.font, args.font_size)
    fig, axes = plt.subplots(3, 4, sharex=True, figsize=(args.width, args.height))
    fig.subplots_adjust(left=.083, right=.985, bottom=.135, top=.863, wspace=args.dataset_gap, hspace=.25)
    for i, metric in enumerate(METRICS):
        for j, network in enumerate(NETWORKS):
            axis = axes[i, j]
            draw_panel(axis, data, network, metric, args)
            if i == 0:
                axis.set_title(network, fontsize=args.font_size + .3, weight="bold", pad=3)
            if j == 0:
                axis.set_ylabel(metric, fontsize=args.font_size, labelpad=2.5)
    add_legend(fig)
    fig.supxlabel("Total privacy budget ε", x=.53, y=.004, fontsize=args.font_size)
    save(fig, "gdd_ablation_curves", args)

def draw_columns(data, args):
    configure_font(args.font, args.font_size)
    legend = plt.figure(figsize=(args.column_width, .21))
    add_legend(legend, narrow=True)
    save(legend, "gdd_ablation_key", args)
    for network in NETWORKS:
        fig, axes = plt.subplots(1, 3, sharex=True, figsize=(args.column_width, args.strip_height))
        fig.subplots_adjust(left=.125, right=.98, bottom=.205, top=.756, wspace=.59)
        fig.text(.018, .895, network, fontsize=args.font_size + .2, weight="bold", ha="left", va="center")
        for axis, metric in zip(axes, METRICS):
            draw_panel(axis, data, network, metric, args, narrow=True)
            position = axis.get_position()
            first_metric = metric == METRICS[0]
            heading_x = position.x1 if first_metric else (position.x0 + position.x1) / 2
            fig.text(heading_x, .895, metric, fontsize=args.font_size, ha="right" if first_metric else "center", va="center")
        save(fig, "gdd_ablation_" + SLUGS[network], args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/e2_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=HERE / "icde_sections/figures")
    parser.add_argument("--layout", choices=("wide", "column", "both"), default="both")
    parser.add_argument("--font", default="Arial")
    parser.add_argument("--font-size", type=float, default=8.2)
    parser.add_argument("--width", type=float, default=7.16, help="Double-column width in inches")
    parser.add_argument("--height", type=float, default=2.83, help="Double-column figure height in inches")
    parser.add_argument("--dataset-gap", type=float, default=.48, help="Gap between datasets in the wide layout")
    parser.add_argument("--column-width", type=float, default=3.49, help="Single-column width in inches")
    parser.add_argument("--strip-height", type=float, default=.94, help="Height per single-column network strip in inches")
    parser.add_argument("--ylim", nargs=4, action="append", default=[], metavar=("DATASET", "METRIC", "LOW", "HIGH"),
                        help='Override one panel, e.g. --ylim "DBLP" EMD 1 15. May be repeated.')
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    args = parser.parse_args(argv)
    if min(args.font_size, args.width, args.height, args.column_width, args.strip_height, args.dpi) <= 0 or args.dataset_gap < 0:
        parser.error("Dimensions, font size and DPI must be positive. The dataset gap must be nonnegative.")
    args.panel_limits = {}
    for network, metric, low, high in args.ylim:
        if network not in NETWORKS or metric not in METRICS:
            parser.error(f"Unknown --ylim panel: {network}, {metric}")
        try:
            low, high = float(low), float(high)
        except ValueError:
            parser.error("--ylim bounds must be numbers.")
        if not np.isfinite([low, high]).all() or high <= low or (metric != "KS" and low <= 0):
            parser.error("Use finite, increasing bounds and a positive lower bound for logarithmic axes.")
        args.panel_limits[network, metric] = low, high
    data = read_data(args.data)
    if args.layout in ("wide", "both"):
        draw(data, args)
    if args.layout in ("column", "both"):
        draw_columns(data, args)


if __name__ == "__main__":
    main()
