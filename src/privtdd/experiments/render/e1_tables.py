"""Write three IEEEtran table layouts from the original E1 point estimates and intervals."""
import argparse
import csv
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
NETWORKS = ("Bio-DMELA", "DBLP", "Higgs social", "Orkut")
BUDGETS = tuple(i / 2 for i in range(1, 9))
METHODS = ("PrivTDD", "TriTR-Hist", "LDPGraph-T", "LDPGraph-S")
METRICS = ("MSE", "EMD", "KS")
SCALED_HEADERS = (r"$10^3$\,MSE", "EMD", r"$100$\,KS")
SCALES = {"MSE": 1000, "EMD": 1, "KS": 100}
RAW_DIGITS = {"MSE": 5, "EMD": 2, "KS": 4}
LABEL = "tab:gdd:e1:accuracy"
MARK_NOTE = ("Gray cells identify PrivTDD. Bold marks the lowest mean across the four methods "
             "at each dataset, budget and metric.")
SAMPLE_NOTE = ("Means use 20 releases per method on Bio-DMELA and DBLP and 10 on Higgs social and Orkut. "
               r"PrivTDD has eight at $\varepsilon=0.5$ on Higgs social and $\varepsilon=4$ on Orkut.")
END = r" \\"
RIGHT_X = r">{\raggedleft\arraybackslash}X"


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float


def read_data(path):
    data = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            ds, ep, metric, method = row["dataset"], float(row["epsilon"]), row["metric"], row["method"]
            key = ds, ep, metric, method
            estimate = Estimate(*(float(row[c]) for c in ("mean", "lower95", "upper95")))
            if key in data or not all(math.isfinite(v) and v >= 0 for v in vars(estimate).values()):
                raise ValueError(f"Duplicate or invalid E1 record: {key}")
            planned = 20 if ds in NETWORKS[:2] else 10
            partial = method == "PrivTDD" and (ds, ep) in {("Higgs social", 0.5), ("Orkut", 4)}
            if int(row["planned"]) != planned or int(row["n"]) != (8 if partial else planned):
                raise ValueError(f"Sample counts changed. Check the table notes for {key}.")
            if not estimate.lower <= estimate.mean <= estimate.upper:
                raise ValueError(f"Check the confidence interval for {key}.")
            data[key] = estimate
    if set(data) != set(product(NETWORKS, BUDGETS, METRICS, METHODS)):
        raise ValueError("Expected four networks, eight budgets, three metrics and four methods.")
    return data


def cells_for(data, ds, ep, metric, *, scaled):
    estimates = [data[ds, ep, metric, method] for method in METHODS]
    best = min(item.mean for item in estimates)
    scale = SCALES[metric] if scaled else 1
    digits = 2 if scaled else RAW_DIGITS[metric]

    def fmt(value):
        return f"{value * scale:.{digits}f}"

    if len({fmt(item.mean) for item in estimates}) != len({item.mean for item in estimates}):
        raise ValueError(f"Displayed precision conceals a method difference at {ds}, {ep}, {metric}.")
    cells = {}
    for method, item in zip(METHODS, estimates):
        mean = fmt(item.mean)
        if item.mean == best:
            mean = rf"\textbf{{{mean}}}"
        background = r"\cellcolor[gray]{0.90}" if method == "PrivTDD" else ""
        bounds = f"{fmt(item.lower)}--{fmt(item.upper)}"
        cells[method] = (background + mean, background + bounds)
    return cells


def row(items):
    return " & ".join(items) + END


def span(label, rows):
    return rf"\multirow{{{rows}}}{{*}}{{{label}}}"


def table_start(caption, size, baseline, *, continued=False):
    lines = [r"\begin{table*}[!t]", r"\centering"]
    if continued:
        # An unnumbered continuation preserves the original counter, label and list-of-tables entry.
        lines.append(rf"\csname @makecaption\endcsname{{\tablename~\ref{{{LABEL}}} (continued)}}{{{caption}}}")
    else:
        lines.extend([rf"\caption{{{caption}}}", rf"\label{{{LABEL}}}"])
    lines.extend([rf"\fontsize{{{size}}}{{{baseline}}}\selectfont", r"\setlength{\tabcolsep}{2.4pt}"])
    return lines


def table_end(note):
    return [r"\bottomrule", r"\end{tabularx}", r"\par\vspace{3pt}",
            r"\begin{minipage}{\linewidth}\fontsize{8}{9.3}\selectfont\raggedright " + note,
            r"\end{minipage}", r"\end{table*}"]


def wide(data):
    lines = table_start("Estimation error on four networks across eight privacy budget settings", 8.5, 10)
    lines.extend([r"\renewcommand{\arraystretch}{1.08}",
                  rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}lc *{{12}}{{{RIGHT_X}}}@{{}}}}",
                  r"\toprule",
                  row(["", ""] + [rf"\multicolumn{{3}}{{c}}{{{name}}}" for name in METHODS]),
                  r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(l){12-14}",
                  row(["Datasets", r"$\varepsilon$"] + list(SCALED_HEADERS) * 4), r"\midrule"])
    for i, ds in enumerate(NETWORKS):
        if i:
            lines.append(r"\midrule")
        for j, ep in enumerate(BUDGETS):
            estimates = [cells_for(data, ds, ep, m, scaled=True) for m in METRICS]
            means = [metric[name][0] for name in METHODS for metric in estimates]
            lines.append(row([span(ds, 8) if j == 0 else "", f"{ep:g}", *means]))
    return lines + table_end(f"{MARK_NOTE} MSE is multiplied by 1,000 and KS is in percentage points. {SAMPLE_NOTE}")


def spread(data):
    lines = []
    for half, pair in enumerate((NETWORKS[:2], NETWORKS[2:])):
        title = f"Recovery error with pointwise 95\\% bootstrap intervals on {pair[0]} and {pair[1]}"
        lines.extend(table_start(title, 8.5, 9.7, continued=bool(half)))
        lines.extend([r"\renewcommand{\arraystretch}{1.02}",
                      r"\begin{tabularx}{\linewidth}{@{}lcl"
                      + "*{4}{" + RIGHT_X + r">{\fontsize{8}{9}\selectfont}r}@{}}",
                      r"\toprule", row(["", "", ""] + [rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in METHODS]),
                      r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}",
                      row(["Datasets", r"$\varepsilon$", "Metric"] + ["Mean", r"95\% CI"] * 4), r"\midrule"])
        for i, ds in enumerate(pair):
            if i:
                lines.append(r"\midrule")
            for j, ep in enumerate(BUDGETS):
                if j:
                    lines.append(r"\addlinespace[2.3pt]")
                for k, metric in enumerate(METRICS):
                    cells = [span(ds, 24) if j == k == 0 else "", span(f"{ep:g}", 3) if k == 0 else "", SCALED_HEADERS[k]]
                    estimates = cells_for(data, ds, ep, metric, scaled=True)
                    for method in METHODS:
                        cells.extend(estimates[method])
                    lines.append(row(cells))
        sample = ("Each setting uses 20 releases per method." if not half else
                  r"Each setting uses 10 releases per method except PrivTDD on Higgs social at $\varepsilon=0.5$ and Orkut at $\varepsilon=4$, with eight each.")
        note = (f"{MARK_NOTE} CI gives the pointwise bootstrap interval. "
                f"MSE is multiplied by 1,000 and KS is in percentage points. {sample}")
        lines.extend(table_end(note))
    return lines


def trends(data):
    lines = table_start("Recovery error by privacy budget in original metric units", 9, 10)
    lines.extend([r"\renewcommand{\arraystretch}{1.00}",
                  rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}lll *{{8}}{{{RIGHT_X}}}}}", r"\toprule",
                  row(["", "", "", r"\multicolumn{8}{c}{Total privacy budget $\varepsilon$}"]),
                  r"\cmidrule(l){4-11}", row(["Datasets", "Metric", "Method"] + [f"{ep:g}" for ep in BUDGETS]), r"\midrule"])
    for i, ds in enumerate(NETWORKS):
        if i:
            lines.append(r"\midrule")
        for k, metric in enumerate(METRICS):
            if k:
                lines.append(r"\addlinespace[2.2pt]")
            vectors = [cells_for(data, ds, ep, metric, scaled=False) for ep in BUDGETS]
            for j, method in enumerate(METHODS):
                labels = [span(ds, 12) if k == j == 0 else "", span(metric, 4) if j == 0 else "", method]
                lines.append(row(labels + [vec[method][0] for vec in vectors]))
    return lines + table_end(f"{MARK_NOTE} {SAMPLE_NOTE}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/e1_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=HERE / "e1_options")
    args = parser.parse_args(argv)
    data = read_data(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for layout in (wide, spread, trends):
        path = args.out_dir / f"e1_{layout.__name__}.tex"
        path.write_text("\n".join(layout(data)) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
