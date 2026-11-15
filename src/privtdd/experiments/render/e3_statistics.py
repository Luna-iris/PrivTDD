"""Rebuild the E3 manuscript and supplemental tables from matched releases."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TABLES = HERE / "icde_sections/tables"
NETWORKS = ("Bio-DMELA", "DBLP", "Higgs social", "Orkut")
BUDGETS = (0.5, 2.0)
PAIRS = ("(0.80,0.90)", "(0.90,0.95)", "(0.95,0.99)")
DEFAULT = PAIRS[1]
ALTERNATIVES = (PAIRS[0], PAIRS[2])
METRICS = ("MSE", "EMD", "KS")


def stats(frame):
    means, contrasts = [], []
    for network in NETWORKS:
        for budget in BUDGETS:
            group = frame[(frame.dataset == network) & (frame.epsilon == budget)]
            by_seed = group.groupby("seed")
            if (by_seed.report_sha256.nunique().ne(1).any() or by_seed.truth_sha256.nunique().ne(1).any()
                    or by_seed.size().ne(3).any()):
                raise ValueError(f"Unmatched reports at {network}, epsilon={budget}")
            n = len(by_seed)
            indices = np.random.default_rng(731).integers(n, size=(2000, n))
            for metric in METRICS:
                pivot = group.pivot(index="seed", columns="quantiles", values=metric).sort_index()
                if set(pivot.columns) != set(PAIRS) or not np.isfinite(pivot.to_numpy()).all():
                    raise ValueError(f"Incomplete outputs for {network}, epsilon={budget}, {metric}")
                default = pivot[DEFAULT].to_numpy()
                default_draws = default[indices].mean(axis=1)
                for pair in PAIRS:
                    values = pivot[pair].to_numpy()
                    draws = values[indices].mean(axis=1)
                    lo, hi = np.quantile(draws, [.025, .975])
                    key = dict(dataset=network, epsilon=budget, n=n, quantiles=pair, metric=metric)
                    means.append(dict(**key, mean=values.mean(), lower95=lo, upper95=hi))
                    if pair == DEFAULT:
                        continue
                    differences = values - default
                    delta_lo, delta_hi = np.quantile(differences[indices].mean(axis=1), [.025, .975])
                    pct_lo, pct_hi = np.quantile(100 * (draws / default_draws - 1), [.025, .975])
                    contrasts.append(dict(**key, reference=DEFAULT, mean_alternative=values.mean(),
                        mean_default=default.mean(), delta=differences.mean(), delta_lower95=delta_lo,
                        delta_upper95=delta_hi, percent=100 * (values.mean() / default.mean() - 1),
                        percent_lower95=pct_lo, percent_upper95=pct_hi))
    return pd.DataFrame(means), pd.DataFrame(contrasts)


def math(value):
    return "$" + value + "$"


def row(values):
    return " & ".join(values) + r" \\"


def write_table(filename, lines):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_table(contrasts):
    indexed = contrasts.set_index(["dataset", "epsilon", "quantiles", "metric"])
    lines = [r"\begin{table}[!t]", r"\centering",
        r"\caption{Percent change in mean recovery error relative to the default region quantiles $(0.90,0.95)$.}",
        r"\label{tab:gdd:e3:boundaries}", r"\fontsize{8}{9.4}\selectfont\setlength{\tabcolsep}{1.25pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lc rrr @{\hspace{5pt}\extracolsep{\fill}}rrr@{}}",
        r"\toprule", row(["", "", r"\multicolumn{3}{c}{$(0.80,0.90)$}", r"\multicolumn{3}{c}{$(0.95,0.99)$}"]),
        r"\cmidrule(lr){3-5}\cmidrule(l){6-8}", row(["Dataset", r"$\varepsilon$", *METRICS, *METRICS]), r"\midrule"]
    for i, network in enumerate(NETWORKS):
        if i:
            lines.append(r"\addlinespace[2.4pt]")
        for j, budget in enumerate(BUDGETS):
            values = [rf"\multirow{{2}}{{*}}{{{network}}}" if j == 0 else "", f"{budget:g}"]
            for pair in ALTERNATIVES:
                for metric in METRICS:
                    pct = indexed.loc[(network, budget, pair, metric), "percent"]
                    if pct == 0:
                        values.append("$0.0$")
                    elif abs(pct) < .05:
                        values.append(r"$\approx 0$")
                    else:
                        values.append(math(f"{pct:+.1f}"))
            lines.append(row(values))
    lines.extend([r"\bottomrule\end{tabular*}", r"\par\vspace{2pt}\begin{minipage}{\columnwidth}",
        r"\fontsize{8}{9.4}\selectfont\raggedright Negative values indicate lower error. "
        r"The symbol $\approx 0$ denotes a nonzero change below $0.05\%$ in magnitude. "
        r"Each setting uses ten matched reports, except Higgs social at $\varepsilon=0.5$ (eight).",
        r"\end{minipage}", r"\end{table}"])
    write_table("e3_boundary_changes.tex", lines)


def supplemental_table(frame, *, percent):
    pairs = ALTERNATIVES if percent else PAIRS
    title = ("Relative change in mean recovery error from the default with paired 95\\% bootstrap intervals (percent)"
             if percent else "Mean recovery errors in original units with pointwise 95\\% bootstrap intervals")
    tag = "contrasts" if percent else "means"
    indexed = frame.set_index(["dataset", "epsilon", "quantiles", "metric"])
    lines = [r"\begin{table*}[tp]", r"\centering", rf"\caption{{{title}}}",
             rf"\label{{tab:gdd:e3:supp:{tag}}}", r"\fontsize{8.3}{9.7}\selectfont\setlength{\tabcolsep}{1.4pt}",
             r"\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcc l rr rr rr@{}}", r"\toprule",
             row(["", "", "", ""] + [rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in METRICS]),
             r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
             row(["Dataset", r"$\varepsilon$", "$n$", "Quantiles"] + (["Change", r"95\% CI"] if percent else ["Mean", r"95\% CI"]) * 3),
             r"\midrule"]
    point, lo, hi = ("percent", "percent_lower95", "percent_upper95") if percent else ("mean", "lower95", "upper95")
    for i, network in enumerate(NETWORKS):
        if i:
            lines.append(r"\midrule")
        for j, budget in enumerate(BUDGETS):
            if j:
                lines.append(r"\addlinespace[2.4pt]")
            for k, pair in enumerate(pairs):
                n = indexed.loc[(network, budget, pair, "EMD"), "n"]
                label = rf"$\mathbf{{{pair}}}$" if pair == DEFAULT else math(pair)
                values = [rf"\multirow{{{len(pairs)*2}}}{{*}}{{{network}}}" if j == k == 0 else "",
                          rf"\multirow{{{len(pairs)}}}{{*}}{{{budget:g}}}" if k == 0 else "",
                          rf"\multirow{{{len(pairs)}}}{{*}}{{{n}}}" if k == 0 else "", label]
                for metric in METRICS:
                    result = indexed.loc[(network, budget, pair, metric)]
                    if percent:
                        values.extend([math(f"{result[point]:+.3g}"), math(f"[{result[lo]:+.3g},\\,{result[hi]:+.3g}]")])
                    else:
                        digits = 7 if network == "Higgs social" and budget == 2 and metric != "EMD" else 5
                        values.extend([math(f"{result[point]:.{digits}g}"), math(f"[{result[lo]:.4g},\\,{result[hi]:.4g}]")])
                lines.append(row(values))
    note = (r"Percent change is $100(\bar e_{\rm alt}/\bar e_{\rm def}-1)$. The default is $(0.90,0.95)$. "
            r"Each interval uses 2,000 bootstrap resamples of matched report indices, with the ratio recomputed in each draw."
            if percent else r"Bold quantiles denote the default. Each interval uses 2,000 bootstrap resamples of the matched reports.")
    lines.extend([r"\bottomrule\end{tabular*}",
                  r"\par\vspace{2pt}\begin{minipage}{\textwidth}\fontsize{8}{9.4}\selectfont\raggedright " + note,
                  r"\end{minipage}", r"\end{table*}"])
    write_table(f"e3_supp_{tag}.tex", lines)


def main(data_dir, output_dir):
    global DATA, TABLES
    DATA, TABLES = Path(data_dir), Path(output_dir)
    releases = pd.read_csv(DATA / "releases.csv", float_precision="round_trip")
    means, contrasts = stats(releases)
    TABLES.mkdir(parents=True, exist_ok=True)
    means.to_csv(TABLES / "e3_means.csv", index=False)
    contrasts.to_csv(TABLES / "e3_paired_changes.csv", index=False)
    main_table(contrasts)
    supplemental_table(means, percent=False)
    supplemental_table(contrasts, percent=True)
    return means, contrasts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
