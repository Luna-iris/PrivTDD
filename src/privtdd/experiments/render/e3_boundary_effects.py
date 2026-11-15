"""Compare the tested endpoint quantiles and rank PrivTDD against the E1 baselines."""
import numpy as np
import pandas as pd

from .e3_statistics import DATA, NETWORKS, BUDGETS, PAIRS, DEFAULT, METRICS, math, row, write_table

KEY = ["dataset", "epsilon", "seed"]
BASELINES = ("TriTR-Hist", "LDPGraph-T", "LDPGraph-S")
LEVELS = ("lower", "default", "upper")


def calculate(e1, e3):
    if e1.duplicated(KEY + ["method"]).any() or e3.duplicated(KEY + ["quantiles"]).any():
        raise ValueError("Duplicate method or quantile outputs for a release.")
    full = e3.loc[e3["quantiles"] == DEFAULT].set_index(KEY).sort_index()
    original = e1.loc[e1.method == "PrivTDD"].set_index(KEY).sort_index()
    if not original.index.equals(full.index):
        raise ValueError("E1 and E3 use different release indices.")
    for field in ["report_sha256", "truth_sha256", "prediction_sha256", *METRICS]:
        np.testing.assert_array_equal(original[field], full[field])
    checks, comparisons = [], []
    for ds in NETWORKS:
        for eps in BUDGETS:
            variants = e3[(e3.dataset == ds) & (e3.epsilon == eps)]
            seeds = variants.seed.unique()
            baseline = e1[(e1.dataset == ds) & (e1.epsilon == eps) & e1.method.isin(BASELINES)]
            if (variants.groupby("seed").report_sha256.nunique().ne(1).any()
                    or pd.concat([variants, baseline]).truth_sha256.nunique() != 1):
                raise ValueError(f"Inconsistent E3 reports or evaluation truth at {ds}, {eps}")
            n = len(seeds)
            resample = np.random.default_rng(731).integers(n, size=(2000, n))
            for metric in METRICS:
                w = variants.pivot(index="seed", columns="quantiles", values=metric).sort_index()
                b = baseline.pivot(index="seed", columns="method", values=metric).sort_index()
                if (not w.index.equals(b.index) or set(w.columns) != set(PAIRS) or set(b.columns) != set(BASELINES)
                        or not np.isfinite(w.to_numpy()).all() or not np.isfinite(b.to_numpy()).all()):
                    raise ValueError(f"Incomplete comparison at {ds}, {eps}, {metric}")
                means = w.loc[:, list(PAIRS)].mean()
                baseline_means = b.mean()
                common = dict(dataset=ds, epsilon=eps, metric=metric, n=n)
                checks.append(dict(**common, **baseline_means.to_dict(), best_baseline=baseline_means.idxmin(),
                    **{f"mean_{name}": value for name, value in zip(LEVELS, means)},
                    **{f"rank_{name}": 1 + int((baseline_means < value).sum()) for name, value in zip(LEVELS, means)}))
                lower, upper = w[PAIRS[0]].to_numpy(), w[PAIRS[2]].to_numpy()
                diff = upper - lower
                dlo, dhi = np.quantile(diff[resample].mean(axis=1), [.025, .975])
                plo, phi = np.quantile(100 * (upper[resample].mean(axis=1) / lower[resample].mean(axis=1) - 1),
                                       [.025, .975])
                comparisons.append(dict(**common, upper_minus_lower=diff.mean(), delta_lower95=dlo, delta_upper95=dhi,
                    percent=100 * (upper.mean() / lower.mean() - 1), percent_lower95=plo, percent_upper95=phi,
                    count_upper_better=(diff < 0).sum(), count_equal=(diff == 0).sum(), count_lower_better=(diff > 0).sum()))
    return pd.DataFrame(checks), pd.DataFrame(comparisons)


def supplement(changes):
    values = changes.set_index(["dataset", "epsilon", "metric"])
    lines = [r"\begin{table*}[tp]\centering",
             r"\caption{Paired comparisons of the upper and lower region quantiles.}",
             r"\label{tab:gdd:e3:supp:endpoints}", r"\fontsize{8.3}{9.6}\selectfont\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcc rr rr rr@{}}\toprule",
             row(["", "", ""] + [rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in METRICS]),
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
             row(["Dataset", r"$\varepsilon$", "$n$"] + [r"Change (\%)", r"95\% CI"] * 3), r"\midrule"]
    for i, ds in enumerate(NETWORKS):
        if i:
            lines.append(r"\addlinespace[2pt]")
        for j, eps in enumerate(BUDGETS):
            cells = [rf"\multirow{{2}}{{*}}{{{ds}}}" if j == 0 else "", f"{eps:g}", str(int(values.loc[(ds,eps,"EMD"),"n"]))]
            for m in METRICS:
                v = values.loc[(ds, eps, m)]
                cells.extend([math(f"{v.percent:+.3g}"), math(f"[{v.percent_lower95:+.3g},\\,{v.percent_upper95:+.3g}]")])
            lines.append(row(cells))
    lines.extend([r"\bottomrule\end{tabular*}", r"\par\vspace{2pt}\begin{minipage}{\textwidth}",
                  r"\fontsize{8}{9.4}\selectfont\raggedright Change is $100(\bar e_{\rm upper}/\bar e_{\rm lower}-1)$. "
                  r"Pointwise intervals use 2,000 resamples of matched reports within each dataset and budget.",
                  r"\end{minipage}\end{table*}"])
    write_table("e3_supp_endpoints.tex", lines)


def main(data_dir, output_dir):
    from pathlib import Path
    from . import e3_statistics
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    e3_statistics.TABLES = output_dir
    baseline = pd.read_csv(data_dir / "e1_matched_releases.csv", float_precision="round_trip")
    variants = pd.read_csv(data_dir / "releases.csv", float_precision="round_trip")
    ranks, changes = calculate(baseline, variants)
    ranks.to_csv(output_dir / "e3_baseline_rank_checks.csv", index=False)
    changes.to_csv(output_dir / "e3_endpoint_contrasts.csv", index=False)
    supplement(changes)
    return ranks, changes


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    main(args.data_dir,args.out_dir)
