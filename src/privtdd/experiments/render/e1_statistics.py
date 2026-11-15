"""Build all E1 tables from releases.csv and jobs.csv. No experiment execution."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["MSE", "EMD", "KS"]
METHODS = ["PrivTDD", "TriTR-Hist", "LDPGraph-T", "LDPGraph-S"]
DATASETS = ["Bio-DMELA", "DBLP", "Higgs social", "Orkut"]
GROUP = ["dataset", "epsilon", "method"]


def interval(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    if len(values) == 1:
        low = high = sd = np.nan
    else:
        rng = np.random.default_rng(731)
        means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        sd = values.std(ddof=1)
    return dict(n=len(values), mean=values.mean(), sd=sd, lower95=low, upper95=high)


def summarize(folder):
    results = pd.read_csv(folder / "releases.csv", float_precision="round_trip").sort_values("seed")
    jobs = pd.read_csv(folder / "jobs.csv")
    assert set(results.job_id) == set(jobs.loc[jobs.status.eq("complete"), "job_id"])
    assert not jobs.duplicated(GROUP + ["seed"]).any()
    planned = jobs.groupby(GROUP).size()
    summary = []
    for key, group in results.groupby(GROUP):
        for metric in METRICS:
            summary.append(dict(zip(GROUP, key), metric=metric, planned=planned[key],
                                **interval(group[metric])))
    stats = pd.DataFrame(summary)
    stats.to_csv(folder / "summary.csv", index=False)

    pair_on = ["dataset", "epsilon", "seed"]
    subset = results[results.method.eq("LDPGraph-T")]
    pairs = subset.merge(results[results.method.eq("LDPGraph-S")], on=pair_on,
                         suffixes=("_T", "_S"), how="inner", validate="one_to_one")
    assert pairs.report_sha256_T.notna().all()
    assert (pairs.report_sha256_T == pairs.report_sha256_S).all()
    diffs = pairs[pair_on].copy()
    diffs["report_sha256"] = pairs.report_sha256_T
    for m in METRICS:
        diffs[m] = pairs[m + "_S"] - pairs[m + "_T"]
    diffs.to_csv(folder / "paired_releases.csv", index=False)
    paired = []
    for (ds, eps), group in diffs.sort_values("seed").groupby(pair_on[:2]):
        for m in METRICS:
            paired.append(dict(dataset=ds, epsilon=eps, metric=m, **interval(group[m])))
    pd.DataFrame(paired).to_csv(folder / "paired_summary.csv", index=False)

    # Sensitivity of the ranking when all methods use the same run indices.
    # The three protocols still use independent random streams.
    common = results.groupby(pair_on).method.transform("nunique").eq(len(METHODS))
    common_results = results.loc[common].groupby(GROUP)[METRICS].mean()
    balanced = common_results.stack().rename("mean_common_seeds").reset_index()
    balanced = balanced.rename(columns={"level_3": "metric"})
    balanced["n_common"] = [int(common.loc[
        (results.dataset == r.dataset) & (results.epsilon == r.epsilon)
        & (results.method == r.method)].sum()) for r in balanced.itertuples()]
    comparisons = stats.merge(balanced, on=GROUP + ["metric"])
    setting = ["dataset", "epsilon", "metric"]
    comparisons["rank_available"] = comparisons.groupby(setting)["mean"].rank(method="min")
    comparisons["rank_common"] = comparisons.groupby(setting)["mean_common_seeds"].rank(method="min")
    comparisons.to_csv(folder / "ranks.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    summarize(parser.parse_args().folder)
