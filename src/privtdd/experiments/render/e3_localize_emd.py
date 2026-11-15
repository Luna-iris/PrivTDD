"""Localize EMD and the maximum CDF gap using each release's default cutpoints."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .e3_statistics import DATA, PAIRS, row, write_table

NAMES = {"bio-dmela": "Bio-DMELA", "dblp": "DBLP", "higgs": "Higgs social", "orkut": "Orkut"}
METHODS = dict(zip(("q80_90", "q90_95", "q95_99"), PAIRS))
REGIONS = ("zero", "body", "transition", "tail")
KEYS = ["dataset", "epsilon", "seed"]


@lru_cache(maxsize=4)
def truth_histogram(path, digest):
    with path.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == digest
    counts = np.load(path, allow_pickle=False)
    x, freq = np.unique(counts, return_counts=True)
    return x, freq / len(counts)


def components(truth, prediction, b1, b2):
    keys, mass = truth
    with np.load(prediction, allow_pickle=False) as saved:
        support, freq = saved["keys"], saved["counts"]
    # The gap is constant from an integer support point up to, but excluding, the next one.
    # Insert the starts of the positive body, transition and tail before summing those gaps.
    starts = [1, b1 + 1, b2 + 1]
    grid = np.unique(np.r_[0, keys, support, starts])
    gap = np.zeros(len(grid))
    np.add.at(gap, np.searchsorted(grid, keys), mass)
    np.add.at(gap, np.searchsorted(grid, support), -freq / freq.sum())
    absolute_gap = np.abs(np.cumsum(gap))
    areas = absolute_gap[:-1] * np.diff(grid)
    regions = np.searchsorted(starts, grid[:-1], side="right")
    peak = np.argmax(absolute_gap)
    parts = np.bincount(regions, weights=areas, minlength=4)
    # Count two is a post-hoc diagnostic at the most common upper-pair KS peak on DBLP.
    return dict(zip(REGIONS, parts), total=parts.sum(), KS=absolute_gap[peak], ks_peak_t=int(grid[peak]),
                truth_cdf_at_2=mass[keys <= 2].sum(), estimated_cdf_at_2=freq[support <= 2].sum() / freq.sum())


def collect(root):
    published = pd.read_csv(DATA / "releases.csv", float_precision="round_trip").set_index(KEYS + ["quantiles"])
    jobs = json.loads((root / "forpaper/plan.json").read_bytes())["jobs"]
    jobs = {(NAMES[t["dataset"]], t["epsilon"], t["seed"]): t["id"] for t in jobs
            if t["panel"] == "main" and t.get("method") == "LDP-GDD" and t["dataset"] in NAMES}
    records = []
    for suite in ("paper_required", "paper_required_E3_large"):
        data = pd.read_csv(root / suite / "E3/releases.csv", float_precision="round_trip")
        for r in data.itertuples():
            key = (NAMES[r.dataset], r.epsilon, r.seed)
            pair = METHODS[r.method]
            official = published.loc[(*key, pair)]
            default = published.loc[(*key, PAIRS[1])]
            assert r.report_sha256 == official.report_sha256 == default.report_sha256
            e1, = json.loads((root / "forpaper/jobs" / jobs[key] / "records.json").read_bytes())
            assert e1["truth_sha256"] == official.truth_sha256
            path = Path(r.prediction)
            with path.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == official.prediction_sha256
            b1, b2 = int(default.b1), int(default.b2)
            diagnostic = components(truth_histogram(Path(e1["truth"]), official.truth_sha256), path, b1, b2)
            np.testing.assert_allclose(diagnostic["total"], official.EMD, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(diagnostic["KS"], official.KS, rtol=1e-10, atol=1e-10)
            records.append(dict(zip(KEYS, key), quantiles=pair, default_b1=b1, default_b2=b2, **diagnostic))
    frame = pd.DataFrame(records).set_index(KEYS + ["quantiles"])
    assert frame.index.is_unique and set(frame.index) == set(published.index)
    return frame.sort_index().reset_index()


def contrasts(frame):
    out = []
    for (ds, eps), group in frame.groupby(KEYS[:2], sort=False):
        n = group.seed.nunique()
        idx = np.random.default_rng(731).integers(n, size=(2000, n))
        for metric in (*REGIONS, "total"):
            table = group.pivot(index="seed", columns="quantiles", values=metric).sort_index()
            delta = (table[PAIRS[2]] - table[PAIRS[0]]).to_numpy()
            lo, hi = np.quantile(delta[idx].mean(axis=1), [.025, .975])
            from_default = (table[PAIRS[2]] - table[PAIRS[1]]).to_numpy()
            default_lo, default_hi = np.quantile(from_default[idx].mean(axis=1), [.025, .975])
            out.append(dict(dataset=ds, epsilon=eps, n=n, region=metric, mean_lower=table[PAIRS[0]].mean(),
                            mean_default=table[PAIRS[1]].mean(), mean_upper=table[PAIRS[2]].mean(),
                            upper_minus_lower=delta.mean(), lower95=lo, upper95=hi,
                            upper_minus_default=from_default.mean(), upper_minus_default_lower95=default_lo,
                            upper_minus_default_upper95=default_hi))
    return pd.DataFrame(out)


def write_dblp_table(frame):
    cell = frame[(frame.dataset == "DBLP") & (frame.epsilon == 2)]
    peaks = cell.pivot(index="seed", columns="quantiles", values="ks_peak_t").reindex(columns=PAIRS)
    if peaks.shape != (10, 3) or peaks.isna().any().any() or cell.truth_cdf_at_2.nunique() != 1:
        raise ValueError("Expected all three boundary pairs on the same ten DBLP runs.")
    means = cell.groupby("quantiles").mean(numeric_only=True).reindex(PAIRS)
    np.testing.assert_allclose(means[list(REGIONS)].sum(axis=1), means.total, rtol=1e-10)
    lines = [r"\begin{table}[!htbp]\centering",
             r"\caption{Regional EMD and low-count cumulative estimates on DBLP at $\varepsilon=2$.}",
             r"\label{tab:gdd:e3:supp:dblp}",
             r"\fontsize{8.2}{9.8}\selectfont\setlength{\tabcolsep}{1.6pt}",
             r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l rrr@{}}\toprule",
             row(["Positive-count percentiles", "80/90", "90/95", "95/99"]), r"\midrule"]
    for key, label in zip((*REGIONS, "total"),
                          (r"EMD at $t=0$", "Body EMD", "Transition EMD", "Tail EMD", "Total EMD")):
        lines.append(row([label, *[f"{v:.3f}" for v in means[key]]]))
    lines.extend([r"\midrule",
                  row([r"Estimated $F(2)$ (\%)", *[f"{100*v:.2f}" for v in means.estimated_cdf_at_2]]),
                  row([r"True $F(2)$ (\%)", rf"\multicolumn{{3}}{{c}}{{{100*cell.truth_cdf_at_2.iloc[0]:.2f}}}"]),
                  row([r"Runs with KS peak at $t=2$", *[str(v) for v in peaks.eq(2).sum()]]),
                  r"\bottomrule\end{tabular*}",
                  r"\par\vspace{2pt}\begin{minipage}{\columnwidth}\fontsize{8}{9.5}\selectfont\raggedright "
                  r"Means use ten paired runs. Within each run, all fits use the default boundaries for scoring the "
                  r"four EMD contributions. $F(2)$ is the fraction of vertices in at most two triangles. "
                  r"We inspected the KS peak locations to select this illustrative threshold.",
                  r"\end{minipage}\end{table}"])
    write_table("e3_supp_dblp.tex", lines)


def main(data_dir, output_dir):
    from . import e3_statistics
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    e3_statistics.TABLES = output_dir
    rows = pd.read_csv(data_dir / "emd_by_region.csv", float_precision="round_trip")
    contrasts(rows).to_csv(output_dir / "e3_emd_region_contrasts.csv", index=False)
    write_dblp_table(rows)
    return rows


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args()
    main(args.data_dir,args.out_dir)
