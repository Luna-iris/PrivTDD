"""Rebuild the paper's numerical tables and plots from immutable release records.

This is a statistical reproduction, not an execution of the private estimators.
The input snapshot is hash-verified and never edited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

METRICS = ("MSE", "EMD", "KS")
NETWORKS = ("Bio-DMELA", "DBLP", "Higgs social", "Orkut")


def repository_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "reproduction/manifest.json").is_file():
            return parent
    raise FileNotFoundError("Supply the artifact repository root containing reproduction/manifest.json")


def _read(path):
    return pd.read_csv(path, float_precision="round_trip")


def verify_snapshot(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "reproduction/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["frozen_files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root / "results/published"):
            raise ValueError("Frozen-file manifest escapes results/published")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen evidence changed: {item['path']}")
    snapshot = root / "results/published" / manifest["snapshot"]
    e1, jobs = _read(snapshot / "e1/releases.csv"), _read(snapshot / "e1/jobs.csv")
    if len(e1) != 1916 or len(jobs) != 1920 or jobs.status.eq("complete").sum() != 1916:
        raise ValueError("Unexpected E1 coverage")
    if set(e1.job_id) != set(jobs.loc[jobs.status.eq("complete"), "job_id"]):
        raise ValueError("E1 completed IDs differ from the published release set")
    e2 = _read(snapshot / "e2/releases.csv").query("suite == 'E2'")
    groups = e2.groupby(["dataset", "epsilon", "seed"])
    if len(groups) != 476 or len(e2) != 2380 or not groups.size().eq(5).all():
        raise ValueError("Unexpected E2 five-method matched coverage")
    if not groups.report_sha256.nunique().eq(1).all():
        raise ValueError("E2 contrasts do not share a private report")
    e3 = _read(snapshot / "e3/releases.csv")
    if len(e3) != 234 or len(_read(snapshot / "e3/coverage.csv")) != 240:
        raise ValueError("Unexpected E3 coverage")
    if len(e3.groupby(["dataset", "epsilon", "seed"])) != 78:
        raise ValueError("Unexpected E3 matched reports")
    e4 = _read(snapshot / "e4/large40_runs.csv")
    if len(e4) != 50 or len(e4[e4.rate < 1]) != 40:
        raise ValueError("Expected 40 sampled E4 releases and ten paired controls")
    return dict(snapshot=manifest["snapshot"],verified_files=len(manifest["frozen_files"]),
                e1_complete=1916,e1_planned=1920,e2_matched=476,e3_complete=234,e3_planned=240,
                e4_sampled=40,unresolved=[x for x in manifest["items"] if x.get("status")=="unresolved"])


def _compare_stats(actual, expected, keys, columns):
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f"Statistics have a different grouping: {keys}")
    for column in columns:
        np.testing.assert_allclose(a[column], b[column], rtol=2e-12, atol=2e-12,
                                   err_msg=f"Frozen statistics mismatch for {column}")


def render_e4(e1, sampled, out):
    """The published 100% row uses ten E1 runs, not the five paired E4 controls."""
    from .render.e1_statistics import interval
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, code in (("Higgs social", "higgs"), ("Orkut", "orkut")):
        for rate in (.0001, .001, .01, .1, 1.):
            if rate == 1:
                group = e1[(e1.dataset == dataset) & (e1.epsilon == 2) & (e1.method == "PrivTDD")]
                expected = list(range(20261028,20261038))
                source = "E1 ten-run published full-enumeration control"
            else:
                group = sampled[(sampled.dataset == code) & (sampled.epsilon == 2) & (sampled.rate == rate)]
                expected = list(range(20261028,20261033))
                source = "E4 five-run sampled setting"
            group = group.sort_values("seed")
            if list(group.seed.astype(int)) != expected:
                raise ValueError(f"Unexpected E4 seed set: {dataset}, rate={rate}")
            for metric in METRICS:
                rows.append(dict(dataset=dataset,epsilon=2,rate=rate,metric=metric,source=source,
                                 **interval(group[metric])))
    data = pd.DataFrame(rows)
    data.to_csv(out / "e4_summary.csv", index=False)
    indexed = data.set_index(["dataset","rate","metric"])
    lines=[r"\begin{table}[htbp]",r"\centering",
           r"\caption{TDD estimation error by neighbor-pair sampling rate at $\varepsilon=2$.}",
           r"\label{tab:gdd:e4:sampling}",r"\footnotesize",r"\setlength{\tabcolsep}{2pt}",
           r"\renewcommand{\arraystretch}{1.08}",
           r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{\hspace{8pt}}ccc@{}}",
           r"\toprule",r"\multirow{2}{*}{\shortstack[l]{Sampling rate}} & \multicolumn{3}{c}{Higgs social} & \multicolumn{3}{c}{Orkut} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",r" & MSE & EMD & KS & MSE & EMD & KS \\",r"\midrule"]
    for rate in (.0001,.001,.01,.1,1.):
        label=f"{rate*100:g}\\%" + (" (E1)" if rate==1 else "")
        values=[f"{indexed.loc[(ds,rate,m),'mean']:.{digits}f}" for ds in ("Higgs social","Orkut") for m,digits in zip(METRICS,(5,2,4))]
        lines.append(" & ".join([label,*values])+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular*}",r"\end{table}"])
    (out / "e4_sampling.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return data


def render_components(e2_dir, output_dir):
    """Recompute additional ablations on each variant's own matched Full subset."""
    from .render.e1_statistics import interval
    source=Path(e2_dir); output=Path(output_dir)
    raw=_read(source/'additional_ablation_releases.csv')
    full=_read(source/'releases.csv').query("suite == 'E2' and method == 'PrivTDD'")
    keys=['dataset','epsilon','seed','report_sha256','truth_sha256']
    paired=raw.merge(full[keys+list(METRICS)],on=keys,suffixes=('','_full'),validate='many_to_one')
    if len(paired)!=len(raw) or paired.duplicated(['dataset','epsilon','seed','method']).any():
        raise ValueError('Additional ablation lost its matched Full/report binding')
    summaries=[]
    for key,group in paired.sort_values('seed').groupby(['dataset','epsilon','method']):
        for metric in METRICS:
            xs,ys=group[metric].to_numpy(),group[metric+'_full'].to_numpy()
            stats=interval(xs-ys)
            summaries.append(dict(zip(('dataset','epsilon','method'),key),metric=metric,n=len(xs),
                variant_mean=xs.mean(),matched_full_mean=ys.mean(),delta_variant_minus_full=stats['mean'],
                delta_lower95=stats['lower95'],delta_upper95=stats['upper95']))
    result=pd.DataFrame(summaries)
    _compare_stats(result,_read(source/'additional/E2_component_pairs_all_available.csv'),
                   ['dataset','epsilon','method','metric'],['n','variant_mean','matched_full_mean','delta_variant_minus_full','delta_lower95','delta_upper95'])
    result.to_csv(output/'e2_component_pairs.csv',index=False)
    paired.to_csv(output/'e2_component_paired_releases.csv',index=False)
    return result


def render_all(root=None, output_dir=None):
    root = Path(root or repository_root()).resolve()
    output = Path(output_dir or root / "results/reproduced/paper-v1").resolve()
    if output.is_relative_to(root / "results/published") or (root / "results/published").is_relative_to(output):
        raise ValueError("Rendering output must be separate from immutable published evidence")
    report = verify_snapshot(root)
    snapshot = root / "results/published/paper-v1"
    from privtdd.evaluation.e3_diagnostics import verify_derived
    report['e3_prose_diagnostics'] = verify_derived(snapshot/'e3')
    output.mkdir(parents=True,exist_ok=True)
    for suite in ("e1","e2","e3","e4"):
        shutil.copytree(snapshot/suite,output/"data"/suite,dirs_exist_ok=True)
    tables, figures = output/"tables",output/"figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    from .render import e1_statistics,e2_statistics,e3_statistics,e1_tables,e2_plot,e3_plot
    e1_statistics.summarize(output/"data/e1")
    e2_statistics.main(output/"data/e2")
    e3_means,e3_pairs=e3_statistics.stats(_read(snapshot/"e3/releases.csv"))
    e3_means.to_csv(output/"data/e3/means.csv",index=False)
    e3_pairs.to_csv(output/"data/e3/paired_changes.csv",index=False)
    _compare_stats(_read(output/"data/e1/summary.csv"),_read(snapshot/"e1/summary.csv"),
                   ["dataset","epsilon","method","metric"],["mean","sd","lower95","upper95","n"])
    e2core=_read(output/"data/e2/means.csv").query("suite == 'E2'")
    _compare_stats(e2core,_read(snapshot/"e2/expected_core_summary.csv"),
                   ["dataset","epsilon","method","metric"],["mean","sd","lower95","upper95","n"])
    e2core.to_csv(output/"data/e2/core_summary.csv",index=False)
    _compare_stats(e3_means,_read(snapshot/"e3/means.csv"),
                   ["dataset","epsilon","quantiles","metric"],["mean","lower95","upper95","n"])
    e1_tables.main(["--data",str(output/"data/e1/summary.csv"),"--out-dir",str(tables)])
    # Preserve current published nomenclature without changing old evidence bytes.
    table=tables/"e1_wide.tex"
    text=table.read_text(encoding="utf-8").replace(r"\multicolumn{3}{c}{PrivTDD}",r"\multicolumn{3}{c}{PrivTDD (Ours)}")
    table.write_text(text,encoding="utf-8")
    e2_plot.main(["--data",str(output/"data/e2/core_summary.csv"),"--out-dir",str(figures),"--layout","column","--formats","pdf","png"])
    e3_plot.main(output/"data/e3",figures,"Higgs social")
    from .render import e3_boundary_effects, e3_localize_emd
    e3_statistics.main(output/"data/e3",tables)
    e3_boundary_effects.main(output/"data/e3",tables)
    e3_localize_emd.main(output/"data/e3",tables)
    e4=render_e4(_read(snapshot/"e1/releases.csv"),_read(snapshot/"e4/large40_runs.csv"),tables)
    component_summary=render_components(snapshot/'e2',tables)
    report.update(e1_table_values=384,e2_curve_means=len(e2core),e3_all_network_means=len(e3_means),e4_table_values=len(e4),
                  additional_component_summary_rows=len(component_summary),output=str(output),mode="statistics and figures only; no estimator execution",
                  diagnostic_inputs="data/e2/additional and data/e3/emd_by_region.csv retain their original available coverage")
    report["generated_files"]=[dict(path=p.relative_to(output).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for folder in (tables,figures) for p in sorted(folder.iterdir()) if p.is_file()]
    (output/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    verify_snapshot(root)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=None)
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args(argv)
    result=verify_snapshot(args.root or repository_root()) if args.check_only else render_all(args.root,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
