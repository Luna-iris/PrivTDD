"""Reproduce the E3 prose diagnostics on fixed, true-positive percentile ranges.

Ground truth is used ONLY for evaluation, never for estimator fitting.
This diagnostic was reconstructed during artifact preparation. It is not
represented as the historical script that originally generated the prose.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def truth_summary(path):
    truth = np.load(path, allow_pickle=False)
    if truth.ndim != 1 or len(truth) == 0 or np.any(truth < 0):
        raise ValueError('Expected nonempty nonnegative per-node triangle degrees')
    positive = truth[truth > 0]
    if not len(positive):
        raise ValueError('Positive triangle degrees are required for this diagnostic')
    quantiles = np.quantile(positive, [.9, .99, .999], method='linear')
    # EMD is a sum over integer cutoffs t. Above a real quantile q means
    # t >= floor(q)+1; the three resulting integer ranges partition the sum.
    starts = np.floor(quantiles).astype(np.int64) + 1
    keys, count = np.unique(truth, return_counts=True)
    top_count = int(np.ceil(.01 * len(truth)))
    summary = dict(nodes=len(truth), positive_nodes=len(positive),
                   top_one_percent_nodes=top_count,
                   top_one_percent_triangle_degree_share=float(np.sort(truth)[-top_count:].sum(dtype=np.float64)/truth.sum(dtype=np.float64)),
                   quantile_method='linear', quantile_levels=[.9,.99,.999],
                   quantile_values=quantiles.tolist(), integer_range_starts=starts.tolist(),
                   truth_sha256=sha256(path))
    return (keys, count / len(truth)), starts, summary


def region_errors(truth_histogram, starts, prediction):
    keys, mass = truth_histogram
    with np.load(prediction, allow_pickle=False) as saved:
        support, counts = saved['keys'], saved['counts']
    if counts.sum() <= 0 or np.any(counts < 0):
        raise ValueError('Invalid predicted histogram')
    predicted = counts / counts.sum()
    grid = np.unique(np.r_[0, keys, support, starts])
    gap = np.zeros(len(grid))
    np.add.at(gap, np.searchsorted(grid, keys), mass)
    np.add.at(gap, np.searchsorted(grid, support), -predicted)
    gap = np.cumsum(gap)  # true CDF minus estimated CDF
    widths = np.diff(grid)
    regions = np.searchsorted(starts, grid[:-1], side='right')
    absolute = np.bincount(regions, weights=np.abs(gap[:-1])*widths, minlength=4)
    signed = np.bincount(regions, weights=gap[:-1]*widths, minlength=4)
    names = ('at_or_below90', 'above90_through99', 'above99_through999', 'above999')
    result = dict(zip(names, map(float, absolute)))
    result.update({f'{k}_signed':float(v) for k,v in zip(names,signed)})
    result['EMD'] = float(absolute.sum())
    return result


def aggregate(frame, summaries):
    """Recompute prose statements from saved per-run derived inputs."""
    results = {}
    for dataset, expected_n in [('higgs',8),('orkut',10)]:
        part = frame[frame.dataset.eq(dataset)]
        pivot = part.pivot(index='seed', columns='method', values=['above90_through99','above999','EMD'])
        if len(pivot) != expected_n or pivot.isna().any().any():
            raise ValueError(f'Incomplete E3 prose pairs for {dataset}')
        deltas = {m:pivot[m]['q95_99']-pivot[m]['q80_90'] for m in ['above90_through99','above999','EMD']}
        results[dataset] = dict(matched_runs=len(pivot),
            top_one_percent_share=summaries[dataset]['top_one_percent_triangle_degree_share'],
            changes_upper_minus_lower={m:float(d.mean()) for m,d in deltas.items()},
            tail_decreases_every_run=bool((deltas['above999']<0).all()))
    expected = {'higgs':{'share':47.6,'tail':-37.06,'total':-24.70},
                'orkut':{'share':32.0,'middle':43.85,'total':45.04}}
    for ds,v in expected.items():
        result=results[ds]
        if round(result['top_one_percent_share']*100,1)!=v['share']:
            raise ValueError('Top-one-percent share differs from the paper')
        delta=result['changes_upper_minus_lower']
        for name,key in [('tail','above999'),('middle','above90_through99'),('total','EMD')]:
            if name in v and round(delta[key],2)!=v[name]:
                raise ValueError(f'E3 prose differs for {ds}/{name}')
    return results


def from_manifest(manifest_path, root, output):
    """Read a portable manifest and recompute diagnostics from prediction arrays."""
    root, output = Path(root).resolve(), Path(output)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    rows, summaries = [], {}
    for dataset, spec in manifest['datasets'].items():
        truth_path = root / spec['truth']
        histogram, starts, summary = truth_summary(truth_path)
        if summary['truth_sha256'] != spec['truth_sha256']:
            raise ValueError(f'Truth checksum mismatch for {dataset}')
        summaries[dataset] = summary
        for item in spec['predictions']:
            pred = root / item['path']
            if sha256(pred) != item['sha256']:
                raise ValueError(f'Prediction checksum mismatch: {pred}')
            values = region_errors(histogram,starts,pred)
            if not np.isclose(values['EMD'],item['expected_EMD'],rtol=1e-10,atol=1e-10):
                raise ValueError(f'Prediction EMD differs: {pred}')
            rows.append(dict(dataset=dataset,epsilon=.5,seed=item['seed'],method=item['method'],
                             report_sha256=item['report_sha256'],prediction_sha256=item['sha256'],
                             truth_sha256=summary['truth_sha256'],**values))
    frame = pd.DataFrame(rows).sort_values(['dataset','seed','method'])
    result = aggregate(frame,summaries)
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'prose_regions.csv',index=False)
    (output/'prose_truth_summary.json').write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf-8')
    (output/'prose_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def verify_derived(snapshot):
    snapshot = Path(snapshot)
    return aggregate(pd.read_csv(snapshot/'prose_regions.csv',float_precision='round_trip'),
                     json.loads((snapshot/'prose_truth_summary.json').read_text(encoding='utf-8')))
