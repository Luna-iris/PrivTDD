"""PrivTDD experiment entry point: a single case or dataset x budget x seed grid; evaluate only after prediction."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import numpy as np
from privtdd.release.pipeline import estimate_tdd_from_reports
from privtdd.config import PrivTDDConfig
from privtdd.io.storage import json_write, atomic_text_write
from privtdd.io.memory import atomic_npz
from privtdd.reporting.collection import prepare_graph, collect_reports_joint


def file_digest(path):
    """[Execution identity] Streaming SHA256 for input/code/cache identity only; not part of estimation."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def output_lock(directory, name):
    """[Concurrency] Exclusive output directory; ordinary exceptions release the lock. After forced interruption, verify the process has stopped before removing the lock."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/name
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError as error:
        raise RuntimeError(f'Output is locked: {lock}; check the owning process first') from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def build_cases(args, parser):
    """[Experiment orchestration] Expand independent cases; datasets default to public D=n-1, or an explicitly declared bound.
    
    args: Parsed command line; parser: reports configuration errors. Return (cases, batch).
    The dataset cache must contain indptr.npy, indices.npy and graph.json; do not estimate the cap from the original graph.
    Record only the truth path and its existence; read its content after prediction is complete.
    """
    if args.epsilon is not None and args.epsilons is not None:
        parser.error('Use --epsilon or --epsilons, not both')
    if args.seed is not None and args.seeds is not None:
        parser.error('Use --seed or --seeds, not both')
    budgets = args.epsilons if args.epsilons is not None else [args.epsilon]
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 20260908]
    if any(e is None or not math.isfinite(e) or e <= 0 for e in budgets):
        parser.error('Positive finite --epsilon or --epsilons is required')
    if any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds) or len(set(budgets)) != len(budgets):
        parser.error('Seeds must be nonnegative; duplicate seeds/budgets are not separate trials')
    if args.reports and (args.seeds is not None or args.epsilons is not None):
        parser.error('A --reports file is one transcript: use singular --epsilon; --seeds cannot redraw it')
    if args.degree_prior and not args.reports:
        parser.error('--degree-prior is only for one matching --reports transcript')
    if args.datasets and (args.public_cap is not None or args.truth):
        parser.error('--datasets uses --public-cap-policy and cached triangles.npy; omit --public-cap/--truth')
    if args.no_evaluate and args.truth:
        parser.error('--no-evaluate conflicts with --truth')
    batch = args.datasets is not None or args.epsilons is not None or args.seeds is not None
    sources = []
    if args.datasets:
        if len(set(args.datasets)) != len(args.datasets):
            parser.error('Duplicate datasets are not separate trials')
        for dataset in args.datasets:
            if not dataset or dataset in ('.', '..') or any(c in dataset for c in '/\\:'):
                parser.error('Dataset names must be simple cache-directory names')
            folder = args.cache_root/dataset
            metadata = json.loads((folder/'graph.json').read_text(encoding='utf-8'))
            cap = (metadata['nodes']-1 if getattr(args, 'public_cap_policy', 'n-minus-one') == 'n-minus-one'
                   else metadata['public_degree_cap'])
            if isinstance(cap, bool) or not isinstance(cap, int) or not 2 <= cap < metadata['nodes']:
                parser.error(f'Invalid declared cap in {folder}/graph.json')
            sources.append((dataset, 'graph', folder, cap, None if args.no_evaluate else folder/'triangles.npy'))
    else:
        if args.public_cap is None or args.public_cap < 2:
            parser.error('--graph/--reports requires --public-cap >= 2')
        source = args.graph or args.reports
        sources.append((source.stem, 'graph' if args.graph else 'reports', source, args.public_cap, args.truth))
    cases = []
    for dataset, mode, source, cap, truth in sources:
        files = [source/'indptr.npy', source/'indices.npy'] if source.is_dir() and mode == 'graph' else [source]
        for path in files + ([truth] if truth else []) + ([args.degree_prior] if args.degree_prior else []):
            if not path.is_file():
                parser.error(f'Missing input: {path}')
        for epsilon in budgets:
            for seed in seeds:
                # repr preserves float identity; g with default precision could collide.
                label = str(int(epsilon)) if epsilon.is_integer() else repr(epsilon)
                key = f'{dataset}_eps{label}_seed{seed}' if mode == 'graph' else f'{dataset}_eps{label}_reports'
                cases.append(dict(key=key, dataset=dataset, mode=mode, source=str(source.resolve()),
                    input_files=[str(p.resolve()) for p in files], public_cap=cap, epsilon=epsilon,
                    seed=seed if mode == 'graph' else None, truth=str(truth.resolve()) if truth else None,
                    degree_prior=str(args.degree_prior.resolve()) if args.degree_prior else None,
                    report_cap_offset=getattr(args, 'report_cap_offset', None),
                    output=str((args.output/key if batch else args.output).resolve())))
    return cases, batch


def case_identity(case, config, digests, code_digest):
    """[Execution identity] Bind input content, budget, seed, numerical/model configuration and code; do not merge experiments.
    
    digests: File hashes reused by path within this command; code_digest: digest of the local method and shared distribution implementation.
    Only bind the truth path without opening it before prediction; truth is not part of model identity. Threads and blocking do not affect statistical inputs.
    """
    inputs = {}
    for path in case['input_files'] + ([case['degree_prior']] if case['degree_prior'] else []):
        if path not in digests:
            digests[path] = file_digest(path)
        inputs[path] = digests[path]
    settings = asdict(config)
    for name in ('workers', 'blas_threads', 'report_block_size'):
        settings.pop(name)
    binding = dict(mode=case['mode'], inputs=inputs, public_cap=case['public_cap'], epsilon=case['epsilon'],
                   seed=case['seed'], settings=settings, code_digest=code_digest,
                   report_cap_offset=case.get('report_cap_offset'))
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()


def run_case(case, config, identity):
    """[Experiment execution] One report set -> S2-S6 -> S7 safeguard -> save prediction -> optional evaluation; reuse completed cases.
    
    case: Single-case definition from build_cases; config: common PrivTDDConfig; identity: input/code fingerprint for this case.
    Graph mode caches this case's reports.npz so resuming does not traverse the original graph again; do not reuse reports across seeds or budgets.
    """
    out = Path(case['output'])
    started = time.perf_counter()
    with output_lock(out, 'CASE.lock'):
        manifest = out/'case_manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8'))['identity'] != identity:
                raise ValueError(f'Different inputs/code/settings at {out}; use a new output directory')
        else:
            if any(p.name != 'CASE.lock' for p in out.iterdir()):
                raise ValueError(f'Nonempty output without case identity: {out}')
            json_write(manifest, dict(identity=identity, case=case, config=asdict(config)))
        complete = out/'completion.json'
        prediction = out/'prediction.npz'
        if complete.exists():
            saved = json.loads(complete.read_text(encoding='utf-8'))
            if saved['prediction_sha256'] != file_digest(prediction):
                raise ValueError(f'Completed prediction changed: {prediction}')
            if not (out/'run.json').is_file():
                raise ValueError(f'Completed run metadata missing: {out}')
            reused = True
        else:
            released = out/'reports.npz' if case['mode'] == 'graph' else Path(case['source'])
            collection_performed = not released.exists()
            collection_started = time.perf_counter()
            if not released.exists():
                source = Path(case['source'])
                mapped = []
                try:
                    if source.is_dir():
                        for name in ('indptr.npy', 'indices.npy'):
                            mapped.append(np.load(source/name, mmap_mode='r'))
                        ip, ix = mapped
                    else:
                        with np.load(source) as z:
                            ip, ix = z['indptr'], z['indices']
                    ip, ix = prepare_graph((ip, ix), len(ip)-1, case['public_cap'])
                    reports = collect_reports_joint(ip, ix, case['public_cap'], case['epsilon'], case['seed'],
                                                    block_size=config.report_block_size, workers=config.workers)
                    atomic_npz(released, **reports)
                finally:
                    # Explicitly release mappings owned by this function, so references retained by exceptions/test callbacks do not keep graph files locked.
                    for array in mapped:
                        array._mmap.close()
                del ip, ix, reports
            collection_seconds = time.perf_counter()-collection_started if collection_performed else None
            posterior_started = time.perf_counter()
            with np.load(released) as z:
                caps, noisy = z['caps'], z['noisy_triangle_report']
                report_offset = z['cap_offset'] if 'cap_offset' in z.files else case.get('report_cap_offset')
                if report_offset is None:
                    raise ValueError('Report file lacks cap_offset. Use --report-cap-offset 0 only for known L=0 reports; historical L=150 reports must be regenerated')
                if case.get('report_cap_offset') is not None and int(report_offset) != case['report_cap_offset']:
                    raise ValueError('Declared report cap_offset conflicts with report metadata')
            prior = None
            if case['degree_prior']:
                with np.load(case['degree_prior']) as z:
                    prior = z['prior']
            histogram, info = estimate_tdd_from_reports(caps, noisy, case['public_cap'], case['epsilon'],
                config=config, workdir=out/'work', degree_prior=prior, return_debug=True,
                report_cap_offset=report_offset)
            keys = np.array(sorted(histogram), np.int64)
            atomic_npz(prediction, keys=keys, counts=np.array([histogram[int(k)] for k in keys]))
            json_write(out/'run.json', dict(input_mode=case['mode'], input_path=case['source'],
                collection_seed=case['seed'], identity=identity, method=info,
                timing=dict(collection_performed=collection_performed, collection_seconds=collection_seconds,
                            postprocessing_seconds=time.perf_counter()-posterior_started)))
            reused = False
        # Open truth only after the prediction file exists; changing truth triggers reevaluation, not refitting.
        metrics = {}
        evaluation = None
        evaluation_started = time.perf_counter()
        if case['truth']:
            evaluation = dict(path=case['truth'], sha256=file_digest(case['truth']))
            if reused and saved.get('evaluation') == evaluation and (out/'metrics.json').is_file():
                metrics = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
            else:
                metrics = evaluate_saved_prediction(prediction, case['truth'], case['public_cap'])
                json_write(out/'metrics.json', metrics)
        evaluation_seconds = time.perf_counter()-evaluation_started
        seconds = time.perf_counter()-started
        # [Timing] With reused reports, collection time is None; do not label postprocessing time as original-graph end-to-end time.
        timing = json.loads((out/'run.json').read_text(encoding='utf-8')).get('timing', {})
        if not reused or saved.get('evaluation') != evaluation:
            json_write(complete, dict(identity=identity, prediction_sha256=file_digest(prediction),
                                     evaluation=evaluation, complete=True, last_execution_seconds=seconds,
                                     timing=dict(timing, evaluation_seconds=evaluation_seconds,
                                                 scope='fresh_graph_to_metrics' if timing.get('collection_performed') and not reused else 'reused_reports_or_prediction')))
        return dict(dataset=case['dataset'], epsilon=case['epsilon'], seed=case['seed'], status='complete',
                    reused=reused, invocation_seconds=seconds, output=case['output'],
                    collection_seconds=timing.get('collection_seconds'),
                    postprocessing_seconds=timing.get('postprocessing_seconds'), **metrics)


def write_batch_summary(directory, cases, rows):
    """[Experiment summary] Atomically write per-case CSV, dataset/budget means and sample standard deviations, and progress JSON.
    
    rows contains only states of cases requested by this command; count failures separately and leave sample standard deviation blank for n=1.
    Means aggregate independent output metrics, without merging reports or posteriors; resumed elapsed time is not time from a fresh start.
    """
    records = list(rows.values())
    fields = ['dataset', 'epsilon', 'seed', 'status', 'reused', 'MSE', 'EMD', 'KS', 'invocation_seconds', 'collection_seconds', 'postprocessing_seconds', 'output', 'error']
    def write_csv(name, records, fields):
        """[Output] CSV reconstructible from per-case results; warn if the file is busy without interrupting estimation."""
        path = directory/name
        with io.StringIO(newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
            return atomic_text_write(path, stream.getvalue(), encoding='utf-8-sig', required=False)
    results_written = write_csv('results.csv', records, fields)
    summary = []
    for dataset, epsilon in sorted({(c['dataset'], c['epsilon']) for c in cases}):
        group = [r for r in records if r['dataset'] == dataset and r['epsilon'] == epsilon]
        done = [r for r in group if r['status'] == 'complete']
        row = dict(dataset=dataset, epsilon=epsilon,
            requested=sum(c['dataset'] == dataset and c['epsilon'] == epsilon for c in cases),
            completed=len(done), evaluated=sum('MSE' in r for r in done), failed=sum(r['status'] == 'failed' for r in group))
        for metric in ('MSE', 'EMD', 'KS'):
            values = [r[metric] for r in done if metric in r]
            row[metric+'_mean'] = float(np.mean(values)) if values else None
            row[metric+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    summary_written = write_csv('summary.csv', summary, ['dataset', 'epsilon', 'requested', 'completed', 'evaluated', 'failed']+
              [m+s for m in ('MSE', 'EMD', 'KS') for s in ('_mean', '_std')])
    progress_written = json_write(directory/'batch_progress.json', dict(required=len(cases), completed=sum(r['status']=='complete' for r in records),
        failed=sum(r['status']=='failed' for r in records), cases=rows))
    return results_written and summary_written and progress_written


# [PrivTDD] experiment CLI / evaluation after release; experiment interface/evaluation (not part of the estimator); [Experiment evaluation, outside the method] Compute MSE/EMD/KS from predictions already written to disk.
def evaluate_saved_prediction(prediction, truth_path, public_cap):
    """[Experiment evaluation, outside the method] Compute MSE/EMD/KS from predictions already written to disk.
    
    prediction: NPZ with keys/counts; truth_path: NPY with true per-node triangle counts; public_cap: public D.
    MSE is squared node-count error over the full public T domain; EMD is one-dimensional W1 between normalized distributions.
    """
    with np.load(prediction) as z:
        keys, counts = z['keys'], z['counts']
    truth = np.load(truth_path)
    if truth.ndim != 1 or truth.dtype.kind not in 'iu' or np.any(truth < 0):
        raise ValueError('Truth must be one integer triangle count per node')
    if abs(counts.sum()-len(truth)) > 1e-6 or np.any(truth > public_cap*(public_cap-1)//2):
        raise ValueError('Truth population/domain mismatch')
    tk, tc = np.unique(truth, return_counts=True)
    union = np.union1d(tk, keys)
    a, b = np.zeros(len(union)), np.zeros(len(union))
    a[np.searchsorted(union, tk)] = tc
    b[np.searchsorted(union, keys)] = counts
    delta = np.cumsum(a/a.sum()-b/b.sum())
    return dict(MSE=float(np.sum((a-b)**2)/(public_cap*(public_cap-1)//2+1)),
                EMD=float(np.abs(delta[:-1])@np.diff(union)), KS=float(np.max(abs(delta))))


# [PrivTDD] experiment CLI / evaluation after release; experiment interface/evaluation (not part of the estimator); [Experiment entry point] argv is a command-line argument list or None; --help lists input and numerical settings.
def main(argv=None):
    """[Experiment entry point] argv is a command-line argument list or None; --help lists input and numerical settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--graph', type=Path, help='CSR NPZ or directory with indptr.npy/indices.npy; starts from graph')
    source.add_argument('--reports', type=Path, help='Private NPZ: caps + noisy_triangle_report; skips collection')
    source.add_argument('--datasets', nargs='+', help='Cache directory names, e.g. higgs orkut dblp gplus')
    parser.add_argument('--cache-root', type=Path, default=Path('datasets/processed'))
    parser.add_argument('--public-cap', type=int, help='Required for graph/reports; datasets use --public-cap-policy (default n-1)')
    parser.add_argument('--public-cap-policy', choices=('n-minus-one', 'declared'), default='n-minus-one',
                        help='Datasets: n-minus-one (default) matches the safeguard experiment; declared uses graph.json public_degree_cap')
    parser.add_argument('--report-cap-offset', type=int, choices=(0,),
                        help='Explicit L=0 declaration for an untagged reports NPZ; cannot convert L=150 reports')
    parser.add_argument('--epsilon', type=float, help='Budget of one transcript')
    parser.add_argument('--epsilons', nargs='+', type=float, help='Independent budgets for graph/dataset grid')
    parser.add_argument('--seed', type=int, help='One collection seed; default 20260908; reports mode does not redraw')
    parser.add_argument('--seeds', nargs='+', type=int, help='Independent collection seeds for graph/dataset grid')
    parser.add_argument('--continue-on-error', action='store_true', help='Record failed cases and continue; no automatic retries')
    parser.add_argument('--no-evaluate', action='store_true', help='Datasets: do not open triangles.npy or compute metrics')
    parser.add_argument('--dry-run', action='store_true', help='Print cases and paths only; do not collect, fit or create outputs')
    parser.add_argument('--degree-prior', type=Path, help='Optional same-transcript degree NPZ (prior); reports mode only')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; validated cache in output/work')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--blas-threads', type=int, default=2)
    parser.add_argument('--posterior-backend', choices=('integrated', 'legacy'), default='integrated')
    parser.add_argument('--geometry-draws', type=int, default=65536, help='Numerical draws per replicate, not private observations')
    parser.add_argument('--geometry-replicates', type=int, default=4)
    parser.add_argument('--numerical-w1-limit', type=float, default=1e-6)
    parser.add_argument('--compression-w1-budget', type=float, default=2.)
    parser.add_argument('--report-block-size', type=int, default=256)
    parser.add_argument('--acceleration', choices=('exact', 'binned', 'fast'), default='exact',
                        help='binned: degree/Q bins; fast: also noise-scaled T bins, lazy endpoints and batched small groups')
    parser.add_argument('--degree-grid-size', type=int, default=None,
                        help='0: original integer-degree fit; >=68: restricted grid. Overrides preset')
    parser.add_argument('--report-bin-width', type=float, default=None,
                        help='Q bin width / Laplace scale; 0 disables. Overrides preset; same cap only')
    parser.add_argument('--degree-grid-tolerance', type=float, default=1e-5,
                        help='Restricted-grid per-cap objective gap; exact integer path retains 1e-7')
    parser.add_argument('--triangle-bin-width', type=float, default=None,
                        help='T bin width mapped to Q / minimum eligible noise scale; fast preset 1/16; 0: old W1 compression')
    parser.add_argument('--structural-lazy-endpoints', action=argparse.BooleanOptionalAction, default=None,
                        help='Skip unused hard endpoint generation without changing retained random streams; fast preset on')
    parser.add_argument('--batch-small-groups', action=argparse.BooleanOptionalAction, default=None,
                        help='Batch the same channel across small cap groups; fast preset on')
    parser.add_argument('--batch-f1', action=argparse.BooleanOptionalAction, default=None,
                        help='Batch F1 channels/integrals without changing scales or components; fast preset on')
    parser.add_argument('--compiled-structure', action=argparse.BooleanOptionalAction, default=None,
                        help='Compile original structural draw loops and reuse fixed root kernels; fast preset on')
    parser.add_argument('--safeguard', action=argparse.BooleanOptionalAction, default=True,
                        help='Final capacity-feasible W1 safeguard, enabled for every acceleration preset')
    parser.add_argument('--safeguard-tau', type=float, default=1., help='Public final W1 radius multiplier; default 1')
    parser.add_argument('--truth', type=Path, help='Optional evaluation-only triangles.npy; opened only after prediction is saved')
    args = parser.parse_args(argv)
    cases, batch = build_cases(args, parser)
    config = PrivTDDConfig(workers=args.workers, blas_threads=args.blas_threads, posterior_backend=args.posterior_backend,
                         geometry_draws=args.geometry_draws, geometry_replicates=args.geometry_replicates,
                         numerical_W1_limit=args.numerical_w1_limit, compression_W1_budget=args.compression_w1_budget,
                         report_block_size=args.report_block_size,
                         degree_grid_size=args.degree_grid_size if args.degree_grid_size is not None else (512 if args.acceleration != 'exact' else 0),
                         report_bin_width=args.report_bin_width if args.report_bin_width is not None else (1/16 if args.acceleration != 'exact' else 0.),
                         degree_grid_tolerance=args.degree_grid_tolerance,
                         triangle_bin_width=args.triangle_bin_width if args.triangle_bin_width is not None else (1/16 if args.acceleration == 'fast' else 0.),
                         structural_lazy_endpoints=args.structural_lazy_endpoints if args.structural_lazy_endpoints is not None else args.acceleration == 'fast',
                         batch_small_groups=args.batch_small_groups if args.batch_small_groups is not None else args.acceleration == 'fast',
                         batch_f1=args.batch_f1 if args.batch_f1 is not None else args.acceleration == 'fast',
                         compiled_structure=args.compiled_structure if args.compiled_structure is not None else args.acceleration == 'fast',
                         safeguard=args.safeguard, safeguard_tau=args.safeguard_tau)
    if args.dry_run:
        print(json.dumps(dict(cases=len(cases), batch=batch, config=asdict(config), plan=cases), ensure_ascii=False, indent=2))
        return
    from privtdd.io.identity import source_digest
    code_digest = source_digest()
    digests = {}
    if not batch:
        result = run_case(cases[0], config, case_identity(cases[0], config, digests, code_digest))
        print(result)
        return
    with output_lock(args.output, 'BATCH.lock'):
        # Check identities for the entire batch first; incompatible configuration must not overwrite the old batch's summaries or progress.
        identities = {}
        for case in cases:
            identity = case_identity(case, config, digests, code_digest)
            manifest = Path(case['output'])/'case_manifest.json'
            if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['identity'] != identity:
                raise ValueError(f"Different inputs/code/settings at {case['output']}; use a new output directory")
            identities[case['key']] = identity
        rows = {c['key']:dict(dataset=c['dataset'], epsilon=c['epsilon'], seed=c['seed'], status='pending', output=c['output']) for c in cases}
        json_write(args.output/'batch_plan.json', dict(cases=cases, config=asdict(config), code_digest=code_digest))
        write_batch_summary(args.output, cases, rows)
        for index, case in enumerate(cases, 1):
            key = case['key']
            print(f'[{index}/{len(cases)}] {key}', flush=True)
            rows[key]['status'] = 'running'
            write_batch_summary(args.output, cases, rows)
            try:
                rows[key] = run_case(case, config, identities[key])
            except Exception as error:
                rows[key].update(status='failed', error=f'{type(error).__name__}: {error}')
                write_batch_summary(args.output, cases, rows)
                if not args.continue_on_error:
                    raise
            except KeyboardInterrupt:
                rows[key]['status'] = 'interrupted'
                write_batch_summary(args.output, cases, rows)
                raise
            summaries_written = write_batch_summary(args.output, cases, rows)
        if any(r['status']=='failed' for r in rows.values()):
            raise SystemExit(1)
        if summaries_written:
            print(f'Completed {len(cases)} independent cases: {args.output}/summary.csv')
        else:
            print(f'Completed {len(cases)} independent cases; summary files may be stale. '
                  'Close applications holding output files and rerun the same command to rebuild summaries; '
                  'completed predictions will be reused.')


if __name__ == '__main__':
    main()
