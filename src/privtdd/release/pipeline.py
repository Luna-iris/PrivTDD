"""PrivTDD: base distribution, structural references, correction, and release."""
from dataclasses import asdict
import gc
import json
import math
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from threadpoolctl import threadpool_limits
from privtdd.config import (PrivTDDConfig, DEGREE_FRACTION, RR_FRACTION, RELEASE_FRACTION,
    DEGREE_TOLERANCE, POSITIVE_QUANTILES, JOINT_COORDINATES, JOINT_TOLERANCE, JOINT_MAX_ITERATIONS)
from privtdd.reporting.collection import prepare_graph, collect_reports_joint
from privtdd.degree.fit import fit_cold
from privtdd.degree.canonical import canonicalize_tail
from privtdd.base_distribution.construction import build_f1_joint
from privtdd.references.construction import build_structural_joint
from privtdd.correction.forward import build_family
from privtdd.correction.optimizer import fit_joint
from privtdd.posterior.update import update_posterior_joint
from privtdd.io.memory import atomic_npz
from privtdd.io.storage import json_write, cache_identity, initialize_cache
import privtdd.config as method_config
from privtdd.release.transport_guard import capacity_reference, safeguard, capacity_violation


# [PrivTDD] S1-S7 complete entry point; core method or probability/structure primitive (not necessarily a separate contribution); [Core S5] Constrained report likelihood in seven shared coordinates plus the identity quadratic penalty on the original 16 coordinates.
def fit_joint7(family, *, directory=None):
    """[Core S5] Constrained report likelihood in seven shared coordinates plus the identity quadratic penalty on the original 16 coordinates.
    
    family: Forward tables and probability feasible domain returned by build_family. Return original 16-dimensional coordinates, node densities and solver statistics.
    directory: Optional working directory whose input/code identity is already bound by the caller; save or resume a warm start for the same
    objective. None solves in memory only. This directory must not point to historical outputs from other experiments.
    Map only the specified seven columns to the optimizer; the linear structural KL cost is zero and no truth metrics participate.
    """
    mapping = np.eye(16)[:, JOINT_COORDINATES]
    transformed = dict(family, delta_tables=mapping.T@family['delta_tables'], A=family['A']@mapping)
    warm_path = None if directory is None else Path(directory)/'joint_warm_start.json'
    progress_path = None if directory is None else Path(directory)/'joint_recovery_progress.json'
    # [PrivTDD] S5; implementation cache; Save the seven-dimensional recovery initial point and original objective from this same solve; do not save the original graph or accept unconverged results.
    def save_warm(point, objective):
        """[S5 implementation] Save only the seven-dimensional point about to enter recovery and its value for the same objective."""
        if warm_path is not None:
            json_write(warm_path, dict(point=point.tolist(), objective=objective))
    if warm_path is not None and warm_path.exists():
        from privtdd.correction.roundoff import refine_joint
        warm = json.loads(warm_path.read_text(encoding='utf-8'))
        z, stats = refine_joint(transformed, initial=warm['point'], quadratic=mapping.T@mapping,
                                max_iterations=80, tolerance=JOINT_TOLERANCE,
                                expected_initial_objective=warm['objective'], progress_path=progress_path)
        if stats['objective'] > 1e-10:
            raise ArithmeticError('Recovered joint objective is worse than the feasible F1 anchor')
        stats['recovery_trigger'] = 'same_workdir_warm_start'
    else:
        z, stats = fit_joint(transformed, range(7), max_iterations=JOINT_MAX_ITERATIONS,
                             tolerance=JOINT_TOLERANCE, quadratic=mapping.T@mapping,
                             recovery_checkpoint=save_warm, recovery_progress_path=progress_path)
    coordinates = mapping@z
    # Retain the original matrix shapes and multiplication order to reduce unnecessary floating-point differences.
    normalizer = family['predictive_table'][:, None]+family['delta_tables'].T@coordinates[None, :].T
    if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
        raise ArithmeticError('Positive finite candidate likelihood required')
    return coordinates, normalizer, stats


# [PrivTDD] S1-S7 complete entry point; S2-S7 orchestration; Run stages sequentially in directory, whose identity is bound and whose exclusive ownership is managed by the caller.
def _run_joint(caps, noisy, public_cap, epsilon, config, directory, degree_prior):
    """[S2-S7 orchestration] Run sequentially in directory, whose identity is bound and whose exclusive ownership is managed by the caller.
    
    caps/noisy/public_cap/epsilon: Current experiment; config: PrivTDDConfig; degree_prior: optional degree estimate from the same reports.
    Execute each stage once or reuse a matching cache; there is no independent validation, baseline evaluation or truth input.
    """
    ed, er, el = DEGREE_FRACTION*epsilon, RR_FRACTION*epsilon, RELEASE_FRACTION*epsilon
    gap = 2/(1+math.exp(-er))-1
    p = (1+gap)/2
    report, scale = gap*noisy, np.maximum(caps-1, 0)/el
    # All approximation parameters belong to this cache identity; the full original caps remain in use for S2/S4.
    report_caps, report_counts = caps, None
    approximation = dict(degree_grid_size=config.degree_grid_size, report_bin_width=config.report_bin_width,
                         triangle_bin_width=config.triangle_bin_width,
                         degree_grid_tolerance=config.degree_grid_tolerance,
                         empirical_baseline_acceptance='not_evaluated_by_estimator')
    if config.report_bin_width > 0:
        from privtdd.numerics.quantization import bin_reports
        report_caps, report, report_counts, approximation['reports'] = bin_reports(caps, report, el, config.report_bin_width)
        scale = np.maximum(report_caps-1, 0)/el
    timings = {}
    current_stage, stage_started = None, time.perf_counter()
    # [PrivTDD] S1-S7 complete entry point; memory and parallel execution; [Progress] name identifies the current stage; only record progress without starting additional computation.
    def stage(name):
        """[Progress] name identifies the current stage; only record progress without starting additional computation."""
        nonlocal current_stage, stage_started
        now = time.perf_counter()
        if current_stage is not None:
            timings[current_stage] = now-stage_started
        current_stage, stage_started = name, now
        json_write(directory/'progress.json', dict(stage=name, completed_stage_seconds=timings))

    def degree_checkpoint(iteration, gap, nll):
        """[S2 progress] iteration/gap/nll only record the current solve; no independent verification."""
        json_write(directory/'progress.json', dict(stage='degree_fit', iteration=int(iteration),
            gap=float(gap), nll_per_cap=float(nll), stage_seconds=time.perf_counter()-stage_started))

    def full_degree_checkpoint(point, iteration, gap, nll):
        """[S2 callback adapter] point is the current prior from the original solver callback; only forward progress."""
        degree_checkpoint(iteration, gap, nll)
    stage('degree_fit')
    if (directory/'degree.npz').exists():
        with np.load(directory/'degree.npz') as z:
            prior = z['prior']
    else:
        if degree_prior is None:
            if config.degree_grid_size:
                from privtdd.numerics.quantization import fit_degree_grid
                prior, stats = fit_degree_grid(caps, public_cap, ed, config.degree_grid_size,
                    tolerance=config.degree_grid_tolerance, checkpoint=degree_checkpoint)
            else:
                prior, stats = fit_cold(caps, public_cap, ed, tolerance=DEGREE_TOLERANCE, checkpoint=full_degree_checkpoint)
        else:
            prior, stats = degree_prior.copy(), dict(provided_degree_prior=True)
        prior, tie = canonicalize_tail(prior, ed)
        json_write(directory/'degree_fit.json', dict(fit=stats, canonicalization=tie))
        atomic_npz(directory/'degree.npz', prior=prior)
    stage('F1')
    f1, f1_stats = build_f1_joint(report_caps, report, scale, prior, p, ed, el, directory, config,
                                 report_counts=report_counts)
    positive = f1.global_keys > 0
    if positive.any():
        cdf = np.cumsum(f1.global_counts[positive])
        cdf /= cdf[-1]
        boundaries = tuple(int(f1.global_keys[positive][np.searchsorted(cdf, q)]) for q in POSITIVE_QUANTILES)
    else:
        boundaries = (0, 0)
    stage('structural_sources')
    (beta, geometry), structural_stats = build_structural_joint(caps, prior, ed, float(noisy.mean()), directory, config)
    sources = (f1, beta, geometry)
    stage('correction_fit')
    if (directory/'joint_context.npz').exists():
        with np.load(directory/'joint_context.npz') as z:
            coordinates, normalizer, gate = z['coordinates'], z['normalizer'], z['gate_basis']
        joint_stats = json.loads((directory/'joint_fit.json').read_text(encoding='utf-8'))
    else:
        # [S5 implementation] Persist the minimal forward tables needed for fitting in the working directory with the same identity; recovery after interruption
        # does not regenerate expensive channels. Keep sources separately in memory; S6 reconstructs conditional rows from the same sources.
        channel_path = directory/'joint_channel.npz'
        if channel_path.exists():
            with np.load(channel_path) as z:
                family = {name: z[name] for name in z.files}
        else:
            family = build_family(sources, report_caps, report, prior, p, ed, el, boundaries, workers=config.workers,
                                  batch_small_groups=config.batch_small_groups)
            if report_counts is not None:
                family['counts'] = report_counts.astype(float)
                family['n'] = len(caps)
            fields = ('counts', 'predictive_table', 'delta_tables', 'A', 'b', 'gate_basis')
            atomic_npz(channel_path, **{name: family[name] for name in fields})
        coordinates, normalizer, joint_stats = fit_joint7(family, directory=directory)
        gate = family['gate_basis']
        json_write(directory/'joint_fit.json', joint_stats)
        atomic_npz(directory/'joint_context.npz', coordinates=coordinates, normalizer=normalizer, gate_basis=gate)
        del family
        gc.collect()
    stage('posterior')
    family = dict(sources=sources, support=f1.support, pi=f1.pi, boundaries=boundaries, gate_basis=gate)
    hist = update_posterior_joint(family, coordinates, normalizer, report_caps, report, prior, p, ed, el,
                                 workers=config.workers, report_counts=report_counts, batch_small_groups=config.batch_small_groups)
    keys = np.array(sorted(hist), np.int64)
    raw = (keys, np.array([hist[int(k)] for k in keys]))
    atomic_npz(directory/'prediction_raw.npz', keys=raw[0], counts=raw[1])
    guard_info = dict(enabled=config.safeguard, tau=config.safeguard_tau, D=public_cap)
    if config.safeguard:
        stage('safeguard')
        # Use original debiased T reports, never binned/scaled S3 working Q.
        direct, reference, reference_joint, cap_law = capacity_reference(noisy, prior, public_cap)
        if capacity_violation(raw, cap_law) > 2e-9:
            raise ArithmeticError('Raw posterior violates fitted degree capacity')
        guarded, stats = safeguard(raw, reference, tau=config.safeguard_tau, D=public_cap)
        violation = capacity_violation(guarded, cap_law)
        if violation > 2e-9:
            raise ArithmeticError('Safeguard output violates fitted degree capacity')
        for name, histogram in (('safeguard_direct', direct), ('safeguard_reference', reference)):
            atomic_npz(directory/(name+'.npz'), keys=histogram[0], counts=histogram[1])
        atomic_npz(directory/'safeguard_reference_joint.npz', **reference_joint)
        guard_info.update(stats, capacity_cdf_violation=violation,
            degree_marginal_preserved=True, reference_input='original_unbinned_debiased_report',
            minimax_theorem_certified=False)
        hist = dict(zip(map(int, guarded[0]), map(float, guarded[1])))
        keys, counts = guarded
    else:
        keys, counts = raw
    atomic_npz(directory/'prediction.npz', keys=keys, counts=counts)
    method = 'privtdd_v2_binned_v1' if config.degree_grid_size or config.report_bin_width else 'privtdd_v2'
    if config.triangle_bin_width:
        method = 'privtdd_v2_noise_binned_v2'
    # Projection completion changes the statistical working channel, so new
    # runs must not be labelled as the historical omission-based estimator.
    method += '_projected_v1'
    method += f'_L{method_config.CAP_OFFSET}' + ('_safeguard_v1' if config.safeguard else '_raw')
    approximation.update(projected_domain_complete=True,
        projection_small_cap_model='conditional uniform edges / hypergeometric',
        projection_large_cap_model='mean thinning + RR normal + Laplace',
        kernel_interpolation=False, final_metric_error_bound_claimed=False)
    stage('save_result')
    info = dict(method=method, n=len(caps), public_cap=public_cap, epsilon=epsilon,
                eps_degree=ed, eps_rr=er, eps_release=el, posterior_updates=1,
                cap_offset=method_config.CAP_OFFSET, safeguard=guard_info,
                numerical_precision_certified=False, config=asdict(config), boundaries=boundaries,
                coordinates=coordinates.tolist(), F1=f1_stats, joint=joint_stats, geometry=structural_stats,
                approximation=approximation, stage_seconds=timings,
                timing_scope='current invocation, possibly using same-case cached stages')
    json_write(directory/'result.json', info)
    stage('complete')
    return hist, info


# [PrivTDD] S1-S7 complete entry point; core method or probability/structure primitive (not necessarily a separate contribution); [Public S2-S7 entry point] One released report set at a common budget -> count GDD.
def estimate_tdd_from_reports(caps, noisy_triangle_report, public_cap, epsilon, *, config=None,
                                   workdir=None, degree_prior=None, return_debug=False, report_cap_offset=None):
    """[Public S2-S7 entry point] One L0 report set at a common budget -> one posterior update -> safeguard -> count GDD.
    
    caps: Integer private caps of length n; noisy_triangle_report: Noisy T reports already debiased for RR.
    public_cap: Declared integer degree bound D; epsilon: Total budget that generated these reports.
    config: PrivTDDConfig or None for defaults; workdir: New cache directory or one with matching identity, or None for a temporary directory.
    degree_prior: Optional degree estimate of length D+1 from the same reports; skips degree fitting but retains common tail canonicalization.
    return_debug: False returns dict[t]=node count; True returns (histogram, metadata).
    report_cap_offset: Optional reporting-mechanism label; when supplied it must match current L, while None uses the current L contract.
    Bare arrays cannot reveal their generating mechanism; pass released['cap_offset'] to reject misuse of historical reports.
    Postprocessing receives only the inputs above, with no original graph or true triangle counts.
    """
    config = PrivTDDConfig() if config is None else config
    if not isinstance(config, PrivTDDConfig):
        raise TypeError('config must be PrivTDDConfig')
    if report_cap_offset is not None:
        declared = np.asarray(report_cap_offset)
        if (declared.shape != () or declared.dtype.kind not in 'iu' or
                int(declared) != method_config.CAP_OFFSET):
            raise ValueError(f'Report cap_offset must match current L={method_config.CAP_OFFSET}; regenerate reports from the graph')
    raw_caps = np.asarray(caps)
    noisy = np.asarray(noisy_triangle_report, float)
    D = int(public_cap)
    if D != public_cap or D < 2 or raw_caps.dtype.kind not in 'iu' or raw_caps.ndim != 1 or len(raw_caps) < 4:
        raise ValueError('Integer cap vector and public_cap>=2 required')
    caps = raw_caps.astype(np.int64, copy=False)
    if noisy.shape != caps.shape or not np.isfinite(noisy).all() or np.any(caps < 0) or np.any(caps > D):
        raise ValueError('Invalid released reports')
    if np.any(noisy[caps <= 1] != 0) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Invalid deterministic reports or epsilon')
    p = 1/(1+math.exp(-RR_FRACTION*epsilon))
    if not .5 < p < 1:
        raise ValueError('Numerically nondegenerate RR required')
    if degree_prior is not None:
        degree_prior = np.asarray(degree_prior, float)
        if degree_prior.shape != (D+1,) or np.any(degree_prior < 0) or not np.isfinite(degree_prior).all() or abs(degree_prior.sum()-1) > 2e-12:
            raise ValueError('degree_prior must be the full nonnegative degree simplex')
    identity = cache_identity(caps, noisy, D, epsilon, config, degree_prior)
    # [PrivTDD] S1-S7 complete entry point; memory and parallel execution; [Cache isolation] Own the working directory exclusively; read completed cache entries without refitting.
    def execute(path):
        """[Cache isolation] Own the working directory exclusively; read completed cache entries without refitting."""
        directory = initialize_cache(path, identity)
        lock = directory/'RUN.lock'
        try:
            handle = lock.open('x', encoding='utf-8')
        except FileExistsError as error:
            raise RuntimeError('Workdir is in use; remove RUN.lock only after the interrupted process has ended') from error
        try:
            handle.write(str(os.getpid()))
            handle.close()
            progress = directory/'progress.json'
            if progress.exists() and json.loads(progress.read_text(encoding='utf-8')).get('stage') == 'complete':
                with np.load(directory/'prediction.npz') as z:
                    hist = dict(zip(map(int, z['keys']), map(float, z['counts'])))
                return hist, json.loads((directory/'result.json').read_text(encoding='utf-8'))
            with threadpool_limits(limits=config.blas_threads):
                return _run_joint(caps, noisy, D, epsilon, config, directory, degree_prior)
        finally:
            handle.close()
            lock.unlink(missing_ok=True)
    if workdir is None:
        with TemporaryDirectory(prefix='privtdd_') as temporary:
            result = execute(temporary)
    else:
        result = execute(Path(workdir))
    return result if return_debug else result[0]


# [PrivTDD] S1-S7 complete entry point; core method or probability/structure primitive (not necessarily a separate contribution); [Complete public S1-S7 entry point] Simple undirected original graph -> interactive reports -> seven-parameter correction, posterior and TDD release.
def estimate_tdd(adj_list, node_num, deg_max, epsilon, seed=20260908, *, config=None,
                          workdir=None, return_debug=False):
    """[Complete public S1-S7 entry point] Original graph -> L0 interactive reports -> seven-parameter correction -> posterior -> safeguarded TDD.
    
    adj_list/node_num/deg_max: Adjacency structure, public n and public D; epsilon: Total budget.
    seed: Experiment reproducibility seed; config/workdir/return_debug: See estimate_tdd_from_reports.
    Old FULL, k, M and delta arguments are not accepted: the interactive mechanism is not an alias for the old FULL call.
    Return a histogram of floating-point node counts with total mass n; do not round or use truth to choose the output.
    """
    config = PrivTDDConfig() if config is None else config
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Nonnegative integer experimental seed required')
    ip, ix = prepare_graph(adj_list, node_num, deg_max)
    released = collect_reports_joint(ip, ix, deg_max, epsilon, int(seed), block_size=config.report_block_size, workers=config.workers)
    return estimate_tdd_from_reports(released['caps'], released['noisy_triangle_report'], deg_max, epsilon,
                                          config=config, workdir=workdir, return_debug=return_debug,
                                          report_cap_offset=released['cap_offset'])
