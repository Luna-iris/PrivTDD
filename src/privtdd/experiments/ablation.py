"""Joint fast ablations: all variant definitions, switches, inference and CLI live here.

The production Joint estimator is reused without monkey-patching or changing its
defaults. Truth is read only by the output evaluator, never by variant fitting.
E is registered but cannot run until a consistent projected model is validated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize
from threadpoolctl import threadpool_limits

from privtdd.config import (
    PrivTDDConfig, JOINT_COORDINATES, JOINT_TOLERANCE, JOINT_MAX_ITERATIONS,
    DEGREE_FRACTION, RR_FRACTION, RELEASE_FRACTION, CAP_OFFSET,
)
from privtdd.release.pipeline import fit_joint7
from privtdd.correction.optimizer import fit_joint
from privtdd.correction.forward import build_family
from privtdd.posterior.update import update_posterior_joint
from privtdd.posterior.rows import prior_profiles
from privtdd.correction.regions import region_ids
from privtdd.numerics.quantization import bin_reports
from privtdd.io.storage import (
    StoredConditional, load_sources, save_sources, json_write, atomic_text_write,
)
from privtdd.io.memory import atomic_npz
from privtdd.io.runner import (
    build_cases, case_identity, file_digest, output_lock, run_case,
    evaluate_saved_prediction,
)

DATASETS = ('gplus', 'higgs', 'orkut')
EPSILONS = (.5, 1., 1.5, 2., 2.5, 3., 4.)
SEEDS = tuple(range(20260908, 20260913))
OFFSETS = (0, 25, 50, 100, 150)
MASS_COORDINATES = (0, 1, 3)
SHAPE_COORDINATES = (4, 5, 6, 7)
GEO_COORDINATES = (0, 1, 4, 5)
BETA_COORDINATES = (3, 6, 7)
E_REASON = (
    'Deferred: current fast drops d>cap states. E requires consistent projected '
    'S3 likelihood/recovery, S4 Geo mean calibration, S5 forward and S6 adjoint, '
    'plus offset-aware collection/degree channel/grid/tail. Changing CAP_OFFSET '
    'alone is not a valid E experiment. A projected L=150 bridge is required.'
)


@dataclass(frozen=True)
class Variant:
    name: str
    experiment: str
    policy: str
    posterior: bool = False
    strength: float | None = None
    source_mode: str = 'both'
    active: tuple[int, ...] = JOINT_COORDINATES


def variant_registry():
    """One authoritative registry, including aliases of the existing A/C/D."""
    variants = [
        Variant('A_joint7_post', 'A', 'learned', True),
        Variant('B_degree', 'B', 'degree'),
        Variant('C_f1', 'C', 'zero'),
        Variant('C_f1_post', 'C', 'zero', True),
        Variant('D_joint7', 'D', 'learned'),
    ]
    for alpha in (.25, .5, .75):
        for post in (False, True):
            variants.append(Variant(f'F_fixed{int(alpha*100):02d}'+('_post' if post else ''),
                                    'F', 'fixed', post, alpha))
    for label, policy, active in (('G_mass', 'mass', MASS_COORDINATES),
                                  ('H_shape', 'shape', SHAPE_COORDINATES)):
        for post in (False, True):
            variants.append(Variant(label+('_post' if post else ''), label[0], policy, post, active=active))
    for label, mode, active in (('I_no_beta', 'no_beta', GEO_COORDINATES),
                                ('J_no_geo', 'no_geo', BETA_COORDINATES)):
        for post in (False, True):
            variants.append(Variant(label+('_post' if post else ''), label[0], 'source', post,
                                    source_mode=mode, active=active))
    return {v.name: v for v in variants}


def select_variants(experiments):
    """E stays visible in the plan; it never silently runs the retained-only model."""
    registry = variant_registry()
    allowed = set('ABCDEFGHIJ') | {'all'}
    if not experiments or any(x not in allowed for x in experiments):
        raise ValueError('Experiments must be all or letters A through J')
    if len(experiments) != len(set(experiments)) or ('all' in experiments and len(experiments) != 1):
        raise ValueError('Do not duplicate experiments or combine all with individual letters')
    selected = set('ABCDEFGHIJ') if experiments == ['all'] else set(experiments)
    return [v for v in registry.values() if v.experiment in selected], ('E' in selected)


def fast_config(*, workers=8, blas_threads=2, geometry_draws=65536, geometry_replicates=4):
    """Exactly the current run_joint --acceleration fast preset."""
    return PrivTDDConfig(workers=workers, blas_threads=blas_threads,
        geometry_draws=geometry_draws, geometry_replicates=geometry_replicates,
        degree_grid_size=512, report_bin_width=1/16, triangle_bin_width=1/16,
        structural_lazy_endpoints=True, batch_small_groups=True,
        batch_f1=True, compiled_structure=True)


def fixed_coordinates(family, strength):
    """Euclidean projection onto the SAME seven-coordinate polytope and [0,1].

    This function deliberately accepts no reports, counts, likelihood table or
    truth in its objective: only family A/b are read. Data-derived feasibility
    is retained; correction strength is never selected by observed likelihood.
    """
    if strength not in (.25, .5, .75):
        raise ValueError('Registered fixed strengths are 0.25, 0.5, 0.75')
    ids = np.asarray(JOINT_COORDINATES)
    original = np.asarray(family['A'], float)[:, ids]
    rhs = np.asarray(family['b'], float)
    scale = np.maximum(np.max(abs(original), axis=1), abs(rhs))
    scale[scale == 0] = 1
    A, b = original/scale[:, None], rhs/scale
    target = np.full(len(ids), strength)
    constraint = dict(type='ineq', fun=lambda z: b-A@z, jac=lambda z: -A)
    answer = minimize(lambda z: (.5*float((z-target)@(z-target)), z-target),
        np.zeros(len(ids)), jac=True, method='SLSQP', bounds=[(0., 1.)]*len(ids),
        constraints=[constraint], options=dict(ftol=1e-14, maxiter=2000))
    point = np.clip(answer.x, 0., 1.)
    # Check first-order optimality independently of the QP solver's success flag.
    gradient = point-target
    lp = linprog(gradient, A_ub=A, b_ub=b, bounds=[(0., 1.)]*len(ids), method='highs',
                 options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    if not lp.success:
        raise ArithmeticError('Fixed projection optimality check failed')
    lam = np.minimum(lp.ineqlin.marginals, 0.)
    lower = np.maximum(lp.lower.marginals, 0.)
    upper = np.minimum(lp.upper.marginals, 0.)
    residual = gradient-A.T@lam-lower-upper
    dual = float(b@lam+upper.sum()+np.minimum(residual, 0.).sum())
    gap = max(0., float(gradient@point)-dual)
    violation = float(np.max(original@point-rhs))
    if gap > 1e-8 or violation > 2e-12 or not np.isfinite(point).all():
        raise ArithmeticError(f'Unresolved fixed projection: gap={gap}, violation={violation}')
    result = np.zeros(16)
    result[ids] = point
    return result, dict(policy='predeclared_feasible_projection', strength=strength,
        projection_gap=gap, solver_message=str(answer.message),
        distance_to_requested=float(np.linalg.norm(point-target)),
        degenerate_zero=bool(np.max(abs(point)) <= 1e-10), report_objective_used=False,
        feasible_domain_is_data_derived=True, truth_used=False)


def fit_variant(family, variant, directory):
    """Fix or fit coordinates, preserving the original penalty and stopping rule."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'coefficients.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        return np.asarray(saved['coordinates']), saved['fit']
    if variant.policy == 'zero':
        x, stats = np.zeros(16), dict(policy='F1_anchor', report_objective_used=False)
    elif variant.policy == 'fixed':
        x, stats = fixed_coordinates(family, variant.strength)
    elif variant.active == JOINT_COORDINATES:
        x, _, stats = fit_joint7(family, directory=directory)
    else:
        # Restricted original coordinates; no extra parameters or altered penalty.
        x, stats = fit_joint(family, variant.active, max_iterations=JOINT_MAX_ITERATIONS,
                             tolerance=JOINT_TOLERANCE)
    if np.max(family['A']@x-family['b']) > 2e-12 or np.any(x < -2e-12) or np.any(x > 1+2e-12):
        raise ArithmeticError('Invalid ablation coefficients')
    json_write(path, dict(coordinates=x, fit=stats))
    return x, stats


def stable_candidate_density(family, coordinates, density, shared, config):
    """Repair cancellation in base + signed directions with the same positive law.

    Only ill-conditioned observation rows are reevaluated. The original S6
    prior_profiles, degree weights, RR/Laplace kernels and report multiplicities
    are retained. No density floor or relaxed conservation tolerance is used.
    """
    from privtdd.base_distribution.channel import prepare_groups
    from privtdd.degree.groups import groups_by_degree
    from privtdd.correction.sparse_channel import (
        integer_law_channel, use_batched_channel, batched_integer_channel,
    )
    from privtdd.correction.projected_sparse_channel import projected_sparse_channel
    envelope = (np.abs(family['predictive_table'])+
                np.abs(family['delta_tables']).T@np.abs(coordinates))
    suspect = (~np.isfinite(density)) | (density <= 1e-4*envelope)
    selected = np.flatnonzero(suspect)
    audit = dict(density_evaluation='affine_with_positive_sum_cancellation_repair',
                 cancellation_rows=int(len(selected)))
    if not len(selected):
        return density, audit
    positions = np.full(len(density), -1, dtype=int)
    positions[selected] = np.arange(len(selected))
    repaired = np.zeros(len(selected))
    groups = groups_by_degree(prepare_groups(shared['caps'], shared['prior'], shared['ed']))
    X = np.asarray(coordinates).reshape(1, 16)
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        for j, degree in enumerate(family['support']):
            entries = groups.get(int(degree), [])
            restricted = [(cap, ids[suspect[ids]], weight) for cap, ids, weight in entries
                          if np.any(suspect[ids])]
            if not restricted:
                continue
            keys, law, _ = prior_profiles(family, j, X)
            # Projected states must use the same channel as build_family;
            # ordinary-only sets retain the existing batched implementation.
            ordinary_only = all(cap >= degree or cap <= 1 for cap, _, _ in restricted)
            if ordinary_only and config.batch_small_groups and use_batched_channel(entries, int(degree), len(keys)):
                ids = np.concatenate([e[1] for e in restricted])
                weights = np.concatenate([np.full(len(e[1]), e[2]) for e in restricted])
                values = batched_integer_channel(shared['report'][ids],
                    np.maximum(shared['caps'][ids]-1, 0)/shared['el'],
                    int(degree), keys, shared['p'], law)[:, 0]
                repaired[positions[ids]] += weights*values
            else:
                def task(entry):
                    cap, ids, weight = entry
                    if cap <= 1:
                        if np.any(shared['report'][ids] != 0):
                            raise ArithmeticError('Nonzero deterministic report')
                        values = np.ones(len(ids))
                    else:
                        values, _ = projected_sparse_channel(keys, law, int(degree), cap, shared['p'],
                            (cap-1)/shared['el'], shared['report'][ids])
                        values = values[:, 0]
                    return ids, weight*values
                for ids, values in pool.map(task, restricted):
                    repaired[positions[ids]] += values
    # A repair must agree in absolute scale with the affine expression; a real
    # channel/cache mismatch is an error, not a reason to normalize it away.
    difference = np.abs(repaired-density[selected])
    bound = 2e-12*envelope[selected]
    if (not np.isfinite(density[selected]).all() or np.any(difference > bound)
            or np.any(repaired <= 0) or not np.isfinite(repaired).all()):
        raise ArithmeticError('Candidate density repair exceeds roundoff bound')
    audit.update(maximum_density_absolute_correction=float(difference.max()),
                 maximum_density_relative_correction=float(np.max(difference/repaired)))
    result = density.copy()
    result[selected] = repaired
    return result, audit


def observation_score(family, coordinates, *, shared=None, config=None):
    """S5 in-sample composite likelihood; NOT a held-out or true-risk score."""
    base = np.asarray(family['predictive_table'])
    x = np.asarray(coordinates)
    # Same multiplication order as fit_joint7.
    density = (base[:, None]+family['delta_tables'].T@x[None, :].T)[:, 0]
    density_audit = {}
    if shared is not None:
        density, density_audit = stable_candidate_density(family, x, density, shared, config)
    weights = family['counts']
    if np.any(density <= 0) or not np.isfinite(density).all():
        raise ArithmeticError('Nonpositive candidate report density')
    nll = -math.fsum((weights*np.log(density)).tolist())
    relative = (density-base)/base
    log_ratio = np.empty_like(density)
    safe = np.abs(relative) < .5
    log_ratio[safe] = np.log1p(relative[safe])
    log_ratio[~safe] = np.log(density[~safe])-np.log(base[~safe])
    delta = -math.fsum((weights*log_ratio).tolist())
    penalty = .5*float(x@x)
    return density, dict(nll=nll, nll_per_node=nll/weights.sum(),
        nll_minus_F1=delta, penalty=penalty, penalized_minus_F1=delta+penalty,
        score_stage='frozen_prior_before_S6', in_sample=True,
        likelihood_interpretation='current approximate retained-state composite channel', **density_audit)


def frozen_histogram(family, coordinates, n):
    """Output S5 before posterior, auditing every original-degree conditional row."""
    hist = {}
    row_l1 = 0.
    worst_mass = 0.
    minimum = 1.
    region_mass = np.zeros(4)
    for j, d in enumerate(family['support']):
        keys, laws, audit = prior_profiles(family, j, np.asarray(coordinates).reshape(1, 16))
        law = laws[:, 0]
        fkeys, fprob = family['sources'][0].row(j)
        anchor = np.zeros(len(keys))
        anchor[np.searchsorted(keys, fkeys)] = fprob
        weight = family['pi'][j]
        row_l1 += weight*float(abs(law-anchor).sum())
        region_mass += weight*np.bincount(region_ids(keys, family['boundaries']), weights=law, minlength=4)
        worst_mass = max(worst_mass, audit['maximum_mass_error'])
        minimum = min(minimum, audit['minimum_before_roundoff_clipping'])
        if np.any(keys < 0) or np.any(keys > int(d)*(int(d)-1)//2):
            raise ArithmeticError('Conditional support exceeds degree capacity')
        for key, value in zip(keys, n*weight*law):
            if value:
                hist[int(key)] = hist.get(int(key), 0.)+float(value)
    return hist, dict(weighted_conditional_L1_from_F1=row_l1,
        region_mass_before_S6=region_mass, maximum_conditional_mass_error=worst_mass,
        minimum_conditional_probability=minimum, fixed_degree_marginal=True)


def degree_only(caps, prior, eps_degree, config, directory):
    """B: cap-only Beta predictor. No noisy triangle report/Geo/F1 is accepted.

    Keep the original degree budget: the discarded triangle budget is not
    reassigned. Degree alone does not identify GDD; the Beta assumption is explicit.
    """
    from privtdd.references.context import structural_context, prepare_compositions
    from privtdd.references.independent_edge import generate_beta
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory/'beta.npz'
    if cache.exists():
        beta = load_sources(cache)[0]
    else:
        context, _ = structural_context(caps, prior, eps_degree)
        packet, _ = prepare_compositions(context, draws=config.geometry_draws,
            replicates=config.geometry_replicates, reuse_root_kernels=config.compiled_structure)
        beta, _ = generate_beta(packet, compiled=config.compiled_structure)
        save_sources(cache, (beta,))
    return beta.population_histogram(prior[beta.support], len(caps))


def source_family(shared, source_mode, directory, config):
    """Remove a source entirely, including empty-region fallback shapes.

    Replacing the missing source by F1 is a neutral placeholder for the existing
    three-slot API. Its coordinates are also frozen at zero. Boundaries/pi stay
    fixed to the common F1. Merely masking directions would leak fallback shapes.
    """
    if source_mode == 'both':
        return shared['family']
    directory.mkdir(parents=True, exist_ok=True)
    f1, beta, geo = shared['family']['sources']
    if source_mode == 'no_beta':
        sources = (f1, f1, geo)
    elif source_mode == 'no_geo':
        sources = (f1, beta, f1)
    else:
        raise ValueError(source_mode)
    path = directory/'channel.npz'
    if path.exists():
        with np.load(path) as z:
            family = {k: z[k] for k in z.files}
    else:
        family = build_family(sources, shared['caps'], shared['report'], shared['prior'],
            shared['p'], shared['ed'], shared['el'], shared['family']['boundaries'],
            workers=config.workers, batch_small_groups=config.batch_small_groups)
        family['counts'] = shared['counts']
        atomic_npz(path, **{k: family[k] for k in ('counts', 'predictive_table', 'delta_tables', 'A', 'b', 'gate_basis')})
    family.update(sources=sources, support=f1.support, pi=f1.pi,
                  boundaries=shared['family']['boundaries'], n=shared['n'])
    return family


def load_shared(reference, config):
    """Read the unmodified production run's frozen stages, retaining report weights."""
    work = reference/'work'
    with np.load(reference/'reports.npz') as z:
        raw_caps, noisy = z['caps'], z['noisy_triangle_report']
    info = json.loads((work/'result.json').read_text(encoding='utf-8'))
    epsilon = info['epsilon']
    ed, er, el = DEGREE_FRACTION*epsilon, RR_FRACTION*epsilon, RELEASE_FRACTION*epsilon
    gap = 2/(1+math.exp(-er))-1
    caps, report, counts, _ = bin_reports(raw_caps, gap*noisy, el, config.report_bin_width)
    with np.load(work/'degree.npz') as z:
        prior = z['prior']
    with np.load(work/'F1.npz') as z:
        f1 = StoredConditional(work/'f1_rows', z['support'], z['degree_mass'],
            z['keys'], z['counts'], z['native_degree_mass'])
    with np.load(work/'joint_channel.npz') as z:
        family = {k: z[k] for k in z.files}
    with np.load(work/'joint_context.npz') as z:
        coordinates = z['coordinates']
    family.update(sources=(f1, *load_sources(work/'sources.npz')), support=f1.support,
                  pi=f1.pi, boundaries=tuple(info['boundaries']), n=len(raw_caps))
    np.testing.assert_array_equal(counts, family['counts'])
    return dict(family=family, raw_caps=raw_caps, raw_noisy=noisy, caps=caps, report=report, counts=counts,
                prior=prior, ed=ed, el=el, p=(1+gap)/2, n=len(raw_caps),
                coordinates=coordinates, joint_stats=info['joint'], original_info=info)


def validate_histogram(histogram, n, public_cap):
    keys = np.array(sorted(histogram), dtype=np.int64)
    counts = np.array([histogram[int(k)] for k in keys], dtype=float)
    if not len(keys) or np.any(keys < 0) or np.any(keys > public_cap*(public_cap-1)//2):
        raise ArithmeticError('Histogram outside public triangle domain')
    if not np.isfinite(counts).all() or np.any(counts < 0) or abs(counts.sum()/n-1) > 1e-9:
        raise ArithmeticError('Histogram population/positivity violation')
    return keys, counts


def completed_variant(directory):
    """Never trust a completion marker without checking its saved artifacts."""
    marker = directory/'completion.json'
    if not marker.exists():
        return False
    done = json.loads(marker.read_text(encoding='utf-8'))
    for name, digest in done['sha256'].items():
        if file_digest(directory/name) != digest:
            raise ValueError(f'Completed ablation artifact changed: {directory/name}')
    return True


def save_variant(directory, variant, histogram, diagnostics, n, public_cap):
    directory.mkdir(parents=True, exist_ok=True)
    keys, counts = validate_histogram(histogram, n, public_cap)
    atomic_npz(directory/'prediction.npz', keys=keys, counts=counts)
    json_write(directory/'diagnostics.json', diagnostics)
    json_write(directory/'variant.json', asdict(variant))
    json_write(directory/'completion.json', dict(status='complete',
        sha256={name: file_digest(directory/name) for name in ('prediction.npz', 'diagnostics.json', 'variant.json')}))


def variant_histogram(shared, family, x, variant, config):
    """Run only the requested S6 switch; always retain a pre-S6 diagnostic."""
    hist, audit = frozen_histogram(family, x, shared['n'])
    normalizer, score = observation_score(family, x, shared=shared, config=config)
    if variant.posterior:
        hist = update_posterior_joint(family, x, normalizer, shared['caps'], shared['report'],
            shared['prior'], shared['p'], shared['ed'], shared['el'], workers=config.workers,
            report_counts=shared['counts'], batch_small_groups=config.batch_small_groups)
    return hist, dict(**audit, **score)


def run_ablation_case(case, config, identity, variants):
    """One transcript, shared production A, then isolated candidate outputs.

    A is computed using run_joint.run_case unchanged. All truth paths are withheld
    until ALL requested candidate predictions for this case have been frozen.
    """
    out = Path(case['output'])
    rows = []
    with output_lock(out, 'ABLATION.lock'), threadpool_limits(limits=config.blas_threads):
        manifest = out/'ablation_identity.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8'))['identity'] != identity:
                raise ValueError(f'Different ablation inputs/code/settings: {out}; use a new output')
        else:
            if any(p.name != 'ABLATION.lock' for p in out.iterdir()):
                raise ValueError(f'Nonempty unbound ablation directory: {out}')
            json_write(manifest, dict(identity=identity, case=case, config=asdict(config)))
        shared = None
        families = {}
        for variant in variants:
            directory = out/'variants'/variant.name
            if completed_variant(directory):
                continue
            if shared is None:
                # The original runner owns its cache and resumes its own stages.
                reference = out/'reference'
                ref_case = dict(case, output=str(reference), truth=None)
                run_case(ref_case, config, identity)
                if case['mode'] == 'reports':
                    # Original reports runner reads the external file; keep a local
                    # immutable transcript for the common stage loader.
                    with np.load(case['source']) as z:
                        atomic_npz(reference/'reports.npz', caps=z['caps'], noisy_triangle_report=z['noisy_triangle_report'])
                shared = load_shared(reference, config)
            started = time.perf_counter()
            json_write(out/'progress.json', dict(stage=variant.name))
            if variant.policy == 'degree':
                histogram = degree_only(shared['raw_caps'], shared['prior'], shared['ed'], config, out/'degree_only')
                diagnostics = dict(policy='cap_only_beta', noisy_triangle_report_used=False,
                    Geo_calibration_used=False, additional_epsilon=0.,
                    degree_budget_reallocated=False, effective_cap_budget=shared['ed'],
                    fixed_degree_marginal=True, observation_score_not_computed=True)
            else:
                if variant.source_mode not in families:
                    families[variant.source_mode] = source_family(shared, variant.source_mode,
                        out/'families'/variant.source_mode, config)
                family = families[variant.source_mode]
                # S5 fit shared between its pre/post outputs; full learned reuses A.
                policy_key = f'{variant.policy}_{variant.strength}_{variant.source_mode}'
                if variant.policy == 'learned':
                    x, fit = shared['coordinates'], shared['joint_stats']
                else:
                    x, fit = fit_variant(family, variant, out/'fits'/policy_key)
                if variant.name == 'A_joint7_post':
                    with np.load(out/'reference'/'prediction.npz') as z:
                        histogram = dict(zip(z['keys'].tolist(), z['counts'].tolist()))
                    _, audit = frozen_histogram(family, x, shared['n'])
                    _, score = observation_score(family, x)
                    diagnostics = dict(**audit, **score, baseline_is_unmodified_production=True)
                else:
                    histogram, diagnostics = variant_histogram(shared, family, x, variant, config)
                diagnostics.update(coordinates=x, fit=fit, source_mode=variant.source_mode)
            diagnostics.update(posterior_updates=int(variant.posterior), cap_offset=CAP_OFFSET,
                execution_code_sha256=code_fingerprint(),
                truth_used_for_inference=False, method_seconds_current_invocation=time.perf_counter()-started,
                timing_scope='marginal_variant_work_with_shared_cached_stages_not_end_to_end')
            save_variant(directory, variant, histogram, diagnostics, shared['n'], case['public_cap'])
        # No truth or baseline-risk feedback before this point.
        truth_digest = file_digest(case['truth']) if case['truth'] else None
        for variant in variants:
            directory = out/'variants'/variant.name
            diagnostics = json.loads((directory/'diagnostics.json').read_text(encoding='utf-8'))
            metrics = {}
            if case['truth']:
                metrics = evaluate_saved_prediction(directory/'prediction.npz', case['truth'], case['public_cap'])
                json_write(directory/'metrics.json', dict(**metrics, truth_sha256=truth_digest))
            rows.append(dict(dataset=case['dataset'], epsilon=case['epsilon'], seed=case['seed'],
                variant=variant.name, status='complete', output=str(directory), **metrics,
                nll_per_node=diagnostics.get('nll_per_node'), penalty=diagnostics.get('penalty'),
                nll_minus_F1=diagnostics.get('nll_minus_F1'),
                variant_seconds=diagnostics['method_seconds_current_invocation']))
        # Detect fixed-strength collapse explicitly; never select a best alpha.
        fixed = {}
        for v in variants:
            if v.policy == 'fixed' and not v.posterior:
                diagnostic = json.loads((out/'variants'/v.name/'diagnostics.json').read_text(encoding='utf-8'))
                fixed[v.name] = diagnostic['coordinates']
        duplicates = [[a, b] for i, a in enumerate(fixed) for b in list(fixed)[i+1:]
                      if np.max(abs(np.asarray(fixed[a])-fixed[b])) <= 1e-10]
        json_write(out/'fixed_strength_audit.json', dict(equivalent_pairs=duplicates,
            tested_strengths=[.25, .5, .75], truth_based_selection=False))
        json_write(out/'progress.json', dict(stage='complete', outputs=len(rows)))
    return rows


def write_csv(path, rows, fields):
    with io.StringIO(newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        atomic_text_write(path, stream.getvalue(), encoding='utf-8-sig')


def summarize(directory, cases, variants, rows, deferred):
    """Requested-case coverage, sample SD, and same-seed deltas; no seed pooling."""
    fields = ['dataset', 'epsilon', 'seed', 'variant', 'status', 'MSE', 'KS', 'EMD',
              'nll_per_node', 'nll_minus_F1', 'penalty', 'variant_seconds', 'output', 'error']
    write_csv(directory/'results.csv', rows, fields)
    summary = []
    group_keys = sorted({(c['dataset'], c['epsilon']) for c in cases})
    for dataset, epsilon in group_keys:
        requested = sum(c['dataset'] == dataset and c['epsilon'] == epsilon for c in cases)
        for variant in variants:
            records = [r for r in rows if r['dataset'] == dataset and r['epsilon'] == epsilon and r['variant'] == variant.name]
            done = [r for r in records if r['status'] == 'complete']
            item = dict(dataset=dataset, epsilon=epsilon, variant=variant.name, requested=requested,
                completed=len(done), failed=sum(r['status'] == 'failed' for r in records),
                coverage_complete=len(done) == requested)
            for metric in ('MSE', 'KS', 'EMD'):
                values = [r[metric] for r in done if metric in r]
                item[metric+'_n'] = len(values)
                item[metric+'_mean'] = float(np.mean(values)) if values else None
                item[metric+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(item)
    write_csv(directory/'summary.csv', summary,
        ['dataset', 'epsilon', 'variant', 'requested', 'completed', 'failed', 'coverage_complete']+
        [m+s for m in ('MSE', 'KS', 'EMD') for s in ('_n', '_mean', '_std')])
    index = {(r['dataset'], r['epsilon'], r['seed'], r['variant']): r for r in rows if r['status'] == 'complete'}
    paired = []
    for row in rows:
        if row['status'] != 'complete':
            continue
        reference = 'A_joint7_post' if row['variant'].endswith('_post') or row['variant'] == 'B_degree' else 'D_joint7'
        if row['variant'] == reference:
            continue
        anchor = index.get((row['dataset'], row['epsilon'], row['seed'], reference))
        if anchor is None:
            continue
        paired.append(dict(contrast='vs_learned_same_posterior', dataset=row['dataset'], epsilon=row['epsilon'], seed=row['seed'],
            variant=row['variant'], reference=reference,
            **{m+'_delta': row[m]-anchor[m] for m in ('MSE', 'KS', 'EMD') if m in row and m in anchor}))
    for row in rows:
        if row['status'] != 'complete' or not row['variant'].endswith('_post'):
            continue
        reference = 'D_joint7' if row['variant'] == 'A_joint7_post' else row['variant'][:-5]
        anchor = index.get((row['dataset'], row['epsilon'], row['seed'], reference))
        if anchor is not None:
            paired.append(dict(contrast='posterior_effect', dataset=row['dataset'], epsilon=row['epsilon'],
                seed=row['seed'], variant=row['variant'], reference=reference,
                **{m+'_delta': row[m]-anchor[m] for m in ('MSE', 'KS', 'EMD') if m in row and m in anchor}))
    write_csv(directory/'paired_deltas.csv', paired,
        ['contrast', 'dataset', 'epsilon', 'seed', 'variant', 'reference', 'MSE_delta', 'KS_delta', 'EMD_delta'])
    paired_summary = []
    for key in sorted({(r['contrast'], r['dataset'], r['epsilon'], r['variant'], r['reference']) for r in paired}):
        contrast, dataset, epsilon, variant, reference = key
        group = [r for r in paired if (r['contrast'], r['dataset'], r['epsilon'], r['variant'], r['reference']) == key]
        item = dict(contrast=contrast, dataset=dataset, epsilon=epsilon, variant=variant, reference=reference)
        for metric in ('MSE', 'KS', 'EMD'):
            values = [r[metric+'_delta'] for r in group if metric+'_delta' in r]
            item[metric+'_pairs'] = len(values)
            item[metric+'_delta_mean'] = float(np.mean(values)) if values else None
            item[metric+'_delta_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        paired_summary.append(item)
    write_csv(directory/'paired_summary.csv', paired_summary,
        ['contrast', 'dataset', 'epsilon', 'variant', 'reference']+
        [m+s for m in ('MSE', 'KS', 'EMD') for s in ('_pairs', '_delta_mean', '_delta_std')])
    failed = sum(r['status'] == 'failed' for r in rows)
    completed = sum(r['status'] == 'complete' for r in rows)
    requested = len(cases)*len(variants)
    status = 'failed' if failed else ('running' if completed < requested else ('complete_with_deferred_E' if deferred else 'complete'))
    json_write(directory/'suite_status.json', dict(status=status, completed_outputs=completed,
        requested_runnable_outputs=requested, failed_outputs=failed,
        E=dict(status='deferred', offsets=OFFSETS, reason=E_REASON) if deferred else dict(status='not_requested')))


def code_fingerprint():
    from privtdd.io.identity import source_digest
    return source_digest('package')


def compatible_legacy_fingerprint():
    # Moved/renamed sources have a new identity. Never accept the research-tree cache.
    return None


def resume_identity(case, config, digests, fingerprint):
    """Accept unchanged inputs/settings under one explicitly compatible revision."""
    current = case_identity(case, config, digests, fingerprint)
    manifest = Path(case['output'])/'ablation_identity.json'
    if not manifest.exists():
        return current
    saved = json.loads(manifest.read_text(encoding='utf-8'))['identity']
    if saved == current:
        return current
    legacy = compatible_legacy_fingerprint()
    if legacy and saved == case_identity(case, config, digests, legacy):
        return saved
    raise ValueError(f"Different ablation inputs/code/settings at {case['output']}; use a new output")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--datasets', nargs='+', help='Default: gplus higgs orkut')
    sources.add_argument('--graph', type=Path, help='CSR NPZ or directory')
    sources.add_argument('--reports', type=Path, help='One tagged L=0 transcript from the current mechanism')
    parser.add_argument('--cache-root', type=Path, default=Path('datasets/processed'))
    parser.add_argument('--public-cap', type=int)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--epsilons', type=float, nargs='+')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--no-evaluate', action='store_true')
    parser.add_argument('--experiments', nargs='+', default=['all'], choices=['all', *'ABCDEFGHIJ'])
    parser.add_argument('--output', type=Path, default=Path('results/runs/ablation'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--blas-threads', type=int, default=2)
    parser.add_argument('--geometry-draws', type=int, default=65536, help='Use 65536 for formal runs')
    parser.add_argument('--geometry-replicates', type=int, default=4, help='Use 4 for formal runs')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    args = parser.parse_args(argv)
    try:
        variants, deferred = select_variants(args.experiments)
        config = fast_config(workers=args.workers, blas_threads=args.blas_threads,
            geometry_draws=args.geometry_draws, geometry_replicates=args.geometry_replicates)
    except ValueError as error:
        parser.error(str(error))
    if not variants:
        print(json.dumps(dict(E=dict(status='deferred', offsets=OFFSETS, reason=E_REASON)), indent=2))
        if not args.dry_run:
            raise SystemExit(2)
        return
    if not any((args.datasets, args.graph, args.reports)):
        args.datasets = list(DATASETS)
    if args.epsilon is None and args.epsilons is None:
        if args.reports:
            parser.error('--reports requires its actual --epsilon; budgets cannot be inferred')
        args.epsilons = list(EPSILONS)
    if args.seed is None and args.seeds is None and not args.reports:
        args.seeds = list(SEEDS)
    args.degree_prior = None  # Shared degree fit is always derived from this transcript.
    cases, _ = build_cases(args, parser)
    # Always one case subdirectory; avoid root lock/manifest collision for single runs.
    for case in cases:
        case['output'] = str((args.output/case['key']).resolve())
    plan = dict(cases=cases, variants=[asdict(v) for v in variants], config=asdict(config),
        runnable_outputs=len(cases)*len(variants), acceleration='fast', cap_offset=CAP_OFFSET,
        fixed_strengths=[.25, .5, .75], E=dict(status='deferred' if deferred else 'not_requested',
        offsets=OFFSETS, reason=E_REASON), baseline='unmodified current Joint7 fast',
        formal_integration_budget=config.geometry_draws == 65536 and config.geometry_replicates == 4)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    fingerprint = code_fingerprint()
    digests, identities = {}, {}
    with output_lock(args.output, 'ABLATION_BATCH.lock'):
        # Check EVERY requested case before changing the plan or writing results.
        for case in cases:
            identity = resume_identity(case, config, digests, fingerprint)
            identities[case['key']] = identity
        json_write(args.output/'ablation_plan.json', dict(plan, code_sha256=fingerprint,
            compatible_legacy_code_sha256=compatible_legacy_fingerprint(),
            numerical_fix='2026-10-24: positive-sum cancellation repair for candidate densities',
            case_identities=identities))
        rows = []
        summarize(args.output, cases, variants, rows, deferred)
        for number, case in enumerate(cases, 1):
            print(f"[{number}/{len(cases)}] {case['key']} ({len(variants)} outputs)", flush=True)
            try:
                rows.extend(run_ablation_case(case, config, identities[case['key']], variants))
            except Exception as error:
                # Frozen variants remain resumable; this case is not advertised as complete.
                rows.extend(dict(dataset=case['dataset'], epsilon=case['epsilon'], seed=case['seed'],
                    variant=v.name, status='failed', error=f'{type(error).__name__}: {error}') for v in variants)
                summarize(args.output, cases, variants, rows, deferred)
                if not args.continue_on_error:
                    raise
            summarize(args.output, cases, variants, rows, deferred)
        if any(r['status'] == 'failed' for r in rows):
            raise SystemExit(1)
        print(f"Completed {len(rows)} runnable outputs: {args.output/'summary.csv'}", flush=True)
        if deferred:
            print('E is DEFERRED, not completed. See ablation_plan.json and README_ablation.md.', flush=True)


if __name__ == '__main__':
    main()
