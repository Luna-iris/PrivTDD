"""S2-S7 for the rotated-sampling experiment; observation kernels use the sampling frontend's original centered-count units.

S3, S5, and S6 all use SampledChannel; the degree channel, structural references, joint constraints,
optimizer, and final safeguard come from the main method. This entry point accepts neither adjacency lists nor ground truth.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from privtdd.distributions.conditional import ConditionalDistribution
from privtdd.correction import basis as stable_basis_joint
from privtdd.release.pipeline import fit_joint7
from privtdd.base_distribution.cells import closure_edges
from privtdd.base_distribution.optimizer import optimize_prior
from privtdd.correction.feasible_domain import coefficient_domain, degree_gate_basis
from privtdd.config import (
    CLOSURE_CELLS, CLOSURE_TOLERANCE, DEGREE_TOLERANCE, POSITIVE_QUANTILES,
)
from privtdd.degree.canonical import canonicalize_tail
from privtdd.degree.fit import fit_cold
from privtdd.degree.groups import groups_by_degree
from privtdd.base_distribution.channel import prepare_groups
from privtdd.io.memory import atomic_npz
from privtdd.posterior.rows import prior_profiles
from privtdd.numerics.quantization import fit_degree_grid
from privtdd.release.transport_guard import capacity_reference, capacity_violation, safeguard
from privtdd.io.storage import json_write
from privtdd.references.construction import build_structural_joint
from .sampled_channel import SampledChannel


def _bin_reports(caps, q, channel, width):
    """Merge rows with the same cap and original Q/b bucket. Restore group multiplicities in S3, S5, and S6."""
    if width < 0 or not np.isfinite(width):
        raise ValueError("Invalid report bin width")
    scale = channel.scale(caps)
    label = q
    if width > 0:
        standardized = np.divide(q, scale, out=np.zeros_like(q), where=scale > 0)
        label = np.floor(standardized / width)
    order = np.lexsort((label, caps))
    c, y, label = caps[order], q[order], label[order]
    first = np.r_[0, 1 + np.flatnonzero((c[1:] != c[:-1]) | (label[1:] != label[:-1]))]
    counts = np.diff(np.r_[first, len(q)])
    means = np.add.reduceat(y / np.repeat(counts, counts), first)
    return c[first], means, counts


def _closure_grid(degree, edges, level, bands=()):
    """Construct positive-weight integer quadrature for the original atoms at 0 and 1 and the 63 uniform closure-rate cells.

    For small capacities, retain every integer T. For large capacities, subdivide each cell and use a geometric grid to refine
    the low-participation region. Partition the actual continuous cells at midpoints of adjacent integer representative points so each
    original cell has probability mass 1; approximate only its within-cell integral rather than replacing the whole cell with one value.
    """
    w = degree * (degree - 1) // 2
    if w <= 256:
        keys = np.arange(w + 1, dtype=np.int64)
    else:
        linear = edges[:-1, None] + np.diff(edges)[:, None] * np.arange(4 * level) / (4 * level)
        geometric = np.exp(np.arange(np.log(32.), np.log(w), .085 / level))
        # High-signal reports from high-degree nodes can form narrow peaks between two global grid points. The local grid
        # is used for both S3 likelihood integration and the posterior, with identical probability masses and representative points.
        local = []
        for unit, left, right in bands:
            step = max(1, unit // level)
            local.append(np.arange(left, right + 1, step, dtype=np.int64))
            local.append(np.asarray([right], dtype=np.int64))
        global_keys = np.rint(np.r_[np.arange(65), w * linear.ravel(), geometric, w]).astype(np.int64)
        keys = np.unique(np.concatenate([global_keys, *local]))
    if not w:
        return keys, csr_matrix(np.ones((1, len(edges) + 1)))
    # Each representative point corresponds to an original T interval; integrate the intersection length with each closure-rate cell exactly.
    boundaries = np.r_[0., (keys[:-1].astype(float) + keys[1:]) / (2. * w), 1.]
    split = np.unique(np.r_[boundaries, edges])
    mid = (split[:-1] + split[1:]) / 2
    rows = np.searchsorted(boundaries[1:], mid)
    cells = np.searchsorted(edges[1:], mid)
    masses = np.diff(split) / np.diff(edges)[cells]
    basis = csr_matrix((np.r_[1., masses, 1.],
                        (np.r_[0, rows, len(keys) - 1], np.r_[0, cells + 1, len(edges)])),
                       shape=(len(keys), len(edges) + 1))
    if not np.allclose(np.asarray(basis.sum(axis=0)), 1., rtol=0., atol=1e-11):
        raise ArithmeticError("Closure quadrature lost probability mass")
    return keys, basis


def _operator(channel, degree, cap, keys, reports, weights, *, adjoint=False):
    """Apply the forward or transpose product of the same K. Block along both axes to avoid allocating an n-by-Tmax array."""
    if cap <= 1:
        if np.any(reports != 0):
            raise ValueError("Nonzero deterministic sampled report")
        shape = len(keys) if adjoint else len(reports)
        return np.broadcast_to(np.asarray(weights.sum(axis=0)).ravel(),
                               (shape, weights.shape[1])).copy()
    out = np.zeros((len(keys) if adjoint else len(reports), weights.shape[1]))
    for j in range(0, len(keys), 1024):
        part = slice(j, min(j + 1024, len(keys)))
        step = max(1, 8192 // (part.stop - j))
        for i in range(0, len(reports), step):
            block = slice(i, min(i + step, len(reports)))
            kernel = channel.density(degree, cap, keys[part], reports[block])
            if adjoint:
                out[part] += kernel.T @ weights[block]
            else:
                out[block] += kernel @ weights[part]
    return out


def _report_bands(channel, degree, active, q, global_keys):
    """Refine near narrow peaks revealed by released Q; merge overlapping intervals first to avoid expanding a separate grid for each node."""
    w = degree * (degree - 1) // 2
    intervals = {}
    for cap, ids, _ in active:
        h = min(degree, cap)
        # E[Q|T] = slope*T; the Laplace scale gives the report-peak width in original T units.
        slope = channel.gap * channel.s * (h * (h - 1) / 2) / w
        width = float(channel.scale(cap)) / slope
        centers = np.clip(q[ids] / slope, 0., w)
        idx = np.clip(np.searchsorted(global_keys, centers), 1, len(global_keys) - 1)
        coarse = global_keys[idx] - global_keys[idx - 1] > width / 2
        if not coarse.any():
            continue
        unit = 1 << max(0, int(np.floor(np.log2(max(1., width)))))
        locations = np.unique(np.floor(centers[coarse] / unit).astype(np.int64))
        # Cover at least 48 Laplace scales on each side. The actual kernel also includes RR and sampling variance;
        # the original full-domain grid still integrates the remaining region, without truncating tails or omitting mass.
        radius = int(np.ceil(48 * width / unit)) + 1
        group = intervals.setdefault(unit, [])
        for location in locations:
            left = max(0, (int(location) - radius) * unit)
            right = min(w, (int(location) + radius + 1) * unit)
            group.append((left, right))
    bands = []
    for unit, group in intervals.items():
        ordered = sorted(group)
        left, right = ordered[0]
        for lo, hi in ordered[1:]:
            if lo <= right:
                right = max(right, hi)
            else:
                bands.append((unit, left, right))
                left, right = lo, hi
        bands.append((unit, left, right))
    return bands


def _checked_grid(channel, degree, entries, q, edges):
    """Compare two integration resolutions using a public accuracy rule, then use the selected grid in every phase for that degree."""
    grid = _closure_grid(degree, edges, 1)
    active = [entry for entry in entries if entry[0] > 1]
    if degree * (degree - 1) // 2 <= 256 or not active:
        return grid, dict(level=1, probe_relative_change=0., probes=0)
    bands = _report_bands(channel, degree, active, q, grid[0])
    if bands:
        grid = _closure_grid(degree, edges, 1, bands)
    # Check groups covering typical and extreme cap and Q values. This is a refinement diagnostic, not a full-domain error bound.
    positions = np.unique(np.linspace(0, len(active) - 1, min(5, len(active))).astype(int))
    probes = []
    for idx in positions:
        cap, ids, _ = active[idx]
        # Probes must be actual reports; an interpolation between distant narrow peaks is not an observation.
        values = np.quantile(q[ids], np.linspace(0., 1., min(7, len(ids))), method='nearest')
        probes.append((cap, np.unique(values)))
    def measure(pair):
        keys, basis = pair
        return np.vstack([_operator(channel, degree, cap, keys, ys, basis)
                          for cap, ys in probes])
    previous = measure(grid)
    for level in (2, 4, 8, 16):
        grid = _closure_grid(degree, edges, level, bands)
        current = measure(grid)
        # Observation likelihoods for both endpoint atoms are computed directly; do not let their high densities conceal errors in neighboring cells.
        cells = current[:, 1:-1]
        error = np.max(np.abs(previous[:, 1:-1] - cells) /
                       np.maximum(cells.max(axis=1, keepdims=True), np.finfo(float).tiny))
        if error <= .01:
            return grid, dict(level=level, probe_relative_change=float(error),
                              probes=len(current), refined_report_bands=len(bands))
        previous = current
    raise ArithmeticError(f"F1 sampling quadrature did not resolve degree {degree}: change {error:g}")


def _fit_f1(channel, groups, by_degree, q, counts, support, pi, workers):
    edges = closure_edges(CLOSURE_CELLS)
    matrix = np.zeros((len(q), CLOSURE_CELLS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def make_grid(d):
            return _checked_grid(channel, int(d), by_degree.get(int(d), []), q, edges)
        checked = list(pool.map(make_grid, support))
        grids = {int(d): result[0] for d, result in zip(support, checked)}

        def integrate_group(group):
            cap, ids = group['cap'], group['nodes']
            if cap <= 1:
                matrix[ids] = group['posterior'].sum()
                return
            total = np.zeros((len(ids), CLOSURE_CELLS))
            for degree, weight in zip(group['degree'], group['posterior']):
                d = int(degree)
                keys, cell = grids[d]
                total += weight * _operator(channel, d, cap, keys, q[ids], cell)
            matrix[ids] = total
        for _ in pool.map(integrate_group, groups):
            pass
    maxima = matrix.max(axis=1)
    if np.any(maxima <= 0) or not np.all(np.isfinite(matrix)):
        raise ArithmeticError("F1 sampled report probability is not resolved")
    # optimize_prior scales rows in place; multiply normalizer by the original maxima to restore its scale.
    theta, _, fit = optimize_prior(matrix, counts, np.ones(CLOSURE_CELLS) / CLOSURE_CELLS,
                                   tolerance=CLOSURE_TOLERANCE)
    if not fit['converged']:
        raise ArithmeticError("F1 sampled optimizer did not converge")
    normalizer = (matrix @ theta) * maxima
    del matrix
    n = counts.sum()

    def condition_degree(d):
        d = int(d)
        keys, cell = grids[d]
        prior_t = cell @ theta
        evidence = np.zeros(len(keys))
        for cap, ids, weight in by_degree.get(d, []):
            inverse = (weight * counts[ids] / n / normalizer[ids])[:, None]
            evidence += _operator(channel, d, cap, keys, q[ids], inverse, adjoint=True)[:, 0]
        mass = evidence * prior_t
        total = float(mass.sum())
        return keys, mass / total if total > 0 else prior_t, total
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(condition_degree, support))
    keys, probabilities, native = zip(*rows)
    if not np.isclose(sum(native), 1., atol=1e-8, rtol=0.):
        raise ArithmeticError("F1 sampled forward/adjoint population mismatch")
    f1 = ConditionalDistribution("F1", support, keys, probabilities, np.asarray(native))
    fit['quadrature'] = {str(d): dict(result[1], support_points=len(result[0][0]))
                         for d, result in zip(support, checked)}
    return f1, fit


def _joint_posterior(channel, by_degree, q, counts, f1, sources, boundaries, pi, workers, directory):
    """The S5 fit and S6 adjoint share exactly the same kernels, representative reports, and original integer T support."""
    support = f1.support
    gate, _ = degree_gate_basis(support, pi)
    bases = np.zeros((len(support), 12))
    directions = np.zeros((len(support), 12, 16))
    predictive = np.zeros(len(q))
    delta = np.zeros((16, len(q)))
    family = dict(sources=sources, support=support, pi=pi, boundaries=boundaries, gate_basis=gate)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for j, degree in enumerate(support):
            d = int(degree)
            keys, shapes, bases[j], directions[j], _ = stable_basis_joint.aligned_row(
                sources, j, boundaries, gate[j])
            def forward(entry):
                cap, ids, weight = entry
                projected = _operator(channel, d, cap, keys, q[ids], shapes.T)
                predictive[ids] += weight * (projected @ bases[j])
                delta[:, ids] += weight * (projected @ directions[j]).T
            for _ in pool.map(forward, by_degree.get(d, [])):
                pass
        constraints, rhs = coefficient_domain(bases, directions)
        family.update(predictive_table=predictive, delta_tables=delta, A=constraints, b=rhs, counts=counts)
        if np.any(predictive <= 0) or not np.isfinite(predictive).all():
            raise ArithmeticError("Sampled S5 forward probabilities invalid")
        coords, norm, fit = fit_joint7(family, directory=directory)
        inverse = counts[:, None] / counts.sum() / norm
        hist = {}
        native = []
        for j, degree in enumerate(support):
            d = int(degree)
            keys, probabilities, _ = prior_profiles(family, j, coords[None, :])
            def adjoint(entry):
                cap, ids, weight = entry
                return _operator(channel, d, cap, keys, q[ids], weight * inverse[ids], adjoint=True)[:, 0]
            evidence = np.zeros(len(keys))
            for value in pool.map(adjoint, by_degree.get(d, [])):
                evidence += value
            mass = probabilities[:, 0] * evidence
            total = float(mass.sum())
            native.append(total)
            row = mass / total if total > 0 else probabilities[:, 0]
            for key, value in zip(keys, counts.sum() * pi[j] * row):
                if value:
                    hist[int(key)] = hist.get(int(key), 0.) + float(value)
    if abs(sum(native) - 1) > 1e-8 or abs(sum(hist.values()) / counts.sum() - 1) > 1e-9:
        raise ArithmeticError("Sampled S5/S6 forward and adjoint disagree")
    keys = np.asarray(sorted(hist), np.int64)
    return (keys, np.asarray([hist[int(k)] for k in keys])), fit


def recover(caps, noisy, n, allocation, k, config, directory):
    """Map released cap and doubly debiased noisy_T to the original graph's T histogram. The caller isolates directories by identity."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    caps = np.asarray(caps)
    noisy = np.asarray(noisy, dtype=float)
    if (caps.shape != (n,) or noisy.shape != (n,) or caps.dtype.kind not in "iu" or
            np.any(caps < 0) or np.any(caps >= n) or not np.isfinite(noisy).all() or
            np.any(noisy[caps <= 1] != 0)):
        raise ValueError("Invalid sampled cap/report input")
    caps = caps.astype(np.int64, copy=False)
    ed, er, el = (float(allocation[name]) for name in ('eps_deg', 'eps_rr', 'eps_rel'))
    channel = SampledChannel(n, k, er, el)
    s = k / (n + n % 2 - 1)
    raw_q = noisy * (s * channel.gap)
    c, q, counts = _bin_reports(caps, raw_q, channel, config.report_bin_width)
    start = time.perf_counter()
    with threadpool_limits(limits=config.blas_threads):
        if config.degree_grid_size:
            prior, degree_fit = fit_degree_grid(caps, n - 1, ed, config.degree_grid_size,
                                                tolerance=config.degree_grid_tolerance)
        else:
            prior, degree_fit = fit_cold(caps, n - 1, ed, tolerance=DEGREE_TOLERANCE)
        prior, _ = canonicalize_tail(prior, ed)
        support = np.flatnonzero(prior > 0)
        pi = prior[support]
        groups = prepare_groups(c, prior, ed)
        by_degree = groups_by_degree(groups)
        f1, f1_fit = _fit_f1(channel, groups, by_degree, q, counts, support, pi, config.workers)
        histogram = f1.population_histogram(pi, n)
        positive = np.asarray(sorted(key for key, value in histogram.items() if key > 0 and value > 0))
        boundaries = (0, 0)
        if len(positive):
            cdf = np.cumsum([histogram[key] for key in positive])
            boundaries = tuple(int(positive[np.searchsorted(cdf, x * cdf[-1])]) for x in POSITIVE_QUANTILES)
        structures, _ = build_structural_joint(caps, prior, ed, float(noisy.mean()), directory, config)
        raw, joint_fit = _joint_posterior(channel, by_degree, q, counts, f1, (f1, *structures),
                                          boundaries, pi, config.workers, directory)
        final = raw
        guard = dict(enabled=config.safeguard)
        if config.safeguard:
            _, reference, _, cap_law = capacity_reference(noisy, prior, n - 1)
            if capacity_violation(raw, cap_law) > 2e-9:
                raise ArithmeticError("Unadjusted sampled posterior violates degree capacity")
            final, guard_stats = safeguard(raw, reference, tau=config.safeguard_tau, D=n - 1)
            guard.update(guard_stats)
            if capacity_violation(final, cap_law) > 2e-9:
                raise ArithmeticError("Sampled safeguard violates degree capacity")
    atomic_npz(directory / 'prediction_raw.npz', keys=raw[0], counts=raw[1])
    json_write(directory / 'sampled_fit.json', dict(
        degree_fit=degree_fit, f1=f1_fit, joint=joint_fit, safeguard=guard, n=n,
        k=k, s_effective=s, represented_nodes=int(counts.sum()), representative_rows=len(q),
        report_bin_width_in_actual_noise_units=config.report_bin_width,
        max_omitted_degree_mass=max(g['omitted_numerical'] for g in groups),
        posterior_updates=1, seconds=time.perf_counter() - start,
        closure_integration="positive integer quadrature of all 65 components, adaptive refinement probes",
        small_capacity_exact_integer_integration_limit=256, large_capacity_probe_threshold=.01,
        original_fast_triangle_compression_used=False,
        true_graph_or_truth_used=False))
    path = directory / 'prediction.npz'
    atomic_npz(path, keys=final[0], counts=final[1])
    return path
