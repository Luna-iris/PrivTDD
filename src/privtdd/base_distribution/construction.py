"""S3: fit a shared closure-rate mixture -> integrate node posteriors -> obtain F1 conditional on fixed degree."""
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import numpy as np
from privtdd.config import CLOSURE_CELLS, CLOSURE_TOLERANCE
from privtdd.base_distribution.channel import build_matrix, prepare_groups
from privtdd.base_distribution.optimizer import optimize_prior
from privtdd.io.memory import LazyRows, atomic_npz
from privtdd.base_distribution.degree_posterior import compress_integer_cdf, compress_noise_cdf
from privtdd.base_distribution.integral import FastDegreePosterior
from privtdd.base_distribution.integrated import IntegratedDegreePosterior
from privtdd.io.storage import StoredConditional, json_write


# [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); [Core S3 with blocking/parallel execution] Obtain R_F1(T|d) with fixed pi from the same reports.
def build_f1_joint(caps, report, scale, prior, p, eps_degree, eps_release, directory, config, *, report_counts=None):
    """[Core S3 with blocking/parallel execution] Obtain R_F1(T|d) with fixed pi from the same reports.
    
    caps: Private caps; report: Q=RR_gap*noisy_triangle_report; scale: Laplace scale before RR debiasing.
    prior: Canonical prior on all integer degrees; p: RR true-positive probability; eps_degree/eps_release: allocated budgets.
    directory: Cache whose identity is already bound by the caller; config: numerical and parallel settings.
    report_counts: Optional positive integer multiplicities for representative reports; None means one node per row.
    Weights enter the likelihood, posterior integration and population counts; bins must not be treated as new equally weighted nodes.
    Return a lazily loaded conditional distribution and fit statistics; do not read truth or rerun independent validation.
    """
    counts = np.ones(len(report)) if report_counts is None else np.asarray(report_counts, float)
    if counts.shape != report.shape or not np.isfinite(counts).all() or np.any(counts <= 0) or np.any(counts != np.floor(counts)):
        raise ValueError('Positive integer multiplicities aligned with reports required')
    n = int(counts.sum())
    support = np.flatnonzero(prior > 0)
    pi = prior[support]
    saved = directory/'F1_fit.npz'
    if saved.exists():
        with np.load(saved) as z:
            theta, R, edges = z['theta'], z['normalizer'], z['edges']
        fit = json.loads((directory/'F1_fit.json').read_text(encoding='utf-8'))
        groups = prepare_groups(caps, prior, eps_degree)
    else:
        A, omission, groups, edges, channel = build_matrix(report, caps, prior, p, eps_degree, eps_release,
                                                         grid_size=CLOSURE_CELLS, workers=config.workers, batch_degrees=config.batch_f1)
        maxima = A.max(axis=1)
        theta, _, fit = optimize_prior(A, counts, np.full(CLOSURE_CELLS, 1/CLOSURE_CELLS), tolerance=CLOSURE_TOLERANCE)
        if not fit['converged']:
            raise ArithmeticError('F1 optimizer did not converge')
        R = (A@theta)*maxima
        fit.update(channel=channel, average_missing_posterior_upper=float(counts@(omission/(R+omission))/n),
                   population_nodes=n, representative_rows=len(report))
        json_write(directory/'F1_fit.json', fit)
        atomic_npz(saved, theta=theta, normalizer=R, edges=edges, prior=prior)
        del A, maxima, omission
        gc.collect()
    rowdir = directory/'f1_rows'
    rowdir.mkdir(exist_ok=True)
    rows = LazyRows(groups, R, n)
    W = support*(support-1)//2
    budget = float(np.sqrt(pi[W > 64]*W[W > 64]).sum())
    from privtdd.base_distribution.projected_posterior import FastProjectedDegreePosterior
    constructor = FastProjectedDegreePosterior

    # [PrivTDD] S3 F1; memory and parallel execution; [Independent task / core posterior] Integrate one (d,W_d,pi_d) and compress under the specified transport budget.
    def recover_degree(arguments):
        """[Independent task / core posterior] Integrate one (d,W_d,pi_d) and compress under the specified transport budget."""
        d, capacity, weight = arguments
        path = rowdir/f'd{int(d)}.npz'
        if path.exists():
            with np.load(path) as z:
                return int(d), float(weight), z['keys'], z['probability'], float(z['native_mass']), float(z['compression_bound']), str(z['compression_mode'])
        ids, coefficient = rows.get(int(d))
        if report_counts is not None:
            coefficient = coefficient*counts[ids]
        row = constructor(int(d), report[ids] if len(ids) else np.array([0.]),
                          caps[ids] if len(ids) else np.array([0]),
                          scale[ids] if len(ids) else np.array([0.]),
                          coefficient if len(ids) else np.array([1.]), theta, edges, p,
                          numerical_W1_limit=config.numerical_W1_limit,batch_small_groups=config.batch_f1,
                          posterior_backend=config.posterior_backend)
        native = row.mass if len(ids) else 0.
        tolerance = config.compression_W1_budget*math.sqrt(capacity/weight)/budget if capacity > 64 else 1.
        if config.triangle_bin_width > 0:
            # Retain the original fast-path T resolution. The equivalent Laplace scale mapped from projected large capacity back to original T
            # is W/w*(cap-1)/(eps_release*gap), which for d>cap is no smaller than the scale used here.
            # Small capacities still use the declared hypergeometric working model for integration; no final GDD error bound is claimed.
            signal_step = config.triangle_bin_width*max(int(d)-1, 1)/(eps_release*(2*p-1))
            compressed = compress_noise_cdf(row.cdf, int(capacity), signal_step, fallback_tolerance=tolerance)
        else:
            compressed = compress_integer_cdf(row.cdf, int(capacity), tolerance)
        keys, mass = compressed['keys'], compressed['probability']
        row_mode = compressed.get('mode', 'adaptive_W1')
        atomic_npz(path, keys=keys, probability=mass, native_mass=native, compression_bound=compressed['bound'], compression_mode=row_mode)
        return int(d), float(weight), keys, mass, native, compressed['bound'], row_mode

    hist, native = {}, []
    compression_bound, total_keys = 0., 0
    compression_modes = {}
    # map returns in input order, so thread scheduling does not change degree accumulation order.
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        for d, weight, keys, mass, native_mass, row_bound, row_mode in pool.map(recover_degree, zip(support, W, pi)):
            compression_modes[row_mode] = compression_modes.get(row_mode, 0)+1
            compression_bound += weight*row_bound
            total_keys += len(keys)
            for k, v in zip(keys, n*weight*mass):
                if v:
                    hist[int(k)] = hist.get(int(k), 0.)+float(v)
            native.append(native_mass)
            if len(native) % 25 == 0:
                json_write(directory/'progress.json', dict(stage='F1_rows', completed=len(native), total=len(support)))
    keys = np.array(sorted(hist), np.int64)
    counts = np.array([hist[int(k)] for k in keys])
    native = np.asarray(native)
    if abs(counts.sum()/n-1) > 1e-9 or abs(native.sum()-1) > 1e-9:
        raise ArithmeticError('F1 population mass')
    atomic_npz(directory/'F1.npz', keys=keys, counts=counts, support=support, degree_mass=pi, native_degree_mass=native)
    fit.update(compression=dict(mode='noise_scaled_with_adaptive_guard' if config.triangle_bin_width else 'adaptive_W1',
        triangle_bin_width=config.triangle_bin_width, weighted_W1_bound=compression_bound,
        conditional_support_points=total_keys, rows_by_mode=compression_modes, max_noise_bins=4096,
        final_GDD_bound_claimed=False))
    return StoredConditional(rowdir, support, pi, keys, counts, native), fit
