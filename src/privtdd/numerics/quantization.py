"""Optional numerical approximations: a private-cap-driven degree grid and same-cap report bins retaining node counts.

Accept only released cap/Q, without reading the original graph, truth or baselines. This is not a new privacy mechanism.
The degree grid restricts the fit space; report binning approximates integration. Neither guarantees final GDD risk.
"""
import numpy as np
from scipy.optimize import minimize, brentq
from privtdd.config import CAP_OFFSET, DEGREE_GRID_COVERAGE_RADIUS
from privtdd.degree.channel import log_cap_probability
from privtdd.degree.operator import CapDegreeOperator
from privtdd.numerics.likelihood import mixture_objective


def refine_grid_weights(channel, frequency, initial, tolerance, *, max_iterations=600, checkpoint=None):
    """[S2 numerical recovery for the same objective] Feasible Newton steps on a fixed grid, preserving the original simplex and gap.
    
    channel/frequency: Row-scaled cap channel and exact node frequencies; initial: current weights of this solve.
    tolerance: Original grid-gap requirement; max_iterations: recovery limit 600; checkpoint: progress callback for the same solve.
    With Hessian-diagonal preconditioning, damping of 1e-8 only selects directions and adds no statistical penalty.
    Use 35 backtracking steps and Armijo coefficient 1e-4; if no Newton step is accepted, use pairwise exact line search on the original objective.
    """
    from scipy.linalg import solve
    prior = np.asarray(initial, float).copy()
    for iteration in range(max_iterations+1):
        mixture = channel@prior
        value, gradient = mixture_objective(prior, channel, frequency)
        gap = max(0., float(prior@gradient-gradient.min()))
        if checkpoint is not None and (iteration % 10 == 0 or gap <= tolerance):
            checkpoint(iteration, gap, value)
        if gap <= tolerance:
            return prior, dict(iterations=iteration, gap=gap, objective_unchanged=True)
        if iteration == max_iterations:
            break
        best = int(np.argmin(gradient))
        active = np.union1d(np.flatnonzero(prior > 0), [best])
        B = (channel[:, active]/mixture[:, None])*np.sqrt(frequency[:, None])
        H = B.T@B
        diagonal = np.diag(H)
        scale = 1/np.sqrt(np.maximum(diagonal, max(diagonal.max()*1e-12, np.finfo(float).tiny)))
        H *= scale[:, None]*scale[None, :]
        H.flat[::len(active)+1] += 1e-8
        rhs = np.column_stack(((float(prior@gradient)-gradient[active])*scale, scale))
        uv = solve(H, rhs, assume_a='pos')
        direction = np.zeros_like(prior)
        direction[active] = scale*(uv[:, 0]-(scale@uv[:, 0])/(scale@uv[:, 1])*uv[:, 1])
        direction[int(np.argmax(prior))] -= direction.sum()
        derivative = float(gradient@direction)
        accepted = False
        if derivative < 0 and not np.any((prior == 0)&(direction < 0)):
            neg = np.flatnonzero(direction < 0)
            ratios = prior[neg]/-direction[neg]
            maximum = min(1., float(ratios.min()))
            step = maximum
            for backtrack in range(35):
                candidate = prior+step*direction
                if backtrack == 0 and maximum < 1:
                    candidate[neg[ratios <= maximum*(1+4*np.finfo(float).eps)]] = 0.
                candidate = np.maximum(candidate, 0.); candidate /= candidate.sum()
                change = (channel@(candidate-prior))/mixture
                improvement = np.inf if np.any(change <= -1) else -float(frequency@np.log1p(change))
                if step > 0 and improvement <= 1e-4*step*derivative+1e-18:
                    prior = candidate; accepted = True; break
                step *= .5
        if not accepted:
            active = np.flatnonzero(prior > 0)
            worst = int(active[np.argmax(prior[active]*(gradient[active]-gradient[best]))])
            direction = channel[:, best]-channel[:, worst]
            maximum = prior[worst]
            def slope(step):
                """[S2 numerical utility] Negative derivative of the same grid objective along a pairwise transfer direction."""
                den = mixture+step*direction
                return -np.inf if np.any(den <= 0) else float(frequency@(direction/den))
            step = maximum if slope(maximum) >= 0 else brentq(slope, 0., maximum,
                xtol=np.finfo(float).tiny, rtol=4*np.finfo(float).eps, maxiter=200)
            prior[best] += step; prior[worst] -= step
    raise ArithmeticError(f'Grid Newton recovery did not meet unchanged tolerance: gap={gap:g}')


def degree_grid(caps, public_cap, grid_size):
    """[S2 numerical approximation] Choose integer degree candidates from a public coverage grid and private-cap quantiles.
    
    caps: Released integer caps for this experiment; public_cap: public D; grid_size: maximum candidate count, >=68.
    Always retain 0..64, max(0,D-L) and D; L=CAP_OFFSET, default 0 in the main path.
    Roughly half the remaining slots use a public log grid and half use empirical quantiles of clip(cap-L,0,D).
    Merge duplicates without adding random points; 64 protects low-degree integer resolution and is not chosen using truth.
    Use slots freed by deduplication for cap-driven coverage: each clip(cap-L) is at most 75 from its nearest candidate.
    Coverage radius DEGREE_GRID_COVERAGE_RADIUS is independent of L, retaining the validated numerical grid.
    Respect grid_size; explicitly fail if the available slots cannot provide coverage, without dropping reports or silently exceeding the limit.
    Return integer candidates in ascending order. Retain all integers for small D; the original entry point handles maximum-entropy tail canonicalization.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 68:
        raise ValueError('degree grid size must be an integer >=68')
    if public_cap+1 <= grid_size:
        return np.arange(public_cap+1, dtype=np.int64)
    fixed = np.unique(np.r_[np.arange(65), max(0, public_cap-CAP_OFFSET), public_cap])
    remaining = grid_size-len(fixed)
    public_slots = remaining//2
    public = np.rint(np.expm1(np.linspace(np.log1p(65), np.log1p(public_cap), public_slots))).astype(np.int64)
    # Quantiles depend on private caps, never on original degrees or GDD truth.
    private = np.rint(np.quantile(np.clip(np.asarray(caps)-CAP_OFFSET, 0, public_cap),
                                 np.linspace(0, 1, remaining-public_slots))).astype(np.int64)
    grid = np.unique(np.r_[fixed, public, private])
    centers = np.unique(np.clip(np.asarray(caps)-CAP_OFFSET, 0, public_cap)).astype(np.int64)
    while len(centers):
        right = np.searchsorted(grid, centers)
        distance = np.minimum(abs(centers-grid[np.maximum(right-1, 0)]),
                              abs(centers-grid[np.minimum(right, len(grid)-1)]))
        furthest = int(np.argmax(distance))
        if distance[furthest] <= DEGREE_GRID_COVERAGE_RADIUS:
            break
        if len(grid) == grid_size:
            raise ValueError('Degree grid budget insufficient for private-cap tail coverage; increase degree_grid_size')
        grid = np.sort(np.r_[grid, centers[furthest]])
    return grid


def fit_degree_grid(caps, public_cap, eps_degree, grid_size, *, tolerance=1e-7, checkpoint=None):
    """[S2 numerical approximation / bounded solve] Mixture MLE for the original cap likelihood on a fixed candidate degree grid.
    
    caps/public_cap/eps_degree: Same inputs as the original degree fit; grid_size: common grid-size limit;
    tolerance: Per-report convex gap target on the grid; checkpoint: optional (iteration,gap,nll) callback.
    Merge equal caps using exact node counts only. Initialize with 25 EM steps, then at most 300 SLSQP and 1000 pairwise
    line-search steps; raise if the target is unmet rather than silently accepting. The full integer-domain gap is only an approximation diagnostic,
    not a substitute for within-grid convergence. Blocks of 4096 only limit temporary array construction, without changing likelihood or groups.
    """
    support = degree_grid(caps, public_cap, grid_size)
    observed, counts = np.unique(caps, return_counts=True)
    frequency = counts/counts.sum()
    channel = np.empty((len(observed), len(support)))
    offsets = np.empty(len(observed))
    for start in range(0, len(observed), 4096):
        stop = min(start+4096, len(observed))
        logk = log_cap_probability(observed[start:stop, None], support[None, :], public_cap, eps_degree)
        offsets[start:stop] = logk.max(axis=1)
        channel[start:stop] = np.exp(logk-offsets[start:stop, None])
    prior = np.full(len(support), 1/len(support))
    for _ in range(25):
        prior *= channel.T@(frequency/(channel@prior))
        prior /= prior.sum()
    iterations = 0

    def progress(point):
        """[Progress] Forward only this iteration's objective/gap; do not start another fit."""
        nonlocal iterations
        iterations += 1
        if checkpoint is not None and iterations % 10 == 0:
            value, gradient = mixture_objective(point, channel, frequency)
            checkpoint(iterations, max(0., float(point@gradient-gradient.min())),
                       value-float(frequency@offsets))

    result = minimize(mixture_objective, prior, args=(channel, frequency), jac=True,
        method='SLSQP', bounds=[(0., 1.)]*len(prior),
        constraints=[dict(type='eq', fun=lambda x: float(x.sum()-1), jac=lambda x: np.ones_like(x))],
        callback=progress, options=dict(maxiter=300, ftol=1e-13))
    prior = np.maximum(result.x, 0.)
    prior /= prior.sum()
    for polish in range(1001):
        value, gradient = mixture_objective(prior, channel, frequency)
        gap = max(0., float(prior@gradient-gradient.min()))
        if checkpoint is not None and (polish % 25 == 0 or gap <= tolerance):
            checkpoint(iterations+polish, gap, value-float(frequency@offsets))
        if gap <= tolerance:
            break
        if polish == 1000:
            break
        best = int(np.argmin(gradient))
        active = np.flatnonzero(prior > 0)
        worst = int(active[np.argmax(gradient[active])])
        maximum = prior[worst]
        mixture = channel@prior
        direction = channel[:, best]-channel[:, worst]

        def slope(step):
            """[Numerical solve] One-dimensional derivative of the original grid likelihood along a pairwise mass transfer."""
            den = mixture+step*direction
            return -np.inf if np.any(den <= 0) else float(frequency@(direction/den))

        step = maximum if slope(maximum) >= 0 else brentq(slope, 0., maximum,
            xtol=np.finfo(float).tiny, rtol=4*np.finfo(float).eps, maxiter=200)
        if step <= 0 or best == worst:
            raise ArithmeticError('Binned degree MLE has no resolved descent step')
        prior[best] += step
        prior[worst] -= step
    recovery = None
    if gap > tolerance:
        def recovery_progress(iteration, gap, value):
            """[Progress] Continue numbering within the same grid solve and restore the negative log likelihood to its original scale."""
            if checkpoint is not None:
                checkpoint(iterations+polish+iteration, gap, value-float(frequency@offsets))
        prior, recovery = refine_grid_weights(channel, frequency, prior, tolerance, checkpoint=recovery_progress)
        value, gradient = mixture_objective(prior, channel, frequency)
        gap = max(0., float(prior@gradient-gradient.min()))
    full = np.zeros(public_cap+1)
    full[support] = prior
    # One O(D) score pass documents approximation loss in the *cap* objective.
    # It is neither a rerun nor a GDD accuracy certificate.
    operator = CapDegreeOperator(public_cap, eps_degree)
    predicted = operator.forward(full)
    if np.any(predicted[observed] <= 0):
        raise ArithmeticError('Degree grid cannot represent observed caps numerically')
    weights = np.zeros(public_cap+1)
    weights[observed] = frequency/predicted[observed]
    score = operator.adjoint(weights)
    stats = dict(approximation='restricted_integer_degree_grid_v1', grid=support.tolist(),
        requested_grid_size=grid_size, grid_size=len(support), active_states=int(np.count_nonzero(full)),
        original_domain_size=public_cap+1, iterations=int(result.nit), polish_steps=polish,
        restricted_gap=gap, full_integer_gap=max(0., float(score.max()-full@score)),
        full_integer_optimality_claimed=False, nll_per_cap=value-float(frequency@offsets),
        tolerance=tolerance, recovery=recovery, cap_counts_exact=True, truth_used=False, additional_epsilon=0.)
    return full, stats


def bin_reports(caps, report, eps_release, width):
    """[S3/S5/S6 numerical approximation] Represent reports sharing a cap and Q/b bin by their mean and node count.
    
    caps/report: Released caps and Q=RR_gap*noisy_T, each of length n; eps_release: original release budget;
    width: Dimensionless bin width; Q-bin width is width*b, with b=(cap-1)/eps_release.
    width=0 merges only exactly identical (cap,Q), for equivalence checks. Retain zero unchanged for cap<=1.
    Do not truncate tails, subsample or merge different caps; return representative cap/Q, positive integer counts and approximation statistics.
    For each original report, its distance to the bin mean satisfies |delta Q|/b <= width, apart from floating-point error.
    The Laplace convolution kernel's log density therefore has a corresponding input-perturbation bound; this does not imply a final MSE/EMD bound.
    """
    caps, report = np.asarray(caps, np.int64), np.asarray(report, float)
    if caps.ndim != 1 or report.shape != caps.shape or not len(caps) or np.any(caps < 0):
        raise ValueError('Aligned nonempty cap/report vectors required')
    if not np.isfinite(report).all() or np.any(report[caps <= 1] != 0):
        raise ValueError('Invalid released reports')
    if not np.isfinite(width) or width < 0 or not np.isfinite(eps_release) or eps_release <= 0:
        raise ValueError('Nonnegative finite width and positive release budget required')
    scale = np.maximum(caps-1, 0)/eps_release
    if width == 0:
        bins = report
    else:
        bins = np.zeros(len(report))
        np.divide(report, scale*width, out=bins, where=scale > 0)
        bins = np.floor(bins)
        if not np.isfinite(bins).all():
            raise ValueError('Report bin index overflow; use a larger width')
    order = np.lexsort((bins, caps))
    c, q, labels = caps[order], report[order], bins[order]
    first = np.r_[0, 1+np.flatnonzero((c[1:] != c[:-1]) | (labels[1:] != labels[:-1]))]
    counts = np.diff(np.r_[first, len(c)])
    # Summing per-node q/count avoids overflow of an otherwise finite mean.
    means = np.add.reduceat(q/np.repeat(counts, counts), first)
    representative = np.repeat(means, counts)
    error = np.zeros(len(q))
    np.divide(abs(q-representative), scale[order], out=error, where=scale[order] > 0)
    stats = dict(input_nodes=len(caps), representative_rows=len(first), width_in_noise_units=width,
        compression_factor=float(len(caps)/len(first)), max_displacement_in_noise_units=float(error.max()),
        mean_displacement_in_noise_units=float(error.mean()), population_preserved=int(counts.sum()),
        cap_values_preserved=True, truth_used=False, additional_epsilon=0.)
    return c[first], means, counts, stats
