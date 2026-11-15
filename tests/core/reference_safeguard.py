"""Small independent reference helpers retained from pre-integration validation."""
import time
import numpy as np
from privtdd.reporting import collection as reports

def law(keys, counts):
    k, c = np.asarray(keys), np.asarray(counts, float)
    if k.ndim != 1 or c.shape != k.shape or not len(k):
        raise ValueError('Nonempty aligned histogram required')
    if not np.isfinite(k).all() or np.any(k < 0) or np.any(k != np.floor(k)):
        raise ValueError('Nonnegative integer support required')
    if not np.isfinite(c).all() or np.any(c < 0) or c.sum() <= 0:
        raise ValueError('Finite nonnegative nonzero mass required')
    u, inv = np.unique(k.astype(np.int64), return_inverse=True)
    mass = np.bincount(inv, weights=c, minlength=len(u))
    return u, mass / mass.sum()

def shifted_degree_grid(caps, public_cap, grid_size, offset):
    """Original fast grid, actual cap shift, fixed numerical coverage radius 75."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 68:
        raise ValueError('degree grid size must be an integer >=68')
    if public_cap+1 <= grid_size:
        return np.arange(public_cap+1, dtype=np.int64)
    fixed = np.unique(np.r_[np.arange(65), max(0, public_cap-offset), public_cap])
    remaining = grid_size-len(fixed)
    public_slots = remaining//2
    public = np.rint(np.expm1(np.linspace(np.log1p(65), np.log1p(public_cap), public_slots))).astype(np.int64)
    centers_raw = np.clip(np.asarray(caps)-offset, 0, public_cap)
    private = np.rint(np.quantile(centers_raw, np.linspace(0, 1, remaining-public_slots))).astype(np.int64)
    grid = np.unique(np.r_[fixed, public, private])
    centers = np.unique(centers_raw).astype(np.int64)
    while len(centers):
        right = np.searchsorted(grid, centers)
        distance = np.minimum(abs(centers-grid[np.maximum(right-1, 0)]),
                              abs(centers-grid[np.minimum(right, len(grid)-1)]))
        furthest = int(np.argmax(distance))
        if distance[furthest] <= 75:
            break
        if len(grid) == grid_size:
            raise ValueError('Degree grid budget insufficient for 75-degree coverage; increase --degree-grid-size uniformly')
        grid = np.sort(np.r_[grid, centers[furthest]])
    return grid

def probability_cdf(mass):
    """Block prefix sums avoid long accumulation drift on million-point laws."""
    mass = np.asarray(mass, float)
    block = 1024
    if len(mass) <= block:
        cdf = np.cumsum(mass)
    else:
        full = len(mass)//block*block
        local = np.cumsum(mass[:full].reshape(-1, block), axis=1)
        totals = np.sum(mass[:full].reshape(-1, block), axis=1)
        offsets = np.r_[0., np.cumsum(totals)]
        local += offsets[:-1, None]
        cdf = np.r_[local.ravel(), offsets[-1]+np.cumsum(mass[full:])]
    cdf = np.minimum(cdf, 1.)
    cdf[-1] = 1.
    # Adjacent block sums can differ by a rounding ulp at a flat CDF segment.
    return np.maximum.accumulate(cdf)

def cdf_at(keys, mass, points):
    cdf = np.r_[0., probability_cdf(mass)]
    return cdf[np.searchsorted(keys, points, side='right')]

def w1(a, b):
    ak, ap = law(*a)
    bk, bp = law(*b)
    union = np.union1d(ak, bk)
    return float(np.abs(cdf_at(ak, ap, union)[:-1]-cdf_at(bk, bp, union)[:-1]) @ np.diff(union))

def capacity_reference(noisy, prior, D):
    """Sparse F_A=max(F_dir,F_cap), plus a feasible joint (degree,T) witness.

    For common u~Uniform(0,1), set d=Q_pi(u), T=min(Q_dir(u),W_d).
    The marginal CDF of T is exactly the maximum; no O(D^2) triangle array.
    """
    noisy, prior = np.asarray(noisy, float), np.asarray(prior, float)
    if noisy.ndim != 1 or not len(noisy) or not np.isfinite(noisy).all():
        raise ValueError('Finite nonempty released reports required')
    if prior.shape != (D+1,) or not np.isfinite(prior).all() or np.any(prior < 0) or abs(prior.sum()-1) > 2e-12:
        raise ValueError('Full degree simplex required')
    W = D*(D-1)//2
    direct_keys, direct_counts = np.unique(np.floor(np.clip(noisy, 0, W)+.5).astype(np.int64), return_counts=True)
    direct_mass = direct_counts/len(noisy)
    degree = np.flatnonzero(prior > 0)
    pi = prior[degree]/prior.sum()
    cap_keys, cap_mass = law(degree*(degree-1)//2, pi)
    fd = np.cumsum(direct_counts, dtype=np.int64)/len(noisy)
    fp = probability_cdf(pi)
    cuts = np.unique(np.r_[0., fd, fp])
    mass = np.diff(cuts)
    # Right endpoints implement quantiles on (cuts[i],cuts[i+1]].
    ids_d = np.searchsorted(fp, cuts[1:], side='left')
    ids_t = np.searchsorted(fd, cuts[1:], side='left')
    jd = degree[ids_d]
    jt = np.minimum(direct_keys[ids_t], jd*(jd-1)//2)
    keys, prob = law(jt, mass)
    union = np.union1d(direct_keys, cap_keys)
    expected = np.maximum(cdf_at(direct_keys, direct_mass, union), cdf_at(cap_keys, cap_mass, union))
    if np.max(abs(cdf_at(keys, prob, union)-expected)) > 2e-12:
        raise ArithmeticError('Reference CDF envelope mismatch')
    recovered = np.bincount(jd, weights=mass, minlength=D+1)
    if np.max(abs(recovered-prior)) > 2e-12 or np.any(jt > jd*(jd-1)//2):
        raise ArithmeticError('Reference lost degree marginal or capacity')
    return ((direct_keys, direct_counts.astype(float)), (keys, prob*len(noisy)),
            dict(degree=jd, triangles=jt, probability=mass), (cap_keys, cap_mass))

def safeguard(raw, reference, *, tau, D, delta=None):
    """One global mixture, retaining exact input arrays when it does not trigger."""
    if not np.isfinite(tau) or tau <= 0 or D < 2:
        raise ValueError('Positive finite tau and D>=2 required')
    delta = w1(raw, reference) if delta is None else float(delta)
    radius = tau*D
    alpha = 1. if delta <= radius else radius/delta
    start = time.perf_counter()
    if alpha == 1.:
        output = (raw[0].copy(), raw[1].copy())
    else:
        n = float(np.sum(raw[1]))
        keys = np.union1d(raw[0], reference[0])
        counts = np.zeros(len(keys))
        counts[np.searchsorted(keys, raw[0])] += alpha*raw[1]
        counts[np.searchsorted(keys, reference[0])] += (1-alpha)*reference[1]*n/np.sum(reference[1])
        output = keys, counts
    mix_seconds = time.perf_counter()-start
    observed = w1(output, reference)
    change = w1(output, raw)
    tolerance = 2e-9*max(1., delta, radius)
    if observed > radius+tolerance or abs(change-max(0., delta-radius)) > tolerance:
        raise ArithmeticError('Safeguard transport bound/minimal-change check failed')
    return output, dict(alpha=alpha, triggered=alpha < 1., Delta=delta,
        radius=radius, distance_ratio=delta/radius, W1_to_reference=observed,
        W1_change=change, mix_seconds=mix_seconds)

def load_hist(path):
    with np.load(path) as z:
        return z['keys'], z['counts']

def synthetic_graph(n, family):
    """Exact graph and analytic truth; no random graph selection or truth fitting."""
    if n < 8 or (family == 'tail' and n % 2):
        raise ValueError('Synthetic sizes >=8; tail sizes must be even')
    if family == 'tail':
        rows = [[v for v in range(n) if v != u and v != (u ^ 1)] for u in range(n)]
        truth = np.full(n, (n-2)*(n-4)//2, np.int64)
    else:
        rows = [sorted({(u+s) % n for s in (-2, -1, 1, 2)}) for u in range(n)]
        truth = np.full(n, 3, np.int64)
    ip, ix = reports.prepare_graph(rows, n, n-1)
    return ip, ix, truth
