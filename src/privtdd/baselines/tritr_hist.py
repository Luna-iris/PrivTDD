"""Sparse TriTR-Hist adapter, with the original projection and random streams."""
import math
import time
import numpy as np
from numba import njit, prange
from .histogram import _split_budget, _noisy_hist
from .shared_rr import _compute_raw, randpair
PURE_METHODS=("tritr_gdd",)
IMPLEMENTATION="csr_projected_gdd_replacement_noise_v1"

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


@njit(parallel=True)
def project_csr(indptr, indices, caps, secret):
    """Same stable per-row priority projection as external_gdd._priority_project."""
    n = len(indptr) - 1
    lengths = np.minimum(np.diff(indptr), caps)
    ptr = np.empty(n + 1, dtype=np.int64)
    ptr[0] = 0
    ptr[1:] = np.cumsum(lengths)
    projected = np.empty(ptr[-1], dtype=np.int32)
    for u in prange(n):
        row = indices[indptr[u]:indptr[u + 1]]
        k = lengths[u]
        if k == len(row):
            projected[ptr[u]:ptr[u + 1]] = row
        elif k > 0:
            priorities = np.empty(len(row), dtype=np.float64)
            for j in range(len(row)):
                priorities[j] = randpair(u, row[j], n, secret)
            selected = np.argpartition(priorities, k - 1)[:k]
            projected[ptr[u]:ptr[u + 1]] = np.sort(row[selected])
    return ptr, projected


def run_csr(indptr, indices, deg_max, epsilon, method, seed, *, alpha=150.,
            mu=None, max_wedge_visits=100_000_000_000):
    """Pure relationship-budget methods; no true triangle counts are supplied."""
    if method not in PURE_METHODS:
        raise ValueError('Use a verified pure adaptation: ' + ', '.join(PURE_METHODS))
    eps0, eps1, eps2 = _split_budget(epsilon, method, True)
    n = len(indptr) - 1
    degree = np.diff(indptr)
    rng = np.random.default_rng(seed)
    if method == 'arrfull_gdd':
        eps0 = 0.
        eps2 = (epsilon - 3 * eps1) / 2
        caps = np.full(n, min(n - 1, deg_max), dtype=np.int64)
        p = 1 / (1 + math.exp(-eps1)) if mu is None else float(mu)
        q = p * math.exp(-eps1)
        if not 0 < q < p <= 1 / (1 + math.exp(-eps1)):
            raise ValueError('ARR mu violates the first-round LDP bound')
    else:
        noisy_degree = np.floor(alpha + np.maximum(0., degree + rng.laplace(0., 1 / eps0, size=n)))
        caps = np.clip(noisy_degree, 0, min(n - 1, deg_max)).astype(np.int64)
        p = 1 / (1 + math.exp(-eps1))
        q = 1 - p
    lengths = np.minimum(degree, caps)
    visits = int(np.sum(lengths * (lengths - 1) // 2))
    if visits > max_wedge_visits:
        raise RuntimeError(f'{visits:,} wedges exceed --max-wedge-visits={max_wedge_visits:,}')
    priority_secret = np.random.default_rng(seed + 1101).integers(0, 2**63, dtype=np.uint64)
    log(f'{method} eps={epsilon:g} seed={seed}: project, then {visits:,} shared RR wedges')
    if np.array_equal(lengths, degree):
        ptr, idx = indptr, indices
    else:
        ptr, idx = project_csr(indptr, indices, caps, priority_secret)
    rr_secret = rng.integers(0, 2**63, dtype=np.uint64)
    local = _compute_raw(ptr, idx, ptr, idx, n, rr_secret, p, q, method != 'arrfull_gdd')
    gap = p - q
    if method == 'arrfull_gdd':
        sensitivity = 2 * max(0, min(n - 1, deg_max) - 1)
        scales = np.full(n, sensitivity / eps2)
        private = (local + rng.laplace(0., scales, size=n)) / gap
    else:
        kappa = 1 / (-math.expm1(-eps1))
        # A projected neighbor replacement both removes and adds terms.
        scales = 2 * kappa * caps / eps2
        private = local + rng.laplace(0., scales, size=n)
    hist = _noisy_hist(private, deg_max * (deg_max - 1) // 2, n)
    diag = dict(implementation=IMPLEMENTATION, method=method, seed=seed,
                epsilon_relationship_total=epsilon, delta_relationship=0.,
                eps0=eps0, eps1=eps1, eps2=eps2,
                privacy_status='conservative_pure_relationship_adaptation',
                alpha=alpha, arr_mu=p if method == 'arrfull_gdd' else None,
                rr_p=p, rr_q=q, projected_nodes=int(np.sum(degree > caps)),
                actual_wedge_visits=visits, projection_caps_max=int(caps.max()),
                second_round_noise_scale_median=float(np.median(scales)),
                second_round_noise_scale_max=float(scales.max()),
                noise_scale_units='raw ARR score' if method == 'arrfull_gdd' else 'triangle participation',
                histogram_total=float(sum(hist.values())))
    return hist, diag
