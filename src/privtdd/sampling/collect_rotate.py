"""Post-projection rotated collector. The original graph and per-center exposure counts remain in the local simulator."""
from decimal import Decimal, ROUND_CEILING

import numpy as np
from numba import config as numba_config, get_num_threads, njit, prange, set_num_threads

from .design import _build_rotated_design, _derive_seed
from privtdd.reporting.primitives import (
    _edge_exists_sorted_csr, _pair_uniform_numba, lists_to_csr, project_neighbors, rr_params,
)


def matching_parameters(n, rate):
    """Add a dummy vertex for odd graph sizes; compute design probabilities from the actual integer round count. Full enumeration uses a separate historical branch."""
    mu = Decimal(str(rate))
    if n < 4 or not mu.is_finite() or not 0 < mu <= Decimal("0.5"):
        raise ValueError("The sampling frontend requires n>=4 and 0<rate<=0.5; the historical control provides 100% enumeration")
    ring = n + n % 2 - 1
    k = int((mu * ring).to_integral_value(rounding=ROUND_CEILING))
    return k, k / ring


@njit(cache=True, parallel=True)
def _count_block(ip, ix, pp, pi, side, positions, right, shifts, start, k, p, q, rr_seed):
    """Locate right-side neighbors by binary search within each left-side projected neighbor's cyclic window; query the original graph only for matches."""
    rows = len(pp) - 1
    m = len(right)
    hits = np.zeros(rows, np.int64)
    ones = np.zeros(rows, np.int64)
    for row in prange(rows):
        neighbors = pi[pp[row]:pp[row + 1]]
        right_pos = np.empty(len(neighbors), np.int64)
        size = 0
        for v in neighbors:
            if side[v] == 1:
                right_pos[size] = positions[v]
                size += 1
        ordered = np.sort(right_pos[:size])
        checked = 0
        positives = 0
        for u in neighbors:
            if side[u] != 0:
                continue
            first = (positions[u] + shifts[start + row]) % m
            # Split a window crossing m into two disjoint half-open intervals; k=m covers the entire right side exactly.
            for part in range(2):
                lo = first if part == 0 else 0
                hi = min(first + k, m) if part == 0 else max(0, first + k - m)
                left_idx = np.searchsorted(ordered, lo)
                right_idx = np.searchsorted(ordered, hi)
                checked += right_idx - left_idx
                for j in range(left_idx, right_idx):
                    v = right[ordered[j]]
                    prob = p if _edge_exists_sorted_csr(ip, ix, u, v) else q
                    # As in the original joint implementation, each unordered edge reuses one RR bit across centers and sampling rates.
                    if _pair_uniform_numba(rr_seed, u, v) < prob:
                        positives += 1
        hits[row] = checked
        ones[row] = positives
    return hits, ones


def collect(ip, ix, rate, seed, allocation, *, workers=8, block_size=256):
    """Accept validated undirected CSR and return two released vectors plus separate aggregate diagnostics.

    The recovery procedure may receive caps and noisy_triangle_report. visits counts the neighbor pairs actually queried;
    the candidates denominator counts only post-projection neighbor pairs, whereas active_fraction uses all nodes as its denominator.
    """
    n = len(ip) - 1
    k, design_rate = matching_parameters(n, rate)
    if workers < 1 or block_size < 1:
        raise ValueError("workers and block_size must be positive")
    ed, er, el = (allocation[name] for name in ("eps_deg", "eps_rr", "eps_rel"))
    if not all(np.isfinite(x) and x > 0 for x in (ed, er, el)):
        raise ValueError("All three privacy budgets must be finite and positive")
    p, q, gap = rr_params(er)
    rng = np.random.default_rng(seed)
    degree = np.diff(ip)
    # Keep the first RNG call identical to E1 with L=0. The rotated design uses an independent seed
    # so it does not change the subsequent random stream used for the final Laplace release.
    caps = np.clip(np.floor(degree + rng.laplace(0, 1 / ed, n)), 0, n - 1).astype(np.int64)
    design = _build_rotated_design(n, _derive_seed(seed, 0xC701))
    centered = np.empty(n)
    visits = active = 0
    previous_threads = get_num_threads()
    set_num_threads(min(workers, numba_config.NUMBA_NUM_THREADS))
    try:
        for start in range(0, n, block_size):
            stop = min(start + block_size, n)
            projected = [project_neighbors(ix[ip[v]:ip[v + 1]], int(caps[v]), v, seed + 0x12345)
                         for v in range(start, stop)]
            pp, pi = lists_to_csr(projected)
            a, b = _count_block(ip, ix, pp, pi, design['side'], design['pos'], design['right'],
                                design['center_shift'], start, k, p, q, seed + 0x5A17)
            visits += int(a.sum())
            active += int(np.count_nonzero(a))
            centered[start:stop] = b - q * a
    finally:
        set_num_threads(previous_threads)
    # With the preceding transcript fixed, replacing one projected neighbor removes and adds at most L terms each;
    # each term lies in [-q,1-q], so the centered count changes by at most L. Add noise in local count units first,
    # then debias using the actual s and RR gap; the output has the same triangle-count units as the main method.
    sensitivity = np.minimum(k, np.maximum(caps - 1, 0))
    noisy = (centered + rng.laplace(0, 1, n) * (sensitivity / el)) / (design_rate * gap)
    retained = np.minimum(caps, degree)
    candidates = int(np.sum(retained * (retained - 1) // 2, dtype=np.int64))
    diagnostics = dict(k=k, design_rate=design_rate, visits=visits, candidates=candidates,
                       realized_rate=visits / candidates if candidates else None,
                       active_fraction=active / n)
    return dict(caps=caps, noisy_triangle_report=noisy), diagnostics
