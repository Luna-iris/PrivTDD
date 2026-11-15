"""Private reporting primitives for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from typing import List

from typing import Sequence

from typing import Tuple

import math

from numba import njit, prange

import numpy as np


# Source: gdd_main_table1_strong_privacy:253

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Derive p, q=1-p and gap=p-q from the RR budget.
def rr_params(eps: float) -> Tuple[float, float, float]:
    e = math.exp(float(eps))
    p = e / (1.0 + e)
    q = 1.0 / (1.0 + e)
    return p, q, p - q


# Source: gdd_main_table1_strong_privacy:279

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Generate deterministic public priorities for neighbor projection.
def public_priority_key(center: int, nbr: int, seed: int) -> int:
    """Stable public key used only for deterministic graph projection."""
    x = (
        (int(center) + 0x9E3779B97F4A7C15)
        ^ ((int(nbr) + 1) * 0xBF58476D1CE4E5B9)
        ^ int(seed)
    ) & ((1 << 64) - 1)
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    x ^= (x >> 31)
    return int(x)


# Source: gdd_main_table1_strong_privacy:294

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Retain all neighbors when within cap; otherwise retain cap neighbors by fixed priority.
def project_neighbors(nbrs: set, cap: int, center: int, seed: int) -> List[int]:
    if cap <= 0:
        return []
    if len(nbrs) <= cap:
        return sorted(int(x) for x in nbrs)
    return sorted(
        (int(x) for x in nbrs),
        key=lambda x: public_priority_key(center, x, seed),
    )[:cap]


# Source: gdd_main_table1_strong_privacy:352

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Convert an adjacency sequence to CSR arrays.
def lists_to_csr(rows: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rows)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for u in range(n):
        indptr[u + 1] = indptr[u] + len(rows[u])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    pos = 0
    for u in range(n):
        a = np.asarray(rows[u], dtype=np.int32)
        indices[pos:pos + len(a)] = a
        pos += len(a)
    return indptr, indices


# Source: gdd_main_table1_strong_privacy:366

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Numba integer-mixing primitive preserving the current experiments' shared RR random stream.
@njit(cache=True)
def _splitmix64_numba(x):
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & mask
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    z = z ^ (z >> np.uint64(31))
    return z


# Source: gdd_main_table1_strong_privacy:377

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Map an unordered node pair and seed to the same pseudorandom uniform value to implement shared RR.
@njit(cache=True)
def _pair_uniform_numba(seed, u, v):
    if u > v:
        t = u
        u = v
        v = t
    x = (
        np.uint64(seed)
        ^ (np.uint64(u + 1) * np.uint64(0xD6E8FEB86659FD93))
        ^ (np.uint64(v + 1) * np.uint64(0xA5A35625AA5A3563))
    )
    z = _splitmix64_numba(x)
    # 53-bit mantissa U[0,1)
    return float(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


# Source: gdd_main_table1_strong_privacy:393

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Binary-search a sorted CSR row for an edge; used only by the reporting simulator.
@njit(cache=True)
def _edge_exists_sorted_csr(indptr, indices, u, v):
    lo = int(indptr[u])
    hi = int(indptr[u + 1])
    while lo < hi:
        mid = (lo + hi) // 2
        x = int(indices[mid])
        if x < v:
            lo = mid + 1
        else:
            hi = mid
    return lo < int(indptr[u + 1]) and int(indices[lo]) == v


# Source: gdd_main_table1_strong_privacy:407

# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); Enumerate projected neighbor pairs and accumulate shared RR reports; B/W remain within collection only.
@njit(cache=True)
def _exact_local_wedge_reports_numba(
    adj_indptr,
    adj_indices,
    proj_indptr,
    proj_indices,
    p1,
    p0,
    edge_seed,
):
    n = len(proj_indptr) - 1
    B = np.zeros(n, dtype=np.float64)
    W = np.zeros(n, dtype=np.float64)
    checks = np.int64(0)
    true_checks = np.int64(0)
    for center in range(n):
        a0 = int(proj_indptr[center])
        a1 = int(proj_indptr[center + 1])
        d = a1 - a0
        W[center] = d * (d - 1) / 2.0
        total = 0.0
        for ia in range(a0, a1):
            u = int(proj_indices[ia])
            for ib in range(ia + 1, a1):
                v = int(proj_indices[ib])
                checks += 1
                edge = _edge_exists_sorted_csr(
                    adj_indptr, adj_indices, u, v
                )
                if edge:
                    true_checks += 1
                prob = p1 if edge else p0
                if _pair_uniform_numba(edge_seed, u, v) < prob:
                    total += 1.0
        B[center] = total
    return B, W, checks, true_checks


@njit(cache=True, parallel=True)
def _exact_local_wedge_reports_parallel_numba(adj_indptr, adj_indices, proj_indptr, proj_indices, p1, p0, edge_seed):
    """[S1 equivalent implementation] Parallelize neighbor-position tasks for full shared-RR counting without subsampling or changing random values.
    
    adj_indptr/adj_indices: Read-only original-graph CSR; proj_indptr/proj_indices: Projected-neighbor CSR for the current block.
    p1/p0: Original mechanism's report probabilities for present/absent edges; edge_seed: Original shared seed for unordered node pairs.
    Each task handles one neighbor and its successors; interleave high/low indices to balance work for large-degree nodes.
    partial: Local integer count for each neighbor position (float64 accommodates the original count domain); aggregate B by root.
    Return B/W and check counts matching the serial kernel; collect_reports_joint limits the thread count.
    """
    partial = np.zeros(len(proj_indices), dtype=np.float64)
    checks = np.int64(0)
    true_checks = np.int64(0)
    for slot in prange(len(proj_indices)):
        center = np.searchsorted(proj_indptr, slot, side='right')-1
        start = proj_indptr[center]
        stop = proj_indptr[center+1]
        degree = stop-start
        local = slot-start
        offset = local//2 if local % 2 == 0 else degree-1-local//2
        ia = start+offset
        u = int(proj_indices[ia])
        total = 0.
        checks += stop-ia-1
        for ib in range(ia+1, stop):
            v = int(proj_indices[ib])
            edge = _edge_exists_sorted_csr(adj_indptr, adj_indices, u, v)
            if edge:
                true_checks += 1
            if _pair_uniform_numba(edge_seed, u, v) < (p1 if edge else p0):
                total += 1.
        partial[slot] = total
    n = len(proj_indptr)-1
    B = np.zeros(n, dtype=np.float64)
    W = np.zeros(n, dtype=np.float64)
    for center in range(n):
        start, stop = proj_indptr[center], proj_indptr[center+1]
        degree = stop-start
        W[center] = degree*(degree-1)/2.
        B[center] = partial[start:stop].sum()
    return B, W, checks, true_checks
