"""S1: reporting simulator with the same mechanism and random streams as the current interactive experiments.

Return only released caps and noisy triangle reports to postprocessing; the original graph, B and W do not enter reconstruction.
Deterministic seeds support experiment reproduction; this centralized simulator is not a distributed client deployment.
"""
import numpy as np
from numba import get_num_threads, set_num_threads, config as numba_config
from privtdd.config import CAP_OFFSET, DEGREE_FRACTION, RR_FRACTION, RELEASE_FRACTION
from privtdd.reporting.primitives import rr_params, project_neighbors, lists_to_csr, _exact_local_wedge_reports_numba
from privtdd.reporting.primitives import _exact_local_wedge_reports_parallel_numba


# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); [Input adapter] Convert an adjacency sequence/dictionary or sorted CSR to simple undirected CSR.
def prepare_graph(adj_list, node_num, deg_max):
    """[Input adapter] Convert an adjacency sequence/dictionary or sorted CSR to simple undirected CSR.
    
    adj_list: Adjacency sequence of length n, dictionary mapping nodes to neighbors, or (indptr, indices).
    node_num: Public node count n, at least 4; deg_max: Public degree bound D, with 2<=D<n.
    CSR must have no self-loops or duplicates, with sorted rows; check the domain and symmetry without repairing the graph.
    """
    n, D = int(node_num), int(deg_max)
    if n != node_num or D != deg_max or n < 4 or not 2 <= D < n:
        raise ValueError('Require integer public n>=4 and 2<=D<n')
    if isinstance(adj_list, tuple) and len(adj_list) == 2:
        ip, ix = (np.asarray(a) for a in adj_list)
        if ip.dtype.kind not in 'iu' or ix.dtype.kind not in 'iu':
            raise ValueError('CSR arrays must be integer')
        ip, ix = ip.astype(np.int64, copy=False), ix.astype(np.int64, copy=False)
    else:
        rows = [sorted(adj_list.get(v, ())) if isinstance(adj_list, dict) else sorted(adj_list[v]) for v in range(n)]
        if any(any(int(x) != x for x in row) for row in rows):
            raise ValueError('Integer neighbor identifiers required')
        ip, ix = lists_to_csr(rows)
    if ip.shape != (n+1,) or ix.ndim != 1 or ip[0] != 0 or ip[-1] != len(ix) or np.any(np.diff(ip) < 0):
        raise ValueError('Invalid CSR shape/offsets')
    if np.any(np.diff(ip) > D) or np.any(ix < 0) or np.any(ix >= n):
        raise ValueError('Graph outside public bounded-degree domain')
    # Input validation is not a second graph reconstruction or a privacy query.
    from scipy.sparse import csr_matrix
    graph = csr_matrix((np.ones(len(ix), np.int8), ix, ip), shape=(n, n))
    if not graph.has_canonical_format or np.any(graph.diagonal()) or (graph != graph.T).nnz:
        raise ValueError('Expected sorted, duplicate-free, loop-free undirected CSR')
    return ip, ix


# [PrivTDD] S1 private reporting; core method or probability/structure primitive (not necessarily a separate contribution); [Core S1 / private mechanism] DP cap -> fixed-priority projection -> shared RR -> Laplace.
def collect_reports_joint(indptr, indices, public_cap, epsilon, seed=20260908, *, block_size=256, workers=1):
    """[Core S1 / private mechanism] DP cap -> fixed-priority projection -> shared RR -> Laplace.
    
    indptr/indices: CSR from prepare_graph; public_cap: Public D; epsilon: Total budget.
    seed: Experiment RNG seed; block_size: Implementation block size, retaining the random streams.
    workers: S1 thread count, default 1 for direct calls; the complete entry point passes PrivTDDConfig.workers.
    Respect the Numba runtime thread limit; parallelize neighbor positions while preserving original shared RR values and all neighbor pairs.
    Return caps, noisy_triangle_report and release_scale_before_RR_debias.
    Matches the original slow_ldp_audited.interactive_exact mechanism, without saving B/W to disk or computing baseline histograms.
    """
    if not np.isfinite(epsilon) or epsilon <= 0 or block_size < 1:
        raise ValueError('Positive epsilon and block_size required')
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError('Positive integer report workers required')
    n = len(indptr)-1
    ed, er, el = DEGREE_FRACTION*epsilon, RR_FRACTION*epsilon, RELEASE_FRACTION*epsilon
    rng = np.random.default_rng(seed)
    caps = np.clip(np.floor(np.diff(indptr)+rng.laplace(0, 1/ed, n)+CAP_OFFSET), 0, public_cap).astype(np.int64)
    p, q, gap = rr_params(er)
    if not .5 < p < 1:
        raise ValueError('RR probability must be numerically nondegenerate')
    centered = np.empty(n)
    effective_workers = min(int(workers), numba_config.NUMBA_NUM_THREADS)
    kernel = _exact_local_wedge_reports_parallel_numba if effective_workers > 1 else _exact_local_wedge_reports_numba
    previous_threads = get_num_threads() if effective_workers > 1 else None
    if previous_threads is not None:
        set_num_threads(effective_workers)
    try:
        for start in range(0, n, block_size):
            stop = min(n, start+block_size)
            projected = [project_neighbors(indices[indptr[v]:indptr[v+1]], int(caps[v]), v, seed+0x12345) for v in range(start, stop)]
            pp, pi = lists_to_csr(projected)
            values, wedges, _, _ = kernel(indptr, indices, pp, pi, p, q, seed+0x5A17)
            centered[start:stop] = values-q*wedges
    finally:
        if previous_threads is not None:
            set_num_threads(previous_threads)
    scale = np.maximum(caps-1, 0)/el
    noisy = (centered+rng.laplace(0, 1, n)*scale)/gap
    return dict(caps=caps, noisy_triangle_report=noisy, release_scale_before_RR_debias=scale,
                cap_offset=np.int64(CAP_OFFSET))
