"""Exact simple-graph triangle counts extracted from the research implementation."""
import numpy as np
from numba import njit

@njit(cache=True)
def contains(indptr, indices, u, v):
    lo, stop = indptr[u], indptr[u+1]
    hi = stop
    while lo < hi:
        mid = (lo+hi)//2
        if indices[mid] < v:
            lo = mid+1
        else:
            hi = mid
    return lo < stop and indices[lo] == v


@njit(cache=True, nogil=True)
def validate_and_triangles(indptr, indices):
    """Exact oriented intersection: visit every triangle once; count all vertices."""
    n = len(indptr)-1
    degree = np.diff(indptr)
    fp = np.zeros(n+1, dtype=np.int64)
    for u in range(n):
        for j in range(indptr[u], indptr[u+1]):
            v = indices[j]
            if v < 0 or v >= n or v == u or (j > indptr[u] and indices[j-1] >= v):
                raise ValueError('Invalid simple sorted adjacency')
            if not contains(indptr, indices, v, u):
                raise ValueError('Asymmetric undirected input')
            if degree[u] < degree[v] or (degree[u] == degree[v] and u < v):
                fp[u+1] += 1
    for u in range(n):
        fp[u+1] += fp[u]
    fi = np.empty(fp[-1], dtype=np.int32)
    for u in range(n):
        at = fp[u]
        for v in indices[indptr[u]:indptr[u+1]]:
            if degree[u] < degree[v] or (degree[u] == degree[v] and u < v):
                fi[at] = v
                at += 1
    counts = np.zeros(n, dtype=np.int64)
    for u in range(n):
        for v in fi[fp[u]:fp[u+1]]:
            a, b = fp[u], fp[v]
            while a < fp[u+1] and b < fp[v+1]:
                x, y = fi[a], fi[b]
                if x == y:
                    counts[u] += 1
                    counts[v] += 1
                    counts[x] += 1
                    a += 1
                    b += 1
                elif x < y:
                    a += 1
                else:
                    b += 1
    return counts
