"""Compact simple graph for the unmodified LDPGraph synthesis control flow.

Only degree(), has_edge(), add_edge(), and add_nodes_from() are observed by
that control flow. No sampling, floating-point operations, or RNG consumption
are changed. Edges are stored once as uint64 keys, not Python dictionaries.
"""
from pathlib import Path
import gc
from types import SimpleNamespace

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _slot(table, key):
    x = np.uint64(key)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    mask = len(table) - 1
    pos = np.int64(x & np.uint64(mask))
    while table[pos] != 0 and table[pos] != key:
        pos = (pos + 1) & mask
    return pos


@njit(cache=True)
def _contains(table, key):
    return table[_slot(table, key)] != 0


@njit(cache=True)
def _insert(table, key):
    pos = _slot(table, key)
    if table[pos] != 0:
        return False
    table[pos] = key
    return True


@njit(cache=True)
def _insert_edge(table, degrees, key, u, v):
    if not _insert(table, key):
        return False
    degrees[u] += 1
    degrees[v] += 1
    return True


@njit(cache=True)
def _rehash(old, new):
    for key in old:
        if key:
            _insert(new, key)


@njit(cache=True)
def _fill_csr(table, n, pointers, indices):
    cursor = pointers[:-1].copy()
    for key in table:
        if key:
            value = np.int64(key - np.uint64(1))
            u, v = value // n, value % n
            indices[cursor[u]] = v
            cursor[u] += 1
            indices[cursor[v]] = u
            cursor[v] += 1


@njit(cache=True, parallel=True)
def _sort_rows(pointers, indices):
    for u in prange(len(pointers) - 1):
        indices[pointers[u]:pointers[u + 1]].sort()


class CompactGraph:
    def __init__(self, n, expected_edges=0, scratch=None):
        if not 0 <= n <= np.iinfo(np.int32).max:
            raise ValueError('Compact graph requires an int32 node domain')
        self.n = int(n)
        self.degrees = np.zeros(n, dtype=np.int64)
        self.edge_count = 0
        self.scratch = Path(scratch) if scratch is not None else None
        self.generation = 0
        self.paths = []
        desired = max(16, int(expected_edges / .65) + 1)
        capacity = 1 << (desired - 1).bit_length()
        self.table = self._allocate(capacity)

    def _allocate(self, capacity):
        if self.scratch is None:
            return np.zeros(capacity, dtype=np.uint64)
        self.scratch.mkdir(parents=True, exist_ok=True)
        path = self.scratch / f'edge_table_{self.generation}.bin'
        self.generation += 1
        self.paths.append(path)
        # Fresh mappings are zero-filled by the OS. Avoid touching every page.
        return np.memmap(path, dtype=np.uint64, mode='w+', shape=(capacity,))

    def add_nodes_from(self, nodes):
        # The source declares the complete public domain once, in order.
        if not isinstance(nodes, range) or nodes != range(self.n):
            raise ValueError('Expected range(n) from source synthesis')

    def degree(self, node):
        # Match NetworkX's Python int result in source arithmetic.
        return int(self.degrees[node])

    def _key(self, u, v):
        u, v = int(u), int(v)
        if not (0 <= u < self.n and 0 <= v < self.n) or u == v:
            raise ValueError('Expected a non-self edge in the public node domain')
        if u > v:
            u, v = v, u
        return np.uint64(u * self.n + v + 1)

    def has_edge(self, u, v):
        if u == v:
            return False
        return bool(_contains(self.table, self._key(u, v)))

    def add_edge(self, u, v):
        key = self._key(u, v)
        if self.edge_count + 1 > int(.70 * len(self.table)):
            if _contains(self.table, key):
                return
            old = self.table
            self.table = self._allocate(2 * len(old))
            _rehash(old, self.table)
            if isinstance(old, np.memmap):
                path = Path(old.filename)
                old._mmap.close()
                path.unlink()
            del old
        if _insert_edge(self.table, self.degrees, key, u, v):
            self.edge_count += 1

    def to_csr(self):
        pointers = np.empty(self.n + 1, dtype=np.int64)
        pointers[0] = 0
        np.cumsum(self.degrees, out=pointers[1:])
        assert pointers[-1] == 2 * self.edge_count
        indices = np.empty(2 * self.edge_count, dtype=np.int32)
        _fill_csr(self.table, self.n, pointers, indices)
        _sort_rows(pointers, indices)
        return pointers, indices

    def close(self):
        table = self.table
        self.table = None
        if isinstance(table, np.memmap):
            table._mmap.close()
        del table
        gc.collect()
        for path in self.paths:
            path.unlink(missing_ok=True)


def synthesize(degrees, triangles, seed, scratch=None):
    from privtdd import baselines
    fn = baselines.synthesis_function()
    graphs = []

    def factory():
        graph = CompactGraph(len(degrees), int(np.maximum(degrees, 0).sum()) // 2, scratch)
        graphs.append(graph)
        return graph

    # This function owns a fresh globals dictionary, not the NetworkX module.
    fn.__globals__['nx'] = SimpleNamespace(Graph=factory)
    try:
        return fn(degrees, triangles, seed)
    except BaseException:
        for graph in graphs:
            graph.close()
        raise
