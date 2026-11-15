"""Streaming real-graph preparation preserving the paper's node-label policy."""
from array import array
import hashlib
from pathlib import Path
import tempfile

import numpy as np
from numba import njit

from . import (_check_raw, _source_text, _write_json, default_data_root, digest,
               download, manifest, verify)


@njit(cache=True)
def _fill(u, v, offsets, cursor, neighbors):
    for j in range(len(u)):
        left, right = int(u[j]), int(v[j])
        neighbors[offsets[left] + cursor[left]] = right
        cursor[left] += 1
        neighbors[offsets[right] + cursor[right]] = left
        cursor[right] += 1


def _chunks(path, pairs=1_000_000):
    with Path(path).open('rb') as stream:
        while block := stream.read(pairs * 16):
            yield np.frombuffer(block, dtype='<i8').reshape(-1, 2)


def build_csr(raw_path, spec, output, work):
    """Parse one graph; loops declare nodes but contribute no adjacency entries."""
    output, work = Path(output), Path(work)
    output.mkdir(parents=True, exist_ok=True)
    edge_file = work / 'edges.i64'
    policy = spec['node_mapping']
    first_seen = {} if policy == 'first_seen' else None
    seen = bytearray()
    declared = spec['nodes'] if policy == 'matrix_market_declared' else None
    buffer = array('q')
    if buffer.itemsize != 8:
        raise RuntimeError('8-byte signed work integers required')
    matrix_header = False
    need_dimensions = False
    loops = 0
    with _source_text(raw_path, spec) as lines, edge_file.open('wb') as stream:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lstrip('%').startswith('MatrixMarket'):
                matrix_header = need_dimensions = True
                continue
            if line.startswith(('#', '%')):
                continue
            values = line.split()
            if need_dimensions:
                rows, cols, _ = map(int, values)
                if rows != cols or rows != declared:
                    raise ValueError('MatrixMarket node domain differs from the frozen dataset')
                need_dimensions = False
                continue
            if len(values) < 2:
                raise ValueError('Edge row requires two node IDs')
            u, v = int(values[0]), int(values[1])
            if min(u, v) < 0:
                raise ValueError('Negative original node ID')
            if policy == 'matrix_market_declared':
                u, v = u - 1, v - 1
                if min(u, v) < 0 or max(u, v) >= declared:
                    raise ValueError('MatrixMarket endpoint outside declared domain')
            elif first_seen is not None:
                for original in (u, v):
                    if original not in first_seen:
                        first_seen[original] = len(first_seen)
                u, v = first_seen[u], first_seen[v]
            else:
                largest = max(u, v)
                if largest > 100_000_000:
                    raise ValueError('Unexpected sparse original ID domain')
                if largest >= len(seen):
                    seen.extend(b'\0' * (largest + 1 - len(seen)))
                seen[u] = seen[v] = 1
            if u == v:
                loops += 1
                continue
            buffer.extend((u, v))
            if len(buffer) >= 2_000_000:
                np.asarray(buffer, dtype='<i8').tofile(stream)
                buffer = array('q')
        if buffer:
            np.asarray(buffer, dtype='<i8').tofile(stream)
    if policy == 'matrix_market_declared':
        if not matrix_header:
            raise ValueError('Expected MatrixMarket source')
        original_ids = np.arange(1, declared + 1, dtype=np.int64)
        old_to_new = None
    elif first_seen is not None:
        original_ids = np.fromiter(first_seen, dtype=np.int64)
        old_to_new = None
    else:
        original_ids = np.flatnonzero(np.frombuffer(seen, dtype=np.uint8)).astype(np.int64)
        old_to_new = np.full(len(seen), -1, dtype=np.int64)
        old_to_new[original_ids] = np.arange(len(original_ids))
    n = len(original_ids)
    if n != spec['nodes']:
        raise ValueError(f'Expected {spec["nodes"]} nodes, found {n}')
    degree = np.zeros(n, dtype=np.int64)
    for chunk in _chunks(edge_file):
        mapped = old_to_new[chunk] if old_to_new is not None else chunk
        degree += np.bincount(mapped.ravel(), minlength=n)
    offsets = np.r_[np.int64(0), np.cumsum(degree)]
    if offsets[-1]:
        neighbors = np.memmap(work / 'neighbors.i64', mode='w+', dtype=np.int64, shape=(int(offsets[-1]),))
    else:
        neighbors = np.empty(0, np.int64)
    cursor = np.zeros(n, dtype=np.int64)
    for chunk in _chunks(edge_file):
        mapped = old_to_new[chunk] if old_to_new is not None else chunk
        _fill(mapped[:, 0], mapped[:, 1], offsets, cursor, neighbors)
    compact = 0
    pointers = np.empty(n + 1, dtype=np.int64)
    pointers[0] = 0
    for node in range(n):
        row = np.unique(neighbors[offsets[node]:offsets[node + 1]])
        neighbors[compact:compact + len(row)] = row
        compact += len(row)
        pointers[node + 1] = compact
    if compact != 2 * spec['edges']:
        raise ValueError(f'Expected {spec["edges"]} edges, found {compact // 2}')
    np.save(output / 'indptr.npy', pointers)
    # np.save streams the contiguous memmap without creating a whole-graph copy.
    index_dtype = np.dtype(spec.get('index_dtype', '<i4'))
    if neighbors.dtype == index_dtype:
        np.save(output / 'indices.npy', neighbors[:compact])
    else:
        out = np.lib.format.open_memmap(output / 'indices.npy', mode='w+', dtype=index_dtype, shape=(compact,))
        for start in range(0, compact, 1_000_000):
            out[start:start + 1_000_000] = neighbors[start:min(start + 1_000_000, compact)]
        out.flush()
        del out
    np.save(output / 'new_to_original.npy', original_ids)
    if isinstance(neighbors, np.memmap):
        neighbors.flush()
        neighbors._mmap.close()
    return dict(nodes=n, edges=compact // 2, self_loops_removed=loops, observed_max_degree=int(np.diff(pointers).max(initial=0)))


def prepare_dataset(dataset, data_root=None, *, raw_path=None, allow_download=False, compute_truth=True, verify_identity=True):
    spec = manifest()['datasets'][dataset]
    root = Path(data_root) if data_root is not None else default_data_root()
    target = root / 'processed' / dataset
    if (target / 'graph.json').exists():
        status = verify(dataset, root, full_hash=verify_identity)
        if not status['ok']:
            raise ValueError(status['issues'])
        return target
    if raw_path is None:
        raw_path = root / 'raw' / spec['raw']['download_filename']
        if not Path(raw_path).exists():
            if not allow_download:
                raise FileNotFoundError(f'Provide {raw_path} or enable the official download')
            raw_path = download(dataset, root)
    _check_raw(raw_path, spec)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'prepare-{dataset}-', dir=root) as tmp:
        stats = build_csr(raw_path, spec, target, tmp)
    if compute_truth:
        from .triangles import validate_and_triangles
        ip = np.load(target / 'indptr.npy', mmap_mode='r', allow_pickle=False)
        ix = np.load(target / 'indices.npy', mmap_mode='r', allow_pickle=False)
        np.save(target / 'triangles.npy', validate_and_triangles(ip, ix))
    # Validate arrays before committing the graph completion metadata.
    if verify_identity:
        for name, expected in spec['processed_files'].items():
            if name == 'triangles.npy' and not compute_truth:
                continue
            if digest(target / name) != expected['sha256']:
                raise ValueError(f'Prepared {dataset}/{name} does not match paper snapshot')
    _write_json(target / 'graph.json', dict(**stats, dataset=dataset, public_degree_cap=stats['nodes'] - 1,
                source_url=spec['source_url'], node_mapping=spec['node_mapping'],
                graph_sha256=hashlib.sha256((digest(target / 'indptr.npy') + digest(target / 'indices.npy')).encode()).hexdigest(),
                semantics='simple undirected; loops removed; duplicate edges merged; all declared nodes retained'))
    return target
