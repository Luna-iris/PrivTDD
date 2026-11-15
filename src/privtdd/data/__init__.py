"""Dataset registry, portable graph preparation, and evaluation-only truth.

Real graphs are read only from the requested data root. No historical repository
or external cache is searched. Set PRIVTDD_DATA_ROOT or pass a root explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from scipy.sparse import coo_matrix

DATASETS = ('bio-dmela', 'dblp', 'higgs', 'orkut')
HERE = Path(__file__).resolve().parents[1] / 'experiments'


def artifact_root():
    explicit = os.environ.get('PRIVTDD_ARTIFACT_ROOT')
    if explicit:
        return Path(explicit).expanduser().resolve()
    for path in (Path.cwd(), *Path(__file__).resolve().parents):
        if (path / 'datasets' / 'manifest.json').is_file():
            return path
    return Path.cwd()


ROOT = artifact_root()


def default_data_root():
    return Path(os.environ.get('PRIVTDD_DATA_ROOT', str(artifact_root() / 'datasets'))).expanduser().resolve()


def manifest():
    """Packaged manifest permits installation outside the source checkout."""
    return json.loads((Path(__file__).with_name('registry.json')).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def csr_from_edges(n, u, v):
    u, v = np.asarray(u, np.int64), np.asarray(v, np.int64)
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= n) or np.any(v >= n):
        raise ValueError('Edge endpoint outside declared node domain')
    keep = u != v
    u, v = u[keep], v[keep]
    graph = coo_matrix((np.ones(2 * len(u), np.int32), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    graph.sum_duplicates()
    graph.sort_indices()
    return graph.indptr.astype(np.int64), graph.indices.astype(np.int64)


def save_graph(path, ip, ix, spec):
    from .triangles import validate_and_triangles
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ip, ix = np.asarray(ip, np.int64), np.asarray(ix)
    if ip.ndim != 1 or len(ip) < 2 or ip[0] != 0 or ip[-1] != len(ix) or np.any(np.diff(ip) < 0):
        raise ValueError('Invalid CSR pointers')
    values = validate_and_triangles(ip, ix)
    np.save(path / 'indptr.npy', ip)
    np.save(path / 'indices.npy', ix)
    np.save(path / 'triangles.npy', values)
    meta = dict(spec=spec, nodes=len(ip) - 1, edges=len(ix) // 2, public_degree_cap=len(ip) - 2,
                observed_max_degree=int(np.diff(ip).max(initial=0)),
                semantics='simple undirected; loops removed; duplicates merged; declared nodes retained',
                graph_sha256=hashlib.sha256((digest(path / 'indptr.npy') + digest(path / 'indices.npy')).encode()).hexdigest())
    _write_json(path / 'graph.json', meta)
    return path


def download(dataset, data_root=None):
    """Fetch from the official source and validate the frozen raw-byte digest."""
    spec = manifest()['datasets'][dataset]
    root = Path(data_root) if data_root is not None else default_data_root()
    root = root / 'raw'
    root.mkdir(parents=True, exist_ok=True)
    target = root / spec['raw']['download_filename']
    if not target.exists():
        tmp = target.with_name(target.name + '.part')
        urllib.request.urlretrieve(spec['source_url'], tmp)
        tmp.replace(target)
    _check_raw(target, spec)
    return target


@contextmanager
def _source_text(path, spec):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.lower().endswith(('.mtx', '.edges', '.txt')) and 'readme' not in name.lower()]
            if len(candidates) != 1:
                raise ValueError(f'Ambiguous graph archive: {candidates}')
            with archive.open(candidates[0]) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig') as text:
                yield text
    elif path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8-sig') as text:
            yield text
    else:
        with path.open(encoding='utf-8-sig') as text:
            yield text


def _check_raw(path, spec):
    raw = spec['raw']
    if Path(path).suffix == '.gz' and raw.get('decompressed_sha256'):
        h = hashlib.sha256()
        with gzip.open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        actual, expected = h.hexdigest(), raw['decompressed_sha256']
    else:
        actual = digest(path)
        expected = raw.get('sha256') or raw.get('decompressed_sha256')
    if expected and actual != expected:
        raise ValueError(f'Raw input differs from frozen {spec["dataset_id"]} source: {path}')


def prepare(dataset, data_root=None, *, raw_path=None, allow_download=False, compute_truth=True, verify_identity=True):
    """Build CSR directly with disk-backed intermediate arrays.

    A different node ordering is rejected by frozen-array verification. Existing
    prepared data is checked rather than overwritten. Temporary work is scoped
    to a fresh TemporaryDirectory within the selected data root.
    """
    from .prepare import prepare_dataset
    return prepare_dataset(dataset, data_root, raw_path=raw_path, allow_download=allow_download,
                           compute_truth=compute_truth, verify_identity=verify_identity)


def verify(dataset, data_root=None, *, full_hash=True, path=None):
    spec = manifest()['datasets'][dataset]
    root = Path(data_root) if data_root is not None else default_data_root()
    graph = Path(path) if path is not None else root / 'processed' / dataset
    issues = []
    checked = []
    for name, expected in spec['processed_files'].items():
        item = graph / name
        if not item.is_file():
            issues.append(f'missing {name}')
            continue
        if item.stat().st_size != expected['bytes']:
            issues.append(f'wrong byte size: {name}')
        if full_hash and digest(item) != expected['sha256']:
            issues.append(f'hash mismatch: {name}')
        checked.append(name)
    if (graph / 'indptr.npy').exists() and (graph / 'indices.npy').exists():
        ip = np.load(graph / 'indptr.npy', mmap_mode='r', allow_pickle=False)
        ix = np.load(graph / 'indices.npy', mmap_mode='r', allow_pickle=False)
        if ip.shape != (spec['nodes'] + 1,) or ix.shape != (2 * spec['edges'],) or ip[0] != 0 or ip[-1] != len(ix):
            issues.append('wrong CSR dimensions/pointers')
    return dict(dataset=dataset, path=str(graph), ok=not issues, issues=issues, checked_files=checked,
                full_hash=full_hash, public_degree_cap=spec['nodes'] - 1)


def real_graph(dataset, cache_root=None, *, download=False):
    if dataset not in DATASETS:
        raise ValueError(f'Paper datasets: {DATASETS}; got {dataset!r}')
    root = default_data_root()
    if cache_root is None or Path(cache_root) == HERE / 'data':
        path = root / 'processed' / dataset
    else:
        cache = Path(cache_root).expanduser().resolve()
        # An explicit data root may include a processed/ child.
        path = (cache / 'processed' / dataset) if (cache / 'processed').is_dir() else cache / dataset
    if not (path / 'graph.json').is_file():
        if download:
            return prepare(dataset, root if cache_root is None else cache_root, allow_download=True)
        raise FileNotFoundError(f'{dataset}: prepare datasets at {path}; no external cache fallback is used')
    result = verify(dataset, path=path, full_hash=False)
    if not result['ok']:
        raise ValueError(result['issues'])
    return path


def truth(path):
    path = Path(path)
    file = path / 'triangles.npy'
    if not file.exists():
        from .triangles import validate_and_triangles
        ip = np.load(path / 'indptr.npy', mmap_mode='r', allow_pickle=False)
        ix = np.load(path / 'indices.npy', mmap_mode='r', allow_pickle=False)
        np.save(file, validate_and_triangles(ip, ix))
    return np.load(file, mmap_mode='r', allow_pickle=False)


def synthetic(path, spec):
    """Generate the original public synthetic families with preserved RNG order."""
    from .families import synthetic as generate
    return generate(path, spec)
