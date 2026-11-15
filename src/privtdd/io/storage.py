"""[Implementation] Cache for matching inputs and code versions, atomic writes and conditional-row storage; no model selection."""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
from privtdd.io.memory import atomic_npz
from privtdd.distributions.conditional import ConditionalDistribution, normalize_row


_DEFERRED_TEXT_WRITES = {}


def atomic_text_write(path, text, *, encoding='utf-8', required=True):
    """[I/O] Unique temporary file in the same directory, atomically replaced after handles close; numerical results are unchanged.
    
    path/text/encoding: Target path, complete text and encoding. required=True is for recovery identity,
    parameters and result metadata and still raises on failure; False is only for reconstructible progress/summaries.
    After PermissionError, retry at most five times with waits of 0.05/0.1/0.2/0.4/0.8 seconds (1.55 seconds total).
    If a noncritical file keeps failing, warn and apply a 30-second cooldown for the same path to avoid waiting every iteration.
    Return whether the write succeeded; a skipped write can leave an outdated file. These constants are implementation parameters.
    """
    path = Path(path)
    key = str(path.resolve())
    if not required and time.monotonic() < _DEFERRED_TEXT_WRITES.get(key, 0.):
        return False
    temporary = None
    try:
        # Do not reuse an old .tmp file; avoid name collisions from multiple readers or leftover temporary files.
        fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'w', encoding=encoding, newline='') as stream:
            stream.write(text)
        for attempt in range(6):
            try:
                temporary.replace(path)
                _DEFERRED_TEXT_WRITES.pop(key, None)
                return True
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * 2**attempt)
    except OSError as error:
        if required:
            raise
        _DEFERRED_TEXT_WRITES[key] = time.monotonic()+30.
        print(f'WARNING: nonessential output could not be updated: {path}; '
              f'file may be stale; computation continues; retry after 30s. {error}', file=sys.stderr)
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Do not mask the original save exception; a leftover unique temporary file is not read as a result.


# [PrivTDD] cache across stages; memory and parallel execution; [I/O] Atomically write scalar/NumPy metadata to path; value does not contain the original graph.
def json_write(path, value):
    """[I/O] Save JSON atomically; only display progress may fail, while recovery data and completion markers must be saved successfully."""
    # [PrivTDD] cache across stages; memory and parallel execution; [Utility] Convert NumPy arrays/scalars to native JSON values.
    def convert(item):
        """[Utility] Convert NumPy arrays/scalars to native JSON values."""
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path)
    optional = path.name in ('batch_progress.json', 'joint_recovery_progress.json') or (
        path.name == 'progress.json' and isinstance(value, dict) and value.get('stage') != 'complete')
    return atomic_text_write(path, json.dumps(value, ensure_ascii=False, indent=2, default=convert),
                             required=not optional)


# [PrivTDD] cache across stages; memory and parallel execution; [Cache] Bind inputs, numerical configuration and all method/shared distribution code.
def cache_identity(caps, reports, public_cap, epsilon, config, degree_prior):
    """[Cache] Bind inputs, numerical configuration and all method/shared distribution code.
    
    caps/reports: Released inputs; public_cap/epsilon: public mechanism settings; config: PrivTDDConfig.
    degree_prior: Optional external degree estimate for the same experiment. Thread counts do not affect cache identity; integration paths do.
    The fingerprint only prevents incorrect cache reuse; it is not statistical or LDP certification.
    """
    from dataclasses import asdict
    import privtdd.config as method_config
    settings = asdict(config)
    for name in ('workers', 'blas_threads', 'report_block_size'):
        settings.pop(name)
    h = hashlib.sha256(json.dumps(dict(D=int(public_cap), epsilon=float(epsilon), settings=settings,
        cap_offset=method_config.CAP_OFFSET), sort_keys=True).encode())
    for value in (caps, reports, degree_prior):
        if value is None:
            h.update(b'none')
        else:
            array = np.ascontiguousarray(value)
            h.update(str((array.shape, array.dtype.str)).encode())
            h.update(memoryview(array).cast('B'))
    from privtdd.io.identity import source_digest
    h.update(b'privtdd-cache-v2\0')
    h.update(source_digest().encode('ascii'))
    return h.hexdigest()


# [PrivTDD] cache across stages; memory and parallel execution; [Cache] Use directory only if empty or its identity matches; identity comes from cache_identity.
def initialize_cache(directory, identity):
    """[Cache] Use directory only if empty or its identity matches; identity comes from cache_identity."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'cache_identity.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['identity'] != identity:
            raise ValueError('Cache belongs to different inputs/code/settings; use a new workdir')
    elif any(directory.iterdir()):
        raise ValueError('Nonempty workdir without a joint cache identity; use a new workdir')
    else:
        json_write(path, dict(identity=identity, method='privtdd_v2'))
    return directory


class StoredConditional:
    """[Storage / shared conditional-row semantics] Load one degree row at a time from disk, caching only the most recent row."""
    # [PrivTDD] cache across stages; memory and parallel execution; Store conditional-row paths, fixed pi and the population F1 summary.
    def __init__(self, directory, support, pi, keys, counts, native):
        """directory: Row directory; support/pi: degree states and masses; keys/counts: population F1; native: native degree masses."""
        self.directory = Path(directory)
        self.support, self.pi = support, pi
        self.global_keys, self.global_counts = keys, counts
        self.native_degree_mass = native
        self.name = 'F1'
        self.last, self.last_row = None, None

    # [PrivTDD] cache across stages; memory and parallel execution; [Storage] Load (integer T, conditional probability) by degree index j, reusing distributions.conditional.normalize_row.
    def row(self, j):
        """[Storage] Load (integer T, conditional probability) by degree index j, reusing distributions.conditional.normalize_row."""
        if self.last != j:
            with np.load(self.directory/f'd{int(self.support[j])}.npz') as z:
                self.last_row = normalize_row(z['keys'], z['probability'], int(self.support[j]))
            self.last = j
        return self.last_row


# [PrivTDD] cache across stages; memory and parallel execution; [I/O] Encode conditional distributions in sources as a flat NPZ without pickle.
def save_sources(path, sources):
    """[I/O] Encode conditional distributions in sources as a flat NPZ without pickle."""
    values = {'source_count': len(sources)}
    for index, source in enumerate(sources):
        rows = [source.row(j) for j in range(len(source.support))]
        prefix = f's{index}_'
        values.update({prefix+'support': source.support, prefix+'pi': source.native_degree_mass,
                       prefix+'offsets': np.r_[0, np.cumsum([len(k) for k, _ in rows])],
                       prefix+'keys': np.concatenate([k for k, _ in rows]),
                       prefix+'probabilities': np.concatenate([p for _, p in rows])})
    atomic_npz(Path(path), **values)


# [PrivTDD] cache across stages; memory and parallel execution; [I/O] Load distributions saved by save_sources; distribution objects enforce the probability domain.
def load_sources(path):
    """[I/O] Load distributions saved by save_sources; distribution objects enforce the probability domain."""
    sources = []
    with np.load(path) as z:
        for index in range(int(z['source_count'])):
            prefix = f's{index}_'
            offsets, keys, probabilities = z[prefix+'offsets'], z[prefix+'keys'], z[prefix+'probabilities']
            rows = [(keys[a:b], probabilities[a:b]) for a, b in zip(offsets[:-1], offsets[1:])]
            sources.append(ConditionalDistribution(f'source{index}', z[prefix+'support'], tuple(k for k, _ in rows),
                                                  tuple(p for _, p in rows), z[prefix+'pi']))
    return tuple(sources)
