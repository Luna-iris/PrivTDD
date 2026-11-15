"""Memory-efficient synthesis and a durable graph checkpoint before evaluation."""
from pathlib import Path
import time

import numpy as np
from numba import set_num_threads, config as numba_config

from privtdd.experiments import common as C
from privtdd import data
from .compact_graph import synthesize


def run(original_run, task, path, method, *, synth=False, fresh=None):
    if not synth:
        return original_run(task, path, method, synth=False, fresh=fresh)
    if method != 'LDPGraph-TargetHist':
        raise ValueError('Unexpected synthesis protocol')
    from .external import configure_source, validate_source
    explicit_source = task.get('ldpgraph_source') or task.get('external_code_root')
    if explicit_source:
        configure_source(explicit_source)
    else:
        validate_source()
    from privtdd.io.runner import output_lock
    from privtdd.io.memory import atomic_npz
    from privtdd.io.identity import source_digest
    runtime_digest = lambda: source_digest('package')

    directory, meta = original_run(task, path, method, synth=False, fresh=fresh)
    with output_lock(directory, 'BASELINE.lock'):
        complete = directory / 'synthesis_complete.json'
        if complete.exists():
            saved = C.read_json(complete)
            if saved['source_report_sha256'] != data.digest(directory / 'reports.npz'):
                raise ValueError('Synthesis report changed')
            for name, sha in saved['artifacts'].items():
                if data.digest(directory / name) != sha:
                    raise ValueError('Changed synthesis artifact')
            return directory, meta
        started = time.perf_counter()
        report_hash = data.digest(directory / 'reports.npz')
        with np.load(directory / 'reports.npz') as archive:
            degrees, triangles = archive['target_degree'], archive['target_triangle']
        set_num_threads(min(task['workers'], numba_config.NUMBA_NUM_THREADS))
        checkpoint = directory / 'synthesis_graph_complete.json'
        graph_file = directory / 'synthetic_graph.npz'
        if checkpoint.exists():
            saved = C.read_json(checkpoint)
            if saved['source_report_sha256'] != report_hash or saved['graph_sha256'] != data.digest(graph_file):
                raise ValueError('Synthesis graph checkpoint identity changed')
            with np.load(graph_file) as archive:
                pointers, indices = archive['indptr'], archive['indices']
            graph_meta = saved
        else:
            print('Synthesis: compact uint64 edge set; original RNG and source control flow', flush=True)
            seed = int(task['seed']) + 90000001 + 101
            graph = synthesize(degrees, triangles, seed, directory / 'synthesis_scratch')
            try:
                pointers, indices = graph.to_csr()
                graph_meta = dict(source_report_sha256=report_hash, seed=seed,
                                  edges=graph.edge_count, hash_capacity=len(graph.table),
                                  backend='compact_edge_set_v1', runtime_sha256=runtime_digest())
            finally:
                graph.close()
            atomic_npz(graph_file, indptr=pointers, indices=indices)
            graph_meta.update(graph_sha256=data.digest(graph_file), synthesis_seconds=time.perf_counter()-started)
            C.write_json(checkpoint, graph_meta)
        print('Synthesis graph saved; counting exact triangle participations', flush=True)
        from privtdd.data.triangles import validate_and_triangles
        values = validate_and_triangles(pointers, indices)
        keys, counts = np.unique(values, return_counts=True)
        atomic_npz(directory / 'synth_prediction.npz', keys=keys, counts=counts)
        C.write_json(complete, dict(seconds=time.perf_counter()-started,
            source_report_sha256=report_hash,
            artifacts={name: data.digest(directory / name) for name in ('synthetic_graph.npz', 'synth_prediction.npz')},
            remaining_degree_deficit=int(np.maximum(degrees-np.diff(pointers), 0).sum()),
            changes=['redundant dense adjacency removed', 'stop on zero remaining demand',
                     'compact exact edge storage; direct sorted CSR; graph checkpoint'],
            backend=graph_meta, rng_and_synthesis_rules='unchanged from frozen source adapter'))
    return directory, meta
