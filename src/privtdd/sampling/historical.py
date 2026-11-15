"""Bind the paper's full-enumeration controls read-only. Existing reports and nonconverged tasks never trigger reruns."""
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

from privtdd.experiments import common as C
from privtdd import data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_archive_path(value, history, base=None):
    """Historical manifests store absolute paths. After migration, first resolve the same file within the current directory."""
    normalized = str(value).replace("\\", "/")
    parts = normalized.split("/")
    for marker, anchor in ((history.name, history),):
        if marker in parts:
            relative = parts[parts.index(marker) + 1:]
            candidate = anchor.joinpath(*relative)
            if candidate.exists():
                return candidate.resolve()
    path = Path(normalized)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def load_controls(history_root, datasets, epsilons, seeds):
    """Return verified full-enumeration records keyed by (dataset, epsilon, seed). Write no locks or caches into the historical directory."""
    history = Path(history_root).resolve()
    plan_file = history / "forpaper" / "plan.json"
    plan = C.read_json(plan_file)
    require(plan["suite"] == "forpaper" and not plan["smoke"], "Expected the formal paper archive")
    wanted = set(product(datasets, epsilons, seeds))
    jobs = [t for t in plan["jobs"] if t.get("panel") == "main" and t.get("method") == "LDP-GDD"
            and (t["dataset"], t["epsilon"], t["seed"]) in wanted]
    require(len(jobs) == len(wanted), "Missing or duplicated full reference in the historical plan")
    # Within one preflight check, verify each original graph, truth file, copied report, and other file only once.
    digest = lru_cache(maxsize=None)(data.digest)
    require(plan.get('design_summary', {}).get('source') == 'new standalone execution',
            'Use E1 results from the standalone package; published evidence is not a working cache')
    graphs = {}
    for task in jobs:
        name = task['dataset']
        if name in graphs:
            continue
        graph = C.graph_for(task)
        csr = [graph / file for file in ("indptr.npy", "indices.npy")]
        graph_sha = C.token([digest(f) for f in csr])
        degrees = np.diff(np.load(csr[0], mmap_mode="r", allow_pickle=False))
        require(len(degrees) == task["public_n"], f"Unexpected graph size: {name}")
        graphs[name] = (graph, graph_sha, degrees, digest(graph / "triangles.npy"))

    table_path = data.HERE / "allocations.json"
    allocation_sha = digest(table_path)
    table = C.read_json(table_path)
    allocations = {(r["dataset"], r["epsilon"]): r for r in table["rows"]}
    from privtdd.reporting import collection, primitives
    collector_sha = digest(Path(collection.__file__))
    primitives_sha = digest(Path(primitives.__file__))
    plan_sha = digest(plan_file)
    result = {}
    for task in jobs:
        dataset, epsilon, seed = key = task["dataset"], task["epsilon"], task["seed"]
        require(key not in result and task["code"] == plan["code"]
                and task["id"] == C.token({k: v for k, v in task.items() if k != "id"})[:24],
                f"Historical job identity mismatch: {key}")
        graph, graph_sha, degrees, truth_sha = graphs[dataset]
        allocation = task["allocation_record"]
        n = task["public_n"]
        expected_allocation = allocations[dataset, epsilon]
        expected_fractions = [2 * allocation["eps_deg"] / epsilon, allocation["eps_rr"] / epsilon,
                              2 * allocation["eps_rel"] / epsilon]
        require(task["allocation_table_sha256"] == allocation_sha
                and all(allocation[name] == value for name, value in expected_allocation.items())
                and task["allocation"] == allocation["allocation"] == expected_fractions
                and allocation["L_for_allocation"] == 0 and allocation["D"] == n - 1
                and len(degrees) == n and task["numeric"] == {"safeguard": True, "safeguard_tau": 1.0},
                f"Historical allocation or mechanism mismatch: {key}")
        design = dict(graph=graph_sha, n=n, D=n - 1, epsilon=epsilon, seed=seed,
                      allocation=task["allocation"], mechanism="joint_shared_original_RR_cap0_v1",
                      collector_sha256=collector_sha, primitives_sha256=primitives_sha)
        report_dir = history / "shared_reports" / C.token(design)
        meta = C.read_json(report_dir / "complete.json")
        report_file = report_dir / "reports.npz"
        report_sha = digest(report_file)
        require(meta["design"] == design and meta["sha256"] == report_sha,
                f"Historical full S1 mismatch: {key}")
        # Full-enumeration reports do not separately store per-center exposure counts; recompute them from the bound CSR and caps.
        with np.load(report_file, allow_pickle=False) as arrays:
            retained = np.minimum(degrees, arrays["caps"]).astype(np.int64, copy=False)
        visits = int(np.sum(retained * (retained - 1) // 2, dtype=np.int64))
        require(visits == meta["actual_wedge_visits"], f"Full candidate count mismatch: {key}")
        job_dir = history / "forpaper" / "jobs" / task["id"]
        status = C.read_json(job_dir / "status.json")
        record = dict(dataset=dataset, epsilon=epsilon, seed=seed, rate=1.0, design_rate=1.0,
                      n=n, k=n + n % 2 - 1, visits=visits, candidates=visits,
                      active_fraction=float(np.mean(retained >= 2)), s1_status="complete",
                      status=status["status"], graph_path=str(graph), graph_sha256=graph_sha,
                      truth_path=str(graph / "triangles.npy"), truth_sha256=truth_sha,
                      report_path=str(report_file), report_sha256=report_sha,
                      parent_job_id=task["id"], parent_plan_sha256=plan_sha,
                      historical_code_sha256=task["code"], collector_sha256=collector_sha,
                      primitives_sha256=primitives_sha,
                      allocation=task["allocation"], allocation_table_sha256=allocation_sha,
                      workers=task["workers"], blas_threads=task["blas_threads"],
                      eps_deg=allocation["eps_deg"], eps_rr=allocation["eps_rr"], eps_rel=allocation["eps_rel"],
                      collection_seconds=meta["collection_seconds"],
                      collection_peak_rss_bytes=meta["collection_peak_rss_bytes"],
                      postprocessing_seconds=None, end_to_end_seconds=None, peak_rss_bytes=None,
                      timing_comparable=False, timing_source="historical E1 full",
                      timing_scope="collection: before prepare_graph through atomic_npz; excludes truth",
                      prediction_path=None, prediction_sha256=None, MSE=None, EMD=None, KS=None)
        worker_cost = job_dir / "worker_cost.json"
        if worker_cost.is_file():
            # The historical worker's process-wide RSS may include evaluation. Preserve its original scope and report it separately.
            record["historical_worker_peak_rss_bytes"] = C.read_json(worker_cost).get("worker_peak_rss_bytes")
        if status["status"] == "complete":
            records_file = job_dir / "records.json"
            require(digest(records_file) == status["records_sha256"], f"Changed E1 records: {key}")
            for field in ("artifacts", "external_artifacts"):
                for name, sha in status.get(field, {}).items():
                    require(digest(resolve_archive_path(name, history, job_dir)) == sha,
                            f"Changed E1 artifact: {name}")
            row, = C.read_json(records_file)
            require(all(row[f] == task[f] for f in ("dataset", "epsilon", "seed", "method", "allocation_table_sha256"))
                    and row["L"] == 0 and row["truth_sha256"] == truth_sha and row["report_sha256"] == report_sha
                    and resolve_archive_path(row["truth"], history, job_dir) == graph / "triangles.npy",
                    f"Historical evaluation binding mismatch: {key}")
            prediction = resolve_archive_path(row["prediction"], history, job_dir)
            base = prediction.parent
            manifest = C.read_json(base / "case_manifest.json")
            completion = C.read_json(base / "completion.json")
            # Only read and verify the frozen configuration bytes here; do not import the solver or trigger Numba caching.
            cfg = manifest["config"]
            expected = dict(workers=task["workers"], blas_threads=task["blas_threads"],
                            geometry_draws=65536, geometry_replicates=4, degree_grid_size=512,
                            triangle_bin_width=1 / 16, report_bin_width=1 / 16,
                            posterior_backend="integrated", structural_lazy_endpoints=True,
                            batch_small_groups=True, batch_f1=True, compiled_structure=True,
                            **task["numeric"])
            require(all(cfg.get(name) == value for name, value in expected.items()),
                    f"Historical fast configuration mismatch: {key}")
            binding = C.token(dict(report=report_sha, report_design=design, config=cfg,
                                   closure_cells=task.get("closure_cells", 65), code=task["code"]))
            require(base.name == completion["identity"] == manifest["identity"] == binding
                    and completion["complete"]
                    and digest(prediction) == completion["prediction_sha256"]
                    and digest(base / "reports.npz") == report_sha
                    and resolve_archive_path(manifest["case"]["source"], history, base) == report_file,
                    f"Historical prediction or configuration mismatch: {key}")
            post = completion["timing"]["postprocessing_seconds"]
            record.update(prediction_path=str(prediction), prediction_sha256=digest(prediction),
                          postprocessing_seconds=post, end_to_end_seconds=meta["collection_seconds"] + post,
                          MSE=row["MSE"], EMD=row["W1"], KS=row["KS"])
        else:
            record["failure_reason"] = status.get("error", status["status"])
        result[key] = record
    return result
