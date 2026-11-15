"""Run E4 using ``python -m privtdd.sampling.run`` after installation.

The paper plan contains Higgs and Orkut, epsilon=2, four sampling rates,
and five seeds: forty sampled releases. Matched full-enumeration controls
come from E1. The main paper table separately uses all ten E1 full runs.
Use --plan to inspect controls, --run to execute, and --smoke for a separate
synthetic integration case. Resume with the same command; --retry-failed
explicitly retries recorded failures. Per-cell records retain sampling costs,
metrics and failure reasons; paired summaries retain their matched controls.
"""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import numpy as np
from filelock import FileLock
from threadpoolctl import threadpool_limits

# Module execution keeps child imports independent of the current directory.
if __package__ in (None, ""):
    raise SystemExit("Run this installed module with python -m privtdd.sampling.run")

from privtdd.io.memory import atomic_npz
from privtdd.reporting.collection import prepare_graph
from privtdd.experiments import common as C
from privtdd import data
from .collect_rotate import collect
from .historical import load_controls

HERE = Path(__file__).resolve().parent
MODULE = "privtdd.sampling.run"
DATASETS = ("higgs", "orkut")
BUDGETS = (2.0,)
RATES = (0.0001, 0.001, 0.01, 0.1)
SEEDS = range(20261028, 20261033)


def implementation_id():
    """Use the new package's own fingerprint while also binding the main method and the inherited matching implementation it depends on."""
    files = [p for p in sorted(HERE.glob("*.py")) if not p.name.startswith("test_")]
    parts = [(p.name, data.digest(p)) for p in files]
    return C.token([C.code_digest(), data.digest(HERE / "design.py"), parts])


def smoke_control(root, code):
    """Generate a separate full-enumeration control for the small-graph comparison; the formal entry point only reads existing E1 results."""
    spec = dict(family="circle", n=24, avg_degree=6.0, eta=2.0, graph_seed=712)
    graph = data.synthetic(root / "fixture" / "graph", spec)
    cfg = dict(safeguard=True, safeguard_tau=1.0)
    task = dict(dataset="smoke", epsilon=2.0, seed=20261028, workers=2, blas_threads=1,
                smoke=True, allocation=[0.30, 0.35, 0.35], code=code, numeric=cfg, root=str(root / "fixture"))
    report_dir, meta = C.ensure_reports(task, graph)
    out, _, info = C.ensure_joint(task, graph, load=False)
    prediction = out / "prediction.npz"
    metrics, = C.evaluate(task, graph, {"full": prediction})
    ip, _ = C.graph_arrays(graph)
    with np.load(report_dir / "reports.npz") as z:
        active = float(np.mean(np.minimum(np.diff(ip), z["caps"]) >= 2))
    return dict(dataset="smoke", epsilon=2.0, seed=task["seed"], n=24, rate=1.0, k=23,
                design_rate=1., graph_path=str(graph), graph_sha256=meta["design"]["graph"],
                truth_path=metrics["truth"], truth_sha256=metrics["truth_sha256"], prediction_path=str(prediction),
                prediction_sha256=data.digest(prediction), allocation=task["allocation"],
                eps_deg=.3, eps_rr=.7, eps_rel=.35, workers=2, blas_threads=1,
                status="complete", s1_status="complete", report_path=str(report_dir / "reports.npz"),
                report_sha256=meta["sha256"], visits=meta["actual_wedge_visits"], candidates=meta["actual_wedge_visits"],
                active_fraction=active, collection_seconds=meta["collection_seconds"],
                postprocessing_seconds=info["postprocessing_seconds"], end_to_end_seconds=info["end_to_end_seconds"],
                collection_peak_rss_bytes=meta["collection_peak_rss_bytes"], peak_rss_bytes=None,
                timing_comparable=True, timing_source="synthetic smoke", MSE=metrics["MSE"], EMD=metrics["W1"], KS=metrics["KS"])


def build_plan(controls, code, smoke, comparable):
    """Include 100% enumeration as a reference row in the analysis; only combinations with s<1 become executable tasks."""
    jobs = []
    for i, ctrl in enumerate(controls):
        for rate in ((0.01, 0.1, 0.5) if smoke else RATES):
            task = {f: ctrl[f] for f in ("dataset", "epsilon", "seed", "workers", "blas_threads")}
            task.update(rate=rate, control_index=i, smoke=smoke)
            task["id"] = C.token([task, code, ctrl["graph_sha256"], ctrl["report_sha256"]])[:24]
            jobs.append(task)
    return dict(design=dict(code=code, controls_sha256=C.token(controls), smoke=smoke, timing_comparable=comparable,
                            population="original full graph", primary_rate=0.1, bootstrap_draws=2000,
                            rate_definition="k/(n+n%2-1)"), jobs=jobs)


def check_artifact(file, expected):
    if not file.is_file() or data.digest(file) != expected:
        raise ValueError(f"Result file is missing or its digest does not match: {file}")


def verify_completed(directory, row):
    """Commit the completion marker last. Validate the result row, reports, and prediction before allowing a resumed run to skip the task."""
    marker = directory / "complete.json"
    if not marker.is_file():
        return False
    check_artifact(directory / "result.json", C.read_json(marker)["result_sha256"])
    check_artifact(directory / "reports.npz", row["report_sha256"])
    check_artifact(Path(row["prediction_path"]), row["prediction_sha256"])
    return True


def run_cell(root, task, ctrl):
    """Commit S1 and the final prediction separately. Preserve S1's measured time and visit count if the backend fails."""
    from .recover import recover

    directory = root / "jobs" / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / "result.json"
    report_file = directory / "reports.npz"
    # The subprocess also holds a lock; if a worker survives an interrupted scheduler, a new scheduler cannot concurrently rewrite the same cell.
    with FileLock(str(directory / "worker.lock")):
        row = C.read_json(result_file) if result_file.exists() else dict(
            **task, n=ctrl["n"], status="pending", s1_status="pending",
            full_report_sha256=ctrl["report_sha256"], graph_sha256=ctrl["graph_sha256"],
            eps_deg=ctrl["eps_deg"], eps_rr=ctrl["eps_rr"], eps_rel=ctrl["eps_rel"],
            timing_comparable=ctrl["timing_comparable"], timing_source="sampled run",
            host=platform.node(), platform=platform.platform(), python=sys.version.split()[0],
        )
        if row.get("id") != task["id"] or row.get("full_report_sha256") != ctrl["report_sha256"]:
            raise ValueError("Saved results have a mismatched task or control identity")
        if row.get("status") == "complete" and verify_completed(directory, row):
            return
        try:
            if row["s1_status"] != "complete":
                ip, ix = C.graph_arrays(Path(ctrl["graph_path"]))
                started = time.perf_counter()
                ip, ix = prepare_graph((ip, ix), ctrl["n"], ctrl["n"] - 1)
                with C.ResourceMeter() as meter:
                    reports, diagnostics = collect(ip, ix, task["rate"], task["seed"], ctrl, workers=task["workers"])
                atomic_npz(report_file, **reports)
                elapsed = time.perf_counter() - started
                # Checksums, paired projection checks, and evaluation are outside the timing interval, matching the historical S1 start and end points.
                with np.load(ctrl["report_path"]) as full:
                    if not np.array_equal(reports["caps"], full["caps"]):
                        raise ArithmeticError("Degree reports differ from the original E1 reports and cannot form paired samples")
                if diagnostics["candidates"] != ctrl["visits"]:
                    raise ArithmeticError("The post-projection candidate-pair denominator differs from full enumeration")
                row.update(**diagnostics, s1_status="complete", status="s1_complete", collection_seconds=elapsed,
                           collection_peak_rss_bytes=meter.peak, report_sha256=data.digest(report_file))
                C.write_json(result_file, row)
                del ip, ix, reports
            else:
                check_artifact(report_file, row["report_sha256"])

            if row.get("prediction_sha256"):
                prediction = Path(row["prediction_path"])
                check_artifact(prediction, row["prediction_sha256"])
            else:
                with np.load(report_file) as releases:
                    caps, noisy = releases["caps"], releases["noisy_triangle_report"]
                config = C.config(dict(task, numeric=dict(safeguard=True, safeguard_tau=1.0)))
                # Each retry starts a complete new recovery timing interval, preventing a partial warm start from shortening the reported runtime.
                attempt = directory / "fits" / uuid.uuid4().hex
                attempt.mkdir(parents=True)
                with threadpool_limits(limits=task["blas_threads"]), C.ResourceMeter() as post:
                    prediction = Path(recover(caps, noisy, ctrl["n"], ctrl, row["k"], config, attempt))
                row.update(status="recovered", prediction_path=str(prediction), prediction_sha256=data.digest(prediction),
                           postprocessing_seconds=post.seconds, end_to_end_seconds=row["collection_seconds"] + post.seconds,
                           peak_rss_bytes=max(row["collection_peak_rss_bytes"], post.peak))
                C.write_json(result_file, row)
            # The recovery interface receives only the two released vectors; score against original-graph truth only after writing the prediction to disk.
            values, = C.evaluate(task, Path(ctrl["graph_path"]), {"sampled": prediction})
            if values["truth_sha256"] != ctrl["truth_sha256"]:
                raise ArithmeticError("Original-graph truth does not match the full-enumeration binding")
            row.update(status="complete", failure_reason=None, MSE=values["MSE"], EMD=values["W1"], KS=values["KS"],
                       truth_sha256=values["truth_sha256"])
            C.write_json(result_file, row)
            C.write_json(directory / "complete.json", dict(result_sha256=data.digest(result_file)))
        except Exception as exc:
            row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
            C.write_json(result_file, row)
            raise


def worker(root, job_id):
    plan = C.read_json(root / "plan.json")
    controls = C.read_json(root / "controls.json")
    if implementation_id() != plan["design"]["code"] or C.token(controls) != plan["design"]["controls_sha256"]:
        raise ValueError("The task plan's code or controls have changed; use a new results directory")
    task, = (t for t in plan["jobs"] if t["id"] == job_id)
    run_cell(root, task, controls[task["control_index"]])


def schedule(root, plan, args):
    from .summarize import summarize
    failed = launched = completed = 0
    for index, task in enumerate(plan["jobs"], 1):
        directory = root / "jobs" / task["id"]
        record = directory / "result.json"
        old = C.read_json(record) if record.is_file() else {}
        if old.get("status") == "complete" and verify_completed(directory, old):
            completed += 1
            continue
        if old.get("status") == "failed" and not args.retry_failed:
            failed += 1
            continue
        if args.limit is not None and launched >= args.limit:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        launched += 1
        print(f"[{index}/{len(plan['jobs'])}] {task['dataset']} ε={task['epsilon']} "
              f"s={task['rate']} seed={task['seed']}", flush=True)
        with (directory / "worker.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", MODULE, "--root", str(root), "--worker", task["id"]],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=root,
                                    env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            try:
                exit_code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                raise
        if exit_code:
            failed += 1
            # OOM or external termination may prevent exception logging; preserve phase information that was already committed atomically.
            row = C.read_json(record) if record.exists() else {}
            if not row:
                ctrl = C.read_json(root / "controls.json")[task["control_index"]]
                row = dict(task, s1_status="pending", full_report_sha256=ctrl["report_sha256"],
                           n=ctrl["n"], graph_sha256=ctrl["graph_sha256"],
                           timing_comparable=ctrl["timing_comparable"], timing_source="sampled run")
            if row.get("status") != "failed":
                row.update(status="failed", failure_reason=f"worker exit code {exit_code}; see worker.log")
                C.write_json(record, row)
        else:
            completed += 1
    summarize(root)
    print(f"Completed {completed}, failed {failed}; results directory {root}", flush=True)
    return int(bool(failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="Execute new tasks and summarize; repeated runs validate and skip completed tasks")
    action.add_argument("--plan", action="store_true", help="Read-only preflight of full-enumeration controls and print the plan")
    action.add_argument("--summarize", action="store_true", help="Rebuild summaries from tasks already saved to disk")
    parser.add_argument("--smoke", action="store_true", help="Run only the independent small-graph validation")
    parser.add_argument("--root", type=Path, help="Directory for new experiment results")
    parser.add_argument("--history-root", type=Path, default=data.ROOT / "results/runs/paper",
                        help="Completed E1 parent root from the standalone package")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed cells; valid S1 reports and predictions may be resumed")
    parser.add_argument("--limit", type=int, help="Maximum number of new tasks to launch in this invocation; the frozen plan is unchanged")
    parser.add_argument("--comparable-timing", action="store_true", help="Enable formal runtime ratios only after verifying that historical and current run conditions are comparable")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = (args.root or data.ROOT / "results/runs" / ("e4_smoke" if args.smoke else "e4")).resolve()
    if args.worker:
        worker(root, args.worker)
        return 0
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    history = args.history_root.resolve()
    if root == history or history in root.parents or root in history.parents:
        parser.error("The experiment output and read-only historical directories must not contain one another")
    if args.summarize:
        from .summarize import summarize
        with FileLock(str(root / "suite.lock"), timeout=0):
            summarize(root)
        return 0
    code = implementation_id()
    if args.smoke and not args.run:
        print("Small-graph plan: 1 graph, 1 seed, 3 sampling tasks; test data is generated only with --smoke --run.")
        return 0
    if not args.smoke:
        controls = list(load_controls(history, DATASETS, BUDGETS, SEEDS).values())
        controls.sort(key=lambda c: (DATASETS.index(c["dataset"]), c["epsilon"], c["seed"]))
        for ctrl in controls:
            ctrl["timing_comparable"] = args.comparable_timing
        plan = build_plan(controls, code, False, args.comparable_timing)
        print(f"Historical full enumeration: {len(controls)} S1 reports, {sum(c['status']=='complete' for c in controls)} predictions; "
              f"new sampling plan: {len(plan['jobs'])} cells.", flush=True)
        if not args.run:
            return 0
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "suite.lock"), timeout=0):
        if args.smoke:
            controls = [smoke_control(root, code)]
            plan = build_plan(controls, code, True, True)
        plan_file = root / "plan.json"
        if plan_file.exists() and C.read_json(plan_file) != plan:
            raise ValueError("Code, controls, or experiment settings differ from the frozen plan. Use a new directory.")
        C.write_json(root / "controls.json", controls)
        C.write_json(plan_file, plan)
        return schedule(root, plan, args)


if __name__ == "__main__":
    raise SystemExit(main())
