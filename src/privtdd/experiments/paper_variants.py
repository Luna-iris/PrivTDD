"""Paper E2/E3 postprocessing. Every variant uses its parent Full transcript."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from . import common as C, components as E2
from privtdd import data

FIXED = {"fixed25": .25, "fixed50": .50, "fixed75": .75}
BOUNDARIES = {"q80_90": (.80, .90), "q95_99": (.95, .99)}
METRICS = {"MSE": "MSE", "EMD": "W1", "KS": "KS"}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def record(row, origin, report):
    return dict(origin=origin, report_sha256=report, truth_sha256=row["truth_sha256"],
                prediction=str(Path(row["prediction"]).resolve()),
                prediction_sha256=data.digest(row["prediction"]),
                **{name:row[field] for name,field in METRICS.items()})


def quantile_bounds(keys, counts, quantiles):
    positive = keys > 0
    if not positive.any():
        return (0, 0)
    cdf = np.cumsum(counts[positive])
    return tuple(map(int, keys[positive][np.searchsorted(cdf/cdf[-1], quantiles)]))


def variant_result(task, name, out, full, shared_holder, ablation_id):
    from . import ablation as Ablation
    from privtdd.release.pipeline import fit_joint7
    from privtdd.correction.forward import build_family
    ref = Path(full["prediction"]).parent
    cfg = C.config(task)
    cuts = None
    if name in BOUNDARIES:
        with np.load(ref/"work/F1.npz") as f:
            default = quantile_bounds(f["keys"], f["counts"], (.90,.95))
            cuts = quantile_bounds(f["keys"], f["counts"], BOUNDARIES[name])
        check(list(default)==C.read_json(ref/"work/result.json")["boundaries"], "Default integer boundaries mismatch")
        if cuts == default:
            return dict(full, origin="identical_integer_boundaries", quantiles=BOUNDARIES[name], boundaries=cuts)
    if not shared_holder:
        with C.allocation_context(task["allocation"]), C.closure_context(task.get("closure_cells",65)):
            shared_holder.append(Ablation.load_shared(ref, cfg))
    shared = shared_holder[0]
    family = shared["family"]
    with threadpool_limits(limits=cfg.blas_threads):
        if name == "screc":
            coordinates, fit = np.zeros(16), {"structural_correction":False}
        elif name in FIXED:
            coordinates, fit = Ablation.fixed_coordinates(family, FIXED[name])
        else:
            family = build_family(family["sources"], shared["caps"],shared["report"],shared["prior"],
                                  shared["p"],shared["ed"],shared["el"],cuts,workers=cfg.workers,
                                  batch_small_groups=cfg.batch_small_groups)
            family.update(counts=shared["counts"], n=shared["n"])
            coordinates, _, fit = fit_joint7(family, directory=out)
        final, _, diagnostics = E2.candidate(shared, family, coordinates, cfg)
    C.save_hist(out/"prediction.npz", final)
    C.write_json(out/"fit.json", dict(coordinates=coordinates, fit=fit, boundaries=family["boundaries"],
                                     report_sha256=full["report_sha256"], safeguard=diagnostics["safeguard"]))
    row, = C.evaluate(task, C.graph_for(task), {name:out/"prediction.npz"})
    return dict(record(row, "new_postprocessing", full["report_sha256"]), fit_sha256=data.digest(out/"fit.json"))
