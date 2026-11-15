"""Versioned source identity for the standalone PrivTDD implementation.

Old experiment manifests are immutable. This identity deliberately differs
from the research repository and never authorizes reuse of its work caches.
"""
import hashlib
import json
from pathlib import Path

IDENTITY_SCHEMA = "privtdd-source-v2"
CORE_FILES = ('base_distribution/cells.py', 'base_distribution/channel.py', 'base_distribution/construction.py', 'base_distribution/count_channel.py', 'base_distribution/degree_posterior.py', 'base_distribution/integral.py', 'base_distribution/integrated.py', 'base_distribution/large_count.py', 'base_distribution/optimizer.py', 'base_distribution/projected_count_channel.py', 'base_distribution/projected_posterior.py', 'base_distribution/weights.py', 'config.py', 'correction/basis.py', 'correction/feasible_domain.py', 'correction/forward.py', 'correction/optimizer.py', 'correction/optimizer_bridge.py', 'correction/projected_sparse_channel.py', 'correction/regions.py', 'correction/roundoff.py', 'correction/source_mapping.py', 'correction/sparse_channel.py', 'correction/stopping.py', 'degree/canonical.py', 'degree/channel.py', 'degree/fit.py', 'degree/groups.py', 'degree/newton.py', 'degree/operator.py', 'degree/phases.py', 'degree/reduced.py', 'degree/working_set.py', 'distributions/conditional.py', 'io/identity.py', 'io/memory.py', 'io/runner.py', 'io/storage.py', 'numerics/beta_refinement.py', 'numerics/likelihood.py', 'numerics/probability.py', 'numerics/quantization.py', 'posterior/rows.py', 'posterior/update.py', 'references/calibration.py', 'references/clustering.py', 'references/construction.py', 'references/context.py', 'references/geometry_model.py', 'references/independent_edge.py', 'references/kernels.py', 'references/primitives.py', 'release/pipeline.py', 'release/transport_guard.py', 'reporting/collection.py', 'reporting/primitives.py')


def source_files(scope="core"):
    """Return installed source files independently of the working directory."""
    root = Path(__file__).resolve().parents[1]
    if scope == "core":
        files = [root / name for name in CORE_FILES]
    elif scope == "package":
        files = sorted(root.rglob("*.py"))
    else:
        raise ValueError("scope must be core or package")
    missing = [p.relative_to(root).as_posix() for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete PrivTDD source installation: " + ", ".join(missing))
    return files


def source_digest(scope="core"):
    """Bind file paths and bytes, excluding machine-specific absolute paths."""
    root = Path(__file__).resolve().parents[1]
    records = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
               for p in source_files(scope)]
    payload = dict(schema=IDENTITY_SCHEMA, scope=scope, files=records)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
