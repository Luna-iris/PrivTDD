"""Relationship-accounted TriTR histogram adaptation helpers."""
import numpy as np

def _split_budget(epsilon, method: str, relationship_ledger: bool = True,
                  ratio=(0.1, 0.45, 0.45)):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    weights = np.asarray(ratio, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('ratio must sum to one')
    # An undirected edge edit can change three owned RR bits after rowwise
    # projection: two at its lower endpoint and one at its upper endpoint.
    # Degrees and second-round local outputs each affect two endpoints.
    # Thus charge 2*eps0 + 3*eps1 + 2*eps2. A local-row ledger charges
    # eps0 + 2*eps1 + eps2 for the same replacement-aware adaptation.
    budget = epsilon / ((2 if relationship_ledger else 1) + weights[1])
    return tuple((budget*weights).tolist())


def _noisy_hist(private_participation, max_k: int, n: int):
    xs=np.asarray(private_participation,dtype=float)
    if len(xs)!=n or not np.isfinite(xs).all():
        raise ValueError('invalid private estimates')
    indices=np.clip(np.rint(xs),0,int(max_k)).astype(np.int64)
    keys,ct=np.unique(indices,return_counts=True)
    result={int(k):float(v) for k,v in zip(keys,ct)}
    assert sum(result.values())==n
    return result
