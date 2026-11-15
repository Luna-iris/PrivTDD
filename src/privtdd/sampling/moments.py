"""Exact moments for the public rotated matching design."""
from functools import lru_cache

@lru_cache(maxsize=512)
def _sampled_exposure_moments(degree, k, ring):
    """Exact induced-edge moments of a randomly relabeled k-regular design.

    The factored variance equals the adjacent/disjoint-edge moment formula
    in private_report.fixed_moments, avoiding cancellation near full degree.
    """
    W = degree*(degree-1)/2
    N = ring+1
    variance = (degree*(degree-1)*(N-degree)*(N-degree-1)*k*(ring-k)
                / (2*ring*ring*(N-2)*(N-3)))
    return W*k/ring, max(variance, 0.)
