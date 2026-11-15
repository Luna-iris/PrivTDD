"""Sparse integer-support metrics and publication-level bootstrap intervals."""
from __future__ import annotations
import numpy as np


def law(keys, counts):
    k, c = np.asarray(keys), np.asarray(counts, float)
    if k.ndim != 1 or c.shape != k.shape or not len(k):
        raise ValueError('Nonempty aligned histogram required')
    if not np.isfinite(k).all() or np.any(k < 0) or np.any(k != np.floor(k)):
        raise ValueError('Nonnegative integer support required')
    if not np.isfinite(c).all() or np.any(c < 0) or c.sum() <= 0:
        raise ValueError('Finite nonnegative nonzero mass required')
    u, inv = np.unique(k.astype(np.int64), return_inverse=True)
    mass = np.bincount(inv, weights=c, minlength=len(u))
    return u, mass / mass.sum()


def bin_ids(keys):
    """Fixed public partition: singleton 0..63, then [64,127], [128,255], ..."""
    k = np.asarray(keys, np.int64)
    ids = k.copy()
    tail = k >= 64
    ids[tail] = 64 + np.floor(np.log2(k[tail])).astype(int) - 6
    return ids


def compare(true_keys, true_counts, keys, counts, *, n=None, public_cap=None):
    a, p = law(true_keys, true_counts)
    b, q = law(keys, counts)
    u = np.union1d(a, b)
    delta = np.zeros(len(u))
    delta[np.searchsorted(u, a)] += p
    delta[np.searchsorted(u, b)] -= q
    cdf = np.cumsum(delta)
    grouped = np.bincount(bin_ids(u), weights=delta)
    mean_true, mean_est = float(a @ p), float(b @ q)
    out = dict(W1=float(abs(cdf[:-1]) @ np.diff(u)), KS=float(abs(cdf).max()),
               TV=float(abs(delta).sum()/2), TV_bin=float(abs(grouped).sum()/2),
               log_W1=float(abs(cdf[:-1]) @ np.diff(np.log1p(u))),
               zero_mass_error=float(abs(delta[u == 0]).sum()),
               mean_true=mean_true, mean_est=mean_est,
               total_relative_error=abs(mean_est-mean_true)/mean_true if mean_true else None)
    if n is not None and public_cap is not None:
        out['MSE'] = float(n*n*(delta @ delta)/(public_cap*(public_cap-1)//2+1))
        out['total_absolute_error'] = n*abs(mean_est-mean_true)/3
    return out


def bootstrap(values, *, seed=731, draws=2000):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite completed publication measurements required')
    if len(x) == 1:
        return dict(mean=float(x[0]), lower=None, upper=None, n=1)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(draws, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(x.mean()), lower=float(lo), upper=float(hi), n=len(x))


def hierarchical(groups, *, seed=731, draws=2000):
    """Equal graph weights; resample graphs, then releases within each selected graph."""
    xs = [np.asarray(x, float) for x in groups]
    if not xs or any(not len(x) or not np.isfinite(x).all() for x in xs):
        raise ValueError('Nonempty graph-level groups required')
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        chosen = rng.integers(len(xs), size=len(xs))
        estimates.append(np.mean([rng.choice(xs[j], len(xs[j]), replace=True).mean() for j in chosen]))
    lo, hi = np.quantile(estimates, [.025, .975])
    return dict(mean=float(np.mean([x.mean() for x in xs])), lower=float(lo) if len(xs)>1 else None,
                upper=float(hi) if len(xs)>1 else None, graphs=len(xs), releases=sum(map(len, xs)))


def topk_membership(distances, k=5):
    """Expected inclusion under uniform tie breaking at the kth distance."""
    d = np.asarray(distances, float)
    if len(d) < k or not np.isfinite(d).all():
        raise ValueError('Insufficient finite reference distances')
    threshold = np.partition(d, k-1)[k-1]
    # Exact ties only; never enlarge a tie based on evaluation outcomes.
    before, tied = d < threshold, d == threshold
    w = before.astype(float)
    w[tied] = (k-int(before.sum()))/int(tied.sum())
    return w


def retrieval(true_distances, private_distances, k=5):
    truth = topk_membership(true_distances, k)
    selected = topk_membership(private_distances, k)
    regret = float((selected-truth) @ np.asarray(true_distances)/k)
    if regret < -1e-9:
        raise ArithmeticError('Negative true-distance regret')
    return dict(Recall_at_5=float(truth @ selected/k), regret=max(0., regret))
