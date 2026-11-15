"""Single-scalar observation channel under public rotated sampling. Input Q is in B-qA+Lap units, without division by s or p-q.

The conditional working model matches the joint projection model: given original degree d and original closed-wedge count T, the
T closure labels among W wedges are exchangeable. A is the number of wedges selected after projection, and J|A is Hypergeom(W,T,A).
Recovery uses only the public design and released cap; it never reads the center's actual A, J, or neighbors.
"""
import math
from collections import OrderedDict
from functools import lru_cache, wraps
from itertools import combinations
from threading import RLock

import numpy as np
from numba import njit
from scipy.special import logsumexp
from scipy.stats import binom, hypergeom

from .moments import _sampled_exposure_moments
from privtdd.base_distribution.count_channel import normal_laplace_logpdf
from privtdd.base_distribution.projected_count_channel import retained_count_probabilities
from privtdd.reporting.primitives import rr_params


# In the sparse regime, a discrete mixture explicitly retains P(A=0). For general k, fixed public simulations estimate the
# distribution of A; all h for the same public design share a batch of random-permutation prefixes. High exposures use a moment-matched approximation.
DISCRETE_MEAN_LIMIT = 4.0
PUBLIC_DRAWS = 4096
_TINY = np.finfo(np.float64).tiny
_TABLE_LOCK = RLock()


@njit
def _prefix_count(tree, stop):
    total = 0
    while stop:
        total += tree[stop]
        stop -= stop & -stop
    return total


@njit
def _range_count(tree, ordered, left, right):
    return (_prefix_count(tree, np.searchsorted(ordered, right))
            - _prefix_count(tree, np.searchsorted(ordered, left)))


@njit
def _prefix_exposures(positions, m, k):
    """Insert vertices sequentially; a Fenwick tree counts previously inserted vertices in the opposite cyclic interval without enumerating wedges."""
    ordered = np.sort(positions)
    tree = np.zeros(len(ordered) + 1, dtype=np.int64)
    result = np.empty(len(ordered), dtype=np.int64)
    count = 0
    for i, v in enumerate(positions):
        if v < m:
            side, begin = m, v
        else:
            side, begin = 0, (v-m-k+1) % m
        end = begin + k
        count += _range_count(tree, ordered, side + begin, side + min(end, m))
        if end > m:
            count += _range_count(tree, ordered, side, side + end-m)
        result[i] = count
        j = np.searchsorted(ordered, v) + 1
        while j < len(tree):
            tree[j] += 1
            j += j & -j
    return result


def _calibrate_mean(prob, target):
    """Calibrate the finite public-sample histogram to its known mean using positive exponential weights, preserving nonnegative probabilities."""
    support = np.flatnonzero(prob > 0)
    if len(support) < 2 or not support[0] < target < support[-1]:
        return prob
    x = support - target
    logp = np.log(prob[support])
    lo, hi = -32., 32.
    for _ in range(42):
        mid = (lo+hi)/2
        tilted = logp + mid*x
        if np.dot(x, np.exp(tilted-tilted.max())) > 0:
            hi = mid
        else:
            lo = mid
    tilted = logp+(lo+hi)*.5*x
    prob[support] = np.exp(tilted-logsumexp(tilted))
    return prob


@lru_cache(maxsize=16)
def _public_exposure_table(N, k, biased, draws):
    """Depend only on the public design. At low exposures, condition on selecting one edge to improve accuracy for rare nonzero signals.

    Place a uniformly chosen design edge in the h-subset, then add h-2 vertices uniformly to obtain the size-biased distribution
    P*(a)=a P(a)/E[A]. Inverse weighting avoids observing many all-zero samples; the mean is exact automatically.
    The public design is vertex-transitive; for an odd population, fixing and excluding the dummy vertex gives the same edge-count distribution.
    """
    s, m = k/(N-1), N//2
    mean_limit = .5 if biased else DISCRETE_MEAN_LIMIT
    hmax = min(N-1, int((1+math.sqrt(1+8*mean_limit/s))/2) + 1)
    rng = np.random.default_rng(np.random.SeedSequence([572921, N, k, int(biased)]))
    # Build in batches so memory grows with h and sparse exposure values; do not allocate a draws-by-N public vertex table.
    counts = [np.zeros(1, dtype=np.int64) for _ in range(hmax+1)]
    for start in range(0, draws, 128):
        batch = np.empty((min(draws-start,128),hmax),dtype=np.int64)
        for row in range(len(batch)):
            if biased:
                right = m + int(rng.integers(k))
                rest = rng.choice(N-2, size=hmax-2, replace=False) + 1
                rest[rest >= right] += 1
                vertices = np.concatenate(([0,right],rest))
            else:
                vertices = rng.choice(N, size=hmax, replace=False)
            batch[row] = _prefix_exposures(vertices, m, k)
        for h in range(2,hmax+1):
            freq = np.bincount(batch[:, h-1])
            if len(freq) > len(counts[h]):
                counts[h] = np.pad(counts[h], (0,len(freq)-len(counts[h])))
            counts[h][:len(freq)] += freq
    probabilities = []
    for h, observed in enumerate(counts):
        mean = h*(h-1)*s/2
        prob = observed.astype(float)/draws
        if h < 2:
            prob[0] = 1.
        elif biased and mean <= .5:
            prob[1:] *= mean/np.arange(1,len(prob))
            prob[0] = 1.-prob[1:].sum()
        elif not biased:
            prob = _calibrate_mean(prob,mean)
        probabilities.append(prob)
    return probabilities


@lru_cache(maxsize=8192)
def exposure_distribution(N, k, h, draws=PUBLIC_DRAWS):
    """Probability mass of low-exposure A, indexed by a. The following three classes of public designs admit direct exact summation."""
    if h < 2:
        return np.ones(1)
    mean, variance = _sampled_exposure_moments(h,k,N-1)
    if h == 2:
        return np.array([1-mean,mean])
    if h == 3:
        # A bipartite graph has no triangles, so A is at most 2; its first two moments uniquely determine the three masses.
        f2 = variance + mean*(mean-1)
        prob = np.maximum([1-mean+f2/2,mean-f2,f2/2],0.)
        return prob/prob.sum()
    m = N//2
    if k == 1:
        # Select both endpoints of a pairs and one endpoint of another h-2a pairs. Ratios of neighboring combination counts avoid factorial overflow.
        lo, hi = max(0,h-m), h//2
        support = np.arange(lo,hi,dtype=float)
        log_ratio = (np.log(h-2*support)+np.log(h-2*support-1)-math.log(4)
                     - np.log(support+1)-np.log(m-h+support+1))
        logp = np.r_[0.,np.cumsum(log_ratio)]
        return np.r_[np.zeros(lo),np.exp(logp-logsumexp(logp))]
    if k == m:
        left = np.arange(max(0,h-m),min(h,m)+1)
        # The induced edge count in K_m,m is the product of the numbers of selected vertices on the two sides.
        mass = hypergeom.pmf(left,N,m,h)
        prob = np.bincount(left*(h-left),weights=mass)
        return prob/prob.sum()
    if math.comb(N,h) <= 8192:
        # For small public domains, enumerate all retained positions directly; this also supplies an independent check of the public simulation.
        seen = (int(_prefix_exposures(np.fromiter(sub,dtype=np.int64),m,k)[-1])
                for sub in combinations(range(N),h))
        prob = np.bincount(np.fromiter(seen,dtype=np.int64)).astype(float)
        return prob/prob.sum()
    if mean > DISCRETE_MEAN_LIMIT:
        raise ValueError('Public discrete table requested outside its sparse domain')
    # The first threaded (d,c) requests may ask for the same public table simultaneously; avoid generating the entire permutation batch repeatedly.
    with _TABLE_LOCK:
        return _public_exposure_table(N,k,mean <= .5,draws)[h]


@lru_cache(maxsize=512)
def _rr_matrix(a,p):
    """For each row conditioned on J, compute the complete mass of B given A=a; shared RR across centers does not change a single-center marginal."""
    idx = np.arange(a+1)
    matrix = np.empty((a+1,a+1))
    for j in range(a+1):
        matrix[j] = np.convolve(binom.pmf(idx[:j+1],j,p),
                                binom.pmf(idx[:a-j+1],a-j,1-p))
    return matrix/matrix.sum(axis=1,keepdims=True)


def _closure_moments(W,t,mean_a,var_a):
    if W < 1:
        return np.zeros_like(t,dtype=float), np.zeros_like(t,dtype=float)
    rho = t.astype(float)/W
    mean_j = rho*mean_a
    # The law of total variance avoids subtracting two nearly equal second moments when W is large.
    var_j = rho*rho*var_a
    if W > 1:
        finite = max(0.,(mean_a*(W-mean_a)-var_a)/(W-1))
        var_j += rho*(1-rho)*finite
    return mean_j,var_j


def _byte_cache(mib):
    """Bound the process cache by actual array sizes so long T grids across many (d,c) pairs do not exhaust memory."""
    def decorate(func):
        entries, lock = OrderedDict(), RLock()
        total,limit = 0,mib*1024**2
        @wraps(func)
        def cached(*args):
            nonlocal total
            with lock:
                saved = entries.get(args)
                if saved is not None:
                    entries.move_to_end(args)
                    return saved[0]
            value = func(*args)
            arrays = value if isinstance(value,tuple) else (value,)
            size = (256 + sum(len(arg) for arg in args if isinstance(arg,bytes))
                    + sum(x.nbytes for x in arrays if isinstance(x,np.ndarray)))
            with lock:
                if args not in entries and size <= limit:
                    while entries and total+size > limit:
                        total -= entries.popitem(last=False)[1][1]
                    entries[args] = (value,size)
                    total += size
            return value
        return cached
    return decorate


@_byte_cache(24)
def _conditional_rr(W, t_bytes, a, p):
    """Different caps and report batches share the conditional RR transitions for the same original integer T values."""
    keys = np.frombuffer(t_bytes,dtype=np.int64)
    if a == 0:
        return np.ones((len(keys),1))
    if a == 1:
        success = 1-p+(2*p-1)*keys.astype(float)/W
        return np.column_stack((1-success,success))
    return retained_count_probabilities(W,keys,a) @ _rr_matrix(a,p)


@_byte_cache(40)
def _signal_atoms(N,k,h,W,t_bytes,p,b):
    """Combine positive discrete weights into one matrix multiplication and reuse them across report blocks.

    For each A and any Q, the Laplace-mixture ratio relative to center 0 is bounded above by exp(p*A/b)
    and below by exp(-p*A/b). Use these bounds to prune only tiny masses whose total relative density error is at most 1e-13.
    """
    law = exposure_distribution(N,k,h)
    nz = np.flatnonzero(law>0)
    logmass = np.log(law[nz])
    lower = logsumexp(logmass-p*nz/b)
    cutoff = lower+math.log(1e-13)-math.log(len(nz))
    nz = nz[logmass+p*nz/b >= cutoff]
    small,large = nz[nz<=64],nz[nz>64]
    centers,weights = [],[]
    for a in small:
        a = int(a)
        centers.append(np.arange(a+1)-(1-p)*a)
        weights.append(law[a]*_conditional_rr(W,t_bytes,a,p).T)
    return (np.concatenate(centers) if centers else np.empty(0),
            np.concatenate(weights,axis=0) if weights else None,
            large, law[large])


class SampledChannel:
    """Discrete sparse kernel and high-exposure moment-matched Normal-Laplace working kernel for the independent experiment backend."""

    def __init__(self,n,k,eps_rr,eps_rel):
        if int(n)!=n or n<4 or int(k)!=k or not 1<=k<=(n+n%2)//2:
            raise ValueError('Expected n>=4 and a supported sampled rotation k')
        if not np.isfinite([eps_rr,eps_rel]).all() or min(eps_rr,eps_rel)<=0:
            raise ValueError('RR and release budgets must be finite and positive')
        self.n,self.k = int(n),int(k)
        self.N = self.n+self.n%2
        self.s = self.k/(self.N-1)
        self.p,self.q,self.gap = rr_params(float(eps_rr))
        self.eps_rel = float(eps_rel)
        if not .5 < self.p < 1:
            raise ValueError('Nondegenerate symmetric RR required')

    def scale(self,cap):
        return np.minimum(self.k,np.maximum(np.asarray(cap,dtype=float)-1.,0.))/self.eps_rel

    def _normal(self,y,W,t,ma,va,b):
        mj,vj = _closure_moments(W,t,ma,va)
        sig = np.sqrt(self.p*self.q*ma + self.gap**2*vj)
        return normal_laplace_logpdf(y[:,None]-self.gap*mj,sig,b)

    def logdensity(self,d,cap,t,qvalues):
        """Return log f(Q|d,T,c) indexed by [report row, original triangle-count column], with Q in the original RR-centered units.

        When cap<=1, Q is deterministically 0, so return a point mass. When d<=1 and cap>1, Laplace noise remains.
        High exposures retain design variance, original-label projection, and RR variance; the normal approximation does not preserve higher moments.
        """
        if int(d)!=d or int(cap)!=cap or not 0<=d<self.n or not 0<=cap<self.n:
            raise ValueError('Invalid public cap or candidate degree')
        d,cap = int(d),int(cap)
        W,h = d*(d-1)//2,min(d,cap)
        t,y = np.atleast_1d(t),np.atleast_1d(np.asarray(qvalues,dtype=float))
        if (t.ndim!=1 or y.ndim!=1 or not np.isfinite(t).all() or
                np.any(t!=np.floor(t)) or np.any((t<0)|(t>W)) or not np.isfinite(y).all()):
            raise ValueError('Expected finite reports and integer T in [0,choose(d,2)]')
        t = t.astype(np.int64,copy=False)
        if cap < 2:
            return np.broadcast_to(np.where(y[:,None]==0,0.,-np.inf),(len(y),len(t))).copy()
        b = float(self.scale(cap))
        ma,va = _sampled_exposure_moments(h,self.k,self.N-1)
        if ma > DISCRETE_MEAN_LIMIT:
            return self._normal(y,W,t,ma,va,b)
        # For extreme reports in original units, subtract the row's Laplace tail exponent before taking the positive-weight matrix product
        # to prevent premature underflow of exp(-abs(Q)/b) from incorrectly assigning the same zero probability to different T values.
        centers,weights,large,large_mass = _signal_atoms(
            self.N,self.k,h,W,t.tobytes(),self.p,b)
        if len(centers):
            shift = -np.maximum.reduce([np.zeros(len(y)),centers.min()-y,y-centers.max()])/b
            likelihood = np.exp(-np.abs(y[:,None]-centers)/b-shift[:,None])
            summed = likelihood @ weights
            with np.errstate(divide='ignore'):
                logf = np.log(summed)+shift[:,None]-math.log(2*b)
        else:
            logf = np.full((len(y),len(t)),-np.inf)
        # The sparse A mixture occasionally contains large A values; retain their probability mass and use the exact first two moments given A
        # to approximate the conditional RR signal. Do not discard tails or count the total exposure variance again.
        for a,mass in zip(large,large_mass):
            component = self._normal(y,W,t,float(a),0.,b)
            logf = np.logaddexp(logf,math.log(mass)+component)
        return logf

    def density(self,d,cap,t,qvalues):
        logf = self.logdensity(d,cap,t,qvalues)
        # Use logdensity to compare extremely small tail likelihood ratios; the real-valued density interface retains positive values for the optimizer.
        return np.where(np.isneginf(logf),0.,np.exp(np.maximum(logf,math.log(_TINY))))
