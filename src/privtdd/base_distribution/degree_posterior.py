"""Conditional degree posterior for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.base_distribution.cells import component_loglikelihood

import heapq

from privtdd.base_distribution.cells import integer_cell_weights

from privtdd.base_distribution.count_channel import logpdf_q

import math

from privtdd.base_distribution.cells import normal_laplace_loginterval

import numpy as np



class DegreePosterior:
    # [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); Compute posterior component masses in blocks, using the exact integer RR distribution at small capacities.
    def __init__(self, degree, report, scale, coefficient, theta, edges, p):
        self.degree = int(degree)
        self.W = self.degree*(self.degree-1)//2
        self.y = np.asarray(report, float)
        self.b = np.asarray(scale, float)
        self.a = np.asarray(coefficient, float)
        self.theta = np.asarray(theta, float)
        self.edges = np.asarray(edges, float)
        self.p = p
        if not (self.y.shape == self.b.shape == self.a.shape) or np.any(self.b < 0):
            raise ValueError('Posterior inputs')
        self.component_mass = np.zeros(len(theta))
        self.finite_cdf = None
        finite_mass = np.zeros(self.W+1) if self.W<=64 else None
        if finite_mass is not None:
            T=np.arange(self.W+1)
            prior=integer_cell_weights(self.W,edges)@self.theta[1:-1]
            prior[0]+=self.theta[0];prior[-1]+=self.theta[-1]
        for start in range(0,len(self.y),2048):
            y=self.y[start:start+2048];b=self.b[start:start+2048];a=self.a[start:start+2048]
            active=b>0
            if np.any(y[~active]!=0):raise ValueError('Nonzero deterministic report')
            K=np.ones((len(y),len(theta)))
            K[active]=np.exp(component_loglikelihood(y[active],self.degree,p,b[active],edges))
            self.component_mass+=(a@K)*self.theta
            if finite_mass is not None:
                likelihood=np.ones((len(y),len(T)))
                likelihood[active]=np.exp(logpdf_q(y[active,None],self.degree,T[None,:],p,b[active,None]))
                finite_mass+=(a@likelihood)*prior
        self.mass=float(self.component_mass.sum())
        if not self.mass>0:raise ArithmeticError('Empty degree posterior')
        self.prefix=np.r_[0.,np.cumsum(self.component_mass[1:-1])]
        if finite_mass is not None:
            if abs(finite_mass.sum()/self.mass-1)>1e-10:raise ArithmeticError('Integer posterior normalization mismatch')
            self.finite_cdf=np.cumsum(finite_mass)/self.mass

    # [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); Return the CDF of this degree-conditional posterior at integer keys; return 0/1 outside its support.
    def cdf(self, keys):
        keys = np.asarray(keys, dtype=np.int64)
        answer = np.zeros(keys.shape, float)
        answer[keys >= self.W] = 1.
        valid = (keys >= 0) & (keys < self.W)
        if not valid.any():
            return answer
        k = keys[valid]
        if self.finite_cdf is not None:
            answer[valid] = self.finite_cdf[k]
            return answer
        x = (k+.5)/self.W
        cells = np.minimum(np.searchsorted(self.edges, x, side='right')-1, len(self.edges)-2)
        out = self.component_mass[0] + self.prefix[cells]
        active = self.b > 0
        ya, ba, aa = self.y[active], self.b[active], self.a[active]
        deterministic = float(self.a[~active].sum())
        gap = 2*self.p-1
        sigma = math.sqrt(self.W*self.p*(1-self.p))
        for cell in np.unique(cells):
            ids = np.flatnonzero((cells == cell) & (x > self.edges[cell]))
            if self.theta[cell+1] == 0:
                continue
            left, right = self.edges[cell:cell+2]
            size = max(1, 200000//max(1, len(ya)))
            for start in range(0, len(ids), size):
                sel = ids[start:start+size]
                width = x[sel]-left
                result = deterministic*width/(right-left)
                if len(ya):
                    logmass = normal_laplace_loginterval(
                        ya[:, None]-gap*self.W*x[sel][None, :],
                        ya[:, None]-gap*self.W*left, sigma, ba[:, None])
                    result += (aa @ np.exp(logmass))/(gap*self.W*(right-left))
                out[sel] += self.theta[cell+1]*result
        answer[valid] = out/self.mass
        if np.min(answer) < -1e-11 or np.max(answer) > 1+1e-11:
            raise ArithmeticError('Posterior CDF outside unit interval')
        return np.clip(answer, 0., 1.)



# [PrivTDD] S3 F1; optional numerical approximation; Bin at the noise scale, falling back to the original adaptive compression if the grid is too fine.
def compress_noise_cdf(cdf, W, signal_step, *, fallback_tolerance=None, max_bins=4096):
    """[S3 numerical approximation] Integrate noise-scale T bins in one batch, retaining 0..64 and W.
    
    cdf: Original conditional posterior CDF; W: integer capacity; signal_step: maximum bin length in T units.
    fallback_tolerance: Optional per-degree W1 budget for original adaptive compression; max_bins: 4096 is an implementation workload threshold.
    If the number of noise bins exceeds the threshold and a budget is supplied, use the original adaptive compression to avoid grid expansion at large epsilon.
    Each bin probability is the difference between the original CDF at its endpoints; place mass at the integer midpoint, without subsampling nodes or truncating tails.
    Return the same format as compress_integer_cdf. bound is a transport upper bound for this conditional law,
    not a final GDD error guarantee; it is no longer constrained by the old compression_W1_budget=2.
    """
    W = int(W)
    if W < 0 or not np.isfinite(signal_step) or signal_step <= 0:
        raise ValueError('Nonnegative capacity and positive finite signal step required')
    step = max(1, int(math.floor(signal_step)))
    if isinstance(max_bins, bool) or not isinstance(max_bins, int) or max_bins < 1:
        raise ValueError('Positive integer noise-bin work threshold required')
    number = min(W+1, 65)+(1 if W > 64 else 0)+(max(W-65, 0)+step-1)//step
    if fallback_tolerance is not None:
        if not np.isfinite(fallback_tolerance) or fallback_tolerance <= 0:
            raise ValueError('Positive finite fallback W1 tolerance required')
        if number > max_bins:
            result = compress_integer_cdf(cdf, W, fallback_tolerance)
            result['mode'] = 'adaptive_W1_fallback'
            return result
    head = np.arange(min(W, 64)+1, dtype=np.int64)
    tail = np.arange(65, W, step, dtype=np.int64) if W > 65 else np.array([], np.int64)
    lower = np.r_[head, tail, np.array([W], np.int64) if W > 64 else np.array([], np.int64)]
    upper = np.r_[lower[1:]-1, W]
    values = np.asarray(cdf(upper), float)
    if not np.isfinite(values).all() or values.min() < -1e-10 or values.max() > 1+1e-10:
        raise ArithmeticError('Invalid noise-bin CDF')
    differences = np.diff(np.r_[0., values])
    roundoff = float(max(0., -differences.min()))
    if roundoff > 1e-10 or abs(values[-1]-1) > 1e-9:
        raise ArithmeticError('Nonmonotone or incomplete noise-bin CDF')
    values = np.maximum.accumulate(np.clip(values, 0., 1.))
    probability = np.diff(np.r_[0., values])
    keys = (lower+upper)//2
    bound = float(probability@np.maximum(keys-lower, upper-keys))
    return dict(keys=keys, probability=probability, lower=lower, upper=upper, mode='noise_scaled',
                cdf_left=np.r_[0., values[:-1]], cdf_right=values,
                bound=bound, splits=0, cdf_roundoff=roundoff)


# [PrivTDD] S3 F1; original numerical compression; Retain 0..64/W and recursively split the remaining intervals until the transport budget is met.
def compress_integer_cdf(cdf, W, tolerance):
    """Exact buckets 0..64 and W; remaining intervals have a transport bound.

    Each interval is placed at its integer midpoint. Its mass times the largest
    endpoint distance bounds its W1 error, irrespective of within-bin shape.
    """
    W = int(W)
    if tolerance <= 0:
        raise ValueError('Positive numerical tolerance required')
    head = np.arange(min(W, 64)+1)
    head_cdf = cdf(head)
    records = [(int(k), int(k), float(a), float(b)) for k, a, b in zip(head, np.r_[0., head_cdf[:-1]], head_cdf)]
    queue = []
    roundoff = 0.

    # [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); Insert an integer interval and its endpoint CDF values into the compression heap, recording midpoint transport cost.
    def add(L, R, a, b):
        nonlocal roundoff
        if b < a:
            roundoff = max(roundoff, a-b)
            if a-b > 1e-10:
                raise ArithmeticError('Nonmonotone CDF during compression')
            b = a
        mass = b-a
        mid = (L+R)//2
        bound = mass*max(mid-L, R-mid)
        heapq.heappush(queue, (-bound, L, R, a, b))

    if W > 64:
        last = float(cdf(np.array([W-1]))[0])
        if W > 65:
            add(65, W-1, float(head_cdf[-1]), last)
        records.append((W, W, last, 1.))
    bound = -sum(item[0] for item in queue)
    splits = 0
    while bound > tolerance:
        batch = []
        for unused in range(min(128, len(queue))):
            item = heapq.heappop(queue)
            if item[0] == 0:
                heapq.heappush(queue, item)
                break
            batch.append(item)
        if not batch:
            raise ArithmeticError('Positive compression error without splittable bins')
        mids = np.array([(x[1]+x[2])//2 for x in batch])
        values = cdf(mids)
        for item, mid, value in zip(batch, mids, values):
            old, L, R, a, b = item
            if value < a-1e-10 or value > b+1e-10:
                raise ArithmeticError('CDF inconsistent with parent bin')
            roundoff = max(roundoff, a-value, value-b)
            value = float(np.clip(value, a, b))
            add(L, int(mid), a, value)
            add(int(mid)+1, R, value, b)
        splits += len(batch)
        bound = -sum(item[0] for item in queue)
    records += [(L, R, a, b) for unused, L, R, a, b in queue]
    records.sort()
    lower, upper, cdf_left, cdf_right = map(np.asarray, zip(*records))
    lower=lower.astype(np.int64);upper=upper.astype(np.int64)
    probability = cdf_right-cdf_left
    if np.min(probability) < -1e-10 or abs(probability.sum()-1) > 1e-9:
        raise ArithmeticError('Compressed probability mass')
    probability = np.maximum(probability, 0.)
    keys = (lower+upper)//2
    actual_bound = float(probability @ np.maximum(keys-lower, upper-keys))
    return dict(keys=keys.astype(np.int64), probability=probability,
                lower=lower.astype(np.int64), upper=upper.astype(np.int64),
                cdf_left=cdf_left, cdf_right=cdf_right,
                bound=actual_bound, splits=splits, cdf_roundoff=float(roundoff))
