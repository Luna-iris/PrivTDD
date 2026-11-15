"""Structural reference primitives for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.optimize import brentq

from scipy.special import expit

import math

from scipy.optimize import minimize

from numba import njit

import numpy as np



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Weighted grouping utility for the original structural model; the current fit_beta uses the log_degree branch.
def weighted_groups(values, counts, maximum):
    """At most maximum contiguous empirical quantile groups; never split ties."""
    values, counts = np.asarray(values), np.asarray(counts)
    unique, inverse = np.unique(values, return_inverse=True)
    if len(unique) <= maximum:
        return inverse
    mass = np.bincount(inverse, weights=counts, minlength=len(unique))
    cuts = unique[np.minimum(np.searchsorted(np.cumsum(mass),
                           np.arange(1, maximum)*np.sum(mass)/maximum), len(unique)-1)]
    labels = np.searchsorted(np.unique(cuts), values, side='left')
    return np.unique(labels, return_inverse=True)[1]



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Mean-degree matching objective, gradient and optional Hessian for the type Beta model, including centered identity regularization.
def beta_objective(beta, sizes, targets, penalty=1., hessian=False):
    n = float(np.sum(sizes))
    pair_counts = np.outer(sizes, sizes)-np.diag(sizes)
    z = beta[:, None]+beta[None, :]
    p = expit(z)
    center = float(np.dot(sizes, beta)/n)
    objective = (.5*np.sum(pair_counts*np.logaddexp(0., z))-np.dot(sizes*targets, beta)
                 +.5*penalty*np.dot(sizes, (beta-center)**2))/n
    degree = p @ sizes-np.diag(p)
    gradient = sizes*(degree-targets+penalty*(beta-center))/n
    if not hessian:
        return float(objective), gradient
    curvature = pair_counts*p*(1-p)
    hess = curvature+np.diag(curvature.sum(axis=1))
    hess += penalty*(np.diag(sizes)-np.outer(sizes, sizes)/n)
    return float(objective), gradient, hess/n



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Use equal-width log(1+d) groups at fixed resolution, without tuning groups using dataset truth.
def log_degree_groups(values, counts, maximum):
    """Uniform log(1+d) cells; resolution fixed before seeing any truth metrics."""
    values=np.asarray(values)
    if values.max()==0: return np.zeros(len(values),dtype=np.int64)
    cell=np.minimum(np.floor(maximum*np.log1p(values)/np.log1p(values.max())).astype(int),maximum-1)
    return np.unique(cell,return_inverse=True)[1]



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Solve type connection propensities beta so logistic(beta_g+beta_h) matches private posterior mean degrees.
def fit_beta(means, counts, max_groups=64, *, grouping="quantile"):
    if grouping not in ("quantile", "log_degree"):
        raise ValueError('Unknown beta grouping')
    grouper = log_degree_groups if grouping == "log_degree" else weighted_groups
    labels = grouper(means, counts, max_groups)
    sizes = np.bincount(labels, weights=counts).astype(np.int64)
    targets = np.bincount(labels, weights=counts*means)/sizes
    n = int(np.sum(sizes))
    bound = math.log(max(n, 2))+5
    avg = float(np.dot(sizes, targets)/n)
    initial = .5*(math.log(max(avg, np.finfo(float).tiny))
                  - math.log(max(n-1-avg, np.finfo(float).tiny)))
    fit = minimize(beta_objective, np.full(len(sizes), np.clip(initial, -bound, bound)),
                   args=(sizes, targets), jac=True, method='L-BFGS-B',
                   bounds=[(-bound, bound)]*len(sizes),
                   options=dict(maxiter=500, ftol=1e-15, gtol=1e-12, maxls=40))
    beta = fit.x.copy()
    # [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Projected gradient for bounded Beta optimization, used to assess convergence of the same solve.
    def projected(b, gradient):
        pg = gradient*n/sizes
        pg[(b <= -bound+1e-10) & (pg > 0)] = 0
        pg[(b >= bound-1e-10) & (pg < 0)] = 0
        return float(np.max(np.abs(pg)))
    # Solve the SAME convex objective more accurately if objective-relative
    # L-BFGS stopping hides residuals in small groups. No data-driven tuning.
    polish = 0
    max_polish = min(30, max(0, 500-int(fit.nit)))
    for polish in range(max_polish+1):
        value, gradient, hess = beta_objective(beta, sizes, targets, hessian=True)
        residual = projected(beta, gradient)
        if residual <= 1e-6 or polish == max_polish:
            break
        direction = np.linalg.solve(hess+1e-14*np.eye(len(beta)), -gradient)
        step = 1.
        for _ in range(40):
            candidate = np.clip(beta+step*direction, -bound, bound)
            new_value, _ = beta_objective(candidate, sizes, targets)
            if new_value <= value+1e-4*np.dot(gradient, candidate-beta)+1e-14*max(1, abs(value)):
                beta = candidate
                break
            step *= .5
        else:
            break
    value, gradient = beta_objective(beta, sizes, targets)
    residual = projected(beta, gradient)
    numerical_audit = dict(numerical_solver='original_LBFGS_Newton', extra_newton_iterations=0,
                           model_or_tolerance_changed=False)
    if np.isfinite(beta).all() and residual > 1e-6:
        # Shared by lift, distribute and both ablation rounds. Only retry
        # numerical refinement; do not alter grouping, penalty or KKT gate.
        from privtdd.numerics.beta_refinement import polish_beta
        beta, numerical_audit = polish_beta(beta, sizes, targets, bound, beta_objective)
        value, gradient = beta_objective(beta, sizes, targets)
        residual = projected(beta, gradient)
    if not np.isfinite(beta).all() or residual > 1e-6:
        raise RuntimeError(f'BETA_FIT_FAILED: per-group projected gradient {residual:g}')
    p = expit(beta[:, None]+beta[None, :])
    debug = dict(grouping=grouping, groups=len(sizes), sizes=sizes, targets=targets, beta=beta,
                 objective_per_node=value, kkt_per_group=residual,
                 lbfgs_iterations=fit.nit, lbfgs_message=str(fit.message),
                 newton_refinement_iterations=polish, lambda_beta=1., bound=bound,
                 grouping_rmse=float(np.sqrt(np.dot(counts, (means-targets[labels])**2)/n)),
                 max_grouping_error=float(np.max(np.abs(means-targets[labels]))),
                 degree_prediction=p @ sizes-np.diag(p))
    debug.update(numerical_audit)
    return labels, sizes, beta, p, debug



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Tilt type-specific neighbor-selection probabilities so the expected total equals the given root degree, for the conditional sampling proposal.
def tilted_probabilities(sizes,beta,root,degree):
    remaining=np.asarray(sizes,np.int64).copy();remaining[root]-=1
    if not 0<degree<remaining.sum():raise ValueError('Nontrivial degree required')
    beta=np.asarray(beta,float)
    shift=brentq(lambda x:float(remaining@expit(x+beta)-degree),
                 -100-float(beta.max()),100-float(beta.min()),xtol=1e-12)
    return remaining,expit(shift+beta)



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Use independent binomial proposals and accept only samples with total neighbor count d, yielding exact fixed-degree neighbor compositions.
def conditional_counts(sizes,beta,root,degree,amount,rng,*,tilt_cache=None):
    remaining = np.asarray(sizes, np.int64).copy(); remaining[root] -= 1
    # Endpoint degrees are deterministic; no finite tilt solves their equation.
    if degree < 0 or degree > remaining.sum() or amount < 1:
        raise ValueError('Invalid conditional degree or integration sample count')
    if degree == 0 or degree == remaining.sum():
        row = np.zeros_like(remaining) if degree == 0 else remaining
        return np.repeat(row[None, :], amount, axis=0), 0
    # [Implementation] cache belongs only to the same packet with fixed sizes/beta; it stores no sample outcomes and consumes no random numbers.
    if tilt_cache is None:
        remaining,p=tilted_probabilities(sizes,beta,root,degree)
    else:
        key = (int(root), int(degree))
        if key not in tilt_cache:
            tilt_cache[key] = tilted_probabilities(sizes,beta,root,degree)
        remaining,p = tilt_cache[key]
    variance=float(remaining@(p*(1-p)))
    answer=[];accepted=0;proposals=0
    while accepted<amount:
        batch=min(4096,max(16,int(np.ceil((amount-accepted)*max(1,np.sqrt(2*np.pi*variance))*1.3))))
        counts=rng.binomial(remaining,p,size=(batch,len(p)))
        valid=counts[counts.sum(1)==degree]
        take=min(len(valid),amount-accepted)
        if take:answer.append(valid[:take]);accepted+=take
        proposals+=batch
    return np.concatenate(answer),proposals



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Use two pointers to count ordered pairs of sorted coordinates at distance at most radius.
@njit(cache=True)
def _ordered_close(x,y,radius):
    lo=0;hi=0;count=0
    for value in x:
        while lo<len(y) and y[lo]<value-radius:lo+=1
        while hi<len(y) and y[hi]<=value+radius:hi+=1
        count+=hi-lo
    return count



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Count neighbor pairs while accounting for circular wraparound.
@njit(cache=True)
def geometry_count(positions,offsets,probability):
    """Count all closing pairs by sorted group windows, including wrap-around."""
    result=0;groups=len(offsets)-1
    for g in range(groups):
        x=positions[offsets[g]:offsets[g+1]]
        if not len(x):continue
        for h in range(g,groups):
            y=positions[offsets[h]:offsets[h+1]]
            if not len(y):continue
            radius=probability[g,h]/2
            ordered=_ordered_close(x,y,radius)+len(x)*len(y)-_ordered_close(x,y,1-radius)
            if g==h:result+=(ordered-len(x))//2
            else:result+=ordered
    return result



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Analytically compute the probability that two points on the two root-neighbor arcs satisfy the closure radius.
def circle_closure(a, b, c):
    """P[circular distance(X,Y)<=c/2], X/Y uniform in centered arcs a/b."""
    a,b,c=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float),np.asarray(c,float))
    if np.any((a<=0)|(a>1)|(b<=0)|(b>1)|(c<0)|(c>1)):
        raise ValueError('Arc lengths must satisfy 0<a,b<=1 and 0<=c<=1')
    low=np.minimum(a,b);high=np.maximum(a,b)
    result=np.empty(a.shape)
    flat=c<=high-low
    result[flat]=c[flat]/high[flat]
    full=c>=high+low
    result[full]=1.
    middle=~(flat|full)
    result[middle]=1.-((high[middle]+low[middle]-c[middle])/high[middle]) * (
        (high[middle]+low[middle]-c[middle])/low[middle])/4
    wrap=np.maximum(a+b+c-2,0)
    result+=(wrap/a)*(wrap/b)/4
    if np.any((result<-1e-12)|(result>1+1e-12)):
        raise ArithmeticError('Invalid circular closure probability')
    return np.clip(result,0,1)
