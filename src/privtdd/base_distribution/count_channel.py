"""Reported count observation channel for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.stats import binom

from scipy.special import log_ndtr

from scipy.special import logsumexp

from functools import lru_cache

import math

import numpy as np



# [PrivTDD] S3/S5/S6 shared reporting kernel; core method or probability/structure primitive (not necessarily a separate contribution); Analytically compute the log density of the normal-RR plus Laplace working kernel.
def normal_laplace_logpdf(z,sigma,scale):
    z,s,b=np.broadcast_arrays(np.asarray(z,float),np.asarray(sigma,float),np.asarray(scale,float))
    if np.any(s<0) or np.any(b<=0) or not np.isfinite(s).all() or not np.isfinite(b).all() or np.isnan(z).any():
        raise ValueError('Invalid normal-Laplace parameters')
    out=np.empty(z.shape);zero=s==0;out[zero]=-np.log(2*b[zero])-abs(z[zero])/b[zero]
    active=(~zero)&np.isfinite(z)
    if np.any(active):
        u=z[active]/s[active];a=s[active]/b[active]
        out[active]=.5*a*a-np.log(2*b[active])+np.logaddexp(-z[active]/b[active]+log_ndtr(u-a),z[active]/b[active]+log_ndtr(-u-a))
    out[np.isinf(z)]=-np.inf
    return out



# [PrivTDD] S3/S5/S6 shared reporting kernel; core method or probability/structure primitive (not necessarily a separate contribution); For W<=64, exactly convolve the two binomial RR distributions; return centered RR support and probabilities.
@lru_cache(maxsize=2048)
def exact_rr_distribution(degree,triangles,p):
    W=degree*(degree-1)//2;T=int(triangles)
    if W>64 or not 0<=T<=W or not .5<p<1:raise ValueError('Exact small-count domain')
    a=binom.pmf(np.arange(T+1),T,p);b=binom.pmf(np.arange(W-T+1),W-T,1-p)
    probability=np.convolve(a,b);probability/=probability.sum()
    return np.arange(W+1)-(1-p)*W,probability



# [PrivTDD] S3/S5/S6 shared reporting kernel; core method or probability/structure primitive (not necessarily a separate contribution); Compute the same count-report density using exact RR for W<=64 and normal RR for large W.
def logpdf_q(report,degree,triangles,p,scale):
    """Density for Q=B-qW+Laplace; caller must ensure d<=released cap.

    For W>64, RR is approximated by Normal(gap*T,W*p*q); the Laplace
    convolution is analytic. This is not an exact large-degree RR density.
    Counts are fixed latent integer T, not independently sampled closure labels.
    """
    if not isinstance(degree,(int,np.integer)) or degree<0 or not .5<p<1:raise ValueError('Degree/RR parameters')
    W=degree*(degree-1)//2
    y,t,b=np.broadcast_arrays(np.asarray(report,float),np.asarray(triangles,float),np.asarray(scale,float))
    if np.any(t!=np.floor(t)) or np.any(t<0) or np.any(t>W) or np.any(b<=0):raise ValueError('Fixed T or release scale')
    if W>64:return normal_laplace_logpdf(y-(2*p-1)*t,math.sqrt(W*p*(1-p)),b)
    out=np.empty(y.shape)
    for T in np.unique(t).astype(int):
        centers,prob=exact_rr_distribution(int(degree),int(T),float(p));mask=t==T
        values=-abs(y[mask,None]-centers[None,:])/b[mask,None]-np.log(2*b[mask,None])+np.log(prob)[None,:]
        out[mask]=logsumexp(values,axis=1)
    return out
