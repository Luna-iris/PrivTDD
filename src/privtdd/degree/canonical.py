"""Canonical degree distribution representation for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.optimize import brentq

from scipy.special import logsumexp

import math

import numpy as np
from privtdd.config import CAP_OFFSET



# [PrivTDD] S2 degree prior; core method or probability/structure primitive (not necessarily a separate contribution); Preserve the mass and exponential moment of the tail unidentifiable through the cap channel; choose its unique maximum-entropy representative as a model convention.
def canonicalize_tail(prior,eps_degree,offset=CAP_OFFSET):
    if not np.isfinite(eps_degree) or eps_degree<=0 or not isinstance(offset,(int,np.integer)) or offset<0:raise ValueError('Valid cap channel parameters required')
    p=np.asarray(prior,float)
    if p.ndim!=1 or len(p)<2 or np.any(p<0) or abs(p.sum()-1)>2e-12 or not np.isfinite(p).all():raise ValueError('Simplex degree prior')
    D=len(p)-1;start=max(0,D-offset);old=p[start:];mass=float(old.sum())
    if mass==0:return p.copy(),dict(start=start,tail_mass=0.,unchanged=True,statistical_penalty_added=False)
    if len(old)==1:return p.copy(),dict(start=start,tail_mass=mass,unchanged=True,statistical_penalty_added=False)
    old=old/mass;u=np.exp(-eps_degree*np.arange(len(old)))
    v=(u-u[-1])/(1-u[-1])
    with np.errstate(divide='ignore'):logv=np.log(v);logold=np.log(old)
    target=float(logsumexp(logold+logv))
    if not np.isfinite(target):q=np.zeros_like(old);q[-1]=1.;parameter=-math.inf
    elif target>=0:q=np.zeros_like(old);q[0]=1.;parameter=math.inf
    else:
        def equation(parameter):return float(logsumexp(parameter*v+logv)-logsumexp(parameter*v)-target)
        lower=-1.;upper=1.
        for _ in range(1024):
            if equation(lower)<=0:break
            lower*=2
        else:raise ArithmeticError('Entropy lower bracket')
        for _ in range(1024):
            if equation(upper)>=0:break
            upper*=2
        else:raise ArithmeticError('Entropy upper bracket')
        parameter=brentq(equation,lower,upper,xtol=1e-12,rtol=1e-13,maxiter=2000)
        logits=parameter*v;q=np.exp(logits-logsumexp(logits))
    result=p.copy();result[start:]=mass*q
    error=abs(float(q@u-old@u));oldH=-float(old[old>0]@np.log(old[old>0]));newH=-float(q[q>0]@np.log(q[q>0]))
    if error>3e-12 or newH<oldH-1e-9:raise ArithmeticError('Entropy-class constraints')
    return result,dict(start=start,tail_mass=mass,exponential_moment_error=error,
        conditional_entropy_before=oldH,conditional_entropy_after=newH,
        conditional_mean_degree_before=float(old@np.arange(start,D+1)),
        conditional_mean_degree_after=float(q@np.arange(start,D+1)),
        fitted_lagrange_multiplier=parameter if math.isfinite(parameter) else str(parameter),
        exact_likelihood_equivalence=True,statistical_penalty_added=False,truth_used=False)
