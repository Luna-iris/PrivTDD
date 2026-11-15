"""Private degree observation channel for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

import numpy as np
from privtdd.config import CAP_OFFSET



# [PrivTDD] S2 degree prior; numerical integration and stable computation; Stably compute log Laplace interval probabilities, including infinite endpoints.
def log_laplace_interval(left,right,scale):
    """Log P(left <= Laplace(0,scale) < right), including infinite endpoints."""
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive finite scale required')
    a,b=np.broadcast_arrays(np.asarray(left,float)/scale,np.asarray(right,float)/scale)
    if np.isnan(a).any() or np.isnan(b).any() or np.any(a>=b):raise ValueError('Nonempty ordered intervals required')
    result=np.empty(a.shape);negative=b<=0;positive=a>=0;middle=~(negative|positive)
    result[negative]=-np.log(2)+b[negative]+np.log(-np.expm1(a[negative]-b[negative]))
    result[positive]=-np.log(2)-a[positive]+np.log(-np.expm1(a[positive]-b[positive]))
    result[middle]=np.log(.5*(-np.expm1(a[middle])-np.expm1(-b[middle])))
    return result



# [PrivTDD] S2 degree prior; core method or probability/structure primitive (not necessarily a separate contribution); Compute exact cap probabilities for floor(d+Laplace+L), with L=0 in the main path and clipping to [0,D].
def log_cap_probability(cap,degree,public_cap,eps_degree,offset=CAP_OFFSET):
    """C=clip(floor(d+Laplace(1/eps_degree)+offset),0,public_cap).

    cap and degree broadcast. This is a released-cap likelihood, not a
    projected triangle-count likelihood. No true graph or degree is read here.
    """
    if not isinstance(public_cap,(int,np.integer)) or public_cap<1:raise ValueError('Positive integer public cap required')
    if not np.isfinite(eps_degree) or eps_degree<=0 or not np.isfinite(offset):raise ValueError('Invalid release parameters')
    c,d=np.broadcast_arrays(np.asarray(cap,float),np.asarray(degree,float))
    if not np.isfinite(c).all() or not np.isfinite(d).all() or np.any(c!=np.floor(c)) or np.any(d!=np.floor(d)):
        raise ValueError('Integer cap and latent degree required')
    if np.any(c<0) or np.any(c>public_cap) or np.any(d<0) or np.any(d>public_cap):raise ValueError('Public domain')
    left=c-d-offset;right=left+1.
    left=np.where(c==0,-np.inf,left)
    right=np.where(c==public_cap,np.inf,right)
    return log_laplace_interval(left,right,1/eps_degree)
