"""Conditional closure cells for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.numerics.probability import _normal_laplace_logcdf

from privtdd.base_distribution.count_channel import logpdf_q

import math

import numpy as np



# [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); Construct atoms at rho=0/1 and cosine boundaries for 63 continuous cells.
def closure_edges(size=65):
    if size<4:raise ValueError('Need two atoms and at least two cells')
    grid=.5*(1-np.cos(np.pi*np.arange(size)/(size-1)))
    interior=grid[1:-1]
    return np.r_[0.,.5*(interior[:-1]+interior[1:]),1.]



# [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); Integrate continuous closure-rate cells into small-capacity states using half-integer boundaries for integer T.
def integer_cell_weights(W,edges):
    if W<0 or int(W)!=W:raise ValueError('Integer wedge count')
    e=np.asarray(edges,float)
    if e[0]!=0 or e[-1]!=1 or np.any(np.diff(e)<=0):raise ValueError('Cell partition')
    if W==0:return np.ones((1,len(e)-1))
    k=np.arange(W+1)[:,None]
    lo=np.maximum(e[:-1][None,:],(k-.5)/W)
    hi=np.minimum(e[1:][None,:],(k+.5)/W)
    weights=np.maximum(hi-lo,0)/np.diff(e)[None,:]
    if np.max(abs(weights.sum(axis=0)-1))>2e-12:raise ArithmeticError('Integer cell mass')
    return weights



# [PrivTDD] S3 F1; numerical integration and stable computation; Stably compute the log probability of a normal-Laplace convolution over a specified interval.
def normal_laplace_loginterval(lower,upper,sigma,scale):
    lo,hi,s,b=np.broadcast_arrays(lower,upper,sigma,scale)
    if np.any(lo>=hi):raise ValueError('Nonempty interval')
    left=lo+hi<=0
    high=_normal_laplace_logcdf(np.where(left,hi,-lo),s,b)
    low=_normal_laplace_logcdf(np.where(left,lo,-hi),s,b)
    if np.any(low-high>1e-10):raise ArithmeticError('CDF ordering')
    with np.errstate(divide='ignore',invalid='ignore'):
        value=high+np.log(-np.expm1(np.minimum(low-high,0)))
    value=np.where(np.isneginf(high),-np.inf,value)
    if np.isnan(value).any():raise ArithmeticError('Cell integral unresolved')
    return value



# [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); Compute each closure-rate component likelihood for report Q; the memory module uses blocks of 2048 reports.
def component_loglikelihood(report,degree,p,scale,edges=None):
    """Return Q-density matrix: atom0, integrated positive cells, atom1.

    W<=64: exact rounded-integer T prior and exact RR/Laplace likelihood.
    W>64: continuous T=M*rho with Gaussian RR/exact Laplace convolution;
    integer rounding and RR normal approximation are explicitly working-model
    approximations. No center-point substitution inside positive cells.
    """
    if edges is None:edges=closure_edges()
    edges=np.asarray(edges,float);W=int(degree)*(int(degree)-1)//2
    y,b=np.broadcast_arrays(np.asarray(report,float),np.asarray(scale,float))
    if y.ndim!=1:raise ValueError('One-dimensional report batch')
    if W<=64:
        T=np.arange(W+1);scores=logpdf_q(y[:,None],int(degree),T[None,:],p,b[:,None])
        anchor=scores.max(axis=1);weights=integer_cell_weights(W,edges)
        integrated=np.exp(scores-anchor[:,None])@weights
        with np.errstate(divide='ignore'):cells=anchor[:,None]+np.log(integrated)
        return np.column_stack((scores[:,0],cells,scores[:,-1]))
    gap=2*p-1
    lo=y[:,None]-gap*W*edges[1:][None,:];hi=y[:,None]-gap*W*edges[:-1][None,:]
    cells=normal_laplace_loginterval(lo,hi,math.sqrt(W*p*(1-p)),b[:,None])-np.log(gap*W*np.diff(edges))[None,:]
    return np.column_stack((logpdf_q(y,int(degree),0,p,b),cells,logpdf_q(y,int(degree),W,p,b)))
