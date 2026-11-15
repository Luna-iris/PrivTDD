"""Matched sparse forward/adjoint for original count laws under cap projection."""
import math
import numpy as np
from privtdd.correction.sparse_channel import SparseNormalLaplaceMixture,sparse_component_channel,integer_law_channel,integer_law_adjoint
from privtdd.base_distribution.projected_count_channel import retained_count_probabilities
from privtdd.base_distribution.large_count import smoothing_constants


def projected_sparse_channel(keys,probabilities,degree,cap,p,scale,targets):
    keys=np.asarray(keys);mass=np.asarray(probabilities,float);targets=np.asarray(targets,float)
    if mass.ndim==1:mass=mass[:,None]
    W=int(degree)*(int(degree)-1)//2
    if keys.dtype.kind not in 'iu' or np.any(keys<0) or np.any(keys>W):raise ValueError('Original count support')
    if cap<=1:
        if np.any(targets!=0):raise ValueError('Deterministic report')
        return np.tile(mass.sum(axis=0),(len(targets),1)),np.zeros(mass.shape[1])
    if degree<=cap:return sparse_component_channel(keys,mass,degree,p,scale,targets)
    w=int(cap)*(int(cap)-1)//2
    if w<=64:
        thinning=retained_count_probabilities(W,keys,w)
        return integer_law_channel(np.arange(w+1),thinning.T@mass,int(cap),p,scale,targets)
    sigma,first,second=smoothing_constants(w,p,scale);gap=2*p-1;rho=keys/W
    convolution=SparseNormalLaplaceMixture(gap*w*rho,mass,sigma,scale)
    variance=w*rho*(1-rho)*(W-w)/(W-1)
    error=convolution.absolute_error_bound+.5*gap*gap*second*(variance@mass)
    return convolution.evaluate(targets),error


def projected_sparse_adjoint(report,weights,degree,cap,keys,p,scale):
    """Return evidence and a per-column uniform absolute reference error bound."""
    keys=np.asarray(keys,np.int64);weights=np.asarray(weights,float);report=np.asarray(report,float)
    if weights.ndim==1:weights=weights[:,None]
    W=int(degree)*(int(degree)-1)//2
    if np.any(keys<0) or np.any(keys>W):raise ValueError('Original count support')
    if cap<=1:
        if np.any(report!=0):raise ValueError('Deterministic report')
        return np.tile(weights.sum(axis=0),(len(keys),1)),np.zeros(weights.shape[1])
    if degree<=cap:return integer_law_adjoint(report,weights,degree,keys,p,scale)
    w=int(cap)*(int(cap)-1)//2
    if w<=64:
        values,error=integer_law_adjoint(report,weights,int(cap),np.arange(w+1),p,scale)
        return retained_count_probabilities(W,keys,w)@values,error
    sigma,first,second=smoothing_constants(w,p,scale);gap=2*p-1;rho=keys/W
    convolution=SparseNormalLaplaceMixture(report,weights,sigma,scale)
    variance=w*rho*(1-rho)*(W-w)/(W-1)
    error=convolution.absolute_error_bound+.5*gap*gap*second*float(variance.max(initial=0))*weights.sum(axis=0)
    return convolution.evaluate(gap*w*rho),error
