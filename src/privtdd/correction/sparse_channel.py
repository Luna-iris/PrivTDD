"""Sparse correction observation channel for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.base_distribution.count_channel import exact_rr_distribution

from privtdd.base_distribution.count_channel import logpdf_q

import math

from privtdd.base_distribution.count_channel import normal_laplace_logpdf

import numpy as np


def use_batched_channel(entries, degree, key_count, *, pair_limit=8192):
    """[S5/S6 equivalent implementation] Use a direct batched kernel when many small cap groups are present; otherwise retain the original sparse-tail algorithm.
    
    entries: (cap, report indices, degree weights); degree/key_count: latent degree and integer support size.
    pair_limit: Uniform workload threshold of 8192 report x support pairs per group; does not change the model or precision.
    W<=64 retains the original exact RR branch. Selection does not use report values, truth or fit quality.
    """
    return (int(degree)*(int(degree)-1)//2 > 64 and bool(entries)
            and sum(len(entry[1]) for entry in entries)*int(key_count) <= pair_limit*len(entries))


def batched_integer_channel(report, scales, degree, keys, p, matrix, *, adjoint=False, block_pairs=262144):
    """[S5/S6 equivalent implementation] Batch the same normal-Laplace kernel and its adjoint across caps.
    
    report/scales: Q and actual mechanism scale for each representative report; never merge scales across caps.
    degree/keys/p: Common latent degree, integer T support and RR probability; used only for the original W>64 working kernel.
    matrix: Component mass matrix over T for the forward operation, or posterior coefficient matrix over reports for the adjoint.
    adjoint: False computes K@matrix, True computes K.T@matrix; block_pairs only limits memory.
    Call the original logpdf_q directly without changing its formula; differences from analytic far-tail evaluation are limited to floating-point effects and the original tail approximation.
    """
    report = np.asarray(report, float)
    scales = np.asarray(scales, float)
    keys = np.asarray(keys, np.int64)
    matrix = np.asarray(matrix, float)
    if (report.ndim != 1 or scales.shape != report.shape or matrix.ndim != 2
            or len(matrix) != (len(report) if adjoint else len(keys))
            or np.any(scales < 0) or not np.isfinite(scales).all()
            or not np.isfinite(report).all() or not np.isfinite(matrix).all()
            or int(degree)*(int(degree)-1)//2 <= 64 or not len(keys)):
        raise ValueError('Batched large-capacity channel inputs')
    if np.any(report[scales == 0] != 0):
        raise ValueError('Nonzero deterministic report')
    step = max(1, int(block_pairs)//len(keys))
    output = np.zeros((len(keys) if adjoint else len(report), matrix.shape[1]))
    for start in range(0, len(report), step):
        stop = min(start+step, len(report))
        y, b = report[start:stop], scales[start:stop]
        active = b > 0
        kernel = np.ones((stop-start, len(keys)))
        kernel[active] = np.exp(logpdf_q(y[active, None], int(degree), keys[None, :], p, b[active, None]))
        if adjoint:
            output += kernel.T@matrix[start:stop]
        else:
            output[start:stop] = kernel@matrix
    return output



class SparseNormalLaplaceMixture:
    # [PrivTDD] general utilities; numerical integration and stable computation; Precompute the sparse kernel's sorted centers, weights and analytic-tail prefix/suffix sums.
    def __init__(self,centers,weights,sigma,scale,*,tail_sigma=10.):
        x=np.asarray(centers,float);weights=np.asarray(weights,float)
        if weights.ndim==1:weights=weights[:,None]
        if (x.ndim!=1 or weights.ndim!=2 or len(x)!=len(weights) or not len(x)
            or np.any(weights<0) or not np.isfinite(weights).all() or not np.isfinite(x).all()
            or sigma<0 or scale<=0):raise ValueError('Sparse convolution inputs')
        order=np.argsort(x,kind='stable');self.x=x[order];self.weights=weights[order]
        self.sigma=float(sigma);self.scale=float(scale);self.L=max(float(tail_sigma),sigma/scale+1.)
        self.mass=weights.sum(axis=0)
        with np.errstate(divide='ignore'):logs=np.log(self.weights)
        self.prefix=np.logaddexp.accumulate(logs+self.x[:,None]/scale,axis=0)
        self.suffix=np.logaddexp.accumulate((logs-self.x[:,None]/scale)[::-1],axis=0)[::-1]
        self.density_error_per_unit_mass=(math.exp(-self.L*self.L/2)/math.sqrt(2*math.pi)
            /(scale*(self.L-sigma/scale))) if sigma else 0.
        self.absolute_error_bound=self.mass*self.density_error_per_unit_mass

    # [PrivTDD] general utilities; numerical integration and stable computation; Evaluate sparse convolution at target coordinates, accelerated by a central band and analytic tails.
    def evaluate(self,targets):
        targets=np.asarray(targets,float)
        if targets.ndim!=1 or not np.isfinite(targets).all():raise ValueError('Finite target vector')
        out=np.zeros((len(targets),self.weights.shape[1]));b=self.scale;sig=self.sigma
        band=self.L*sig;common=.5*(sig/b)**2-math.log(2*b)
        active=np.flatnonzero(self.mass>0)
        step=max(1,min(1024,250000//max(1,len(self.x))))
        for offset in range(0,len(targets),step):
            y=targets[offset:offset+step];M=len(y)
            lower=np.searchsorted(self.x,y-band,side='left');upper=np.searchsorted(self.x,y+band,side='right')
            value=np.zeros((M,self.weights.shape[1]))
            valid=lower>0
            value[valid]+=np.exp(self.prefix[lower[valid]-1]-y[valid,None]/b+common)
            valid=upper<len(self.x)
            value[valid]+=np.exp(self.suffix[upper[valid]]+y[valid,None]/b+common)
            number=upper-lower;query=np.repeat(np.arange(M),number)
            if len(query):
                starts=np.r_[0,np.cumsum(number)]
                nodes=np.repeat(lower,number)+np.arange(len(query))-np.repeat(starts[:-1],number)
                density=np.exp(normal_laplace_logpdf(y[query]-self.x[nodes],sig,b))
                for column in active:
                    value[:,column]+=np.bincount(query,weights=density*self.weights[nodes,column],minlength=M)
            out[offset:offset+step]=value
        if np.any(out<0) or not np.isfinite(out).all():raise ArithmeticError('Invalid sparse density')
        return out



# [PrivTDD] general utilities; numerical integration and stable computation; Pass an integer-T conditional distribution through the count-report channel to obtain each report density.
def integer_law_channel(keys,probabilities,degree,p,scale,targets,*,tail_sigma=10.):
    """Columns may be normalized regional shapes, full sources, or zero laws."""
    keys=np.asarray(keys);mass=np.asarray(probabilities,float)
    if mass.ndim==1:mass=mass[:,None]
    W=int(degree)*(int(degree)-1)//2
    if keys.dtype.kind not in 'iu' or np.any(keys<0) or np.any(keys>W):raise ValueError('Integer source support')
    if W<=64:
        rr=np.zeros((W+1,mass.shape[1]))
        for key,row in zip(keys,mass):
            centers,prob=exact_rr_distribution(int(degree),int(key),float(p))
            rr+=prob[:,None]*row[None,:]
        centers=np.arange(W+1)-(1-p)*W
        convolution=SparseNormalLaplaceMixture(centers,rr,0.,scale,tail_sigma=tail_sigma)
    else:
        convolution=SparseNormalLaplaceMixture((2*p-1)*keys,mass,math.sqrt(W*p*(1-p)),scale,tail_sigma=tail_sigma)
    return convolution.evaluate(targets),convolution.absolute_error_bound



# [PrivTDD] general utilities; numerical integration and stable computation; Adjoint of the same count kernel, accumulating each report's evidence for candidate T.
def integer_law_adjoint(report,weights,degree,keys,p,scale,*,tail_sigma=10.):
    """Sum_i weights_i K(Q_i|d,T), using the same kernel as the forward map."""
    report=np.asarray(report,float);weights=np.asarray(weights,float);keys=np.asarray(keys,np.int64)
    if weights.ndim==1:weights=weights[:,None]
    W=int(degree)*(int(degree)-1)//2
    if np.any(keys<0) or np.any(keys>W):raise ValueError('Adjoint integer support')
    if W<=64:
        convolution=SparseNormalLaplaceMixture(report,weights,0.,scale,tail_sigma=tail_sigma)
        centers=np.arange(W+1)-(1-p)*W;values=convolution.evaluate(centers)
        out=np.empty((len(keys),weights.shape[1]))
        for index,key in enumerate(keys):
            unused,prob=exact_rr_distribution(int(degree),int(key),float(p))
            out[index]=prob@values
        return out,np.zeros(weights.shape[1])
    convolution=SparseNormalLaplaceMixture(report,weights,math.sqrt(W*p*(1-p)),scale,tail_sigma=tail_sigma)
    return convolution.evaluate((2*p-1)*keys),convolution.absolute_error_bound



# [PrivTDD] general utilities; numerical integration and stable computation; Skip structural zeros in regional components; evaluate single atoms directly and use sparse convolution otherwise.
def sparse_component_channel(keys,probabilities,degree,p,scale,targets,*,tail_sigma=10.):
    """Avoid scanning zeros in disjoint regional-source component columns."""
    keys=np.asarray(keys,np.int64);mass=np.asarray(probabilities,float)
    if mass.ndim==1:mass=mass[:,None]
    W=int(degree)*(int(degree)-1)//2
    if W<=64:return integer_law_channel(keys,mass,degree,p,scale,targets,tail_sigma=tail_sigma)
    output=np.zeros((len(targets),mass.shape[1]));error=np.zeros(mass.shape[1]);single={}
    for column in range(mass.shape[1]):
        ids=np.flatnonzero(mass[:,column]>0)
        if not len(ids):continue
        if len(ids)==1:
            t=int(keys[ids[0]])
            if t not in single:single[t]=np.exp(logpdf_q(targets,degree,t,p,scale))
            output[:,column]=mass[ids[0],column]*single[t]
        else:
            model=SparseNormalLaplaceMixture((2*p-1)*keys[ids],mass[ids,column],math.sqrt(W*p*(1-p)),scale,tail_sigma=tail_sigma)
            output[:,column]=model.evaluate(targets)[:,0];error[column]=model.absolute_error_bound[0]
    return output,error
