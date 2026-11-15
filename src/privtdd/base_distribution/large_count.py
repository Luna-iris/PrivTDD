"""Projected mean-count approximation with absolute density error bounds.

The reference is hypergeometric thinning followed by the existing Gaussian
RR / Laplace working law. These bounds do NOT certify Gaussian versus exact
RR or the conditional uniform-neighborhood assumption for arbitrary graphs.
"""
import math
import numpy as np
from privtdd.base_distribution.count_channel import normal_laplace_logpdf
from privtdd.base_distribution.cells import closure_edges,component_loglikelihood


def smoothing_constants(w,p,scale):
    b=np.asarray(scale,float)
    if w<=0 or not .5<p<1 or np.any(b<=0) or not np.isfinite(b).all():
        raise ValueError('Positive capacity, scale and RR probability required')
    sigma=math.sqrt(w*p*(1-p))
    # Either convolution factor bounds the corresponding derivative.
    first=np.minimum(1/(2*b*b),math.exp(-.5)/(math.sqrt(2*math.pi)*sigma*sigma))
    second=np.minimum(np.maximum(1/(2*b),1/(math.sqrt(2*math.pi)*sigma))/(b*b),
                      1/(math.sqrt(2*math.pi)*sigma**3))
    return sigma,first,second


def projected_mean_count_density(report,degree,cap,triangles,p,scale):
    """Approximate density and error versus the hypergeometric/Normal mixture."""
    if int(degree)!=degree or int(cap)!=cap or not 2<=cap<degree:raise ValueError('Projected degree/cap')
    W=degree*(degree-1)//2;w=cap*(cap-1)//2
    if w<=64:raise ValueError('Use exact small-count channel')
    y,T,b=np.broadcast_arrays(np.asarray(report,float),np.asarray(triangles,float),np.asarray(scale,float))
    if np.any(T!=np.floor(T)) or np.any((T<0)|(T>W)):raise ValueError('Original integer count')
    sigma,first,second=smoothing_constants(w,p,b);rho=T/W;gap=2*p-1
    variance=w*rho*(1-rho)*(W-w)/(W-1)
    estimate=np.exp(normal_laplace_logpdf(y-gap*w*rho,sigma,b))
    error=np.minimum(1/(2*b),.5*gap*gap*variance*second)
    return estimate,error


def projected_large_component_likelihood(report,degree,cap,p,scale,edges=None):
    """Cell estimates and component-wise absolute error to rounded-T reference.

Continuous closure is integrated analytically at mean retained count w*rho.
The error covers hypergeometric variability and original-T rounding separately.
Endpoint counts have neither error. It excludes floating-point certification.
"""
    if int(degree)!=degree or int(cap)!=cap or not 2<=cap<degree:raise ValueError('Projected degree/cap')
    W=degree*(degree-1)//2;w=cap*(cap-1)//2
    if w<=64:raise ValueError('Use exact small-count channel')
    y,b=np.broadcast_arrays(np.asarray(report,float),np.asarray(scale,float))
    if y.ndim!=1:raise ValueError('Report vector')
    e=closure_edges() if edges is None else np.asarray(edges,float)
    sigma,first,second=smoothing_constants(w,p,b);gap=2*p-1
    estimate=np.exp(component_loglikelihood(y,int(cap),p,b,e))
    # The rounded original rho is at most 1/(2W) outside each cell.
    left=np.maximum(0,e[:-1]-.5/W);right=np.minimum(1,e[1:]+.5/W)
    peak=np.clip(.5,left,right);variance=w*peak*(1-peak)*(W-w)/(W-1)
    hypergeom=.5*gap*gap*second[:,None]*variance[None,:]
    rounding=first[:,None]*(gap*w/(2*W))
    error=np.zeros_like(estimate)
    error[:,1:-1]=np.minimum(1/(2*b[:,None]),hypergeom+rounding)
    return estimate,error,dict(reference='Conditional uniform graph; rounded original count; Gaussian RR working law',
        original_capacity=W,retained_capacity=w,hypergeometric_variability_bounded=True,
        original_count_rounding_bounded=True,exact_RR_error_bounded=False,
        floating_point_error_bounded=False)
