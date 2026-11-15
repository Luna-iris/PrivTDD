"""Exact small retained-count channel under the stated conditional graph model.

Given (d,T), neighbor-neighbor edges are uniform among the choose(d,2)
possible pairs. This is a working model, not a universal projected graph law.
SciPy parameter convention: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
"""
import math
import numpy as np
from scipy.special import logsumexp
from privtdd.base_distribution.count_channel import logpdf_q
from privtdd.base_distribution.cells import closure_edges,integer_cell_weights,component_loglikelihood


def _small_sample_hypergeom(W,T,w):
    """Stable PMF ratios around a mode; work depends on w, not population W."""
    T=np.asarray(T)
    if int(W)!=W or not 0<=W<=2**52 or not 0<=w<=min(W,65) or np.any(T!=np.floor(T)) or np.any((T<0)|(T>W)):
        raise ValueError('Small-sample hypergeometric domain')
    shape=T.shape;values=T.astype(np.int64).ravel();rows=np.arange(len(values))
    lower=np.maximum(0,w-(W-values));upper=np.minimum(w,values)
    # The adjacent-PMF ratio crosses one at this mode. Relative values
    # therefore stay bounded while recursing outwards, then normalize.
    mode=np.clip(((w+1)*(values+1))//(W+2),lower,upper).astype(int)
    relative=np.zeros((len(values),w+1));relative[rows,mode]=1.
    for offset in range(1,w+1):
        k=mode+offset;active=k<=upper;ids=rows[active];j=k[active];v=values[active]
        relative[ids,j]=relative[ids,j-1]*((v-j+1)/(W-v-w+j))*((w-j+1)/j)
        k=mode-offset;active=k>=lower;ids=rows[active];j=k[active];v=values[active]
        relative[ids,j]=relative[ids,j+1]*((W-v-w+j+1)/(v-j))*((j+1)/(w-j))
    relative/=relative.sum(axis=1,keepdims=True)
    return relative.reshape(shape+(w+1,))


def retained_count_probabilities(W,T,w):
    T=np.asarray(T)
    if not 0<=w<=W or w>64 or np.any(T!=np.floor(T)) or np.any((T<0)|(T>W)):
        raise ValueError('Integer hypergeometric domain')
    if W==0:return np.ones(T.shape+(1,))
    probability=_small_sample_hypergeom(W,T,w)
    error=float(np.max(abs(probability.sum(axis=-1)-1)))
    if not np.isfinite(probability).all() or probability.min()<0 or error>2e-10:
        raise ArithmeticError('Projected count normalization')
    return probability/probability.sum(axis=-1,keepdims=True)


def sum_retained_count_probabilities(W,w,lower,upper):
    """Sum Hypergeom(W,T,w) PMFs over an inclusive integer T interval."""
    if not 0<=w<=min(W,64):raise ValueError('Small retained-count domain')
    lo=max(0,int(lower));hi=min(W,int(upper));k=np.arange(w+1)
    if lo>hi:return np.zeros(w+1)
    if w==0:return np.array([float(hi-lo+1)])
    if hi-lo<64:
        return retained_count_probabilities(W,np.arange(lo,hi+1),w).sum(axis=0)
    # The (k+1)-th order statistic of a (w+1)-subset of W+1 positions.
    # Switch to complementary tails when the survival probabilities are
    # close to one; never subtract 1-cdf by hand.
    high_probability=_small_sample_hypergeom(W+1,np.array(hi+1),w+1)
    low_probability=_small_sample_hypergeom(W+1,np.array(lo),w+1)
    high_sf=np.cumsum(high_probability[1:][::-1])[::-1];low_sf=np.cumsum(low_probability[1:][::-1])[::-1]
    low_cdf=np.cumsum(low_probability)[:-1];high_cdf=np.cumsum(high_probability)[:-1]
    difference=np.where(high_sf<=.5,high_sf-low_sf,low_cdf-high_cdf)
    if not np.isfinite(difference).all() or difference.min()<-2e-14:raise ArithmeticError('Order-statistic interval probability')
    answer=(W+1)/(w+1)*np.maximum(difference,0.)
    if abs(answer.sum()/(hi-lo+1)-1)>2e-10:raise ArithmeticError('Integrated hypergeometric mass')
    return answer


def _batched_cell_probabilities(degree,cap,edges):
    """Same integer-rounded hypergeometric integral, batching cell boundaries."""
    W=int(degree)*(int(degree)-1)//2;w=int(cap)*(int(cap)-1)//2
    e=np.asarray(edges,float);L=W*e[:-1];H=W*e[1:];width=H-L
    lo=np.clip(np.floor(L+.5).astype(np.int64),0,W)
    hi=np.clip(np.floor(H+.5).astype(np.int64),0,W)
    first=_small_sample_hypergeom(W,lo,w)
    last=_small_sample_hypergeom(W,hi,w)
    result=first*np.maximum(0,np.minimum(H,lo+.5)-np.maximum(L,lo-.5))[:,None]
    different=hi>lo
    result[different]+=last[different]*np.maximum(0,np.minimum(H,hi+.5)-np.maximum(L,hi-.5))[different,None]
    number=hi-lo-1
    short=np.flatnonzero((number>0)&(number<=64))
    if len(short):
        values=np.concatenate([np.arange(lo[j]+1,hi[j]) for j in short])
        owners=np.repeat(short,number[short])
        np.add.at(result,owners,_small_sample_hypergeom(W,values,w))
    long=np.flatnonzero(number>64)
    if len(long):
        high=_small_sample_hypergeom(W+1,hi[long],w+1)
        low=_small_sample_hypergeom(W+1,lo[long]+1,w+1)
        high_sf=np.cumsum(high[:,1:][:,::-1],axis=1)[:,::-1]
        low_sf=np.cumsum(low[:,1:][:,::-1],axis=1)[:,::-1]
        high_cdf=np.cumsum(high,axis=1)[:,:-1]
        low_cdf=np.cumsum(low,axis=1)[:,:-1]
        diff=np.where(high_sf<=.5,high_sf-low_sf,low_cdf-high_cdf)
        if not np.isfinite(diff).all() or diff.min()<-2e-14:raise ArithmeticError('Cell interval probability')
        result[long]+=(W+1)/(w+1)*np.maximum(diff,0.)
    result/=width[:,None]
    error=float(np.max(abs(result.sum(1)-1)))
    if error>2e-10 or result.min()<0:raise ArithmeticError('Cell mass')
    result/=result.sum(1)[:,None]
    probability=np.zeros((w+1,len(e)+1));probability[0,0]=1;probability[-1,-1]=1
    probability[:,1:-1]=result.T
    return probability,error




def projected_cell_count_probabilities(degree,cap,edges=None):
    """Rows are retained counts; columns match the original 65 closure cells."""
    if int(degree)!=degree or int(cap)!=cap or degree<0 or cap<0:raise ValueError('Degree/cap')
    d=int(degree);a=min(d,int(cap));W=d*(d-1)//2;w=a*(a-1)//2
    if w>64:raise NotImplementedError('Large retained-count approximation is a separate stage')
    e=closure_edges() if edges is None else np.asarray(edges,float)
    if e[0]!=0 or e[-1]!=1 or np.any(np.diff(e)<=0):raise ValueError('Closure partition')
    columns=len(e)+1
    if w==0:return np.ones((1,columns)),dict(maximum_pre_normalization_mass_error=0.,conditional_graph_model=True)
    probability=np.zeros((w+1,columns));probability[0,0]=1.;probability[-1,-1]=1.
    if d<=cap:
        probability[:,1:-1]=integer_cell_weights(W,e)
    else:
        # Equivalent integer-rounding integral, batching all closure-cell endpoints.
        # Short intervals use direct PMF sums; long intervals use the original
        # order-statistic identity. No new approximation or fitted parameter.
        probability,_=_batched_cell_probabilities(degree,cap,e)
    error=float(np.max(abs(probability.sum(axis=0)-1)))
    if error>2e-10 or probability.min()<0:raise ArithmeticError('Projected closure-cell normalization')
    probability/=probability.sum(axis=0,keepdims=True)
    return probability,dict(maximum_pre_normalization_mass_error=error,conditional_graph_model=True,
        original_capacity=W,retained_capacity=w,integer_cell_prior=True,RR_count_exact=True)


def projected_logpdf_q(report,degree,cap,triangles,p,scale):
    if int(degree)!=degree or int(cap)!=cap or degree<0 or cap<0:raise ValueError('Degree/cap')
    d=int(degree);c=int(cap);W=d*(d-1)//2
    y,T,b=np.broadcast_arrays(np.asarray(report,float),np.asarray(triangles,float),np.asarray(scale,float))
    if np.any(T!=np.floor(T)) or np.any((T<0)|(T>W)):raise ValueError('Latent integer count')
    if c<=1:return np.where(y==0,0.,-np.inf)
    if d<=c:return logpdf_q(y,d,T,p,b)
    w=c*(c-1)//2
    if w>64:raise NotImplementedError('Large retained-count approximation is a separate stage')
    probability=retained_count_probabilities(W,T,w)
    with np.errstate(divide='ignore'):lp=np.log(probability)
    channel=logpdf_q(y[...,None],c,np.arange(w+1),p,b[...,None])
    return logsumexp(lp+channel,axis=-1)


def projected_component_loglikelihood(report,degree,cap,p,scale,edges=None):
    y,b=np.broadcast_arrays(np.asarray(report,float),np.asarray(scale,float))
    if y.ndim!=1:raise ValueError('Report vector')
    e=closure_edges() if edges is None else np.asarray(edges,float)
    if cap<=1:return np.broadcast_to(np.where(y==0,0.,-np.inf)[:,None],(len(y),len(e)+1)).copy()
    if degree<=cap:return component_loglikelihood(y,int(degree),p,b,e)
    probability,audit=projected_cell_count_probabilities(degree,cap,e);a=min(degree,cap);w=a*(a-1)//2
    values=logpdf_q(y[:,None],int(a),np.arange(w+1),p,b[:,None]);anchor=values.max(axis=1)
    with np.errstate(divide='ignore'):answer=anchor[:,None]+np.log(np.exp(values-anchor[:,None])@probability)
    return answer


def projected_partial_cell_mass(W, w, left, keys):
    """[S3 equivalent batching] Integrate the existing small-cap working law.

    W: original integer capacity; w: retained capacity, at most 64.
    left: closure cell's left endpoint; keys: integer CDF endpoints, with
    left < (keys+.5)/W < 1. Returns one row per key and one column per
    retained count, integrating Hypergeom(W, round(W*r), w) in r.
    The fixed switch 64 selects direct prefix sums versus the same existing
    order-statistic identity for numerical stability; it is not a new bucket
    width, graph truncation, or statistical approximation.
    """
    keys = np.asarray(keys, np.int64)
    L = W*float(left)
    lo = int(np.floor(L+.5))
    if np.any(keys < lo) or np.any(keys >= W) or not 0 <= w <= min(W,64):
        raise ValueError('Partial projected cell domain')
    first = _small_sample_hypergeom(W, np.array(lo), w)*(lo+.5-L)
    result = np.broadcast_to(first, (len(keys), w+1)).copy()
    number = keys-lo
    short = (number > 0) & (number < 64)
    if short.any():
        values = np.arange(lo+1, lo+int(number[short].max())+1)
        prefix = np.cumsum(_small_sample_hypergeom(W, values, w), axis=0)
        result[short] += prefix[number[short]-1]
    long = number >= 64
    if long.any():
        high = _small_sample_hypergeom(W+1, keys[long]+1, w+1)
        low = _small_sample_hypergeom(W+1, np.array(lo+1), w+1)
        high_sf = np.cumsum(high[:,1:][:,::-1], axis=1)[:,::-1]
        low_sf = np.cumsum(low[1:][::-1])[::-1]
        high_cdf = np.cumsum(high, axis=1)[:,:-1]
        low_cdf = np.cumsum(low)[:-1]
        difference = np.where(high_sf <= .5, high_sf-low_sf, low_cdf-high_cdf)
        if not np.isfinite(difference).all() or difference.min() < -2e-14:
            raise ArithmeticError('Projected interval probability')
        result[long] += (W+1)/(w+1)*np.maximum(difference,0.)
    return result/W
