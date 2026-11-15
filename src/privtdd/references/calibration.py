"""Calibrate the existing one-parameter Geo source in released-count space."""
import math
import numpy as np
from scipy.special import logsumexp
from privtdd.degree.channel import log_cap_probability
from privtdd.references import geometry_model as model


def report_degree_weights(caps,prior,eps_degree):
    """q_d = mean_i P(d|C_i) choose(min(d,C_i),2)/choose(d,2).

    This applies the SAME f(T|d) for every cap, as the downstream likelihood
    does. It does not claim a cap-independent root topology for arbitrary graphs.
    """
    support=np.flatnonzero(prior>0);D=len(prior)-1;unique,counts=np.unique(caps,return_counts=True)
    logw=np.log(prior[support])[None,:]+log_cap_probability(unique[:,None],support[None,:],D,eps_degree)
    weights=np.exp(logw-logsumexp(logw,axis=1)[:,None]);weights/=weights.sum(axis=1)[:,None]
    a=np.minimum(unique[:,None],support[None,:]);W=support.astype(float)*(support-1)/2
    exposure=np.divide(a*(a-1)/2,W[None,:],out=np.zeros_like(weights),where=W[None,:]>0)
    q=counts@(weights*exposure)/len(caps)
    return support,q,dict(report_degree_weight_sum=float(q.sum()),truth_used=False,additional_epsilon=0.)


def weighted_pair_tensor(packet,degree_weights):
    """Reweight saved per-degree compositions without changing their draw counts."""
    pi=packet['pi'];support=packet['support'];q=np.asarray(degree_weights,float)
    if q.shape!=pi.shape or np.any(q<0) or not np.isfinite(q).all():raise ValueError('Degree weights')
    P=packet['probability'];G=len(P);M=np.zeros((G,G,G));diag=np.diag_indices(G)
    R=int(packet['replicates']);draws=int(packet['draws'])
    for (rep,j,root),left,right in zip(packet['blocks'],packet['offsets'][:-1],packet['offsets'][1:]):
        c=np.asarray(packet['neighbor_counts'][left:right],float);pairs=c.T@c;pairs[diag]-=c.sum(axis=0)
        denominator=max(1,int(math.ceil(draws*pi[j])))*R
        M[root]+=.5*q[j]*pairs/denominator
    js=np.flatnonzero(support==2)
    if len(js) and q[js[0]]>0:
        j=int(js[0]);odds=np.exp(packet['beta']-packet['beta'].max());roots=packet['grouped'][:,j]/packet['grouped'][:,j].sum()
        for root in range(G):
            remaining=packet['sizes'].astype(float).copy();remaining[root]-=1
            pairs=np.outer(remaining*odds,remaining*odds);pairs[diag]-=remaining*odds**2
            M[root]+=q[j]*roots[root]*pairs/pairs.sum()
    expected=float(q@(support.astype(float)*(support-1)/2))
    if not math.isclose(float(M.sum()),expected,rel_tol=3e-12,abs_tol=3e-12):raise ArithmeticError('Forward pair mass')
    return M


def calibrate_projected_geometry(packet,caps,prior,eps_degree,private_report_mean):
    support,q,audit=report_degree_weights(caps,prior,eps_degree)
    if not np.array_equal(support,packet['support']):raise ValueError('Fixed degree support mismatch')
    M=weighted_pair_tensor(packet,q);fit=model.calibrate(packet['probability'],M,private_report_mean)
    canonical=model.population_pair_weights(packet)
    fit.update(audit,canonical_endpoint_means=packet['endpoint_means'].tolist(),calibration_space='Mean released debiased count after cap projection',
        canonical_original_count_mean=model.population_mean(packet['probability'],canonical,fit['kappa']),
        conditional_uniform_neighborhood_assumed=True,reference_f_d_shared_across_caps=True)
    return fit,q
