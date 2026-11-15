"""Geometric reference model for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.optimize import brentq

import math

import numpy as np

import privtdd.references.primitives as old



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Construct nested arc widths and mixture weights for kappa, preserving marginal edge probabilities for each type pair.
def arc_mixture(P,kappa):
    P=np.asarray(P,float)
    if not np.isfinite(P).all() or np.any(P<=0) or np.any(P>1) or not 0<=kappa<=1:raise ValueError('Invalid arc mechanism')
    if kappa==0:return np.zeros_like(P),np.ones_like(P),1-P
    if kappa==1:return P.copy(),P.copy(),np.ones_like(P)
    a=kappa*P;b=np.minimum(P/kappa,1.)
    w=np.divide(b-P,b-a,out=np.ones_like(P),where=b!=a)
    if np.min(w)<-2e-14 or np.max(w)>1+2e-14:raise ArithmeticError('Invalid arc mixture weight')
    w=np.clip(w,0,1)
    if np.max(abs(w*a+(1-w)*b-P))>2e-14:raise ArithmeticError('Edge marginal changed')
    return a,b,w



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Compute type-pair closure probabilities between two neighbors of a specified root type.
def conditional_closure(P,root,kappa):
    if kappa==0:return P.copy()
    a,b,w=arc_mixture(P,kappa)
    if kappa==1:
        width=P[root];return old.circle_closure(width[:,None],width[None,:],P)
    v=w[root]*a[root]/P[root];C=np.zeros_like(P)
    for width_left,wl in ((a[root],v),(b[root],1-v)):
        for width_right,wr in ((a[root],v),(b[root],1-v)):
            C+=wl[:,None]*wr[None,:]*(w*old.circle_closure(width_left[:,None],width_right[None,:],a)
                +(1-w)*old.circle_closure(width_left[:,None],width_right[None,:],b))
    if C.min() < -2e-14 or C.max()>1+2e-14:raise ArithmeticError('Invalid conditional closure')
    return np.clip(C,0,1)



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Construct population weights over root type x neighbor-type pairs from fixed-degree neighbor compositions.
def population_pair_weights(packet):
    P=packet['probability'];G=len(P);M=np.zeros((G,G,G));support=packet['support'];pi=packet['pi']
    R=int(packet['replicates']);draws=int(packet['draws']);diagonal=np.diag_indices(G)
    for (rep,j,root),a,b in zip(packet['blocks'],packet['offsets'][:-1],packet['offsets'][1:]):
        c=np.asarray(packet['neighbor_counts'][a:b],float);pairs=c.T@c;pairs[diagonal]-=c.sum(axis=0)
        count=max(1,int(math.ceil(draws*pi[j])))*R
        M[root]+=.5*pi[j]*pairs/count
    # Degree2 was integrated exactly, with no composition draws.
    js=np.flatnonzero(support==2)
    if len(js) and pi[js[0]]>0:
        j=int(js[0]);odds=np.exp(packet['beta']-packet['beta'].max())
        roots=packet['grouped'][:,j]/packet['grouped'][:,j].sum()
        for root in range(G):
            remaining=packet['sizes'].astype(float).copy();remaining[root]-=1
            pair=np.outer(remaining*odds,remaining*odds);pair[diagonal]-=remaining*odds**2
            M[root]+=pi[j]*roots[root]*pair/pair.sum()
    expected=float(pi@(support.astype(float)*(support-1)/2))
    if not math.isclose(float(M.sum()),expected,rel_tol=2e-12,abs_tol=2e-12):raise ArithmeticError('Pair-weight mass mismatch')
    return M



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Integrate conditional closure probabilities over structural weights to obtain the model triangle mean.
def population_mean(P,M,kappa):
    return math.fsum(float(np.sum(M[root]*conditional_closure(P,root,kappa))) for root in range(len(P)))



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Clip the private mean to the range between the two endpoint means and solve for geometric strength kappa in [0,1].
def calibrate(P,M,private_mean):
    lo=population_mean(P,M,0.);hi=population_mean(P,M,1.)
    target=float(np.clip(private_mean,min(lo,hi),max(lo,hi)));evaluations=[]
    # [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Mean residual for the one-dimensional root solve of geometric strength.
    def f(k):
        mu=population_mean(P,M,float(k));evaluations.append(dict(kappa=float(k),mean=mu));return mu-target
    if lo==hi or target==lo:kappa=0.
    elif target==hi:kappa=1.
    else:kappa=float(brentq(f,0.,1.,xtol=1e-10))
    return dict(kappa=kappa,private_mean=float(private_mean),target_mean=target,beta_endpoint_mean=lo,hard_endpoint_mean=hi,
        implied_mean=population_mean(P,M,kappa),clipped=target!=private_mean,coincident_endpoints=lo==hi,
        evaluations=evaluations,root_rule='deterministic Brent solve on the fixed endpoint bracket; no monotonicity or uniqueness claim',
        statistical_parameters=1,additional_epsilon=0.,truth_used=False)



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Analytically compute the structural closure probability for degree 2.
def degree_two_closure(packet,kappa):
    P=packet['probability'];G=len(P);odds=np.exp(packet['beta']-packet['beta'].max());means=np.zeros(G)
    for root in range(G):
        remaining=packet['sizes'].astype(float).copy();remaining[root]-=1
        pairs=np.outer(remaining*odds,remaining*odds);pairs[np.diag_indices(G)]-=remaining*odds**2
        means[root]=np.sum(pairs*conditional_closure(P,root,kappa))/pairs.sum()
    return means
