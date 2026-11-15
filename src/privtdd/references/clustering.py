"""Clustering reference construction for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.distributions.conditional import ConditionalDistribution

import math

import privtdd.references.geometry_model as model

import numpy as np

import privtdd.references.independent_edge as previous



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Given a root type and neighbor composition, sample triangle counts using the inner/outer-ring mechanism of the concentric-circle model.
def draw_count(counts,root,P,arcs,rng):
    counts=np.asarray(counts,np.int64);active=np.flatnonzero(counts);c=counts[active]
    d=int(c.sum())
    if d<2:return 0
    a,b,w=arcs
    offsets=np.r_[0,np.cumsum(c)];positions=np.empty(d)
    for g,typ in enumerate(active):
        # Choose a mixture component BEFORE sorting iid positions.
        v=w[root,typ]*a[root,typ]/P[root,typ]
        width=np.where(rng.random(int(c[g]))<v,a[root,typ],b[root,typ])
        positions[offsets[g]:offsets[g+1]]=np.sort(rng.uniform(-.5,.5,int(c[g]))*width)
    inner=previous.near_pair_counts(positions,offsets,a[np.ix_(active,active)])
    outer=previous.near_pair_counts(positions,offsets,b[np.ix_(active,active)])
    i,j=np.triu_indices(len(active));totals=c[i]*c[j]
    diag=i==j;totals[diag]=c[i[diag]]*(c[i[diag]]-1)//2
    near=inner[i,j];ring=outer[i,j]-near
    if np.any(near<0) or np.any(ring<0) or np.any(near+ring>totals):
        raise ArithmeticError('Invalid nested pair counts')
    return int(near.sum()+rng.binomial(ring,1-w[np.ix_(active,active)][i,j]).sum())



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Aggregate retained count samples by degree into conditional distributions; retain analytic treatment for d=0/1/2.
def assemble(packet,draws,kappa):
    """Reconstruct a conditional source from retained draws and exact d<=2."""
    support=packet['support'];pi=packet['pi'];R=int(packet['replicates'])
    draws=np.asarray(draws)
    if draws.shape!=(len(packet['neighbor_counts']),) or not np.issubdtype(draws.dtype,np.integer):
        raise ValueError('Expected one integer count per saved neighbor composition')
    two=model.degree_two_closure(packet,kappa);keys=[];probs=[];repmeans=np.zeros(R)
    for j,d in enumerate(support):
        if pi[j]==0 or d<2:keys.append(np.array([0],np.int64));probs.append(np.array([1.]));continue
        if d==2:
            mu=float(packet['grouped'][:,j]@two/packet['grouped'][:,j].sum())
            keys.append(np.array([0,1],np.int64));probs.append(np.array([1-mu,mu]));repmeans+=pi[j]*mu
            continue
        pooled=[]
        for rep in range(R):
            ids=np.flatnonzero((packet['blocks'][:,0]==rep)&(packet['blocks'][:,1]==j))
            values=np.concatenate([draws[packet['offsets'][i]:packet['offsets'][i+1]] for i in ids])
            expected=max(1,int(math.ceil(int(packet['draws'])*pi[j])))
            if len(values)!=expected or np.any(values<0) or np.any(values>int(d)*(int(d)-1)//2):
                raise ArithmeticError('Draw count or capacity mismatch')
            pooled.append(values);repmeans[rep]+=pi[j]*values.mean()
        k,n=np.unique(np.concatenate(pooled),return_counts=True);keys.append(k);probs.append(n/n.sum())
    law=ConditionalDistribution('concentric_geo_generated',support,tuple(keys),tuple(probs),pi,
        dict(kappa=kappa,degree_zero_one_two_analytic=True,truth_used=False,additional_epsilon=0.))
    return law,repmeans



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Calibrate concentric-circle kappa and generate Geo conditional laws using the same fixed-degree compositions.
def generate(packet,private_mean,*,endpoint_draws=None,calibration=None,compiled=False):
    # calibration: Calibration already computed for the same packet/private_mean, avoiding repeated root finding on the lazy-endpoint path.
    P=packet['probability']
    if calibration is None:
        M=model.population_pair_weights(packet);audit=model.calibrate(P,M,private_mean)
    else:
        audit=dict(calibration)
    kappa=audit['kappa'];R=int(packet['replicates']);arcs=model.arc_mixture(P,kappa)
    endpoint_means=audit.get('canonical_endpoint_means',[audit['beta_endpoint_mean'],audit['hard_endpoint_mean']])
    if not np.allclose(endpoint_means,packet['endpoint_means'],rtol=2e-12,atol=1e-9):raise ArithmeticError('Geometry endpoint identity')
    rngs={r:np.random.default_rng(np.random.SeedSequence([731+100003*r,4])) for r in range(R)}
    draws=np.empty(len(packet['neighbor_counts']),np.int64)
    if kappa in (0.,1.) and endpoint_draws is not None:
        endpoints=np.asarray(endpoint_draws)
        if endpoints.shape!=(len(draws),2):raise ValueError('Invalid endpoint draws')
        draws[:]=endpoints[:,int(kappa)]
    else:
        for (rep,j,root),a,b in zip(packet['blocks'],packet['offsets'][:-1],packet['offsets'][1:]):
            if compiled:
                from privtdd.references.kernels import geometry_block
                draws[a:b] = geometry_block(packet['neighbor_counts'][a:b], int(root), P, *arcs, rngs[int(rep)])
            else:
                for index in range(a,b):draws[index]=draw_count(packet['neighbor_counts'][index],root,P,arcs,rngs[int(rep)])
    law,repmeans=assemble(packet,draws,kappa)
    audit.update(integrated_mean=float(repmeans.mean()),replicate_integrated_means=repmeans.tolist(),
        replicate_mean_SE=float(repmeans.std(ddof=1)/math.sqrt(R)) if R>1 else None,
        mean_minus_analytic=float(repmeans.mean()-audit.get('canonical_original_count_mean',audit['implied_mean'])),
        integration_convergence_claimed=False,independent_private_reports_claimed=False,
        random_stream='SeedSequence([731+100003*r,4]); immutable existing compositions; endpoint draws reused only at kappa0/1')
    return law,draws,audit
