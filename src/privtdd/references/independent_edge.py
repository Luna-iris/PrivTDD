"""Independent edge reference calculations for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.distributions.conditional import ConditionalDistribution

import math

from numba import njit

import numpy as np

import privtdd.references.primitives as old



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Count nearby type pairs at sorted circular positions, avoiding a neighbor-squared adjacency matrix.
@njit(cache=True)
def near_pair_counts(positions,offsets,probability):
    groups=len(offsets)-1;answer=np.zeros((groups,groups),np.int64)
    for g in range(groups):
        x=positions[offsets[g]:offsets[g+1]]
        for h in range(g,groups):
            y=positions[offsets[h]:offsets[h+1]];radius=probability[g,h]/2
            ordered=old._ordered_close(x,y,radius)+len(x)*len(y)-old._ordered_close(x,y,1-radius)
            answer[g,h]=(ordered-len(x))//2 if g==h else ordered
    return answer



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Given a neighbor composition, compute the conditional triangle mean under independent type-pair closure.
def pair_means(counts,probability):
    counts=np.asarray(counts,float)
    return .5*(np.einsum('ij,ij->i',counts@probability,counts)-counts@np.diag(probability))



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Analytically compute independent-edge and hard-circle endpoint means for degree 2.
def degree_two_means(sizes,beta,probability):
    odds=np.exp(beta-np.max(beta));means=np.zeros((len(sizes),2))
    for root in range(len(sizes)):
        remaining=np.asarray(sizes,float).copy();remaining[root]-=1
        weights=np.outer(remaining*odds,remaining*odds);weights[np.diag_indices(len(sizes))]-=remaining*odds**2
        if weights.sum()<=0:raise ValueError('No degree-two neighborhood')
        arcs=probability[root];C=old.circle_closure(arcs[:,None],arcs[None,:],probability)
        means[root]=[np.sum(weights*probability)/weights.sum(),np.sum(weights*C)/weights.sum()]
    return means



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Auxiliary calibration of the original endpoint generator; the current caller supplies the Beta endpoint mean, so lam=0.
def estimate_strength(private_mean,endpoint_means):
    beta_mean,geo_mean=map(float,endpoint_means)
    if not np.isfinite([private_mean,beta_mean,geo_mean]).all():raise ValueError('Invalid means')
    gap=geo_mean-beta_mean
    eta=float(np.clip((private_mean-beta_mean)/gap,0,1)) if gap!=0 else 0.
    return dict(eta=eta,lambda_value=float(np.cbrt(eta)),private_mean=float(private_mean),
        beta_exact_mean=beta_mean,hard_geo_mean=geo_mean,
        implied_soft_mean=beta_mean+eta*gap,
        clipped=bool(private_mean<min(beta_mean,geo_mean) or private_mean>max(beta_mean,geo_mean)),
        nonidentifiable=gap==0,optimized_parameters_added=0,moment_estimated_parameters=1)



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Reuse independent random streams to generate Beta/hard-circle endpoints; the current soft endpoint reduces to Beta and is not a joint-family source.
def draw_three(counts,root,P,lam,rng_beta,rng_hard,rng_soft,*,beta_only=False):
    if not 0<=lam<=1:raise ValueError('lambda must lie in [0,1]')
    counts=np.asarray(counts,np.int64);active=np.flatnonzero(counts);c=counts[active];d=int(c.sum())
    if d<2:return np.zeros(3,np.int64)
    p=P[np.ix_(active,active)];i,j=np.triu_indices(len(active))
    totals=c[i]*c[j];diag=i==j;totals[diag]=c[i[diag]]*(c[i[diag]]-1)//2
    take=totals>0;i,j,totals=i[take],j[take],totals[take]
    B=int(rng_beta.binomial(totals,p[i,j]).sum())
    # [Equivalent implementation] Random streams are independent; skipping unused hard endpoints does not change Beta sampling.
    if beta_only:
        if lam != 0:
            raise ValueError('Beta-only endpoint requires lambda=0')
        return np.array([B, 0, B], np.int64)
    offsets=np.r_[0,np.cumsum(c)];hard=np.empty(d);soft=np.empty(d)
    for g,a in enumerate(active):
        sl=slice(offsets[g],offsets[g+1]);width=P[root,a]
        hard[sl]=np.sort(rng_hard.uniform(-width/2,width/2,int(c[g])))
        if 0<lam<1:
            positions=np.where(rng_soft.random(int(c[g]))<lam,hard[sl],rng_soft.uniform(-.5,.5,int(c[g])))
            soft[sl]=np.sort(positions)
    H=int(old.geometry_count(hard,offsets,p))
    if lam==0:S=B
    elif lam==1:S=H
    else:
        near=near_pair_counts(soft,offsets,p)[i,j]
        if np.any(near<0) or np.any(near>totals):raise ArithmeticError('Invalid near-pair count')
        S=int((rng_soft.binomial(near,lam+(1-lam)*p[i,j])+rng_soft.binomial(totals-near,(1-lam)*p[i,j])).sum())
    return np.array([B,H,S],np.int64)


def generate_beta(packet, *, compiled=False):
    """[S4 equivalent implementation] Generate only the Beta conditional law actually used by the joint family.
    
    packet: Already generated fixed neighbor compositions, type probabilities and structural random-stream information.
    Use the same SeedSequence, order and binomial calls as Beta in generate; do not generate the hard-circle endpoint.
    Return (Beta conditional distribution, one-dimensional integer samples). Used only when Geo does not require a hard endpoint.
    """
    from privtdd.references.clustering import assemble
    P = packet['probability']
    rngs = {r: np.random.default_rng(np.random.SeedSequence([731+100003*r, 1]))
            for r in range(int(packet['replicates']))}
    draws = np.empty(len(packet['neighbor_counts']), np.int64)
    for (rep, j, root), a, b in zip(packet['blocks'], packet['offsets'][:-1], packet['offsets'][1:]):
        if compiled:
            from privtdd.references.kernels import endpoint_block
            draws[a:b] = endpoint_block(packet['neighbor_counts'][a:b], int(root), P,
                                       rngs[int(rep)], rngs[int(rep)], False)[:, 0]
        else:
            for index in range(a, b):
                draws[index] = draw_three(packet['neighbor_counts'][index], root, P, 0.,
                                         rngs[int(rep)], None, None, beta_only=True)[0]
    source, _ = assemble(packet, draws, 0.)
    law = ConditionalDistribution('beta_exact_conditioned', source.support, source.keys_by_degree, source.probability_by_degree,
        source.native_degree_mass, dict(lambda_value=0., degree_zero_one_two_analytic=True, truth_used=False,
                       additional_epsilon=0., skipped_unused_hard_endpoint=True))
    return law, draws



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Generate independent-edge/hard-circle endpoint conditional laws; currently retain only Beta for the joint family.
def generate(packet,private_mean,*,compiled=False):
    strength=estimate_strength(private_mean,packet['endpoint_means']);lam=strength['lambda_value'];eta=strength['eta']
    support=packet['support'];pi=packet['pi'];P=packet['probability'];replicates=int(packet['replicates'])
    draws=np.empty((len(packet['neighbor_counts']),3),np.int64)
    rngs={r:tuple(np.random.default_rng(np.random.SeedSequence([731+100003*r,stream])) for stream in (1,2,3)) for r in range(replicates)}
    for (rep,j,root),a,b in zip(packet['blocks'],packet['offsets'][:-1],packet['offsets'][1:]):
        if compiled and lam == 0.:
            from privtdd.references.kernels import endpoint_block
            draws[a:b] = endpoint_block(packet['neighbor_counts'][a:b], int(root), P,
                                        rngs[int(rep)][0], rngs[int(rep)][1], True)
        else:
            for index in range(a,b):
                draws[index]=draw_three(packet['neighbor_counts'][index],root,P,lam,*rngs[int(rep)])
    keys=[[] for _ in range(3)];probabilities=[[] for _ in range(3)];rep_means=np.zeros((replicates,3))
    for j,d in enumerate(support):
        if pi[j]==0 or d<2:
            for m in range(3):keys[m].append(np.array([0],np.int64));probabilities[m].append(np.array([1.]))
        elif d==2:
            mu=packet['grouped'][:,j]@packet['two_means']/packet['grouped'][:,j].sum()
            values=[mu[0],mu[1],mu[0]+eta*(mu[1]-mu[0])]
            for m,v in enumerate(values):keys[m].append(np.array([0,1],np.int64));probabilities[m].append(np.array([1-v,v]));rep_means[:,m]+=pi[j]*v
        else:
            block_ids=np.flatnonzero(packet['blocks'][:,1]==j);pooled=[]
            for rep in range(replicates):
                ids=[v for v in block_ids if packet['blocks'][v,0]==rep]
                v=np.concatenate([draws[packet['offsets'][i]:packet['offsets'][i+1]] for i in ids]);pooled.append(v)
                rep_means[rep]+=pi[j]*v.mean(axis=0)
            v=np.concatenate(pooled)
            if np.any(v<0) or np.any(v>int(d)*(int(d)-1)//2):raise ArithmeticError('Count outside capacity')
            for m in range(3):
                k,n=np.unique(v[:,m],return_counts=True);keys[m].append(k);probabilities[m].append(n/n.sum())
    names=('beta_exact_conditioned','hard_geo_generated','soft_geo_generated')
    laws=tuple(ConditionalDistribution(name,support,tuple(keys[m]),tuple(probabilities[m]),pi,
        dict(lambda_value=(0.,1.,lam)[m],degree_zero_one_two_analytic=True,truth_used=False,additional_epsilon=0.)) for m,name in enumerate(names))
    means=rep_means.mean(axis=0)
    audit=dict(strength,integrated_source_means=means.tolist(),replicate_integrated_means=rep_means.tolist(),
        replicate_mean_SE=(np.std(rep_means,axis=0,ddof=1)/math.sqrt(replicates)).tolist() if replicates>1 else None,
        source_mean_minus_analytic_endpoint=[float(means[0]-packet['endpoint_means'][0]),float(means[1]-packet['endpoint_means'][1]),float(means[2]-strength['implied_soft_mean'])],
        integration_convergence_claimed=False,independent_private_reports_claimed=False,truth_used=False,additional_epsilon=0.,
        random_streams='composition seed731+100003*r; separate SeedSequence([seed,1/2/3]) for beta/hard/soft')
    return laws,draws,audit
