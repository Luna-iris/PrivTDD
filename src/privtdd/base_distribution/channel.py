"""Base distribution observation channel for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from privtdd.base_distribution.cells import closure_edges

from privtdd.io.memory import component_loglikelihood

from privtdd.degree.channel import log_cap_probability

from scipy.special import logsumexp
from privtdd.base_distribution.projected_count_channel import projected_component_loglikelihood

import math

import numpy as np



# [Core S3 projection completion] Compute soft P(d|cap), retaining d>cap and omitting only a numerical tail of cumulative mass 1e-16.
def prepare_groups(caps,prior,eps_degree,*,omission_tolerance=1e-16):
    caps=np.asarray(caps,int);prior=np.asarray(prior,float);D=len(prior)-1
    if np.any(prior<0) or abs(prior.sum()-1)>2e-12:raise ValueError('Degree simplex')
    support=np.flatnonzero(prior>0);logpi=np.log(prior[support])
    order=np.argsort(caps,kind='stable');unique,first,counts=np.unique(caps[order],return_index=True,return_counts=True)
    groups=[]
    for cap,start,count in zip(unique,first,counts):
        logp=logpi+log_cap_probability(int(cap),support,D,eps_degree)
        post=np.exp(logp-logsumexp(logp));post/=post.sum()
        # C<=1 has a deterministic zero report for every latent original degree.
        eligible=np.ones(len(support),bool) # Model every d, including projected d>cap.
        projected=math.fsum(post[~eligible].tolist())
        ids=np.flatnonzero(eligible);rank=ids[np.argsort(-post[ids],kind='stable')]
        suffix=np.r_[np.cumsum(post[rank][::-1])[::-1],0.]
        retained=int(np.flatnonzero(suffix<=omission_tolerance)[0])
        kept=np.sort(rank[:retained]);omitted=math.fsum(post[rank[retained:]].tolist())
        groups.append(dict(cap=int(cap),nodes=order[start:start+count],degree=support[kept],posterior=post[kept],
            projected_posterior_mass=float(post[support>cap].sum()),
            omitted_projected=projected,omitted_numerical=omitted))
    return groups



# [PrivTDD] S3 F1; core method or probability/structure primitive (not necessarily a separate contribution); Integrate F1 components with soft degree weights for each node to form an n x 65 likelihood table; disjoint cap groups run in parallel.
def build_matrix(report,caps,prior,p,eps_degree,eps_release,*,omission_tolerance=1e-16,grid_size=65,workers=8,batch_degrees=False):
    report=np.asarray(report,float);caps=np.asarray(caps,int);edges=closure_edges(grid_size)
    if report.shape!=caps.shape or not np.isfinite(report).all():raise ValueError('Report inputs')
    groups=prepare_groups(caps,prior,eps_degree,omission_tolerance=omission_tolerance)
    A=np.zeros((len(report),grid_size));delta=np.zeros(len(report));pairs=0
    # [PrivTDD] S3 F1; memory and parallel execution; Forward component contribution of one cap group; writes to disjoint node rows.
    def group_task(g):
        ids=g['nodes'];cap=g['cap'];b=max(cap-1,0)/eps_release
        if cap<=1:
            if np.any(report[ids]!=0):raise ArithmeticError('Nonzero deterministic projected report')
            A[ids]=g['posterior'].sum();delta[ids]=g['omitted_numerical']
        else:
            ordinary=g['degree']<=cap
            for d,weight in zip(g['degree'][ordinary],g['posterior'][ordinary]):
                A[ids]+=weight*np.exp(component_loglikelihood(report[ids],int(d),p,b,edges))
            projected_task(g)
            delta[ids]=(g['omitted_projected']+g['omitted_numerical'])/(2*b)
        return len(ids)*len(g['degree'])
    def projected_task(g):
        # Large retained-cap closure law depends on cap, not original d.
        cap=g['cap'];ids=g['nodes'];selected=g['degree']>cap
        if cap<=1 or not selected.any():return
        b=(cap-1)/eps_release
        if cap*(cap-1)//2>64:
            A[ids]+=g['posterior'][selected].sum()*np.exp(component_loglikelihood(report[ids],cap,p,b,edges))
        else:
            for d,weight in zip(g['degree'][selected],g['posterior'][selected]):
                A[ids]+=weight*np.exp(projected_component_loglikelihood(report[ids],int(d),cap,p,b,edges))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if batch_degrees:
            from privtdd.degree.groups import groups_by_degree
            active_groups = [dict(g,degree=g['degree'][g['degree']<=g['cap']],
                posterior=g['posterior'][g['degree']<=g['cap']]) for g in groups if g['cap']>1]
            by_degree = groups_by_degree(active_groups)
            for g in groups:
                ids = g['nodes']
                if g['cap'] <= 1:
                    group_task(g)
                else:
                    delta[ids] = (g['omitted_projected']+g['omitted_numerical'])/(2*((g['cap']-1)/eps_release))
            # Finish each d before advancing to the next, preserving ascending-degree accumulation per report; block indices are disjoint.
            for d in sorted(by_degree):
                entries = by_degree[d]
                ids = np.concatenate([e[1] for e in entries])
                weights = np.concatenate([np.full(len(e[1]), e[2]) for e in entries])
                def degree_block(start):
                    """[S3 implementation] Disjoint blocks of 2048 rows for one degree, retaining each report's actual noise scale."""
                    selected = ids[start:start+2048]
                    values = np.exp(component_loglikelihood(report[selected], int(d), p,
                                    (caps[selected]-1)/eps_release, edges))
                    A[selected] += weights[start:start+2048, None]*values
                for unused in pool.map(degree_block, range(0, len(ids), 2048)):
                    pass
            for unused in pool.map(projected_task,groups):pass
            pairs = sum(len(g['nodes'])*len(g['degree']) for g in groups)
        else:
            pairs=sum(pool.map(group_task,groups))
    if np.any(A.max(axis=1)<=0) or not np.isfinite(A).all():raise ArithmeticError('Unresolved closure mixture rows')
    audit=dict(nodes=len(report),components=grid_size,distinct_caps=len(groups),node_degree_pairs=pairs,
        max_retained_degrees=max(len(g['degree']) for g in groups),
        max_numerical_omission=max(g['omitted_numerical'] for g in groups),
        max_projected_omission=max(g['omitted_projected'] for g in groups),
        max_projected_posterior_mass=max(g['projected_posterior_mass'] for g in groups),
        degree_omission_tolerance=omission_tolerance,batch_degrees=batch_degrees,truth_used=False,additional_epsilon=0.)
    return A,delta,groups,edges,audit
