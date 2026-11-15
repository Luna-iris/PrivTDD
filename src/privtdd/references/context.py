"""Shared reference construction context for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.references.independent_edge import degree_two_means

from privtdd.degree.channel import log_cap_probability

from scipy.special import logsumexp

import math

import numpy as np

import privtdd.references.primitives as original

from privtdd.references.independent_edge import pair_means



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Construct integer type counts and root-type probabilities conditional on latent degree from the full cap-based degree posterior.
def structural_context(caps,prior,eps_degree,*,max_groups=128):
    caps=np.asarray(caps,np.int64);prior=np.asarray(prior,float)
    if caps.ndim!=1 or not len(caps) or np.any(prior<0) or abs(prior.sum()-1)>2e-12:
        raise ValueError('Private cap/degree inputs')
    support=np.flatnonzero(prior>0);pi=prior[support].copy();D=len(prior)-1
    unique,counts=np.unique(caps,return_counts=True);n=len(caps)
    logpost=np.log(pi)[None,:]+log_cap_probability(unique[:,None],support[None,:],D,eps_degree)
    logpost-=logsumexp(logpost,axis=1)[:,None]
    post=np.exp(logpost);post/=post.sum(axis=1)[:,None]
    means=post@support
    labels,sizes,beta,P,fit=original.fit_beta(means,counts,max_groups,grouping='log_degree')
    if sizes.sum()!=n:raise ArithmeticError('Structural population changed')
    # The existing structural population uses integer observed cap-group sizes.
    # Canonical latent degree mass remains external/fixed; no fractional sizes
    # are cast to integer and no unobserved graph degrees are used.
    logroot=np.array([logsumexp(np.log(counts[labels==g])[:,None]+logpost[labels==g],axis=0)
                      for g in range(len(sizes))])
    root_probability=np.exp(logroot-logsumexp(logroot,axis=0)[None,:])
    root_probability/=root_probability.sum(axis=0)[None,:]
    native=counts@post/n
    grouped=root_probability*(n*pi)[None,:]
    if np.max(abs(root_probability.sum(axis=0)-1))>2e-12:raise ArithmeticError('Root type normalization')
    context=dict(support=support,pi=pi,grouped=grouped,root_probability=root_probability,
        sizes=sizes,beta=beta,probability=P,cap_values=unique,cap_counts=counts,
        cap_degree_means=means,cap_group_labels=labels,native_cap_posterior_degree_mass=native,n=np.array(n))
    audit=dict(beta_fit=fit,canonical_degree_marginal_fixed=True,
        native_vs_canonical_degree_TV=float(abs(native-pi).sum()/2),
        numerical_structural_groups=len(sizes),global_closure_prior_groups_added=0,
        root_type_rule='P(structural cap-mean group | latent degree) from full private cap posterior',
        integer_group_sizes_sum=int(sizes.sum()),truth_used=False,additional_epsilon=0.)
    return context,audit



# [PrivTDD] S4 structural references; core method or probability/structure primitive (not necessarily a separate contribution); Allocate structural integration across degrees using fixed pi, and generate compositions conditioned exactly on total neighbor count d.
def prepare_compositions(context,*,draws=65536,replicates=4,reuse_root_kernels=False):
    """Reuse exact fixed-degree neighbor sampling, with canonical population pi."""
    if draws<1 or replicates<1:raise ValueError('Positive numerical budgets')
    support=context['support'];pi=context['pi'];P=context['probability'];sizes=context['sizes'];beta=context['beta']
    root_probability=context['root_probability'];two=degree_two_means(sizes,beta,P)
    blocks=[];counts=[];moments=[];proposals=0;strata=[]
    rep_means=np.zeros((replicates,2))
    closure_cache = {}
    tilt_cache = {} if reuse_root_kernels else None
    for rep in range(replicates):
        rng=np.random.default_rng(731+100003*rep)
        for j,d in enumerate(support):
            if d<2:continue
            if d==2:
                rep_means[rep]+=pi[j]*(root_probability[:,j]@two);continue
            amount=max(1,int(math.ceil(draws*pi[j])))
            allocation=rng.multinomial(amount,root_probability[:,j]);values=[]
            for root,number in enumerate(allocation):
                if number==0:continue
                c,used=original.conditional_counts(sizes,beta,root,int(d),int(number),rng,tilt_cache=tilt_cache);proposals+=used
                if reuse_root_kernels:
                    if root not in closure_cache:
                        arcs=P[root];closure_cache[root]=original.circle_closure(arcs[:,None],arcs[None,:],P)
                    C=closure_cache[root]
                else:
                    arcs=P[root];C=original.circle_closure(arcs[:,None],arcs[None,:],P)
                v=np.column_stack((pair_means(c,P),pair_means(c,C)))
                blocks.append((rep,j,root));counts.append(c);moments.append(v);values.append(v)
            values=np.concatenate(values);rep_means[rep]+=pi[j]*values.mean(axis=0)
            strata.append(dict(replicate=rep,degree=int(d),draws=amount))
    covariance=np.zeros((2,2));unresolved=[]
    for j,d in enumerate(support):
        values=[moments[i] for i,block in enumerate(blocks) if block[1]==j]
        if not values:continue
        values=np.concatenate(values)
        if len(values)>1:covariance+=pi[j]**2*np.cov(values,rowvar=False,ddof=1)/len(values)
        else:unresolved.append(int(d))
    offsets=np.r_[0,np.cumsum([len(c) for c in counts])]
    packet=dict(support=support,pi=pi,grouped=context['grouped'],root_probability=root_probability,
        sizes=sizes,beta=beta,probability=P,two_means=two,
        blocks=np.asarray(blocks,np.int64).reshape(-1,3),offsets=offsets,
        neighbor_counts=np.concatenate(counts) if counts else np.empty((0,len(sizes)),np.int64),
        conditional_endpoint_means=np.concatenate(moments) if moments else np.empty((0,2)),
        endpoint_means=rep_means.mean(axis=0),endpoint_mean_covariance=covariance,
        replicate_endpoint_means=rep_means,draws=np.array(draws),replicates=np.array(replicates))
    audit=dict(draws_per_replicate=draws,replicates=replicates,seeds=[731+100003*r for r in range(replicates)],
        conditional_neighbor_proposals=proposals,endpoint_means=packet['endpoint_means'].tolist(),
        endpoint_numerical_SE=np.sqrt(np.maximum(np.diag(covariance),0)).tolist(),
        strata=strata,endpoint_SE_unresolved_degree_strata=unresolved,
        numerical_SE_is_estimated_not_a_rigorous_bound=True,
        statistical_source_parameters_changed=False,canonical_degree_marginal_fixed=True,
        private_reports_independent_claimed=False,truth_used=False,additional_epsilon=0.)
    return packet,audit
