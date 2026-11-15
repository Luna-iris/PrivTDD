"""Base distribution optimization for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.optimize import brentq

from scipy.optimize import minimize

from privtdd.numerics.likelihood import mixture_objective

import numpy as np



# [PrivTDD] S3 F1; numerical solver (not an additional statistical model); Fit the closure-rate mixture on the simplex; in-place row scaling saves memory, and pairwise line searches meet the original optimization gap.
def optimize_prior(channel,counts,initial,tolerance=1e-6):
    maximum=np.max(channel,axis=1);offset=np.log(maximum)
    channel/=maximum[:,None]
    frequency=np.asarray(counts,float)/np.sum(counts);initial=np.asarray(initial,float)
    if np.any(initial<0) or not np.isclose(initial.sum(),1):raise ValueError('Invalid initial simplex weights')
    initial=initial/initial.sum()
    initial_value,initial_gradient=mixture_objective(initial,channel,frequency)
    result=minimize(mixture_objective,initial,args=(channel,frequency),jac=True,method='SLSQP',
        bounds=[(0.,1.)]*len(initial),constraints=[dict(type='eq',fun=lambda x:float(x.sum()-1),jac=lambda x:np.ones_like(x))],
        options=dict(maxiter=3000,ftol=1e-13))
    prior=np.maximum(result.x,0);prior/=prior.sum()
    value,gradient=mixture_objective(prior,channel,frequency)
    # Convex simplex supporting-hyperplane bound: f(pi)-min f <= max score-1.
    gap=max(0.,float(prior@gradient-np.min(gradient)))
    exchanges=0
    # SLSQP may stop on function-value precision before the global convex gap
    # is small. Pairwise exact line searches polish the SAME simplex problem.
    # The trigger is the requested numerical certificate, not a data loss gate.
    while gap>tolerance and exchanges<10000:
        score=-gradient;best=int(np.argmax(score));active=np.flatnonzero(prior>0)
        worst=int(active[np.argmin(score[active])]);maximum=prior[worst]
        if best==worst:break
        mixture=channel@prior;direction=channel[:,best]-channel[:,worst]
        def derivative(step):
            den=mixture+step*direction
            if np.any(den<=0):return -np.inf
            return float(frequency@(direction/den))
        step=maximum if derivative(maximum)>=0 else brentq(derivative,0.,maximum,
            xtol=np.finfo(float).tiny,rtol=4*np.finfo(float).eps,maxiter=200)
        if step==0:break
        prior[best]+=step;prior[worst]-=step;exchanges+=1
        value,gradient=mixture_objective(prior,channel,frequency)
        gap=max(0.,float(prior@gradient-np.min(gradient)))
    status=bool(gap<=tolerance and value<=initial_value+1e-12)
    posterior=None
    return prior,posterior,dict(converged=status,optimizer_success=bool(result.success),
        optimizer_message=str(result.message),iterations=int(result.nit),pairwise_polish_steps=exchanges,
        original_dual_gap=max(0.,float(initial@initial_gradient-np.min(initial_gradient))),
        optimized_dual_gap=gap,dual_gap_tolerance=tolerance,
        original_nll_per_report=initial_value-float(frequency@offset),
        optimized_nll_per_report=value-float(frequency@offset),objective_gain=initial_value-value,
        prior_L1_change=float(np.sum(abs(prior-initial))),
        guarantee='convex optimization gap for the numerical fixed-support channel only; no GDD risk guarantee')
