"""Reduced degree support for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.degree.operator import CapDegreeOperator

from scipy.sparse.linalg import LinearOperator

from scipy.optimize import brentq

from scipy.sparse.linalg import cg

from privtdd.degree.newton import hessian_diagonal

import math

import numpy as np
from privtdd.config import CAP_OFFSET



# [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Temporarily represent a channel-equivalent tail by two endpoints, preserving mass and exponential moment; expand to the canonical representation at the end.
def collapse_equivalent_tail(prior,eps_degree,offset=CAP_OFFSET):
    p=np.asarray(prior,float).copy();D=len(p)-1;start=max(0,D-offset)
    old=p[start:].copy();u=np.exp(-eps_degree*np.arange(len(old)))
    mass=float(old.sum())
    if len(old)>1 and mass>0:
        first=float(old@((u-u[-1])/(1-u[-1])))
        p[start:]=0.;p[start]=first;p[D]=mass-first
    return p,np.r_[np.arange(start+1),D] if start<D else np.arange(D+1)



# [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Construct a centered Newton direction in the probability-sum constraint tangent space; damping affects only the solver.
def centered_direction(op,p,score,curvature,allowed,damping=1e-8):
    best=int(allowed[np.argmax(score[allowed])]);active=np.union1d(np.flatnonzero(p>0),[best])
    diag=hessian_diagonal(op,curvature)[active]
    # Preconditioning only: far-away likelihood columns can underflow to zero.
    # Bound the scaling; the objective, support and full convex-gap test stay unchanged.
    floor=max(float(diag.max(initial=0.))*1e-12,np.finfo(float).tiny)
    scale=1/np.sqrt(np.maximum(diag,floor));constraint=scale
    # Subtracting a constant from the gradient is exactly neutral under
    # sum(delta)=0, and avoids subtracting two large nearly equal solutions.
    rhs=(score[active]-float(p@score))*scale
    # [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Provide an implicit Hessian-vector product for conjugate gradients.
    def matvec(z):
        full=np.zeros(len(p));full[active]=scale*z
        return scale*op.adjoint(curvature*op.forward(full))[active]+damping*z
    H=LinearOperator((len(active),len(active)),matvec=matvec,dtype=float)
    u,iu=cg(H,rhs,rtol=1e-10,atol=0.,maxiter=1500)
    v,iv=cg(H,constraint,rtol=1e-10,atol=0.,maxiter=1500)
    den=float(constraint@v)
    if den<=0 or not np.isfinite(u).all() or not np.isfinite(v).all():raise ArithmeticError('Reduced Newton solve')
    delta=np.zeros(len(p));delta[active]=scale*(u-float(constraint@u)/den*v)
    delta[int(np.argmax(p))]-=delta.sum()
    return delta,dict(cg_info=[int(iu),int(iv)],active_states=len(active),damping=damping,
        nll_directional_derivative=-float(score@delta))



# [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Solve the same degree MLE in equivalent reduced coordinates, checking stopping with full-degree scores.
def refine_reduced(caps,public_cap,eps_degree,initial,*,tolerance=1e-7,max_iterations=200,checkpoint=None):
    p,allowed=collapse_equivalent_tail(initial,eps_degree);op=CapDegreeOperator(public_cap,eps_degree)
    if p.min()<0 or abs(p.sum()-1)>2e-12:raise ValueError('Degree simplex')
    prediction_change=float(abs(op.forward(p)-op.forward(initial)).max())
    if prediction_change>2e-12:raise ArithmeticError('Tail quotient changed observed channel')
    counts=np.bincount(np.asarray(caps,int),minlength=len(p));freq=counts/counts.sum();used=counts>0
    history=[];accepted_steps=[];exchanges=0
    for iteration in range(max_iterations+1):
        y=op.forward(p);weights=np.zeros_like(p);weights[used]=freq[used]/y[used];score=op.adjoint(weights)
        value=-math.fsum((freq[used]*np.log(y[used])).tolist());gap=max(0.,float(score.max()-p@score))
        if iteration%10==0 or gap<=tolerance:
            history.append(dict(iteration=iteration,convex_gap_per_cap=gap,nll_per_cap=value,active_states=int(np.count_nonzero(p))))
            if checkpoint is not None:checkpoint(p.copy(),iteration,gap,value)
        if gap<=tolerance:
            return p,dict(converged=True,iterations=iteration,convex_gap_per_cap=gap,nll_per_cap=value,
                history=history,accepted_steps=accepted_steps,exchanges=exchanges,tolerance=tolerance,
                equivalent_tail_initial_prediction_change=prediction_change,full_integer_support_certificate=True,
                objective_unchanged=True,extra_statistical_penalty=False,truth_used=False,additional_epsilon=0.)
        if iteration==max_iterations:break
        curvature=np.zeros_like(p);curvature[used]=freq[used]/y[used]**2;accepted=False
        for damping in [1e-8,1e-6,1e-4]:
            try:direction,details=centered_direction(op,p,score,curvature,allowed,damping)
            except ArithmeticError:continue
            derivative=-float(score@direction)
            if derivative>=0 or np.any((p==0)&(direction<0)):continue
            neg=np.flatnonzero(direction<0);ratios=p[neg]/-direction[neg]
            maximum=min(1.,float(ratios.min())) if len(neg) else 1.
            step=maximum
            for backtrack in range(35):
                candidate=p+step*direction
                if candidate.min()<-2e-14:raise ArithmeticError('Reduced Newton simplex violation')
                if backtrack==0 and maximum<1:
                    # The limiting constraints are attained exactly; do not
                    # leave cancellation residue as a new tiny active prior.
                    candidate[neg[ratios<=maximum*(1+4*np.finfo(float).eps)]]=0
                candidate=np.maximum(candidate,0);candidate/=candidate.sum()
                change=op.forward(candidate-p)[used]/y[used]
                improvement=math.inf if np.any(change<=-1) else -math.fsum((freq[used]*np.log1p(change)).tolist())
                if step>0 and improvement<=1e-4*step*derivative+1e-18:
                    p=candidate;accepted=True;accepted_steps.append(dict(iteration=iteration,step=step,relative_nll_change=improvement,**details));break
                step*=.5
            if accepted:break
        if not accepted:
            a=int(allowed[np.argmax(score[allowed])]);active=np.flatnonzero(p>0)
            b=int(active[np.argmax(p[active]*(score[a]-score[active]))]);direction=np.zeros_like(p);direction[a]=1;direction[b]=-1
            dy=op.forward(direction)[used];maximum=p[b]
            def slope(step):
                den=y[used]+step*dy
                return -math.inf if np.any(den<=0) else math.fsum((freq[used]*dy/den).tolist())
            if a==b or slope(0)<=0:raise ArithmeticError('No reduced feasible descent')
            step=maximum if slope(maximum)>=0 else maximum*brentq(lambda a:slope(maximum*a),0.,1.,xtol=1e-14,rtol=1e-14)
            p[a]+=step;p[b]-=step;exchanges+=1
    raise ArithmeticError('Reduced Newton did not meet unchanged precision: '+str(history[-3:]))
