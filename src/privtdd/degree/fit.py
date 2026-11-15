"""Degree distribution fitting for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.degree.operator import CapDegreeOperator

from privtdd.degree.reduced import collapse_equivalent_tail

import math

import numpy as np

from privtdd.degree.phases import refine_two_phase



# [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Uniform initialization, 100 accelerated EM steps, then a two-phase degree-mixture MLE; no warm starts from other budgets.
def fit_cold(caps,public_cap,eps_degree,*,warm_iterations=100,tolerance=1e-7,checkpoint=None):
    caps=np.asarray(caps,int);counts=np.bincount(caps,minlength=public_cap+1);freq=counts/counts.sum();used=counts>0
    op=CapDegreeOperator(public_cap,eps_degree)
    p,unused=collapse_equivalent_tail(np.full(public_cap+1,1/(public_cap+1)),eps_degree)
    # [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Evaluate the current numerical objective and optional gradient; the enclosing function supplies the state through its closure.
    def objective(x):
        y=op.forward(x)
        return math.inf if np.any(y[used]<=0) else -math.fsum((freq[used]*np.log(y[used])).tolist())
    # [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); One mixture-weight EM update for the fixed cap channel; not the final GDD posterior step.
    def em(x):
        y=op.forward(x);w=np.zeros_like(y);w[used]=freq[used]/y[used]
        result=x*op.adjoint(w);result/=result.sum();return result
    initial=objective(p);warm=[]
    for iteration in range(warm_iterations):
        value=objective(p);p1=em(p);p2=em(p1);r=p1-p;v=p2-2*p1+p;vv=float(v@v)
        alpha=-max(1.,math.sqrt(float(r@r)/vv)) if vv>0 else -1.
        best=p2;bestvalue=objective(p2)
        for unused in range(16):
            proposed=np.maximum(p-2*alpha*r+alpha*alpha*v,0.);proposed/=proposed.sum()
            proposedvalue=objective(proposed)
            if proposedvalue<=bestvalue:best=proposed;bestvalue=proposedvalue;break
            alpha=(alpha-1)/2
        if bestvalue>value+2e-12:raise ArithmeticError('Warmup likelihood increased')
        p=best
        if iteration%25==0:warm.append(dict(iteration=iteration,nll_per_cap=bestvalue))
    if checkpoint is not None:checkpoint(p.copy(),-1,float('nan'),objective(p))
    result,audit=refine_two_phase(caps,public_cap,eps_degree,p,tolerance=tolerance,checkpoint=checkpoint)
    return result,dict(audit,initialization='uniform full-degree simplex, reduced only through exact cap-tail equivalence',
        cross_epsilon_initialization_used=False,warmup_iterations=warm_iterations,warmup_history=warm,
        initial_nll_per_cap=initial,warmup_final_nll_per_cap=objective(p))
