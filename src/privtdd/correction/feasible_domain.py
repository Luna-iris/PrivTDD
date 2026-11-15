"""Feasible correction domain for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

import numpy as np

import privtdd.correction.regions as previous



# [PrivTDD] S5 joint family; core method or probability/structure primitive (not necessarily a separate contribution); Construct the degree-gating basis of the original 16-dimensional representation; the current seven-parameter correction uses no extra gating coefficients.
def degree_gate_basis(support,pi):
    """Three C1-interpolated hats on log(1+d), knots at repaired-degree Q10/50/90.

    Quantiles are conditional on d>=2. Repeated knots collapse; unused columns
    remain zero. All coefficients are shared across degrees, never graph names.
    """
    support=np.asarray(support);pi=np.asarray(pi)
    eligible=(support>=2)&(pi>0);basis=np.zeros((len(support),3))
    if not eligible.any():
        return basis,dict(knots_degree=[],active_basis=0,quantiles=[.1,.5,.9],truth_used=False)
    ds=support[eligible];weights=pi[eligible];cdf=np.cumsum(weights)/weights.sum()
    knots=np.unique(ds[np.minimum(np.searchsorted(cdf,[.1,.5,.9]),len(ds)-1)])
    logknots=np.log1p(knots.astype(float))
    for j in np.flatnonzero(support>=2):
        x=np.log1p(float(support[j]))
        if len(knots)==1 or x<=logknots[0]:basis[j,0]=1.
        elif x>=logknots[-1]:basis[j,len(knots)-1]=1.
        else:
            right=int(np.searchsorted(logknots,x));left=right-1
            t=(x-logknots[left])/(logknots[right]-logknots[left]);v=t*t*(3-2*t)
            basis[j,left]=1-v;basis[j,right]=v
    return basis,dict(knots_degree=knots.tolist(),active_basis=len(knots),quantiles=[.1,.5,.9],
        interpolation='C1 smoothstep hats on log1p(degree), constant beyond end knots',truth_used=False)



# [PrivTDD] S5 joint family; core method or probability/structure primitive (not necessarily a separate contribution); Expand to the common 16-dimensional representation; coordinates outside the current seven remain zero.
def aligned_row(distributions,j,boundaries,gate_basis):
    keys,shapes,base,old,mass=previous.aligned_row(distributions,j,boundaries)
    # The zero atom is an independent component even if all three sources put
    # exactly zero mass there. Its introduction leaves the old model unchanged.
    if keys[0]!=0:
        keys=np.r_[0,keys];shapes=np.pad(shapes,((0,0),(1,0)))
    shapes[0]=0.;shapes[0,0]=1.
    directions=np.zeros((12,16));directions[:,:8]=old
    pf=float(mass[0,1:].sum());zf=float(mass[0,0])
    if int(distributions[0].support[j])>=2:
        positive=np.flatnonzero(mass[:,1:].sum(axis=1)>0)
        if len(positive):
            ref=int(positive[0]);fraction=mass[ref,1:]/mass[ref,1:].sum()
            for h,weight in enumerate(gate_basis):
                directions[0,8+h]=pf*weight
                directions[0,11+h]=-zf*weight
                for r in range(1,4):
                    directions[3*r,8+h]=-pf*weight*fraction[r-1]
                    directions[3*r,11+h]=zf*weight*fraction[r-1]
    for column,r,s in ((14,1,2),(15,3,1)):
        amount=mass[s,r];directions[3*r,column]=-amount;directions[3*r+s,column]=amount
    if np.max(np.abs(directions.sum(axis=0)))>2e-12:
        raise ArithmeticError('A gate direction changes normalization')
    return keys,shapes,base,directions,mass



# [PrivTDD] S5 joint family; core method or probability/structure primitive (not necessarily a separate contribution); Express nonnegativity of each regional mixture coefficient as linear constraints A x <= b.
def coefficient_domain(base,directions):
    dim=directions.shape[-1];rows=[];rhs=[]
    for value,change in zip(base.reshape(-1),directions.reshape(-1,dim)):
        if np.any(change<0):
            scale=max(float(np.max(np.abs(change))),float(value))
            rows.append(-change/scale);rhs.append(value/scale)
    return (np.array(rows),np.array(rhs)) if rows else (np.zeros((1,dim)),np.ones(1))
