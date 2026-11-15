"""Fitted population regions for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

import numpy as np



# [PrivTDD] S5 joint family; core method or probability/structure primitive (not necessarily a separate contribution); Assign T to four regions: zero, (0,Q90], (Q90,Q95] and (Q95,W].
def region_ids(keys, boundaries):
    """Zero; (0,b90]; (b90,b95]; (b95,capacity]. Empty middle is allowed."""
    keys=np.asarray(keys)
    lo,hi=boundaries
    return np.where(keys==0,0,np.where(keys<=lo,1,np.where(keys<=hi,2,3)))



# [PrivTDD] S5 joint family; core method or probability/structure primitive (not necessarily a separate contribution); Align F1/Geo/Beta to the same integer support and decompose their four-region masses and normalized shapes.
def aligned_row(distributions, j, boundaries):
    """12 regional shapes: (reference, Geometry, Beta) for each of four regions."""
    # Caller order stays F1, Beta, Geometry, matching original distribute.
    rows=[distributions[i].row(j) for i in (0,2,1)]
    keys=np.unique(np.concatenate([k for k,p in rows]))
    laws=np.zeros((3,len(keys)))
    for i,(k,p) in enumerate(rows): laws[i,np.searchsorted(keys,k)]=p
    rid=region_ids(keys,boundaries)
    mass=np.array([[law[rid==r].sum() for r in range(4)] for law in laws])
    shapes=np.zeros((12,len(keys)));base=np.zeros(12);directions=np.zeros((12,8))
    for r in range(4):
        mask=rid==r
        available=np.flatnonzero(mass[:,r]>0)
        if not len(available): continue
        reference=int(available[0])
        shapes[3*r,mask]=laws[reference,mask]/mass[reference,r]
        base[3*r]=mass[0,r]
        for s in (1,2):
            if mass[s,r]>0: shapes[3*r+s,mask]=laws[s,mask]/mass[s,r]
    # Two mass directions per structural source. Their sum at a=1 gives
    # that source's regional masses (within the declared mixture polytope).
    pf=float(mass[0,1:].sum())
    for s in (1,2):
        first=2*(s-1);ps=float(mass[s,1:].sum())
        source_fraction=(mass[s,1:]/ps if ps>0 else mass[0,1:]/pf if pf>0 else np.zeros(3))
        redistribution=pf*(source_fraction-mass[0,1:]/pf) if pf>0 else np.zeros(3)
        transfer=ps-pf
        directions[0,first]=-transfer
        for r in range(1,4):
            directions[3*r,first]=transfer*source_fraction[r-1]
            directions[3*r,first+1]=redistribution[r-1]
    # Source mass is a fixed scale, NOT another optimized multiplier.
    for column,r,s in ((4,1,1),(5,2,1),(6,2,2),(7,3,2)):
        amount=mass[s,r]
        directions[3*r,column]=-amount
        directions[3*r+s,column]=amount
    if not np.allclose(directions.sum(axis=0),0,rtol=0,atol=2e-12):
        raise ArithmeticError('A joint direction changes row normalization')
    for r in range(4):
        if np.max(np.abs(directions[3*r:3*r+3,4:].sum(axis=0)))>2e-12:
            raise ArithmeticError('A shape direction changes regional mass')
    return keys,shapes,base,directions,mass
