"""Stable correction basis for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

import math

import numpy as np

import privtdd.correction.feasible_domain as old



# [PrivTDD] S5 joint family; core method or probability/structure primitive (not necessarily a separate contribution); Absorb rounding residuals using complementary events and regions with large mass, preserving the same mass-transfer formula.
def aligned_row(sources,j,boundaries,gate_basis):
    keys,shapes,base,directions,mass=old.aligned_row(sources,j,boundaries,gate_basis)
    directions=directions.copy()
    pf=math.fsum(mass[0,1:].tolist());zf=float(mass[0,0])
    for s in (1,2):
        first=2*(s-1)
        ps=math.fsum(mass[s,1:].tolist());zs=float(mass[s,0])
        transfer=(ps-pf) if max(pf,ps)<=max(zf,zs) else (zf-zs)
        directions[:,first:first+2]=0.
        if max(pf,ps)==0:
            if transfer!=0:raise ArithmeticError('No positive support for zero transfer')
            continue
        fraction=(mass[s,1:]/ps if ps>0 else mass[0,1:]/pf).copy()
        pivot=int(np.argmax(fraction));other=[r for r in range(3) if r!=pivot]
        fraction[pivot]=1-math.fsum(float(fraction[r]) for r in other)
        exchange=transfer*fraction
        exchange[pivot]=transfer-math.fsum(float(exchange[r]) for r in other)
        redistribute=(mass[s,1:]-mass[0,1:])-exchange
        # Algebraically zero-sum; place only rounding residual in the largest
        # available region, never in a tiny/empty zero or positive component.
        pivot2=int(np.argmax(mass[s,1:]+mass[0,1:]))
        other2=[r for r in range(3) if r!=pivot2]
        redistribute[pivot2]=-math.fsum(float(redistribute[r]) for r in other2)
        directions[0,first]=-transfer
        for r in range(3):
            directions[3*(r+1),first]=exchange[r]
            directions[3*(r+1),first+1]=redistribute[r]
    if np.max(abs(directions.sum(axis=0)))>2e-12:
        raise ArithmeticError('Stable mass direction lost conservation')
    return keys,shapes,base,directions,mass
