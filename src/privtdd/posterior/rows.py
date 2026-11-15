"""Posterior row calculations for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.correction.basis import aligned_row

import numpy as np



# [PrivTDD] S6 single posterior update; core method or probability/structure primitive (not necessarily a separate contribution); Generate degree-conditional prior rows from the frozen joint coordinates without refitting.
def prior_profiles(family,j,coordinates):
    keys,S,base,D,unused=aligned_row(family['sources'],j,family['boundaries'],family['gate_basis'][j])
    probabilities=(base[None,:]+np.asarray(coordinates)@D.T)@S
    minimum=float(probabilities.min());mass_error=float(abs(probabilities.sum(axis=1)-1).max())
    if minimum<-2e-12 or mass_error>2e-10:raise ArithmeticError('Frozen prior law changed')
    probabilities=np.maximum(probabilities,0);probabilities/=probabilities.sum(axis=1)[:,None]
    return keys,probabilities.T,dict(minimum_before_roundoff_clipping=minimum,maximum_mass_error=mass_error)
