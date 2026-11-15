"""Likelihood primitives used by PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

import numpy as np



# [PrivTDD] S3 likelihood utilities; core method or probability/structure primitive (not necessarily a separate contribution); Return the mixture likelihood's mean negative log likelihood and analytic gradient.
def mixture_objective(prior,channel,frequency):
    mixture=channel@prior
    if np.any(mixture<=0):return np.inf,np.zeros_like(prior)
    return -float(frequency@np.log(mixture)),-channel.T@(frequency/mixture)
