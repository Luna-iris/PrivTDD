"""Probability calculations used by PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.special import log_ndtr

import math

import numpy as np



# [PrivTDD] S3 analytic probability utilities; numerical integration and stable computation; Reuse the stable normal-plus-Laplace log-CDF primitive underlying the shared distribution implementation.
def _normal_laplace_logcdf(z, sigma, scale):
    """Stable log CDF of N(0,sigma**2)+Laplace(0,scale), including infinities."""
    z, sigma, scale = np.broadcast_arrays(z, sigma, scale)
    result = np.empty(z.shape, dtype=float)
    negative_infinity = np.isneginf(z); positive_infinity = np.isposinf(z)
    result[negative_infinity] = -np.inf; result[positive_infinity] = 0.
    finite = np.isfinite(z)
    if not np.any(finite):
        return result
    zz, ss, bb = z[finite], sigma[finite], scale[finite]
    value = np.empty(len(zz))
    zero = ss == 0
    neg = zz[zero] <= 0
    value[zero] = np.where(neg, -math.log(2)+np.minimum(zz[zero], 0)/bb[zero],
                          np.log1p(-.5*np.exp(-np.maximum(zz[zero], 0)/bb[zero])))
    active = ~zero
    if np.any(active):
        absolute = -np.abs(zz[active]); sig, b = ss[active], bb[active]
        x = absolute/sig; a = sig/b
        log_base = log_ndtr(x)
        log_minus = -absolute/b+.5*a*a+log_ndtr(x-a)
        log_plus = absolute/b+.5*a*a+log_ndtr(-x-a)
        log_low = np.logaddexp(
            log_base+np.log1p(-.5*np.exp(np.minimum(log_minus-log_base, 0.))),
            -math.log(2)+log_plus)
        value[active] = np.where(zz[active] <= 0, log_low, np.log(-np.expm1(log_low)))
    result[finite] = value
    if np.any(np.isnan(result)) or np.any(result > 1e-12):
        raise FloatingPointError('Normal-Laplace CDF unresolved')
    return result
