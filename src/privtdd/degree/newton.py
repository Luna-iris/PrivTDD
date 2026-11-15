"""Newton steps for PrivTDD degree fitting.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.signal import lfilter

import numpy as np



# [PrivTDD] S2 degree prior; numerical solver (not an additional statistical model); Compute the diagonal of K^T diag(w) K using geometric-sequence scans for Newton preconditioning.
def hessian_diagonal(operator,observation_weights):
    """diag(K.T diag(observation_weights) K), using squared geometric scans."""
    op=operator;w=np.asarray(observation_weights,float)
    extended=np.zeros(op.hi-op.lo+1);extended[op.interior_positions]=w[1:-1]
    r2=op.r*op.r;left=lfilter([1.],[1.,-r2],extended);right=lfilter([1.],[1.,-r2],extended[::-1])[::-1]
    previous=np.zeros(len(op.positions));valid=op.positions>0;previous[valid]=left[op.positions[valid]-1]
    return op.h**2*(right[op.positions]+previous)+op.endpoint0**2*w[0]+op.endpointD**2*w[-1]
