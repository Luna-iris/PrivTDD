"""Correction objective assembly for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.optimize import lsq_linear

import math

from scipy.optimize import nnls

import numpy as np



# [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Compute the optimization stopping gap from the current objective's KKT residual; no repeated fit or truth-based certification.
def strong_convex_certificate(x,gradient,A,b):
    """Upper bound on ORIGINAL objective suboptimality, with strong convexity 1.

    For C y <= h and lambda >= 0, let r=g+C.T lambda and s=h-Cx.
    Strong convexity implies J(x)-min_C J <= .5||r||^2+lambda.T s.
    Nonnegative multipliers need not be optimal. NNLS chooses a useful valid
    witness on near-active faces; rounding slack upward is conservative.
    """
    dim=len(x)
    C=np.vstack((A,np.eye(dim),-np.eye(dim)))
    rhs=np.r_[b,np.ones(dim),np.zeros(dim)]
    slack=rhs-C@x
    best=.5*float(gradient@gradient)
    best_details=dict(KKT_residual_norm=float(np.linalg.norm(gradient)),complementarity=0.,
                      strong_convexity_modulus=1.,witness_constraints=0,
                      certificate_witness=dict(point=x.tolist(),gradient=gradient.tolist(),
                          normals=[],rhs=[],multipliers=[]))
    for threshold in (1e-12,1e-10,1e-8,1e-6):
        active=np.flatnonzero(slack<=threshold)
        if not len(active): continue
        matrix=C[active].T
        try:
            weights,_=nnls(matrix,-gradient,maxiter=max(100,10*len(active)))
        except (RuntimeError, np.linalg.LinAlgError):
            # Degenerate active normals: any nonnegative witness remains valid.
            # Use a bounded least-squares solver without NNLS normal equations.
            try:
                weights = lsq_linear(matrix, -gradient, bounds=(0., np.inf),
                    tol=1e-12, max_iter=500, lsq_solver="exact").x
            except (ValueError, np.linalg.LinAlgError):
                continue
        weights=np.maximum(weights,0.)
        residual=gradient+matrix@weights
        complementarity=math.fsum((weights*np.maximum(slack[active],0.)).tolist())
        bound=.5*float(residual@residual)+complementarity
        if bound<best:
            best=bound
            best_details=dict(KKT_residual_norm=float(np.linalg.norm(residual)),
                complementarity=complementarity,strong_convexity_modulus=1.,
                witness_constraints=int(np.count_nonzero(weights)),
                certificate_witness=dict(point=x.tolist(),gradient=gradient.tolist(),
                    normals=C[active][weights>0].tolist(),rhs=rhs[active][weights>0].tolist(),
                    multipliers=weights[weights>0].tolist()))
    return best,best_details
