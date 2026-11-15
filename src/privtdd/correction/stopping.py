"""Correction convergence checks for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

import math

import numpy as np

from privtdd.correction.optimizer_bridge import strong_convex_certificate



# [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Compute a conservative positive-curvature lower bound for the actual Gram matrix, without adding a ridge.
def gram_lower_bound(G):
    G=np.asarray(G,float);G=(G+G.T)/2;n=len(G)
    eig=float(np.linalg.eigvalsh(G).min())
    if eig<=0:return dict(lower_modulus=0.,cholesky_available=False)
    candidate=.5*eig
    try:L=np.linalg.cholesky(G-candidate*np.eye(n))
    except np.linalg.LinAlgError:return dict(lower_modulus=0.,cholesky_available=False)
    unit=np.finfo(float).eps;gamma=(n+8)*unit/(1-(n+8)*unit)
    residual=np.zeros_like(G);error=np.zeros_like(G)
    for i in range(n):
        for j in range(n):
            products=(L[i]*L[j]).tolist()
            residual[i,j]=G[i,j]-(candidate if i==j else 0.)-math.fsum(products)
            error[i,j]=gamma*math.fsum(abs(v) for v in products)+8*unit*(abs(G[i,j])+candidate)
    allowance=float(np.max(np.sum(abs(residual)+error,axis=1)))
    lower=max(0.,float(np.nextafter(candidate-allowance,-np.inf)))
    return dict(lower_modulus=lower,cholesky_available=True,candidate_modulus=candidate,
        cholesky=L.tolist(),residual_infinity_bound=allowance,
        proof='G-candidate*I=L*L.T+E; ||E||2<=||E||infinity; includes product/accumulation allowance',
        statistical_ridge_added=False)



# [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Combine the KKT residual and actual curvature into an upper bound on the original objective's optimization gap.
def gap_bound(x,gradient,A,b,gram_audit):
    mu=gram_audit['lower_modulus']
    if mu<=0:return math.inf,dict(strong_convexity_modulus=0.,certificate_type='no_positive_Gram_modulus')
    # Any nonnegative multiplier witness is valid. The existing NNLS routine
    # proposes one; its unit-curvature numerical bound is NOT reused.
    unused,details=strong_convex_certificate(x,gradient,A,b)
    witness=details['certificate_witness'];C=np.asarray(witness['normals'],float)
    weights=np.asarray(witness['multipliers'],float);rhs=np.asarray(witness['rhs'],float)
    if len(weights):
        C=C.reshape(len(weights),len(x))
        residual=np.array([gradient[j]+math.fsum((C[:,j]*weights).tolist()) for j in range(len(x))])
        slack=np.maximum(rhs-C@x,0.)
        complementarity=math.fsum((weights*slack).tolist())
    else:residual=gradient.copy();complementarity=0.
    bound=.5*math.fsum((residual*residual).tolist())/mu+complementarity
    return bound,dict(strong_convexity_modulus=mu,KKT_residual_norm=float(np.linalg.norm(residual)),
        complementarity=complementarity,certificate_witness=witness,certificate_type='checked_Gram_strong_convex_KKT')
