"""Numerical-only refinement of the existing grouped convex Beta objective.

Adapt the scaled Newton / stable objective increment already implemented in
degree_beta_convex_polish.py. Keep this package self-contained and reuse the
caller's original beta_objective, groups, penalty, bounds and convergence gate.
"""
import numpy as np
from scipy.special import expit


def objective_increment(beta, delta, sizes, targets, penalty=1.):
    """Algebraic f(beta+delta)-f(beta), avoiding large objective subtraction."""
    n = float(np.sum(sizes))
    pairs = np.outer(sizes, sizes) - np.diag(sizes)
    z = beta[:, None] + beta[None, :]
    dz = delta[:, None] + delta[None, :]
    probability = expit(z)
    change = np.empty_like(z)
    small = np.abs(dz) <= .5
    change[small] = np.log1p(probability[small] * np.expm1(dz[small]))
    change[~small] = np.logaddexp(0., z[~small] + dz[~small]) - np.logaddexp(0., z[~small])
    centered = beta - float(sizes @ beta / n)
    dc = delta - float(sizes @ delta / n)
    return float((.5 * np.sum(pairs * change) - (sizes * targets) @ delta
                  + penalty * (sizes @ (centered * dc))
                  + .5 * penalty * (sizes @ (dc * dc))) / n)


def projected_gradient(beta, gradient, sizes, bound):
    pg = np.asarray(gradient, float) * float(np.sum(sizes)) / sizes
    pg[(beta <= -bound + 1e-10) & (pg > 0)] = 0.
    pg[(beta >= bound - 1e-10) & (pg < 0)] = 0.
    return float(np.max(np.abs(pg)))


def polish_beta(beta, sizes, targets, bound, objective, *, tolerance=1e-6, max_iterations=100):
    """Scale Hessian coordinates and stop at the original per-group KKT gate."""
    beta = np.asarray(beta, float).copy()
    sizes, targets = np.asarray(sizes), np.asarray(targets, float)
    trace = []
    for iteration in range(max_iterations + 1):
        value, gradient, hessian = objective(beta, sizes, targets, hessian=True)
        residual = projected_gradient(beta, gradient, sizes, bound)
        trace.append(dict(iteration=iteration, objective=value, projected_gradient=residual))
        if residual <= tolerance:
            return beta, dict(extra_newton_iterations=iteration, initial_projected_gradient=trace[0]['projected_gradient'],
                              optimization_trace=trace, numerical_solver='scaled_Newton_stable_increment_v1',
                              model_or_tolerance_changed=False)
        if iteration == max_iterations:
            break
        active = ((beta <= -bound + 1e-10) & (gradient > 0)) | ((beta >= bound - 1e-10) & (gradient < 0))
        free = np.flatnonzero(~active)
        matrix = hessian[np.ix_(free, free)]
        scale = np.sqrt(np.maximum(np.diag(matrix), np.finfo(float).tiny))
        direction = np.zeros_like(beta)
        try:
            direction[free] = np.linalg.solve(matrix / scale[:, None] / scale[None, :],
                                              -gradient[free] / scale) / scale
        except np.linalg.LinAlgError:
            direction[free] = -gradient[free] / np.maximum(np.diag(matrix), np.finfo(float).tiny)
        if not np.isfinite(direction).all() or float(gradient @ direction) >= 0:
            raise RuntimeError('BETA_FIT_FAILED: scaled Newton direction unresolved')
        for k in range(60):
            step = 2.**(-k)
            delta = np.clip(beta + step * direction, -bound, bound) - beta
            if not np.any(delta):
                continue
            change = objective_increment(beta, delta, sizes, targets)
            if change <= 1e-4 * float(gradient @ delta):
                beta += delta
                trace[-1].update(step=step, objective_increment=change)
                break
        else:
            raise RuntimeError('BETA_FIT_FAILED: stable objective-increment line search unresolved')
    raise RuntimeError(f'BETA_FIT_FAILED: per-group projected gradient {residual:g} after numerical refinement')
