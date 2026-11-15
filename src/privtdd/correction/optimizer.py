"""Constrained correction optimization for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
Uses the exact objective cache and one bounded numerical recovery step.
"""

from __future__ import annotations

from scipy.optimize import brentq

from privtdd.correction.stopping import gap_bound

from privtdd.correction.stopping import gram_lower_bound

from scipy.optimize import linprog

import math

from scipy.optimize import minimize

import numpy as np



# [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Constrained convex optimization with the original Gram penalty; the stopping gap is internal to the optimizer.
def fit_joint(family, active_coordinates=tuple(range(8)), *, max_iterations=250, tolerance=1e-6, linear_cost=None, quadratic=None,
              recovery_checkpoint=None, recovery_progress_path=None):
    """[S5 numerics] Original constrained objective; exact caching and one stable recovery of at most 80 iterations.
    
    recovery_checkpoint: Optional (point, objective) implementation callback that saves the current active coordinates
    only before entering recovery; it does not read the original graph or truth. recovery_progress_path: Optional
    recovery progress JSON path. Other parameters match the original optimizer; failure does not relax tolerance.
    """
    ids=np.asarray(active_coordinates,int)
    cost=np.zeros(len(ids)) if linear_cost is None else np.asarray(linear_cost,float)[ids]
    if cost.shape!=ids.shape or not np.isfinite(cost).all():raise ValueError('Invalid affine cost')
    full_dim=family['delta_tables'].shape[0]
    Q=np.eye(full_dim) if quadratic is None else np.asarray(quadratic,float)
    if Q.shape!=(full_dim,full_dim) or not np.isfinite(Q).all():raise ValueError('Invalid quadratic')
    G=Q[np.ix_(ids,ids)];symmetry=float(np.max(abs(G-G.T)));G=(G+G.T)/2
    norm=max(1.,float(np.linalg.norm(G,2)));eigenvalues=np.linalg.eigvalsh(G)
    if symmetry>1e-12*norm or eigenvalues.min() < -1e-12*norm:raise ValueError('Non-PSD functional Gram')
    # No ridge is added. Account conservatively for a roundoff-sized negative
    # eigenvalue in the linearization bound on a unit-box domain.
    curvature_allowance=.5*len(ids)*max(0.,-float(eigenvalues.min())+16*len(ids)*np.finfo(float).eps*norm)
    gram_audit=gram_lower_bound(G)
    observed=family['counts']>0
    base=np.asarray(family['predictive_table'])[observed]
    ct=np.asarray(family['counts'])[observed]
    delta=np.asarray(family['delta_tables'])[ids][:,observed].T
    if base.min()<=0 or not np.isfinite(base).all() or not np.isfinite(delta).all():
        raise ArithmeticError('Joint fit requires finite positive observed reference probabilities')
    original_A=np.asarray(family['A'])[:,ids];original_b=np.asarray(family['b'])
    row_scales=np.maximum(np.max(np.abs(original_A),axis=1),np.abs(original_b))
    row_scales[row_scales==0]=1.
    A=original_A/row_scales[:,None];b=original_b/row_scales
    coordinate_upper=np.ones(len(ids))
    for row,rhs in zip(A,b):
        if np.all(row>=0):
            positive=row>0
            coordinate_upper[positive]=np.minimum(coordinate_upper[positive],rhs/row[positive])
    # Restrict only the numerical search for sub-roundoff coordinate ranges.
    # ALL certificates below still use the full original polytope (scaled rows)
    # and [0,1] bounds, so they certify the original problem, not the search face.
    search_upper=coordinate_upper.copy()
    search_upper[search_upper<1e-14]=0.

    ratios=delta/base[:,None]
    if not np.isfinite(ratios).all(): raise ArithmeticError('Unresolved report ratios')
    constraint=dict(type='ineq',fun=lambda z:b-A@z,jac=lambda z:-A)
    dim=len(ids);x=np.zeros(dim);history=[];certificate_details={} 

    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Evaluate the current numerical objective and, when needed, its gradient; the enclosing function supplies the state through its closure.
    def uncached_objective(z):
        change=ratios@z;den=1+change
        if np.any(den<=0) or not np.isfinite(den).all(): return math.inf,np.zeros(dim)
        val=-math.fsum((ct*np.log1p(change)).tolist())+.5*float(z@G@z)+float(cost@z)
        grad=G@z+cost-np.array([math.fsum((ct*ratios[:,j]/den).tolist()) for j in range(dim)])
        return val,grad

    # [Implementation optimization] Exact floating-point byte keys; no coordinate rounding or merging of nearby points. Cache only the current objective.
    # As in the original cached optimizer, FIFO holds at most 32 entries; return a copy of the gradient.
    objective_cache={}
    evaluation_counts=dict(computed=0,reused=0)
    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Evaluate the current numerical objective and optional gradient; the optimizer caches at most 32 points with exact byte keys, while roundoff computes directly and stably.
    def objective(z):
        """[S5 implementation] Reuse objective/gradient values at identical coordinates; never match approximate coordinates."""
        point=np.asarray(z,dtype=float)
        key=point.tobytes()
        if key in objective_cache:
            evaluation_counts['reused']+=1
            value,gradient=objective_cache[key]
        else:
            evaluation_counts['computed']+=1
            value,gradient=uncached_objective(point)
            if len(objective_cache)>=32:objective_cache.pop(next(iter(objective_cache)))
            objective_cache[key]=(value,gradient.copy())
        return value,gradient.copy()

    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Correct a numerical candidate into the specified feasible domain, preserving the original probability constraints.
    def feasible(z):
        # SLSQP/LP points on homogeneous faces can have roundoff-sized positive
        # residuals. Radially retracting such a point to a zero-RHS face would
        # erase every unrelated coordinate. Accept only roundoff here; final
        # original-domain checks and certificates are unchanged.
        z=np.clip(z,0,search_upper);used=A@z;bad=used>b+1e-13
        if np.any(bad):
            z*=max(0.,float(np.min(b[bad]/used[bad])))*(1-8*np.finfo(float).eps)
        return z

    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Compute the optimization gap at the current iteration to stop or continue the same solve.
    def certificate(z,g, refine=False):
        scale=max(1.,float(np.max(np.abs(g))))
        lp=linprog(g/scale,A_ub=A,b_ub=b,bounds=[(0.,1.)]*dim,method='highs',
            options=dict(dual_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9))
        if not lp.success:
            # A nearly degenerate polytope may defeat the auxiliary LP. Its
            # failure is not a certificate that the statistical fit is invalid.
            # Try a different LP setup, and leave the certificate unresolved if both fail.
            lp=minimize_linear_support(g/scale,A,b,dim)
        if not lp.success:
            certificate_details.clear()
            strong,details=gap_bound(z,g,A,b,gram_audit)
            certificate_details.update(linear_gap=None,linear_solver_message=str(lp.message),**details)
            return strong,np.zeros(dim)
        # Dual-feasible lower bound, including a conservative stationarity
        # residual term on [0,1]^dim. This does not trust primal LP.fun alone.
        lam=np.minimum(lp.ineqlin.marginals,0.)
        lower=np.maximum(lp.lower.marginals,0.)
        upper=np.minimum(lp.upper.marginals,0.)
        residual=g/scale-A.T@lam-lower-upper
        bound=scale*(math.fsum((b*lam).tolist())+math.fsum(upper.tolist())+
                     math.fsum(np.minimum(residual,0.).tolist()))
        gap=max(0.,math.fsum((g*z).tolist())-bound)
        gap+=curvature_allowance
        linear_gap=gap
        certificate_details.clear()
        if refine and gap>tolerance:
            strong,details=gap_bound(z,g,A,b,gram_audit)
            if strong<gap:
                gap=strong;certificate_details.update(**details)
        if not certificate_details:
            certificate_details.update(certificate_type='linear_support_PSD_prior')
        certificate_details.update(linear_gap=linear_gap,quadratic_curvature_allowance=curvature_allowance)
        return gap,feasible(lp.x)

    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Take a one-dimensional feasible descent step along the supplied direction.
    def line(z,direction):
        if not np.any(direction): return z
        _,g=objective(z)
        if float(g@direction)>=0: return z
        # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Directional derivative for the one-dimensional line search.
        def slope(t):
            val,grad=objective(z+t*direction)
            return math.inf if not math.isfinite(val) else float(grad@direction)
        step=1. if slope(1.)<=0 else brentq(slope,0.,1.,xtol=1e-15,rtol=1e-14,maxiter=200)
        return feasible(z+step*direction)

    # Scale the ENTIRE objective (including the unit prior) for SLSQP only.
    # All reporting and certificates remain in original count units.
    scale=float(ct.sum())
    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Divide the entire objective and gradient by the sample count for SLSQP; retain the original units for the stopping gap.
    def scaled_objective(z):
        """[S5 numerics] Scale the entire objective and gradient by the same factor for SLSQP."""
        value,gradient=objective(z)
        return value/scale,gradient/scale
    for iteration in range(max_iterations):
        value,gradient=objective(x)
        gap,vertex=certificate(x,gradient,refine=iteration>=10)
        history.append(dict(iteration=iteration,objective=value,convex_gap=gap,**certificate_details))
        if gap<=tolerance and value<=1e-10:
            full=np.zeros(family['delta_tables'].shape[0]);full[ids]=x
            if np.max(family['A']@full-family['b'])>2e-12:
                raise ArithmeticError('Joint fit violates coefficient constraints')
            return full,dict(coordinates=full.tolist(),active_coordinates=ids.tolist(),objective=value,
                convex_gap=gap,penalty_coefficient=1.,history=history,objective_evaluations=dict(evaluation_counts),true_risk_certificate=False,
                numerical_solver='active_row_scaling_original_domain_certificate',
                projection_roundoff_tolerance=1e-13,
                constraint_row_scales=row_scales.tolist(),coordinate_upper_bounds=coordinate_upper.tolist(),
                numerical_search_upper_bounds=search_upper.tolist(),
                **certificate_details,
                linear_cost=cost.tolist(),
                quadratic_eigenvalues=eigenvalues.tolist(),statistical_ridge_added=False,
                quadratic_strong_convexity=gram_audit,
                objective_type='count marginal composite likelihood plus functional Gram prior plus affine structural KL')
        x=line(x,vertex-x)
        result=minimize(scaled_objective,x,jac=True,
            method='SLSQP',bounds=list(zip(np.zeros(dim),search_upper)),constraints=[constraint],
            options=dict(ftol=1e-15,maxiter=350))
        candidate=feasible(result.x)
        if objective(candidate)[0]<objective(x)[0]: x=candidate
        # A feasible constrained Newton step refines the small-gradient
        # optimum beyond SLSQP's normalized stopping threshold.
        value,g=objective(x);den=1+ratios@x
        H=G+(ratios/den[:,None]).T@(ct[:,None]*(ratios/den[:,None]))
        hscale=max(1.,float(np.max(np.abs(H))),float(np.max(np.abs(g))))
        qp=minimize(lambda dz:((float(g@dz)+.5*float(dz@H@dz))/hscale,(g+H@dz)/hscale),
            np.zeros(dim),jac=True,method='SLSQP',bounds=list(zip(-x,search_upper-x)),
            constraints=[dict(type='ineq',fun=lambda dz:b-A@(x+dz),jac=lambda dz:-A)],
            options=dict(ftol=1e-15,maxiter=200))
        candidate=feasible(x+qp.x)
        x=line(x,candidate-x)
    # [Numerical recovery] If the main solver budget is exhausted, perform one bounded recovery from the last feasible point of the same objective.
    # Do not read graph names, truth or other budgets; do not accept results that fail the original tolerance.
    from privtdd.correction.roundoff import refine_joint
    recovery_family=dict(counts=ct,predictive_table=base,delta_tables=delta.T,
                         A=original_A,b=original_b)
    expected_initial=objective(x)[0]
    if recovery_checkpoint is not None:recovery_checkpoint(x.copy(),expected_initial)
    del ratios  # After handoff, stop calling the ordinary solver callback and release the old reports x active-coordinates ratio table.
    recovered,stats=refine_joint(recovery_family,initial=x,quadratic=G,linear_cost=cost,
                                tolerance=tolerance,max_iterations=80,
                                expected_initial_objective=expected_initial,
                                progress_path=recovery_progress_path)
    full=np.zeros(full_dim);full[ids]=recovered
    if np.max(family['A']@full-family['b'])>2e-12 or stats['objective']>1e-10:
        raise ArithmeticError('Recovered joint point violates original constraints or F1 objective anchor')
    stats.update(coordinates=full.tolist(),active_coordinates=ids.tolist(),
                 primary_history=history,objective_evaluations=dict(evaluation_counts),
                 recovery_trigger='primary_iteration_limit',true_risk_certificate=False)
    return full,stats



# [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Use linear programming to find a support lower bound for the gradient over the feasible domain, for the stopping-gap calculation.
def minimize_linear_support(gradient,A,b,dim):
    return linprog(gradient,A_ub=A,b_ub=b,bounds=[(0.,1.)]*dim,method='highs-ds',
        options=dict(presolve=False,dual_feasibility_tolerance=1e-9,
                     primal_feasibility_tolerance=1e-9))
