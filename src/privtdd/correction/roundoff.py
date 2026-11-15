"""S5 numerical recovery: stable objective differences and Newton steps on feasible faces; no additional statistical model.

Extracted from the original roundoff optimizer, without dependencies on its runner/audit scripts.
Called once by correction.optimizer after the ordinary solve budget is exhausted; a feasible point for the same objective can also be supplied.
Inputs are only forward tables constructed from released reports; truth and warm starts from other experiments are not accepted.
"""
import math
import numpy as np
from privtdd.correction.stopping import gram_lower_bound, gap_bound
from privtdd.io.storage import json_write


def face_newton_direction(hessian, gradient, point, free, certificate):
    """[S5 numerical implementation] When the ordinary Newton direction is blocked, seek descent along an active face of the original constraints.
    
    hessian: Original objective Hessian on free coordinates; gradient/point: full gradient and current point;
    free: Free coordinates after the original box-constraint screening; certificate: KKT gap record for this iteration's original feasible domain.
    Use only original constraint normals with multiplier>0 and slack<=1e-12 in the existing nonnegative-multiplier evidence.
    Scale by the Hessian diagonal, use SVD to obtain the null space, then solve the reduced Newton system; do not add a ridge.
    The SVD rank threshold is max(matrix shape)*machine epsilon*largest singular value. Constraints only select directions;
    accepted points still satisfy the original full feasible domain, and stopping still uses the original full-objective gap.
    Return the full direction and implementation diagnostics; an empty tangent space returns zero direction without falsely claiming convergence.
    """
    dim=len(point);direction=np.zeros(dim)
    witness=certificate['certificate_witness']
    normals=np.asarray(witness['normals'],float).reshape(-1,dim)
    rhs=np.asarray(witness['rhs'],float)
    weights=np.asarray(witness['multipliers'],float)
    chosen=(weights>0)&(rhs-normals@point<=1e-12)
    scaling=1/np.sqrt(np.diag(hessian))
    rows=normals[chosen][:,free]*scaling[None,:]
    norms=np.linalg.norm(rows,axis=1)
    rows=rows[norms>0]/norms[norms>0,None]
    if len(rows):
        _,singular,vt=np.linalg.svd(rows,full_matrices=True)
        rank=int(np.count_nonzero(singular>max(rows.shape)*np.finfo(float).eps*singular[0]))
        basis=vt[rank:].T
    else:
        rank=0;basis=np.eye(len(free))
    if basis.shape[1]:
        scaled=scaling[:,None]*hessian*scaling[None,:]
        reduced=basis.T@scaled@basis
        direction[free]=scaling*(basis@np.linalg.solve(reduced,-basis.T@(scaling*gradient[free])))
    return direction,dict(face_rank=rank,tangent_dimension=int(basis.shape[1]),
                          face_constraints=int(len(rows)))


# [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); S5 numerical recovery: resume from a feasible point of the same objective using stable objective differences and bounded Newton steps, preserving the original feasible domain and tolerance.
def refine_joint(family, *, initial, max_iterations=80, tolerance=1e-6,
                 quadratic=None, linear_cost=None, progress_path=None,
                 expected_initial_objective=None):
    """[S5 numerical implementation] Refine the same objective in active coordinates without changing the model, penalty or feasible domain.
    
    family: counts, predictive_table, delta_tables, A and b mapped to active coordinates.
    initial: Feasible coordinates for this same objective; only an initial point, which must satisfy the original stopping condition again.
    max_iterations: Recovery iteration limit, actually min(max_iterations,80); not repeated experiments.
    tolerance: Original objective gap threshold, fixed to 1e-6 by the main entry point; not a GDD error bound.
    quadratic: Gram matrix in active space; the main entry point maps the identity penalty on the original 16 coordinates.
    linear_cost: Affine cost for the same objective, all zeros in the main entry point; preserves the ordinary optimizer's general semantics.
    progress_path: Optional numerical progress JSON, unwritten by default; not an independent verification workflow.
    expected_initial_objective: Recorded objective at this same initial point; reject recovery if the difference exceeds 1e-7.
    
    Fixed numerical controls: boundary threshold 1e-12, feasibility rounding slack 2e-12, Hessian block size
    65536, at most 60 step halvings per direction, and boundary-step multiplier 1-16*machine epsilon.
    If the ordinary direction is blocked, add only one tangent Newton direction on the original active face; do not add recovery iterations.
    Tangent constraint slopes account for both componentwise dot-product error and norm error from SVD in scaled space,
    both with coefficient 16*machine epsilon; these only screen step sizes and do not change final feasibility slack.
    Return coordinates and solver status; raise if the original tolerance is unmet instead of producing a false success.
    """
    dim=family['delta_tables'].shape[0]
    Q=np.eye(dim) if quadratic is None else np.asarray(quadratic,float)
    cost=np.zeros(dim) if linear_cost is None else np.asarray(linear_cost,float)
    ct=np.asarray(family['counts'],float);base=np.asarray(family['predictive_table'],float)
    delta=np.asarray(family['delta_tables'],float)
    original_A=np.asarray(family['A'],float);original_b=np.asarray(family['b'],float)
    x=np.asarray(initial,float).copy();history=[]
    if (Q.shape!=(dim,dim) or cost.shape!=(dim,) or x.shape!=(dim,)
            or ct.ndim!=1 or base.shape!=ct.shape or delta.shape!=(dim,len(ct))
            or original_A.ndim!=2 or original_A.shape[1]!=dim
            or original_b.shape!=(len(original_A),)):
        raise ValueError('Invalid active-coordinate recovery shapes')
    if (max_iterations<1 or not np.isfinite(tolerance) or tolerance<=0
            or not all(np.isfinite(a).all() for a in (Q,cost,x,ct,base,delta,original_A,original_b))
            or np.any(ct<0) or not ct.sum()>0 or np.any(base<=0)):
        raise ValueError('Finite positive objective inputs and numerical limits required')
    if not np.array_equal(Q,Q.T):
        raise ValueError('Recovery requires a symmetric Gram matrix')
    gram=gram_lower_bound(Q)
    if gram['lower_modulus']<=0:
        raise ArithmeticError('Stable Newton recovery requires positive original Gram curvature')
    ratios=delta.T/base[:,None]
    if not np.isfinite(ratios).all():
        raise ArithmeticError('Unresolved report ratios')
    scales=np.maximum(np.max(abs(original_A),axis=1),abs(original_b));scales[scales==0]=1
    A=original_A/scales[:,None];b=original_b/scales

    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Evaluate the current numerical objective and optional gradient; the optimizer caches at most 32 points with exact byte keys, while roundoff computes directly and stably.
    def objective(z):
        """[S5 numerics] Original count likelihood, quadratic penalty and affine cost; accumulate stably with fsum."""
        change=ratios@z;den=1+change
        if den.min()<=0 or not np.isfinite(den).all():
            raise ArithmeticError('Positive likelihood required')
        value=-math.fsum((ct*np.log1p(change)).tolist())+.5*float(z@Q@z)+float(cost@z)
        gradient=Q@z+cost-np.array([math.fsum((ct*ratios[:,j]/den).tolist()) for j in range(dim)])
        return value,gradient,den

    # [PrivTDD] S5 optimization; numerical solver (not an additional statistical model); Check numerical feasibility under the original A/b and [0,1], without projection or relaxed model constraints.
    def feasible(z):
        """[S5 numerics] Check the original A/b and [0,1]; do not relax or learn new constraints."""
        return bool(np.isfinite(z).all() and z.min()>=0 and z.max()<=1
                    and np.max(original_A@z-original_b)<=2e-12)

    if not feasible(x):raise ArithmeticError('Warm start is not feasible')
    starting,unused,unused_den=objective(x)
    if expected_initial_objective is not None and (not np.isfinite(expected_initial_objective)
            or abs(starting-expected_initial_objective)>1e-7):
        raise ArithmeticError('Rebuilt objective differs from private warm-start objective')
    for iteration in range(min(max_iterations,80)):
        value,g,den=objective(x)
        gap,certificate=gap_bound(x,g,A,b,gram)
        record=dict(iteration=iteration,objective=value,convex_gap=float(gap),point=x.tolist(),gradient=g.tolist(),**certificate)
        history.append(record)
        if progress_path is not None:json_write(progress_path,dict(history=history,terminal=bool(gap<=tolerance)))
        if gap<=tolerance:
            if not feasible(x):raise ArithmeticError('Original constraints violated')
            return x,dict(objective=value,convex_gap=float(gap),coordinates=x.tolist(),history=history,
                penalty_coefficient=1.,statistical_ridge_added=False,numerical_solver='stable_difference_face_Newton',
                original_tolerance=tolerance,warm_start_used=True,warm_start_objective=starting,
                original_problem_and_constraints_preserved=True,true_risk_certificate=False,**certificate)
        # Boundaries only select search directions; always compute the stopping gap over the original full feasible domain.
        lower=(x<1e-12)&(g>0);upper=(x>1-1e-12)&(g<0)
        snapped=x.copy();snapped[lower]=0.;snapped[upper]=1.
        if not np.array_equal(snapped,x) and feasible(snapped):
            x=snapped;value,g,den=objective(x)
        free=np.flatnonzero(~(lower|upper))
        if not len(free):raise ArithmeticError('No free Newton coordinate but original certificate unresolved')
        H=Q[np.ix_(free,free)].copy()
        for start in range(0,len(ct),65536):
            stop=start+65536
            B=ratios[start:stop,free]/den[start:stop,None]
            H+=B.T@(ct[start:stop,None]*B)
        scaling=1/np.sqrt(np.diag(H))
        direction=np.zeros(dim)
        direction[free]=scaling*np.linalg.solve(scaling[:,None]*H*scaling[None,:],-scaling*g[free])
        def take_step(direction, *, tangent=False):
            """[S5 numerics] Stable difference line search for the same objective; tangent handles the tangent space and dot-product rounding.
            
            direction is the full candidate direction; return the accepted point or None. Both paths preserve the full original
            constraints and strict descent; rounding slack alone never permits an uphill step.
            """
            if float(g@direction)>=0:return None
            limit=1.
            positive=direction>0;negative=direction<0
            if positive.any():limit=min(limit,float(np.min((1-x[positive])/direction[positive])))
            if negative.any():limit=min(limit,float(np.min(-x[negative]/direction[negative])))
            slope=A@direction
            # SVD constructs the null space in Hessian-scaled coordinates; its error is norm-based and cannot be measured solely by
            # the dot-product error sum(abs(A*d)). In particular, when the direction mainly moves along other free coordinates,
            # that measure underestimates the normal residual and truncates an intended tangent direction to a zero step.
            # Only screen out positive slopes at rounding scale below; original feasibility and a strictly negative objective difference remain necessary.
            allowance=0.
            if tangent:
                scaled_rows=A[:,free]*scaling[None,:]
                scaled_direction=direction[free]/scaling
                allowance=16*np.finfo(float).eps*(abs(A)@abs(direction)+
                    np.linalg.norm(scaled_rows,axis=1)*np.linalg.norm(scaled_direction))
            active=slope>allowance
            if active.any():limit=min(limit,float(np.min(np.maximum(b[active]-A[active]@x,0)/slope[active])))
            if limit<1:limit*=1-16*np.finfo(float).eps
            if limit<=0:return None
            for backtrack in range(60):
                proposed=x+limit*direction
                if feasible(proposed):
                    step=proposed-x
                    relative=(ratios@step)/den
                    if np.all(relative>-1):
                        # Compute the difference directly to avoid losing a tiny descent when subtracting two large negative values.
                        change=-math.fsum((ct*np.log1p(relative)).tolist())+float((Q@x)@step)+.5*float(step@Q@step)+float(cost@step)
                        if change<0:
                            record.update(stable_objective_change=change,step_length=limit,
                                          step_solver='active_face_Newton' if tangent else 'box_face_Newton')
                            return proposed
                limit*=.5
            return None
        proposed=take_step(direction)
        if proposed is None:
            direction,face_stats=face_newton_direction(H,g,x,free,certificate)
            record.update(**face_stats)
            proposed=take_step(direction,tangent=True)
        if proposed is None:raise ArithmeticError('Stable objective decrease unresolved on box and active constraint faces')
        x=proposed
    raise ArithmeticError('Original convex-gap tolerance unresolved after bounded recovery')
