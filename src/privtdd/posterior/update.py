"""S6: one update conditioning the joint distribution on reports, then restore the fixed degree marginal."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from privtdd.correction.sparse_channel import integer_law_adjoint, use_batched_channel, batched_integer_channel
from privtdd.posterior.rows import prior_profiles
from privtdd.base_distribution.channel import prepare_groups
from privtdd.degree.groups import groups_by_degree
from privtdd.correction.projected_sparse_channel import projected_sparse_adjoint


# [PrivTDD] S6 single posterior update; core method or probability/structure primitive (not necessarily a separate contribution); [Core S6 with ordered parallel execution] Compute sum_i w_id K_i(Q_i|d,t)/(n R_i).
def posterior_evidence(keys, degree, entries, report, p, eps_release, inverse, pool, *, batch_small_groups=False):
    """[Core S6 with ordered parallel execution] Compute sum_i w_id K_i(Q_i|d,t)/(n R_i).
    
    keys/degree: T support and fixed degree; entries: (cap, node indices, degree-posterior weights); report: Q.
    p/eps_release: Reporting mechanism; inverse: 1/(nR), with columns corresponding to candidates; pool: reusable thread pool.
    Use the projection channel matching S5; retain the original T support without enabling kernel interpolation from the old faster implementation.
    batch_small_groups: Implementation switch; small cap groups use a blocked matrix adjoint of the same kernel, changing floating-point summation order.
    """
    evidence = np.zeros((len(keys), inverse.shape[1]))
    # [PrivTDD] S6 single posterior update; memory and parallel execution; [Independent task] Posterior numerator contribution from one cap group; does not create new private reports.
    def cap_task(entry):
        """[Independent task] Posterior numerator contribution from one cap group; does not create new private reports."""
        cap, ids, weight = entry
        weights = weight*inverse[ids]
        if cap <= 1:
            if np.any(report[ids] != 0):
                raise ArithmeticError('Nonzero deterministic report')
            return weights.sum(axis=0)
        values, _ = projected_sparse_adjoint(report[ids], weights, degree, cap, keys, p, (cap-1)/eps_release)
        return values
    ordinary = [entry for entry in entries if entry[0] >= degree or entry[0] <= 1]
    if batch_small_groups and use_batched_channel(ordinary, degree, len(keys)):
        ids = np.concatenate([entry[1] for entry in ordinary])
        scales = np.concatenate([np.full(len(entry[1]), max(entry[0]-1, 0)/eps_release) for entry in ordinary])
        weights = np.concatenate([entry[2]*inverse[entry[1]] for entry in ordinary])
        evidence = batched_integer_channel(report[ids], scales, degree, keys, p, weights, adjoint=True)
        remaining = [entry for entry in entries if 1 < entry[0] < degree]
    else:
        remaining = entries
    for value in pool.map(cap_task, remaining):
        evidence += value
    if np.any(evidence < 0) or not np.isfinite(evidence).all():
        raise ArithmeticError('Invalid posterior evidence')
    return evidence


# [PrivTDD] S6 single posterior update; core method or probability/structure primitive (not necessarily a separate contribution); [Core S6] One empirical Bayes update, normalized per degree and aggregated with n*pi.
def update_posterior_joint(family, coordinates, normalizer, caps, report, prior, p, eps_degree, eps_release, *, workers=8, report_counts=None, batch_small_groups=False):
    """[Core S6] One empirical Bayes update, normalized per degree and aggregated with n*pi.
    
    family/coordinates: Frozen joint family and 16-dimensional coefficients; normalizer: joint predictive density per node.
    caps/report/prior/p/eps_degree/eps_release: Exactly the same private inputs and channel as in the forward fit.
    workers: Threads for cap tasks. Return a count histogram; retain the degree's joint prior when there is no evidence.
    report_counts: Representative-report multiplicities, or None for the original per-node path; each bin contributes its multiplicity times the posterior.
    batch_small_groups: Batch common-kernel computation for small cap groups without approximating report scales or posterior coefficients.
    """
    counts = np.ones(len(report)) if report_counts is None else np.asarray(report_counts, float)
    if counts.shape != report.shape or not np.isfinite(counts).all() or np.any(counts <= 0) or np.any(counts != np.floor(counts)):
        raise ValueError('Positive integer multiplicities aligned with reports required')
    n = int(counts.sum())
    entries = groups_by_degree(prepare_groups(caps, prior, eps_degree))
    inverse = 1/(n*np.asarray(normalizer).reshape(len(report), 1))
    if report_counts is not None:
        inverse *= counts[:, None]
    X = np.asarray(coordinates).reshape(1, 16)
    hist, native = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for j, d in enumerate(family['support']):
            keys, probabilities, _ = prior_profiles(family, j, X)
            evidence = posterior_evidence(keys, int(d), entries.get(int(d), []), report, p, eps_release, inverse, pool,
                                          batch_small_groups=batch_small_groups)
            mass = probabilities[:, 0]*evidence[:, 0]
            total = float(mass.sum())
            native.append(total)
            row = mass/total if total > 0 else probabilities[:, 0]
            if row.min() < -1e-12 or abs(row.sum()-1) > 1e-10:
                raise ArithmeticError('Posterior probability')
            for k, v in zip(keys, n*family['pi'][j]*row):
                if v:
                    hist[int(k)] = hist.get(int(k), 0.)+float(v)
    if abs(sum(hist.values())/n-1) > 1e-9 or abs(sum(native)-1) > 1e-8:
        raise ArithmeticError('Posterior population')
    return hist
