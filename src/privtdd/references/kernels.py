"""S4 equivalent implementation: NumPy retains the original random streams; Numba compiles only deterministic geometric counting."""
import numpy as np
from numba import njit
from privtdd.references.primitives import geometry_count
from privtdd.references.independent_edge import near_pair_counts


def endpoint_block(count_matrix, root, P, rng_beta, rng_hard, include_hard):
    """[S4 implementation] Batch lambda=0 endpoints without changing type-pair order or the binomial sampler.
    
    count_matrix: Neighbor-type count matrix; root/P: root type and connection probabilities.
    rng_beta/rng_hard: Original independent NumPy random streams; include_hard: whether to compute the hard endpoint.
    262144 is the block-size target for temporary type-pair arrays, not a limit on sample counts or model support.
    Return (Beta, hard, Beta) for each row; the middle column is zero if hard is not computed.
    """
    counts = np.asarray(count_matrix, np.int64)
    output = np.zeros((len(counts), 3), np.int64)
    i, j = np.triu_indices(counts.shape[1])
    diag = i == j
    chunk = max(1, 262144 // max(1, len(i)))
    probabilities = P[i, j]
    for start in range(0, len(counts), chunk):
        c = counts[start:start + chunk]
        totals = c[:, i] * c[:, j]
        totals[:, diag] = c[:, i[diag]] * (c[:, i[diag]] - 1) // 2
        rows, cols = np.nonzero(totals > 0)
        samples = rng_beta.binomial(totals[rows, cols], probabilities[cols])
        np.add.at(output[:, 0], start + rows, samples)
    output[:, 2] = output[:, 0]
    if include_hard:
        for row, count in enumerate(counts):
            active = np.flatnonzero(count)
            c = count[active]
            degree = int(c.sum())
            if degree < 2:
                continue
            offsets = np.r_[0, np.cumsum(c)]
            positions = np.empty(degree)
            for g, typ in enumerate(active):
                width = P[root, typ]
                positions[offsets[g]:offsets[g+1]] = np.sort(
                    rng_hard.uniform(-width/2, width/2, int(c[g])))
            output[row, 1] = geometry_count(positions, offsets, P[np.ix_(active, active)])
    return output


@njit(cache=True)
def _geometry_rings(counts, root, P, inner_width, outer_width, arc_weight, uniforms):
    """[S4 implementation] Determine positions and inner/outer-ring counts from supplied uniforms; do not generate random numbers.
    
    counts/root/P: One neighbor composition, root type and connection probabilities.
    inner_width/outer_width/arc_weight: The two arc widths and weights of the original concentric-circle model.
    uniforms: 2*d random values in the original type order, choosing an arc before generating positions for each type.
    Return the inner-ring edge count, outer-ring candidate-pair counts and corresponding connection probabilities, in the original upper-triangle order.
    """
    active = np.flatnonzero(counts)
    c = counts[active]
    degree = int(c.sum())
    offsets = np.empty(len(active)+1, np.int64)
    offsets[0] = 0
    positions = np.empty(degree)
    inner = np.empty((len(active), len(active)))
    outer = np.empty_like(inner)
    for g in range(len(active)):
        typ = active[g]
        offsets[g+1] = offsets[g]+c[g]
        v = arc_weight[root, typ]*inner_width[root, typ]/P[root, typ]
        base = 2*offsets[g]
        location = np.empty(c[g])
        for k in range(c[g]):
            width = inner_width[root, typ] if uniforms[base+k] < v else outer_width[root, typ]
            location[k] = (uniforms[base+c[g]+k]-.5)*width
        positions[offsets[g]:offsets[g+1]] = np.sort(location)
        for h in range(len(active)):
            inner[g, h] = inner_width[typ, active[h]]
            outer[g, h] = outer_width[typ, active[h]]
    near = near_pair_counts(positions, offsets, inner)
    wide = near_pair_counts(positions, offsets, outer)
    pairs = len(active)*(len(active)+1)//2
    rings = np.empty(pairs, np.int64)
    probabilities = np.empty(pairs)
    value = 0
    index = 0
    for g in range(len(active)):
        for h in range(g, len(active)):
            total = c[g]*(c[g]-1)//2 if g == h else c[g]*c[h]
            ring = wide[g, h]-near[g, h]
            if near[g, h] < 0 or ring < 0 or wide[g, h] > total:
                raise ArithmeticError('Invalid nested pair counts')
            value += near[g, h]
            rings[index] = ring
            probabilities[index] = 1-arc_weight[active[g], active[h]]
            index += 1
    return value, rings, probabilities


def geometry_block(count_matrix, root, P, inner_width, outer_width, arc_weight, rng):
    """[S4 implementation] Original NumPy random stream for Geo sampling plus deterministic compiled counting.
    
    Inputs match draw_count, extended to multiple composition rows; rng is the original stream=4.
    Each row still draws 2*d uniforms and binomial counts for the original upper-triangle type pairs; sample counts are not reduced.
    """
    output = np.zeros(len(count_matrix), np.int64)
    for row, counts in enumerate(count_matrix):
        degree = int(counts.sum())
        if degree < 2:
            continue
        near, rings, probabilities = _geometry_rings(
            counts, root, P, inner_width, outer_width, arc_weight, rng.random(2*degree))
        output[row] = near + rng.binomial(rings, probabilities).sum()
    return output
