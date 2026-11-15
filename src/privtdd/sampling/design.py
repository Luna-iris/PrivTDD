"""Public cyclic design; original RNG order and seed derivation retained."""
from random import Random
import numpy as np
_ROT64_MASK=(1<<64)-1

def _derive_seed(base_seed, tag):
    x = (int(base_seed) &  _ROT64_MASK) ^ (int(tag) &  _ROT64_MASK)
    return int(_splitmix64(x) & 0x7FFFFFFFFFFFFFFF)


def _splitmix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) &  _ROT64_MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) &  _ROT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) &  _ROT64_MASK
    z ^= z >> 31
    return z &  _ROT64_MASK


def _build_rotated_design(node_num, seed):
    """Data-independent cyclic K_{m,m} design; exactly one matching per center."""
    rng = Random(int(seed))#from random
    n = int(node_num)
    nodes = list(range(n))
    rng.shuffle(nodes)
    if n % 2:
        nodes.append(-1)
    n_even = len(nodes)
    if n_even <= 1:
        return {
            "left": np.asarray([], dtype=np.int64),
            "right": np.asarray([], dtype=np.int64),
            "side": np.full(n, -1, dtype=np.int8),
            "pos": np.full(n, -1, dtype=np.int64),
            "center_shift": np.zeros(n, dtype=np.int64),
            "N_even": max(n_even, 1),
            "m": 0,
            "ring": 1,
        }
    m = n_even // 2
    ring = n_even - 1
    left = np.asarray(
        [int(nodes[-1])] + [int(nodes[(-i) % ring]) for i in range(1, m)],
        dtype=np.int64,
    )
    right = np.asarray(
        [int(nodes[0])] + [int(nodes[i]) for i in range(1, m)],
        dtype=np.int64,
    )
    side = np.full(n, -1, dtype=np.int8)
    pos = np.full(n, -1, dtype=np.int64)
    for i, node in enumerate(left):
        if node >= 0:
            side[node], pos[node] = 0, i
    for i, node in enumerate(right):
        if node >= 0:
            side[node], pos[node] = 1, i
    center_order = list(range(n))
    rng.shuffle(center_order)
    center_shift = np.empty(n, dtype=np.int64)
    for j, center in enumerate(center_order):
        center_shift[int(center)] = int(j % m)
    return {
        "left": left,
        "right": right,
        "side": side,
        "pos": pos,
        "center_shift": center_shift,
        "N_even": int(n_even),
        "m": int(m),
        "ring": int(ring),
    }
