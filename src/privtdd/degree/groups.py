"""Degree observation grouping for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations



# [PrivTDD] S2 degree prior; core method or probability/structure primitive (not necessarily a separate contribution); Invert cap-group indices into a list of (cap, node indices, weights) for each d.
def groups_by_degree(groups):
    result={}
    for group in groups:
        for d,weight in zip(group['degree'],group['posterior']):
            if weight>0:result.setdefault(int(d),[]).append((group['cap'],group['nodes'],float(weight)))
    return result
