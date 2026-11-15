"""Reference source mapping for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

import numpy as np



# [PrivTDD] S5 joint family; core method or probability/structure primitive (not necessarily a separate contribution); Map the full Geo/Beta sources to the common 16 coordinates for endpoint identities in the shared representation.
def source_map():
    """Map (Geo, Beta) mixture weights into the old regional coordinates."""
    mapping = np.zeros((16, 2))
    mapping[[0, 1, 4, 5, 15], 0] = 1.
    mapping[[2, 3, 6, 7, 14], 1] = 1.
    return mapping
