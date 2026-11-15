# Paper datasets

The four paper inputs are Bio-DMELA, DBLP, Higgs social, and Orkut. `manifest.json`
records official sources, preprocessing, frozen raw-input digests, node counts,
edge counts, array dtypes, and the SHA-256 of every prepared array. It is also
packaged as `privtdd.data/registry.json` for installed use.

Real data under `raw/`, `processed/`, and `truth/` is local working data and is
excluded from the public repository/package. The code license does not grant
rights to these datasets. Redistribution permission has not been established;
use the linked official sources and their applicable terms. Local copied paper
inputs are available under `processed/<dataset>/`; they are not an authorization
to upload those files.

| Dataset | Nodes | Undirected edges | Node ordering |
|---|---:|---:|---|
| Bio-DMELA | 7,393 | 25,569 | Preserve the MatrixMarket declared 1-based domain, including isolates |
| DBLP | 317,080 | 1,049,866 | Sort observed original IDs, then map to 0..n-1 |
| Higgs social | 456,626 | 12,508,413 | Assign IDs on first appearance in the original edge stream |
| Orkut | 3,072,441 | 117,185,083 | Sort observed original IDs, then map to 0..n-1 |

Preprocessing forms the undirected union, removes loops, merges duplicate edges,
and sorts adjacency rows. Nodes appearing only in loops are retained. Bio-DMELA's
single-percent `%MatrixMarket` header is supported. Higgs removes 23 self-loops;
do not substitute retweet, reciprocal, sampled, or largest-component variants.
DBLP and Orkut's upstream files already represent the upstream largest component.

`new_to_original.npy` stores the reproducible node mapping as int64. `indptr.npy`
and `triangles.npy` are int64; `indices.npy` is int64 for Bio-DMELA and int32 for
the other three graphs, matching the frozen paper arrays. `source_graph.json`
retains historical metadata locally. The new `graph.json` distinguishes the
observed maximum degree from the paper's public degree cap **n-1**.

Python preparation API (after installation):

```python
from privtdd.data import prepare, verify

# Download only when explicitly enabled; otherwise supply the official file.
prepare("bio-dmela", "datasets", allow_download=True)
prepare("higgs", "datasets", raw_path="/path/to/higgs-social_network.edgelist.gz")
assert verify("bio-dmela", "datasets", full_hash=True)["ok"]
```

`PRIVTDD_DATA_ROOT` can point to a dataset root containing `processed/` and
`raw/`. No old research repository is searched. A corrupt or differently ordered
input raises an error rather than silently selecting another cache.

Preparation streams the edge input into a temporary disk file and builds CSR
using memory-mapped neighbor arrays, retaining memory proportional to the node
domain and bounded chunks. Temporary disk usage can exceed final CSR size.
The four selected graph/truth snapshots total approximately 1.032 GiB before ID
maps and metadata. That is not a peak-memory or whole-experiment storage estimate.
Triangle truth is calculated by exact oriented intersections and is used only
after predictions are committed for evaluation.

Higgs's registry records the digest of the decompressed original source, so the
same official content is accepted in gzip or uncompressed form. The compressed
container itself is not claimed to have a previously frozen byte hash.

`examples/triangle_with_isolates.mtx` is a self-created tiny graph with known
truth and may be distributed with this repository. It is not a paper dataset.
