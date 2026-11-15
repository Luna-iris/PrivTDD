# Third-party sources and distribution boundary

The root license covers the authors' code to the extent they hold the rights.
External projects, third-party portions, and datasets retain their own terms.
No upstream LDPGraph implementation, paper PDF, example data, results, or nested
Git repository is included in this artifact.

## TriTR-Hist

The included modules `privtdd.baselines.tritr_hist`, `histogram`, and `shared_rr`
are the research project's local triangle-degree-distribution adaptation. They
retain the frozen relationship budget `2*eps0 + 3*eps1 + 2*eps2 = epsilon`,
projection replacement sensitivity, pair-indexed shared RR, per-node counting,
clipping, rounding, and histogram construction. They are not a separate exact
reproduction of SIGMOD TDP-VC, and no approximate-DP TriTR variant is in the
paper runner.

Reference: Guo et al., *Locally Private Subgraph Counting via Noisy Adjacency
Matrix and Differential Privacy on Randomized Data*,
[arXiv:2507.06508v2](https://arxiv.org/html/2507.06508v2).
Related TDP-VC paper: [He et al., PACMMOD 2025](https://doi.org/10.1145/3725348).

Author artifact: [anonymous repository](https://anonymous.4open.science/r/Locally_Private_Subgraph_Counting_via_NAM_DPRD-A421/).
This is the URL cited by the paper's Open Science section and reference [1],
verified on 2026-11-15. The anonymous repository is an upstream reference,
not this project's distribution-estimation adapter. The source citation is
also available in `docs/baseline_references.bib`.

The local author-reference snapshot contains an MIT notice identifying
“The Subgraph-LDP-NAM-DPRD Contributors”, retained verbatim in
`LICENSES/TriTR-reference-MIT.txt`. Its `Init.py`, `OURS-TriTR.py`,
`OURS-TriTR-Star.py` and license digests are recorded in
`provenance/tritr_reference.json`. All four local files match the retrieved
anonymous repository files byte for byte: `BasicTools/Init.py`,
`Algorithm-Comparison-FB/Codes/OURS-TriTR.py`,
`Algorithm-Comparison-FB/Codes/OURS-TriTR-Star.py`, and `LICENSE`.
The record pins their observed SHA-256 values and retrieval date; no immutable
remote commit identifier is claimed for the anonymous service.
The full author archive is not bundled. `provenance/data_baseline_module_mapping.json`
records the local files and extracted functions used in this migration.

## LDPGraph-T and LDPGraph-S

Upstream: [AwfAwe/LDPGraph, fixed commit](https://github.com/AwfAwe/LDPGraph/tree/b903ddc7d93aa85dae951c536de3eededad69c5d).
Paper: [LDPGraph](https://arxiv.org/html/2610.09642v1).
The audited commit contains no LICENSE file, and the repository's GitHub license
field was null when checked on 2026-11-15. A public repository is not itself a
redistribution license. Obtain and use upstream code under the applicable terms;
the artifact does not grant upstream rights.

To run synthesis with an appropriately obtained checkout, supply its path as
`--external-code-root` to the paper runner. The lower-level Python adapter also
accepts `PRIVTDD_LDPGRAPH_SOURCE`; the top-level paper CLI requires the explicit flag.
The required files are `LDPGraph_main.py` and `utils.py`. The adapter checks the
fixed Git blob identities with explicit CRLF normalization. Missing or changed
files fail clearly. No automatic vendor download is performed.

The local adapter preserves the source's squared consistency weight
`(noisy_degree*eps_degree/eps_triangle)^2`, its threshold search, truncation and
rounding, and synthesis control flow. It applies the documented relationship
budget conversion, fixes shared-RR cache behavior with the pair-indexed simulator,
uses stable-priority projection, and checks the public `n-1` domain. Synthesis
extracts selected statements from the separately supplied files at runtime;
it removes the redundant dense adjacency matrix, stops on zero remaining degree
demand, and uses compact exact edge storage with a durable graph checkpoint.
These are documented adaptations, not a claim that the authors' code was run
unchanged. LDPGraph-T and LDPGraph-S share the same private report arrays.

The external-source path only separates distribution. Any independent permission
needed for upstream use or derived portions remains governed by upstream terms.
Review source provenance before public release; do not relicense upstream content
by copying it into a source archive, wheel, release ZIP, or container image.

## Datasets and dependencies

`datasets/manifest.json` identifies official SNAP/Network Repository sources,
citations, sizes, preprocessing, and current redistribution status. Real raw
graphs, processed arrays, ID maps, truth caches, reports, and synthetic baseline
graphs are excluded from public packages by default. An external archive or DOI
does not replace a dataset's redistribution requirements.

NumPy, SciPy, Numba, NetworkX, psutil, filelock, threadpoolctl, Matplotlib and
other declared packages are installed from their distributions. Their respective
licenses and notices remain applicable; this repository does not vendor them.

## LaTeX bibliography style

`paper/manuscript/icde_sections/IEEEtran.bst` is the retained bibliography style
by Michael Shell and other credited contributors. Its embedded notice declares
LPPL version 1.3; its original copyright, contribution notices and license
header are preserved. It is not relicensed under the root MIT license and is
not part of the Python wheel. Other LaTeX packages are supplied by the user's
LaTeX distribution.
