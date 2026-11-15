# Method and naming

PrivTDD estimates the count histogram of node triangle degrees. Its total
histogram mass is the public node count; the target is not the global number
of triangles and does not divide each node's triangle degree by three.

The package follows the manuscript's four phases. `reporting.collection`
simulates the private degree and shared-RR relational reports; `degree` and
`base_distribution` recover their population and conditional structure.
`references` constructs the degree-driven and clustering references.
`correction` fits regional probability-mass and within-region shape changes.
`posterior` reweights reports and `release` aggregates and applies the final
transport safeguard.

The public API uses `PrivTDDConfig`, `estimate_tdd`, and
`estimate_tdd_from_reports`. Technical joint-distribution operations remain
joint operations; their names do not denote an older method. Historical F1
means the base distribution fit. SCRec means F1 plus posterior update and
the final safeguard, not F1-only. `provenance/module_mapping.json` lists the
renamed modules and three renamed public symbols.

The paper runner explicitly selects L=0, tau=1, K=65, quantiles 0.90/0.95,
the fast 512-degree grid, report/T bins 1/16, structural integration 65536x4,
and one posterior update. The scientific and numerical tolerances are retained.
The generic API also supports other numerical settings; these should not be
called the paper preset. The three budgets use the frozen allocation table,
with 2*eps_deg + eps_rr + 2*eps_rel = total epsilon. `privtdd.budget` derives
the current manuscript's bound-optimal split from public inputs only.

Truth enters dataset validation and evaluation only. In particular, the E3
true-positive percentile diagnostics are post-hoc evaluation, not region
boundaries used by the estimator. The method's own regions use the fitted
base population distribution. Small-capacity exact likelihoods and
large-capacity working approximations retain their original distinction.

The original package's ConditionalDistribution and beta refinement helpers
are now internal modules. The data layer contains only the required graph
operations and triangle counter, not the former C9 estimator. Sampling contains
the required matching/rotation and moment operations, not the old BodyBeta
pipeline. Source identities are based on package-relative content and have a
new schema; historical source identities are preserved in immutable evidence.
