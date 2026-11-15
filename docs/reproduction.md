# Reproduction scope and statistical conventions

`reproduction/manifest.json` maps stable paper labels to configurations,
lightweight source tables, generators and hashes. Frozen results are at
`results/published/paper-v1`; fresh results are separate.

E1 has 1,916 completed outputs of 1,920 planned. The four missing PrivTDD fits
are Higgs epsilon=0.5 seeds 20261033/20261036 and Orkut epsilon=4 seeds
20261028/20261034. E2's five main curves use 476 matched report sets. The two
additional reference ablations each have 411 matched runs and the other five
additional variants each have 410; each contrast has its own matched Full set.
E3 has 234/240 outputs, or 78 matched three-configuration sets. It uses ten
seeds per network/budget, including on small graphs; E1 small graphs use twenty.

E4 has forty sampled outputs: two networks, epsilon=2, four rates and five
seeds. Its published 100% row uses the ten E1 runs, not the five matched E4
controls. Shared seed indices across independent protocols or sampling rates
do not by themselves establish shared reports. E2 and LDPGraph T/S matching
are verified using report hashes.

MSE is count-histogram squared error divided by choose(n-1,2)+1. EMD is the
integer-support integral of the normalized CDF gap; sparse support intervals
retain their widths. KS is the maximum normalized CDF gap. E1 display scaling
is MSE x1000 and KS x100. Confidence intervals are pointwise percentile bootstrap
intervals from 2,000 resamples with seed 731. Paired differences resample
matched run indices together. Failure and missing output are not zero error.

The figure command recomputes frozen statistics and tests their agreement with
the published values. It also regenerates the 546 additional E2 summary/contrast
rows and verifies E3 prose diagnostics reconstructed from the saved predictions.
The active content of the native S1–S3 supplement source is preserved; Chinese
editing comments are translated or removed in the public copy. An extended
document adds E4 and implementation details. This preservation is distinguished from the
separate numerical recomputation of its source tables.

`provenance/prediction_archive.json` and `prediction_references.csv` index the
selected final predictions and original metadata. Local content-addressed
objects are stored in the ignored `results/archive` directory. They are not
included in the GitHub candidate. A future external archive must retain these
hashes and have the applicable distribution conditions resolved. Raw reports
and large intermediate work caches are regenerable and not part of this archive.

The E3 reconstruction input manifest references `datasets/processed` and
`results/archive/predictions` by relative path. Its generator validates all
prediction and truth hashes before re-evaluating the fixed true-percentile
ranges. Derived per-run contributions suffice to recheck the prose without
loading the graph. The original historical diagnostic generator was not found;
the replacement is explicitly identified as a verified reconstruction.

Fresh execution has its own plan, code hash, completion markers, costs and
failure records. Repeating the same command validates completed artifacts and
resumes missing work. Explicit retry is required for recorded failures. A
source/configuration change requires a new output root. Newly successful runs
must not be inserted into the frozen paper snapshot without a new release and
an explanation of the changed sample set.

With the complete local data and prediction archive, independently recompute
all 7,572 referenced rows (22,716 metric comparisons) and the E3 diagnostics:

```bash
python reproduction/checks/verify_prediction_archive.py --root . --output results/checks/archive
```

This check requires the local-only processed truth and archived predictions;
the public figure-reproduction command only needs the bundled frozen tables.
