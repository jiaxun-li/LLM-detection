# Paired intervals for the mean over eleven attacks

The analysis also reports paired clipped-minus-raw human FPR intervals in
`paired_fpr_ci.csv`. Each of the 5,149 human sources contributes once; humans
are not replicated per attack. The same domain-stratified bootstrap draws are
used for TPR and FPR. Positive FPR change means more false positives. An interval
containing zero does not establish equivalence or noninferiority. It is an
interval for the change, not a test against the nominal 5% target. The updated
compact decisions include `human_hits`, and joint replicates include
`fpr_changes`. Rerunning writes a new job-specific directory, preserving the
previous TPR-only output. No model scoring, fitting or recalibration occurs.

`tools/reanalysis/raid_attack_mean_ci.py` adds an analysis of the accepted v4.1
RAID release without changing its results. It streams the original Falcon pack
and four accepted origin packs on Delta. It does not load models, select bounds,
or recalibrate thresholds. Frozen specifications and per-attack references must
match the accepted completion-marker hashes. Every per-attack raw/full/eligible
TPR, clean TPR, and human FPR must match the saved reference to absolute tolerance
1e-12 before bootstrap results are written.

For each of 5,149 test sources, retain its eleven attack decisions. Average these
within source, then resample sources with replacement separately within each
domain, preserving domain sample sizes. Use the same source multiplicities for
all attacks, detectors, and scopes. Because every source has every attack, this
equals the equal-weight attack mean in the report. Report paired clipped-minus-raw
TPR percentile intervals from 2,000 resamples, seed 481516. Fits stay fixed. These
are pointwise intervals, not multiplicity-adjusted intervals or a matched-test-FPR
comparison. Do not average existing per-attack interval endpoints.

The command requires a new output directory and preserves all source artifacts.
Outputs contain a 14-row summary (seven detectors, two scopes), joint bootstrap
replicates, compact source-level decisions, and a completion record with input
and output hashes. Compact decisions permit subsequent local audits without
downloading token features. No intervals for the all-attack aggregate are claimed
until this analysis has run on the actual Delta packs.

The wrapper `tools/delta/delta_raid_attack_mean_ci.sbatch` uses the established
Delta environment and verifies work-filesystem links. Its one GPU reservation
is solely for the existing GPU-type account; computation is CPU-only. It requests
16 GB RAM, four CPUs and a two-hour limit, not a measured duration estimate.
Submit from the repository root after publishing the new files and inspecting
the current checkout and storage. Output is under
`results/raid_aggregate_ci/accepted-v41-job-JOBID/`.

Local regression tests:

```text
python -m unittest tests.test_raid_attack_mean_ci -v
```

They verify exact no-change intervals, constant effects, preservation of
cross-attack correlation, paired scopes, fixed stratum counts, deterministic
resampling and rejection of incomplete attack dimensions. They do not replace
the full-data point-replay check on Delta.
