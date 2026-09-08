# Detector amendment: report origin, fix LRR, reject degenerate candidates

Revision ID: `binocular-origin-lrr-constant-v1`.
Implementation amendment: `official-score-anchored-clipping-v4.2`. Use new result and
RAID revision IDs; do not resume a gate produced by the earlier implementation.
Version 4.1 preserves the saved official numerator when Falcon and Binoculars
score packs are merged; version 4 omitted that field and could not evaluate an
active origin-clipping candidate after otherwise successful gate scoring.
Version 4.2 rejects an LRR clipping rule whose log-rank cap is nonpositive.
Because token log-ranks are nonnegative, a zero cap collapses every denominator
contribution and silently changes LRR into a scaled numerator-only detector.
This is an explicit reanalysis, not a replacement of archived results. The
legacy entry points/configurations retain their old behavior for reproducibility.
Use the new entry points below to apply the changes together. No archived
JSONL, manifest, result, or frozen specification is renamed or overwritten.

## Scientific changes

- New primary nine-cell tables and figures report seven methods: the six
  single-model methods and `binocular_origin`. The evaluation engine retains
  `binocular_gap` only so the prior eight-detector artifacts remain readable and
  reproducible. It is not emitted in the new reported set. The legacy input key
  `binoculars` remains an alias for the exponential gap, never for the published
  ratio.
- Archived gap results retain `exp(mean(NLL - CE))` and their existing oriented
  local-gap clipping. No archived result is deleted or relabeled as origin.
- Origin uses `mean(NLL) / mean(CE)`, with smaller scores more machine-like.
  Origin clipping caps only numerator token NLL at one upper bound; its
  denominator stays fixed. Candidate bounds are the seven existing clean-tuning
  NLL quantiles plus no clipping. This clipped variant is our extension, not an
  upstream detector or a claim of exact equivalence to the theory appendix.
- Primary cells **keep prompt conditioning** and the identical saved continuation
  window for both terms. Their origin result is a conditional ratio adaptation,
  not an exact output-only reproduction. No preparation or GPU rescoring needed.
- RAID origin uses output-only tokenizer-native inputs, right truncation at 512,
  shifted performer NLL, and observer-to-performer CE on all input positions,
  including the final position. Its denominator mask is `input_id != pad_id`;
  numerator mask is shifted attention. BF16 raw reductions match the upstream
  [metrics implementation](https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py).
  A new, separately named score pack is necessary: the legacy CE arrays omit
  the final position. Legacy gap scores continue using the old arrays.
- RAID official raw numerator, denominator, and ratio retain the upstream CUDA
  BF16 values exactly. Serialized BF16 token losses do not encode CUDA's parallel
  summation order and therefore cannot always reconstruct its document numerator
  bit-for-bit. The custom clipped numerator is consequently anchored to the saved
  official numerator and applies a deterministic float64 Winsorization delta:
  `A_clip = A_official + mean(min(NLL,U)) - mean(NLL)`, bounded to `[0,A_official]`.
  No-clipping and bounds changing no token return the official raw scalar exactly;
  active clipping cannot increase its numerator. The denominator remains the
  saved official denominator and the final ratio uses float32 division. This is
  mathematically the ordinary clipped mean in exact arithmetic, while avoiding
  a false claim that NumPy reproduces hardware-dependent CUDA reduction order.
  The observed 511-token production counterexample is a regression fixture.
  Primary conditional ratios keep their existing float64 aggregation on BOTH
  raw and clipped sides and are unaffected by this amendment.
- LRR direction is fixed to `+1` (larger = machine) in revised runs. Formula,
  competition-rank tie handling, and two-component caps are unchanged. A
  nonempty LRR clipping specification is eligible only when its log-rank upper
  cap is finite and strictly positive. Invalid candidates are recorded as
  `nonpositive_lrr_log_rank_cap`; attempting to evaluate one also fails. Other
  learned directions remain unchanged; origin is fixed to `-1`.
- Reject a nonempty clipping candidate if **all pooled clean human and clean
  machine tuning document scores are exactly identical**. No tolerance, no new
  cutoff, no calibration/test inspection. No clipping is always available.
  Also reject structurally fully saturated candidates before averaging: different
  document lengths can otherwise create rounding-only differences despite every
  token being capped to the same constant. LRR requires both components fully
  saturated. Origin is not rejected on numerator saturation alone because its
  varying denominator can still carry information.
  Rejections appear in primary `clipping_rejections.json` and RAID candidate
  diagnostics. Zero TPR alone is not a reason to reject a candidate.
- Existing tuning objectives, rate bins, source splits, calibration and source-
  clustered bootstrap remain unchanged. Revised RAID still reports full and
  eligible universal methods, with four rate-oracle bounds as secondary results.
  Each configuration is calibrated separately; report held-out FPR with TPR.

## Programs to rerun

1. Primary nine cells: `tools/reanalysis/reevaluate_detector_revision.py`, once per cell,
   using each completed source run and a new result ID. Re-evaluate the seven
   reported detectors, not just SQuAD: the LRR rule and candidate safeguards are
   applied consistently to every cell. This is evaluation-only and reuses the
   immutable prompt-conditioned target and Binoculars score packs.
2. RAID: `RAID/revise_detectors.py --stage gate` on the completed source run.
   First scans ALL prepared texts with the pinned native tokenizer, without model
   inference, and verifies the prepared shards exactly cover the same records.
   Empty/one-token/all-pad windows fail with a diagnostic report; no text is
   dropped, repaired or changed automatically. Then selects one complete source
   family per split/domain, before any scores are inspected (normally 24 sources,
   312 rows for eight domains). Exercises real score-pack writing, unchanged
   resume, joining saved Falcon/gap packs, evaluation and plotting. These tiny
   gate results are engineering checks only and never supply full-run thresholds.
   Compares components on identical logits against separately obtained upstream
   `metrics.py`, including two extra long/Unicode probes.
   Records reference SHA256 and immutable model/tokenizer revisions. This gate
   checks raw component parity, not equality of the entire RAID published table.
3. After the gate passes, `--stage score` as four shards. Reads existing prepared
   shards, loads only the Falcon pair, writes `runs/raid_origin/<new-id>/`.
   No new dataset download/index, preparation, or Falcon single-model scoring.
4. `--stage evaluate`: joins old Falcon, old gap, and new origin rows by exact
   keys; reuses the source run's contamination CSV; evaluates/plots/validates eight
   detectors with 2,000 bootstraps in `results/raid/<new-id>/`.

Final RAID counts: 16 universal specifications (8 full + 8 eligible), 32 rate
specifications, 96 attack-summary rows, 32 rate-summary rows, 416 tidy metric rows,
and 7 original-paper sanity rows **for origin only**. Source/score row counts
must still equal the source run; for the previously completed full source this
was 167,323 rows. Never infer current Delta state from these historical counts.

## Delta operations

Use `tools/delta/delta_detector_revision.sbatch`. The user handles Git push/pull.
Before submission, check branch/status, available allocation, disk space, and
that both `readlink -f runs` and `readlink -f results` resolve below `/work/hdd`.
Create `logs` before using the wrapper. Models are loaded offline from the
existing shared cache. A cache miss stops the job instead of downloading models.
Use native Transformers Falcon (`trust_remote_code=False`), as in the completed
RAID source run. Do not load its incompatible cached legacy modeling code.

Environment variables:

- `SOURCE_RUN_ID`: completed source cell/run, never a results-only promotion ID.
- `REVISION_ID`: new ID, distinct from the source.
- `REVISION_STAGE`: `primary`, `gate`, `score`, or `evaluate`.
- `BOOTSTRAP_REPETITIONS`: defaults to 2000.
- `NUM_SHARDS`: defaults to 4; must match the existing prepared shard layout.
- `REFERENCE_METRICS`: gate only, path to the reviewed upstream `metrics.py`.
- `ADOPT_ORIGIN_RUN_ID`: score only; optionally copy a stopped earlier revision's
  raw origin shard after pre/post fingerprint and content-hash checks, then validate
  every adopted row and score only missing keys. Adoption is recorded per shard.

Submission templates (not auto-submitted; set IDs and review resources first):

```bash
# Primary: one job per source cell, no inference. GPU reservation is only for
# the GPU-only allocation; use a CPU account instead if one is available.
export REVISION_STAGE=primary
sbatch --account=bhuc-delta-gpu --gpus-per-node=1 --mem=240G --time=08:00:00 \
  tools/delta/delta_detector_revision.sbatch

# RAID: first obtain/review the upstream reference outside Git, then run gate.
# REFERENCE_METRICS must point to that file. The gate saves its SHA256.
export REVISION_STAGE=gate
sbatch --account=bhuc-delta-gpu --time=02:00:00 \
  tools/delta/delta_detector_revision.sbatch

# Only after gate.complete.json reports pass for this exact revision/source:
export REVISION_STAGE=score
sbatch --account=bhuc-delta-gpu --array=0-3%2 \
  tools/delta/delta_detector_revision.sbatch

# Only after all four scoring shards completed and have completion markers:
export REVISION_STAGE=evaluate
sbatch --account=bhuc-delta-gpu --gpus-per-node=1 --mem=240G --time=08:00:00 \
  tools/delta/delta_detector_revision.sbatch
```

The scoring wrapper defaults to two GPUs/task and two concurrent tasks in this
example. `%4` requires eight available GPUs. Historical four-shard pair scoring
took roughly 70–76 minutes per shard; the new batch-one official reductions need
fresh measurement. Treat 1.5–2 hours with four concurrent shards or 3–4 hours with
two as provisional estimates, excluding queueing, not guarantees.

An interrupted scoring shard resumes under the same source/revision/config.
An incomplete evaluation can also restart with the same inputs; completed outputs
are left alone. Completion now records SHA256/size for every required result and
plot; an already-complete resume verifies them rather than trusting markers.
An interruption between writing the validation report, manifest and final marker
is recoverable without repeating evaluation. Missing/modified artifacts in a
claimed complete result fail explicitly (no silent overwrite or false success).
A failed gate should use a fresh revision ID after diagnosis.
No old result directory should be deleted to make room for this amendment.

The expanded gate scans existing prepared and saved-score files, but does not
rebuild the RAID index or reread the original train.csv. Its GPU inference is
bounded; disk reading and full-data tokenization time still depend on Delta I/O.
It prints preflight progress every 10,000 records. Do not infer its duration from
the old 15-document gate.

## Validation status and limitations

Local no-download tests exercise ratio arithmetic, gap compatibility, fixed LRR,
nonpositive-cap rejection, exact constant rejection, the seven-method primary
reporting path, eight-detector compatibility evaluation, and cached-rate parity.
Torch-dependent component tests and the live upstream gate must pass on Delta;
local CPU tests alone do not establish GPU/model parity. The local environment
does not contain Torch, so no new inference has been performed here.

After publishing v4.2, retry a RAID gate with a new revision ID if producing a
fully versioned v4.2 RAID revision; do not submit a separate CUDA diagnostic.
The wrapper uses the standard Delta module
setup, checks the stage and reference-file arguments, and runs
`tools/validation/check_detector_revision_gate.py` before full-input tokenization or model
loading. That runner requires two visible CUDA GPUs and runs the entire revision
test module, including actual CUDA masked reductions (raw and capped) and
synthetic BF16 cross-entropy outputs. Any failed or skipped test stops the gate.
The same job then proceeds to the bounded real-model pipeline check. Previously
completed primary score packs do not require GPU inference. Their evaluation
does need a new result ID to apply the LRR candidate safeguard and seven-method
reporting set.

This is a disclosed methodological amendment after observing existing results.
Retain the historical tables and report the revised protocol; improved outcomes
are not a completion requirement. Do not tune further against the held-out tests.
