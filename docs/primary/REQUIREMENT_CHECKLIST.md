# Primary evidence and requirement checklist

Audit date: 2026-09-08. Companion protocol:
[SCIENTIFIC_WORKFLOW.md](SCIENTIFIC_WORKFLOW.md). This checklist distinguishes
implementation, released artifacts and unresolved replication evidence. Unit
tests and completion markers are not themselves proof of scientific claims.

## Released study contract

| Requirement | Evidence | Status/interpretation |
|---|---|---|
| Nine primary cells | Final `primary/bundle_summary.json`, revision manifests | Granite 8B/Mistral 24B/Qwen 32 × three datasets; not 21 completed cells |
| Seven reported methods | `PRIMARY_REPORTED_METHODS`, bundle summary | Six single-model methods plus origin; gap excluded |
| Sample versus metric counts | Construction protocol and bundle summary | 78,000 prepared rows/cell; 392 metric records/cell; 3,528 combined records |
| Source lineage | Revision/source artifact fingerprints | Eight September 4 revisions; Qwen 32-SQuAD September 7 V4.2 |
| Source versus publication report | Original reports plus bundle manifest | Source may say 8/448; publication filter correctly yields 7/392 |
| Preservation | Revision/export entrypoints and hashes | Corrections have separate IDs; documentation audit does not change data |

## Construction and scoring

Paths below are relative to `experiment_core/`.

| Requirement | Implementation | Evidence/caveat |
|---|---|---|
| Deterministic sources | `preparation/pipeline.py:select_source_manifest` | Target-independent signature, 260-word filter, packing/deduplication; verify saved overlap for a new paired cross-model analysis |
| Disjoint 500/500/2000 | `preparation/data.py:assign_splits_and_generation_seeds` | Variants remain source-dependent; packed SQuAD source is not one QA pair |
| Seed meaning | Same function/config | 101/202/303 divide sources, not triple generation |
| Sampling settings | Saved revision generation config | Transformers, nominal 30/220 tokens/min 210, temperature 0.8/top-p 0.95 |
| Actual overrides | Cell revision manifests | Granite guard 8/hidden states true; others guard 12; explicit construction 20 only Qwen WritingPrompts |
| Token-splice interpretation | `random_token_contamination`, `tail_token_contamination` | No grammatical human-edit guarantee |
| Tail cache | `preparation/pipeline.py:build_tail_cache` | Once-scored high mean-NLL donor order, prompt-conditioned suffix attack |
| Reusable features | `detectors/scoring.py:TargetModelScorer`, `BinocularsScorer` | Six target methods per pass; saved Falcon components support conditional origin |
| Context | `prepare_scoring_tokens` | Separately encoded prompt/response, no added specials; response-only evidence, combined 512-token ceiling |
| Rank and entropy | `exact_token_features` | Competition rank, full-vocabulary entropy; ties/reduced precision can differ from other conventions |
| LRR direction | `analysis/evaluation.py:orientation(revised=True)` | Fixed+1, not historical learned direction |
| LRR denominator | `detectors/detector_revision.py:valid_lrr_clipping_spec` | Finite positive log-rank cap; Qwen 32-SQuAD correction in V4.2 |
| Origin formula | `origin_components`, `origin_score` | Conditional same-window ratio, fixed-1; distinct RAID official protocol |
| Entropy gap | `analysis/evaluation.py:detector_local_values` | NLL minus entropy; no variance normalization |

## Tuning, calibration and uncertainty

Functions are in `experiment_core/analysis/evaluation.py` unless noted.

| Requirement | Implementation | Precise contract |
|---|---|---|
| Clean tuning quantiles | `_candidate_specs` | Pooled human+machine tokens, seven levels plus raw;LRR 49 pairs before guards |
| Primary objective | `tune_clipping_spec` | 0.8 attacked pooled AUROC +0.2 clean AUROC; no hard clean-loss budget |
| Mixture weighting | `evaluate`, saved config | 10/20/30/40/50%; 3: 1 random:tail rows; 5% evaluated but not tuned |
| Frozen bound | `primary_frozen_mixture` | Same spec across modes/ratios/FPRs; optional mode oracle disabled |
| Degeneracy | Detector-revision guards | Exact constant clean scores/full saturation; no test-driven tolerance |
| Human calibration | `calibration_threshold` | Separate raw/clipped thresholds from 500 clean humans at 1%/5% |
| Actual FPR | `actual_fpr`, metrics | Target is not a guarantee of equal held-out FPR |
| AUROC/pAUROC | `auroc`, `partial_auroc` | Tie-aware AUROC; partial area / 0.05, not sklearn standardized pAUROC |
| Robustness AUC | `robustness_auc` | Trapezoidal TPR–contamination area divided by range |
| Paired inference | `_metric_bootstrap` | 2,000 cluster resamples, source shared raw/clipped; all draws retained |
| Curve dependence | `_robustness_bootstrap` | Source resample retained across ratios |
| Interval scope | `percentile_interval`, bootstrap calls | Pointwise 95%, fixed fitting/calibration, no multiplicity or selection correction |
| Efficient evaluation | `precompute_evaluation_scores` | Frozen scalars cached once per detector/analysis; no unsupported speedup claim |

## Verification limits before strict replication or paper claims

The local bundle contains final summaries, selected configuration/revision
records, original report provenance and publication outputs. These supersede
old blanket claims that all full generation/CUDA execution is awaiting Delta.
The separately reported Delta layout check passed 105 tests after reorganization;
that engineering check is not a replay of all scientific computations.

- Retrieve original source/scoring manifests for resolved model, tokenizer and
  dataset commits. Null default revision fields are not sufficient pinning.
- Use original prepared/base rows for empirical token lengths, normalization,
  realized ratios and cross-model source overlap. Large JSONL packs were
  intentionally excluded from the download.
- Preserve actual cell-specific guards. A recorded TP setting is not proof of
  tensor parallelism under the Transformers/device-map backend.
- Use original job logs for actual topology, wall time and memory; this audit
  establishes no current queue state, storage balance or hardware benchmark.
- Optional vLLM, Erebus and other Qwen scales are outside verified released
  primary evidence. Presence of code/config does not establish completion.
- Report amendments and earlier result inspection. Avoid unsupported
  preregistration or completely untouched confirmatory-test descriptions.
- RAID has its own benchmark-specific tuning/calibration. The old blanket
  prohibition on external recalibration does not describe the released RAID study.

## Maintenance boundary

No data regeneration or retuning is implied by this checklist. Reanalysis uses
a new ID via `tools/reanalysis/reevaluate_detector_revision.py`, preserving
sources. Plot/export changes are separate from inference. Historical programs
live in ignored `archive/` and are not required by active tests. User-owned
`paper/` materials are outside experiment maintenance.
