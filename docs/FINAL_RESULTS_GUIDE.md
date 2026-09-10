# Final results: provenance and reporting guide

Audited 2026-09-08. This guide identifies the accepted downloadable results and
explains how to turn them into tables without confusing historical and final
methods. It does not announce a new experimental run.

## 1. Accepted local bundle

Relative to the repository root:

`downloads/current/primary-nine-plus-raid-final-20260908T150247Z/`

The directory contains `primary/`, `raid/`, `bundle_manifest.json`,
`bundle.complete.json`, and a README. The September 8 documentation audit
recomputed sizes and SHA256 for all **182 artifacts recorded by the completion
marker**, and all matched. The marker itself is not one of its own hashed
artifacts. These checks establish integrity of this bundle, not a fresh replay
of inference or proof of every methodological assumption.

There are no large JSONL feature packs in the download. Retain the original
prepared data, model-score packs and their manifests on Delta. Older downloads
under `downloads/archive/` remain provenance; do not overwrite them with the
new bundle or treat their gap-based Binoculars tables as final ratio results.

## 2. Primary nine cells

| File under primary/ | Rows / purpose |
|---|---|
| `primary_all_metrics.csv` | 3,528 tidy final metric records |
| `primary_clean_metrics.csv` | 252 clean-condition records, retaining both display modes |
| `primary_clipping_effects.csv` | 1,638 positive-contamination clipped-minus-raw records |
| `primary_robustness_auc.csv` | 504 curve-summary records |
| `primary_cell_inventory.csv` | 9 cell inventory records |
| `primary_protocol_config.json` | Export-time configuration snapshot, not every cell's exact overrides |
| `primary_cells/` | Per-cell metrics, figures and original revision evidence |

Each cell has 392 final metric rows and 11 PNG figures, giving **99 primary
plots**. The seven methods are log likelihood, rank, log rank, LRR, entropy,
entropy gap, and `binocular_origin`. All final primary origin results remain
prompt-conditioned; they are not RAID's output-only scoring protocol.

One metric record is indexed by dataset/model, analysis, detector, mode,
requested ratio, target FPR and aggregation. It is not an individual sample.
Per cell, the product is seven detectors, two modes, seven ratios, two target
FPRs, and two aggregations. The underlying protocol has 3,000 source clusters
and 78,000 prepared/scored rows per cell. Random-condition test TPR uses three
draws per test source; that does not triple the independent source count.

### How to avoid double counting

- For a TPR curve, choose one cell, detector, mode and target FPR; compare raw
  and clipped at each requested ratio.
- Clean machine observations are shared by random and tail displays. Do not
  count the two clean panels as independent clean experiments.
- AUROC does not depend on the chosen classification threshold and is repeated
  across target-FPR records. Choose one FPR record for an AUROC-only table.
- Robustness AUC summarizes a complete contamination curve. Use the dedicated
  curve table; do not average repeated curve values over ratio rows.
- Selected specifications belong to clipped configurations. An empty spec in
  a raw row is not evidence that the selector chose no clipping.
- Use stored paired differences and paired intervals, not a difference of
  marginal CI endpoints. A cross-cell or cross-model paired CI needs verified
  source overlap and a new explicitly defined analysis.

### Which revisions are included?

The complete nine-source list is in the
[primary workflow](primary/SCIENTIFIC_WORKFLOW.md#9-final-artifact-lineage).
Eight cells use the September 4 revision suffix
`-revision-20260904T163234Z`. Qwen32–SQuAD instead uses
`qwen32-squad-promptfix-20260804T133827Z-lrr-v42-20260907T163410Z`.
Only that cell needed the final positive-LRR-cap reevaluation. Do not describe
all nine as rerun under v4.2.

Copied source reports in eight cells legitimately say eight detectors and 448
rows. The publication exporter removed `binocular_gap`, leaving seven and 392.
Qwen32–SQuAD's newer source already reports seven and 392. Original reports and
their hashes are immutable provenance of their original files, not expected
hashes of filtered/regenerated publication artifacts. Use the top-level bundle
hash inventory to validate exported files.

Current `configs/paper.json` differs from some saved configurations. For example,
Granite records a base guard of eight and saved hidden states; the later cells
record a base guard of twelve. Only Qwen WritingPrompts explicitly records the
separate twenty-token construction guard. See the saved-override table in the
primary workflow before writing the Methods section.

## 3. RAID: source evidence versus paper view

Accepted revision: `raid-origin-anchored-v41-20260907T044156Z`.
Original prepared/scored source:
`raid-full-excluded500-bootstrap500-provisional-20260824T132810Z`.
The final revision evaluates with **2,000 bootstraps**, notwithstanding the
historical source ID's bootstrap500 text.

| View | Purpose |
|---|---|
| `raid/source_eight_detector_result/` | Complete accepted revision with gap retained, validation, frozen specs, manifests and source figures |
| `raid/paper_seven_detector_result/` | Filtered publication CSVs/figures omitting gap; not a separate fitted experiment |
| `raid/raid_export_summary.json` | Describes the transformation into the publication view |

| Artifact | Source eight-method rows | Paper seven-method rows |
|---|---:|---:|
| `metrics.csv` | 416 | 364 |
| `attack_summary.csv` | 96 | 84 |
| `contamination_summary.csv` | 32 | 28 |
| `calibration_summary.csv` | 384 | 336 |
| `rate_bound_tradeoff_summary.csv` | 336 | 294 |
| `binoculars_sanity.csv` | 7 | 7 |
| `contamination_records.csv` | 154,452 | 154,452 |

There are **six RAID PNGs in the combined bundle**: three source eight-detector
plots and three paper seven-detector plots. Paper-facing plot names are
`raid_attack_tpr.png`, `raid_contamination_tpr.png`, and
`raid_rate_bound_tradeoff.png`. Together with the 99 primary plots this gives
102 paper-facing figures, not six independent RAID result sets.

The source report records 12,871 sources after excluding 500 development
sources; 167,323 prepared rows and matching rows in each required full score
pack. Splits contain 5,148 tuning, 2,574 calibration and 5,149 test sources.
Each source has one human document plus one selected machine generation under
twelve conditions (clean plus eleven attacks). Metric counts and attack rows
must not be described as independent sample sizes.

### Reading RAID comparisons

- `attack_summary.csv` includes `none` plus all eleven attacks. It includes
  zero-change and dense examples, even for the eligible-trained method.
- `contamination_summary.csv` contains four positive-rate bins up to 0.5. It
  does not by itself summarize zero-change or dense behavior.
- Full universal, eligible universal, and rate-oracle have distinct meanings:
  they differ in tuning populations, not the scorer or final test texts.
- Actual test FPR must accompany TPR. Calibration is per domain and per
  configuration; the target is 5%, not a forced 5% test exceedance rate.
- Per-rate AUROC compares a bin's positives with the applicable clean-human
  test reference; it is not an AUROC between adjacent contamination bins.
- Subtract full and eligible point estimates directly if desired. Do not infer
  a paired full-minus-eligible CI from the two clipped-minus-raw CIs: their
  bootstrap seeds differ. That comparison needs a jointly paired calculation.
- The oracle tradeoff table evaluates bounds beyond their training bin; it
  does not license choosing whichever bound performs best on each test attack.

## 4. Final raw Binoculars check against the RAID table

These numbers are from the accepted **origin** sanity CSV, not the earlier
gap-based provisional table. Values below are percentages, rounded for display.
The published targets are the RAID comparison values recorded in that CSV and
verified against [Dugan et al., RAID, Table 6](https://arxiv.org/html/2405.07940v2).

| Condition | Published TPR | Final raw origin TPR | 95% test-source CI |
|---|---:|---:|---|
| None | 79.6 | 80.00 | 78.91–81.12 |
| Paraphrase | 80.3 | 80.83 | 79.76–81.92 |
| Synonym | 43.5 | 43.58 | 42.26–44.86 |
| Perplexity misspelling | 78.0 | 78.40 | 77.28–79.49 |
| Homoglyph | 37.7 | 37.56 | 36.30–38.82 |
| Whitespace | 70.1 | 70.09 | 68.83–71.26 |
| Article deletion | 74.3 | 74.66 | 73.49–75.90 |

Raw origin held-out FPR is **4.6223%**, shared across these attack conditions;
the calibration target is 5%. Agreement supports the selected baseline sanity
check. It is not an exact replay of the entire RAID paper: the experiment uses
its own source/configuration selection, development exclusion, and split-specific
calibration. Paper-facing claims should say comparable subset/protocol and name
the differences, not full identical replication.

## 5. Statistical and reproducibility limits

The intervals resample test sources while keeping directions, fitted bounds
and human calibration thresholds fixed. They do not include refitting uncertainty
or multiple-comparison correction. Methodological amendments followed earlier
result inspection; do not call the whole released analysis preregistered or
describe all test outcomes as previously unseen confirmatory evidence.

The bundle is sufficient for many plots, summary tables and selected-parameter
audits. It is insufficient to rerun GPU inference, reconstruct every source
alignment, verify cross-model source pairing, or recompute arbitrary score-level
statistics. Those require the original Delta packs and corresponding versioned
code/model/tokenizer revisions. Historical manifest paths predating repository
reorganization are provenance, not instructions to recreate old folder names.

No scientific data or result file was modified during this documentation audit.
Use [the audit record](DOCUMENTATION_AUDIT.md) to see what was independently
checked and what remains an evidence limitation.
