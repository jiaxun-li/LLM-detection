# RAID: which code and results to use

The detailed methods reference is [SCIENTIFIC_DESIGN.md](SCIENTIFIC_DESIGN.md).
Use [DELTA_GUIDE.md](DELTA_GUIDE.md) for operations and the
[detector amendment](../DETECTOR_REVISION.md) for the corrected detector route.

## Final released result

The corrected RAID revision is
`raid-origin-anchored-v41-20260907T044156Z`: 12,871 retained sources,
167,323 scorer rows, and 2,000 paired source-cluster bootstrap replicates.
It follows exclusion of 500 pilot development sources and a source-grouped,
domain-stratified 40/20/40 tuning/calibration/test split. Only calibration
humans fit thresholds, separately by domain and detector configuration, at
target 5% FPR. Always report achieved held-out FPR alongside TPR.

The downloaded bundle is
`downloads/current/primary-nine-plus-raid-final-20260908T150247Z/`.

| Folder below its `raid/` directory | Purpose |
|---|---|
| `paper_seven_detector_result/` | Paper tables/plots: six Falcon methods plus corrected `binocular_origin`; 364 metric rows, three plots |
| `source_eight_detector_result/` | Exact validated result: 416 metric rows, frozen specs, manifests, retained gap diagnostic |
| `raid_export_summary.json` | Counts and reporting-filter provenance |

The paper view is a filter, not a new fit. The gap is not the original Binoculars
ratio. The source validation intentionally still lists eight detectors. Large
JSONL packs were not downloaded and must remain on Delta for reanalysis.

## Code responsibilities

Paths below are relative to the repository root.

| Program / module | Function |
|---|---|
| `RAID/run_raid.py` | Base prepare/score/evaluate/plot/validate workflow; retains historical detector naming |
| `RAID/submit_raid.sh` | Dependency-safe preparation, Falcon and pair-model arrays, global finalization |
| `RAID/run_raid_shard.py` | One scoring partition; do not concurrently run unsharded scoring into one output |
| `RAID/raid_data.py` | CSV/index, family selection, exclusions/splits, edit alignment |
| `RAID/raid_scoring.py` | Shared Falcon and historical pair-model token features |
| `RAID/binocular_origin.py` | Additional output-only BF16 official-style ratio scorer with separate namespaced features |
| `RAID/revision_gate.py` | Full-input tokenizer check and bounded upstream-component/integrated pipeline probe |
| `RAID/revise_detectors.py` | Versioned gate, origin scoring/adoption, corrected global evaluation |
| `RAID/raid_evaluation.py` | Directions, cap selection, calibration, clustered intervals and reports |
| `RAID/raid_plot.py`, `RAID/plot_raid.py` | Plot implementation and base plot entry point |
| `RAID/validate_raid.py` | Base validation; corrected revisions have a separate report/marker contract |
| `RAID/promote_bootstrap.py` | Evaluation-only promotion of accepted base 500-bootstrap results to 2,000; not the subsequent ratio correction |
| `tools/exports/export_final_publication_bundle.py` | Validate/copy exact results and produce seven-detector views without refitting |

**Important:** `configs/raid.json` alone does not select the final ratio
amendment. The corrected route adds
`detector_revision=binocular-origin-lrr-constant-v1` and requires a separate
official-origin pack. Revised LRR has fixed direction +1, ratio Binoculars -1.
Current v4.2 code adds the positive LRR denominator-cap guard; saved RAID v4.1
specifications passed that audit without a rerun.

## What is fitted and reported

- Full universal uses all eleven attacked conditions. Eligible universal uses
  attacked rows with \(0<\rho\leq0.5\). Each fits one specification per detector
  with 0.8 equal-attack mean AUROC plus 0.2 clean AUROC. The clean term is a soft
  weight, not a clean-performance guarantee.
- Secondary rate bounds use \((0,0.05]\), \((0.05,0.10]\), \((0.10,0.20]\),
  and \((0.20,0.50]\). Their objective is mean represented-attack AUROC gain
  subject to at most 0.01 clean tuning AUROC loss. Sparse bins fall back to the
  full-universal specification.
- Each fixed configuration has its own per-domain human threshold. Rates need
  a clean/attacked pair, so rate routing is an oracle diagnostic, not a
  deployment-ready detector or guaranteed performance ceiling.
- Corrected RAID Binoculars clips performer NLL through a float64 clipping
  delta anchored to the stored official numerator. The denominator stays
  unchanged. The gap's exponential rule is not substituted for this ratio.
- Intervals keep directions/caps/thresholds fixed and resample test sources
  within domain. A direct full-versus-eligible interval requires a jointly
  paired contrast unless one method is identically raw.

See the scientific design for formulas, denominator masks, alignment limitations,
objective weighting, bootstrap interpretation, and the upstream comparison.

## Archived work and preservation

`archive/raid_pilots/compare_tuning_methods.py`,
`compare_trimmed_mean.py`, and `compare_binoculars_components.py` are historical
development analyses, not final reporting pipelines. Their 500-source pilot was
excluded before final splitting. The archive is local-only/Git-ignored and is
not required by active final code/tests.

Keep source packs/manifests, frozen choices, and exact revision results.
Reorganization changes code paths/fingerprints, not scientific results. Never
overwrite a historical marker to imply current-code execution. Slurm completion
alone is insufficient: check the appropriate manifest, validation, counts,
completion marker, and artifact hashes.
