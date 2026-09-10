# Robust document detection under contamination

This repository contains the completed **nine-cell controlled contamination
study plus RAID**, comparing raw aggregation with token-level clipping.
The final reported detectors are log likelihood, rank, log rank, LRR, entropy,
entropy gap, and Binoculars-origin. Primary Binoculars remains prompt-conditioned;
RAID uses a separate official output-only scoring protocol.

The controlled constructions splice human-source material into machine
continuations. They are contamination experiments, not certified grammatical
human editing. RAID provides a separate heterogeneous attack benchmark.

## Read these before writing the report

1. [Final results guide](docs/FINAL_RESULTS_GUIDE.md): accepted bundle, exact
   counts, source lineage, table use, and baseline comparison.
2. [Primary scientific workflow](docs/primary/SCIENTIFIC_WORKFLOW.md) and
   [RAID scientific design](docs/raid/SCIENTIFIC_DESIGN.md): actual populations,
   construction, fitting, calibration, estimands and uncertainty.
3. [Detector and clipping definitions](docs/methods/clipping_method.md):
   formulas, directions, candidate grids and precise tuning objectives.
4. [Detector amendment ledger](docs/DETECTOR_REVISION.md): gap versus ratio,
   numerical anchoring, LRR correction and accepted versions.
5. [Documentation audit](docs/DOCUMENTATION_AUDIT.md): checks performed,
   corrected discrepancies and remaining evidence limitations.

The [documentation index](docs/README.md), [codebase guide](docs/CODEBASE_GUIDE.md),
[configuration guide](docs/CONFIGURATION.md), and
[repository map](docs/REPOSITORY_MAP.md) explain the implementation and operations.

## What is completed versus planned?

The nine primary cells cross XSum, SQuAD and WritingPrompts with Granite 3.3 8B
Base, Mistral Small 24B Base and Qwen 2.5 32B. Their source runs contain
3,000 source groups and 78,000 prepared/scored records per cell; final publication
tables have 392 metric records per cell and 3,528 combined.

The final RAID revision retains 12,871 source groups after excluding the
500-source development pilot, with 167,323 prepared/scorer rows and 2,000
bootstrap replicates. It reports full-universal and eligible-universal clipping,
with rate-specific oracle bounds as secondary diagnostics.

The broader 21-cell model matrix still listed in configs/paper.json is a
historical plan, not a claim of 21 completed cells. Beemo, splice diagnostics,
and selector/trim pilots are archived development studies, not additional final
results. See the [archive catalog](docs/ARCHIVE_CATALOG.md).

## Where things live

| Directory / entry point | Function |
|---|---|
| experiment_core/ | Shared preparation, token features, detector aggregation and evaluation |
| RAID/ | Benchmark-specific data selection, origin scoring, fitting and validation |
| tools/ | Delta launch/setup, validation, reanalysis and final export |
| configs/ | Tracked baseline and smoke settings |
| docs/ | Experiment methods, evidence and operations |
| tests/ | Active regression tests, independent of ignored archive |
| run_experiment.py | Base primary pipeline; retains legacy detector behavior |
| plot_tpr_contamination.py | Primary figure rendering |
| downloads/current/ | Local validated final publication bundle |
| downloads/archive/ and archive/ | Local-only historical downloads and programs |
| paper/ | User-owned writing, outside experiment maintenance |

## Reproducing the accepted analysis

A base configuration is not the complete amended method. The historical key
binoculars denotes an exponential-gap score, not the original ratio. Final
corrected analysis uses:

- tools/reanalysis/reevaluate_detector_revision.py for primary saved-feature
  evaluation, retaining prompt context.
- RAID/revise_detectors.py for the separate origin gate/scoring/evaluation path.
- tools/exports/export_final_publication_bundle.py for validated filtering and
  figure export without refitting.

Eight primary accepted revisions are from September 4; Qwen32–SQuAD uses the
later LRR-v4.2 correction. RAID remains the accepted v4.1 run, whose LRR choices
passed the subsequent positive-cap audit. Current v4.2 code does not retroactively
change the versions recorded in completed artifacts.

Use original source/revision manifests and corresponding code/model revisions,
not current JSON defaults alone. Saved per-cell settings differ from defaults.
Large original JSONL features remain on Delta and are deliberately absent from
the publication download. Do not discard them after exporting plots.

## Local checks and Delta

No-download regression checks:

~~~bash
python -m unittest discover -s tests -v
~~~

Torch-dependent tests may skip in a lightweight local environment. A local unit
test does not replace the live upstream-component gate. The user-reported
post-reorganization Delta repository smoke passed all 105 active tests without
skips; it was an engineering check, not regeneration of scientific results.

See [Delta operations](docs/delta/DELTA.md) and
[RAID operations](docs/raid/DELTA_GUIDE.md). Stable branch is main. The friendly
interactive environment name is llm-detection; the physical installed venv
remains delta-smoke to preserve working paths. Before large jobs verify actual
runs/results links resolve under /work/hdd and check allocation/queue state.
The user performs every Git commit, push and pull.

A new configuration or methodological amendment needs a new run/revision ID.
An exact resume must preserve the original identity. Do not rerun preparation
or model inference merely to change documentation, inspect results or redraw
figures.
