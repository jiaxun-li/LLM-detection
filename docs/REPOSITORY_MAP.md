# Repository map

The active released experiments are the corrected nine primary cells and RAID.
This map classifies code by purpose; it does not change the frozen experiment
settings or imply that the larger historical model matrix was completed.

## Where to work

| Location | Purpose |
|---|---|
| `run_experiment.py` | Primary experiment stage entry point: source selection, preparation, scoring, evaluation |
| `plot_tpr_contamination.py` | Primary raw/clipped TPR curves |
| `experiment_core/` | Shared scientific implementation for primary experiments and detector revisions |
| `RAID/` | Active RAID benchmark, sharding, validation, and corrected Binoculars-origin analysis |
| `tools/` | Active Delta setup/launch, smoke tests, detector reanalysis, and final export |
| `configs/` | Primary and RAID settings plus two bounded smoke configurations |
| `docs/` | Scientific methods, operations, configuration guide and this map |
| `tests/` | Regression tests for active code; works without the local archive |
| `archive/` | Local-only secondary studies, exploratory tools, and historical tests; ignored by Git |
| `downloads/current/` | Validated final results, figures, archive, and checksum; machine-local and ignored by Git |
| `downloads/archive/` | Earlier downloaded results, retained for provenance; machine-local and ignored by Git |
| `paper/` | User-owned writing material; outside experiment cleanup |

`experiment_core/` is a Python library. `tools/` contains operational commands.
RAID has its own dataset protocol and therefore its own package and launchers.
These are distinct responsibilities, not duplicate implementations.

## Primary library

| Module in `experiment_core/` | Use |
|---|---|
| `infrastructure/config.py` | Configuration loading, overrides, validation |
| `preparation/data.py` | Source splits, token budgets, contamination construction |
| `preparation/generation.py` | Transformers and optional vLLM generation |
| `preparation/pipeline.py` | Prepare/score/evaluate orchestration |
| `detectors/scoring.py` | Exact token features and resumable score packs |
| `analysis/evaluation.py` | Clipping selection, calibration, metrics, clustered bootstrap |
| `detectors/detector_revision.py` | Corrected detector definitions, orientation, and clipping safeguards |
| `infrastructure/revision_artifacts.py` | Validated revision completion and artifact identity |
| `infrastructure/io.py` | Durable JSONL writing, restart handling, completion markers |
| `infrastructure/runtime.py` | Run manifests, environment provenance, throughput |
| `validation/smoke_validation.py` | Bounded Delta smoke validation |

For the corrected final detector analysis, use
`tools/reanalysis/reevaluate_detector_revision.py` with the protocol in
[docs/DETECTOR_REVISION.md](DETECTOR_REVISION.md). The general primary pipeline retains
historical configurations for reproducibility.

## Active operational scripts

| File in `tools/` | Use |
|---|---|
| `delta/setup_delta_env.sh` | Delta Python/CUDA environment setup |
| `delta/activate_environment.sh` | Activate the existing Delta environment with the friendly `llm-detection` prompt |
| `delta/delta_repository_smoke.sbatch` | No-download layout and environment regression check on Delta |
| `delta/delta_experiment.sbatch` | Launch a primary model/dataset experiment |
| `delta/submit_delta_matrix.sh` | Submit a reviewed multi-cell matrix |
| `delta/delta_smoke_qwen_0_5b.sbatch`, `delta/submit_delta_smoke.sh` | Bounded one-GPU smoke launch |
| `validation/synthetic_delta_smoke.py`, `validation/validate_delta_smoke.py` | Offline orchestration check and completed smoke validation |
| `validation/check_detector_revision_gate.py` | CPU/CUDA numerical revision regressions |
| `delta/delta_detector_revision.sbatch` | Detector revision gate, score, or evaluate job |
| `reanalysis/reevaluate_detector_revision.py` | Corrected primary reanalysis using saved score components |
| `exports/export_final_publication_bundle.py` | Validated final nine-cell + RAID export, including figures |
| `delta/delta_export_final_publication.sbatch` | Delta wrapper for the final exporter |
| `exports/export_helpers.py` | Shared table/CSV utilities; imported by the active final exporter |

## Active RAID package

| File in `RAID/` | Use |
|---|---|
| `run_raid.py`, `raid_pipeline.py` | Dataset preparation through validation |
| `raid_data.py` | Streaming/indexed data selection, source groups, token edit rates |
| `raid_scoring.py` | Falcon features and historical score-pack handling |
| `raid_evaluation.py` | Full/eligible universal and rate-specific clipping, calibration, bootstrap |
| `binocular_origin.py` | Corrected output-only Binoculars ratio components |
| `revise_detectors.py`, `revision_gate.py` | Corrected origin gate, shard adoption/scoring, reanalysis |
| `run_raid_shard.py` | Score one deterministic shard |
| `raid_plot.py`, `plot_raid.py` | Figure implementation and plot-only entry point |
| `validate_raid.py` | Validate a completed benchmark |
| `submit_raid.sh` | Prepare → scoring arrays → finalize submission |
| `delta_raid_prepare.sbatch` | Data preparation and reusable index |
| `delta_raid_falcon_shard.sbatch`, `delta_raid_binoculars_shard.sbatch` | Independent GPU score shards |
| `delta_raid_finalize.sbatch` | Merge, evaluate, plot, validate |
| `promote_bootstrap.py`, `delta_raid_bootstrap_promotion.sbatch` | Reevaluate existing packs with the final bootstrap count |
| `../configs/raid.json`, `../docs/raid/` | Centralized benchmark settings and operations guides |

Legacy Binocular-gap compatibility remains in the shared implementation because
historical packs and provenance require it. Paper-facing final exports report
the seven corrected detectors, including Binocular-origin.

## Archives

See [ARCHIVE_CATALOG.md](ARCHIVE_CATALOG.md) for retained programs and commands.
Archived code is runnable from the repository root with its new path. Its tests
live in `archive/tests/`. Both are ignored by Git, so they remain available locally
but are not included in a fresh clone. Old committed versions remain recoverable
from Git history. Imports referencing old paths must use the new package names.

## Downloads and writing

The canonical local bundle is
`downloads/current/primary-nine-plus-raid-final-20260908T150247Z/`.
It has nine primary cells, 3,528 primary metric rows, 99 primary PNGs, and three
paper-facing RAID PNGs in a nested plot directory. The accompanying tar archive
and SHA-256 file are retained. `bundle.complete.json` inventories 182 other files;
the completion marker itself makes 183 files in the extracted directory.

Older downloads have moved intact to:

- `downloads/archive/beemo/`: old primary/Beemo directory and ZIP. Beemo results
  are unique historical results, not superseded by the current RAID bundle.
- `downloads/archive/legacy-primary/primary-final/`: earlier primary metrics-only export.
- `downloads/archive/legacy-raid/RAID/`: earlier provisional and pre-origin RAID exports.

No downloaded result was deleted. These local moves do not move Delta score
packs or server results, and Git does not synchronize ignored downloads.

`paper/` and `ACCESS_SUBMISSION_CHECKLIST.md` remain user-owned writing/planning
material. `docs/methods/clipping_method.md`, the scientific guides, detector amendment, and
`configs/paper.json` document experimental methodology and stay with the code.
All active experiment guides are under `docs/`. Root `README.md` and `AGENTS.md`
remain discoverable entry documents; user-owned writing material and archived
study notes retain their existing locations.

## Maintenance

Run `python -m unittest discover -s tests -v` from the repository root for active
tests. Optional historical coverage uses
`python -m unittest discover -s archive/tests -v` when the local archive exists.
Some Torch/CUDA checks require Delta and skip on the lightweight Windows runtime.

Generated `__pycache__/`, `.pyc`, and `.DS_Store` are disposable and ignored.
Source and downloaded result archives are both ignored by Git.
This reorganization requires no regeneration, model scoring, or scientific
evaluation on Delta. Relocated source files change code-fingerprint identities;
do not resume an old gated revision as if its gate verified the new source tree.
Existing result bundles and their provenance remain unchanged. The user controls
Git commits, pushes, and pulls.
