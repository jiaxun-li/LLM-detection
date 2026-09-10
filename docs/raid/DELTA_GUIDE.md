# RAID operations: preserved base runs and corrected final results

The accepted paper-facing RAID result is the corrected **Binocular-origin**
revision, not the original `binoculars` mean-gap base result. Existing data and
Falcon packs are reused. Do not submit preparation or full inference to inspect
the completed result. For formulas and scientific selection see
[SCIENTIFIC_DESIGN.md](SCIENTIFIC_DESIGN.md) and the
[detector amendment](../DETECTOR_REVISION.md).

## 1. Canonical source and output identities

| Layer | Recorded identifier/path | Meaning |
|---|---|---|
| Full base source | `raid-full-excluded500-bootstrap500-provisional-20260824T132810Z` | Preserved data, Falcon scores, legacy Binocular scores and contamination cache |
| Historical bootstrap promotion | `raid-full-excluded500-bootstrap2000-final-20260825T065701Z` | Evaluation-only 2,000-bootstrap legacy result, before origin correction |
| Corrected RAID revision | `raid-origin-anchored-v41-20260907T044156Z` | Accepted eight-detector origin/gap source result |
| Final local bundle | `downloads/current/primary-nine-plus-raid-final-20260908T150247Z/` | Validated report/download artifact |

The accepted base has 12,871 sources after excluding all 500 development
sources; tuning/calibration/test counts are 5,148/2,574/5,149. There are 167,323
prepared and scorer rows (13 per source). The accepted revision has 416 metric
rows, eight source detectors, 16 universal specs, 32 rate-oracle specs, and
2,000 bootstrap repetitions. The source eight-detector result retains
`binocular_gap` for provenance. Its paper-facing projection drops that method:
seven detectors, 364 metric rows, and three paper plots. Both the source and
paper subtrees contain three plots, so the full exported RAID subtree has six
PNGs, not three.

The accepted RAID version is v4.1. Current shared code is v4.2 because of the
later primary Qwen32–SQuAD LRR guard; the user audited the selected RAID LRR
specs as valid and did not rerun RAID. Do not relabel the preserved v4.1 output
as v4.2 or assume its gate matches a newly reorganized source tree.

## 2. Delta environment and storage

These are Bash commands on Delta, not PowerShell on Windows:

```bash
cd ~/LLM-detection
git branch --show-current
git status --short --branch
source tools/delta/activate_environment.sh
readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER
accounts
squeue -u "$USER"
```

Stable branch is `main`. Activation displays `llm-detection` without moving the
installed `delta-smoke` environment. Do not reinstall rapidfuzz or other
packages to inspect an existing result. Both storage links and the RAID CSV
must resolve below `/work/hdd`. The shared CSV is normally
`/work/hdd/bhuc/jli101/raid/train.csv`; verify rather than redownload it.

`RAID/submit_raid.sh` defaults to a CPU-type account for preparation/finalize.
Jiaxun's recorded account is GPU-type only. Before any deliberately new base
submission set all five values, including the reserved GPU for CPU phases:

```bash
export GPU_ACCOUNT=bhuc-delta-gpu
export GPU_PARTITION=gpuA100x4
export CPU_ACCOUNT=bhuc-delta-gpu
export CPU_PARTITION=gpuA100x4
export CPU_GPUS_PER_NODE=1
```

CPU-only preparation and evaluation remain CPU algorithms; the GPU reservation
is an account requirement. Missing this override causes Slurm rejection.

## 3. Corrected origin gate, shards and evaluation

Use `tools/delta/delta_detector_revision.sbatch`, not the base workflow launcher.
It supports `REVISION_STAGE=gate`, `score`, `evaluate` (plus `primary` for the
separate primary study). Required variables are `SOURCE_RUN_ID` and a **new**
`REVISION_ID`; optional `NUM_SHARDS` defaults to four, and
`BOOTSTRAP_REPETITIONS` defaults to 2,000.

The gate also requires `REFERENCE_METRICS`: a reviewed local copy of upstream
Binoculars `metrics.py`. That Python module is executed, so do not point it at
unreviewed downloaded code. The wrapper first runs no-download regression
tests, then the real full-input tokenizer preflight and a bounded GPU pipeline
probe. The earlier synthetic test PASS is not the real gate PASS.

The current revision code reads:

```text
runs/raid/<source-run-id>/
  manifest.json
  data.jsonl
  falcon_scores.jsonl
  binoculars_scores.jsonl
  data_shards/part-00000-of-00004.jsonl ... part-00003-of-00004.jsonl
results/raid/<source-run-id>/contamination_records.csv
```

It creates:

```text
runs/raid_origin/<new-revision-id>/
  gate.complete.json
  part-00000-of-00004.jsonl ... part-00003-of-00004.jsonl
  part-00000-of-00004.complete.json ... part-00003-of-00004.complete.json
results/raid/<new-revision-id>/
  revision_manifest.json
  revision.complete.json
  validation_report.json
  frozen_specs.json
  metrics.csv
  attack_summary.csv
  contamination_summary.csv
  binoculars_sanity.csv
  calibration_summary.csv
  rate_bound_tradeoff_summary.csv
  contamination_records.csv
  evaluation_counts.json
  plots/raid_attack_tpr.png
  plots/raid_contamination_tpr.png
  plots/raid_rate_bound_tradeoff.png
```

Submit gate → scoring array → evaluation with `afterok` dependencies only after
reviewing inputs and available resources. The four shards partition one study;
they are not four independent RAID experiments. Every source's 13 rows stays
together. Array `--array=0-3%2`, for example, limits concurrency to two tasks.
Scoring uses two GPUs per task and batch/microbatch one. Evaluation is CPU-only
and can reserve one GPU for the account. Default wrapper RAM/time is 120G/4h;
override deliberately if needed, not from a guessed runtime.

The accepted shard counts were 41,015/42,861/41,899/41,548. Origin adoption
(`ADOPT_ORIGIN_RUN_ID`) is an explicit validated score-stage option: compatible
saved rows may be retained and missing rows scored. Do not copy files manually,
edit markers, or loosen numeric guards to force reuse. A failed shard does not
require erasing successful compatible shards. Its dependent evaluation cannot
complete until a repaired dependency graph points to successful scoring.

The code checks source hashes, gate identity, component provenance, exact row
coverage and output hashes. Current identities include source-code paths, so
the repository reorganization changed fingerprints. For a new computation,
use a new revision and new matching gate; for the already accepted result,
read its preserved artifacts rather than attempting a new-code no-op resume.

## 4. Inspect accepted completion without changing it

Use the final source result, not a filtered paper view, for its validation
contract. This read-only example contains errors inside a subshell:

```bash
(
    set -euo pipefail
    cd ~/LLM-detection
    source tools/delta/activate_environment.sh
    python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('results/raid/raid-origin-anchored-v41-20260907T044156Z')
read = lambda name: json.loads((root / name).read_text())
manifest = read('revision_manifest.json')
report = read('validation_report.json')
assert manifest['completion_status'] == 'complete'
assert report['validation_status'] == 'pass'
assert read('revision.complete.json') == report
assert report['bootstrap_repetitions'] == 2000
assert report['metrics_rows'] == 416
for name, expected in report['artifacts'].items():
    data = (root / name).read_bytes()
    assert len(data) == expected['size'], name
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
print('PRESERVED RAID ARTIFACT CHECK: PASS')
PY
)
```

For active jobs, inspect `sacct` plus the actual `#SBATCH` stdout/stderr paths.
Do not print PASS following a failed Python assertion. A missing log is not
evidence of no errors. Scheduler completion alone does not validate science.

## 5. Mandatory four-shard smoke gate for a new base workflow

This section preserves the engineering launch procedure for a deliberately
new base run; it is **not needed for the completed paper bundle**. The base
launcher creates prepare → Falcon and legacy Binocular arrays → finalize.
Final-origin revision still follows separately.

```bash
(
    set -euo pipefail
    cd ~/LLM-detection
    mkdir -p logs
    export GPU_ACCOUNT=bhuc-delta-gpu GPU_PARTITION=gpuA100x4
    export CPU_ACCOUNT=bhuc-delta-gpu CPU_PARTITION=gpuA100x4 CPU_GPUS_PER_NODE=1
    export RAID_DATA_PATH=/work/hdd/bhuc/$USER/raid/train.csv
    export INDEX_CACHE_DIR=/work/hdd/bhuc/$USER/raid/index-cache
    export NUM_SHARDS=4 LIMIT_SOURCES=64 BOOTSTRAP_REPETITIONS=100 DEBUG_ONLY=1
    unset REUSE_INDEX_PATH ADOPT_PREPARED_RUN_DIR EXCLUDE_SOURCE_IDS_PATH
    export RAID_RUN_ID="raid-sharded-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
    bash RAID/submit_raid.sh
)
```

The submitter writes `/work/hdd/bhuc/$USER/raid/last_workflow.env` after all
submissions succeed. Retain a run-specific copy if launching multiple workflows;
the `last_...` pointer can change. A 64-source base smoke has 832 prepared/scored
rows and is debug-only. Its 100 bootstraps are not report-ready evidence.

The base validator reconstructs launch identity, so pass the same settings and
data path, not just the run ID:

```bash
(
    set -euo pipefail
    cd ~/LLM-detection
    source tools/delta/activate_environment.sh
    source /work/hdd/bhuc/$USER/raid/last_workflow.env
    python RAID/validate_raid.py --run-id "$RAID_RUN_ID" \
        --data-path "$RAID_DATA_PATH" --limit-sources 64 \
        --num-shards 4 --bootstrap-repetitions 100 --debug-only
)
```

This is a base-workflow validator, not the final origin revision validator.
Do not point it at `results/raid/<origin-revision>`.

Preparation can reuse a completed checksum-keyed index or adopt a compatible
completed prepared run (`ADOPT_PREPARED_RUN_DIR`). Interrupted-index salvage
requires the original manifest, matching CSV hash, valid schema and empty WAL;
do not use that as an unreviewed full-run shortcut. Full CSV hashing/indexing
can dominate the runtime even for a small source limit. Never run concurrent
writers against one index or unsharded score pack.

## 6. Historical full run and bootstrap promotion

The 500-source exploratory pilot's entire source list was excluded before the
final split. Preserve its `development_source_ids.json`, not just its size.
An unbounded base launch requires `EXCLUDE_SOURCE_IDS_PATH`; source count,
uniqueness and hash must match the reviewed 500-source exclusion. The original
pilot lived in the separate `LLM-detection-fast-smoke` workspace.

The provisional full pass used `BOOTSTRAP_REPETITIONS=500` and `DEBUG_ONLY=1`.
`RAID/promote_bootstrap.py` and `RAID/delta_raid_bootstrap_promotion.sbatch` then
reevaluated those same legacy score packs under a new promotion ID with 2,000
bootstraps, checking unchanged point estimates. They perform no inference but
do **not** add official-origin components. Therefore promoting the old packs
is not a substitute for `RAID/revise_detectors.py`.

Historical exploratory selectors, trimming and component comparisons are now
under ignored `archive/raid_pilots/`; a fresh clone does not contain them. They
are not part of the final operational sequence. Do not rerun them or choose
different settings after inspecting final test results.

## 7. Report export and Windows download

The active exporter is `tools/exports/export_final_publication_bundle.py`,
wrapped by `tools/delta/delta_export_final_publication.sbatch`. Required wrapper
variables: `FINAL_EXPORT_ID`, `PRIMARY_BUNDLE_ID`, `RAID_REVISION_ID`, and
`FINAL_EXPORT_ROOT`. It copies the validated source result unchanged to
`raid/source_eight_detector_result/`, writes filtered CSVs and plots to
`raid/paper_seven_detector_result/`, and records the projection in
`raid/raid_export_summary.json`. No fitting or inference is performed.

For the accepted bundle, read the existing local `downloads/current/` copy;
do not redownload unless it is missing or fails its checksum/hash inventory.
For any transfer, run `scp` from **Windows PowerShell**, not the Delta terminal.
Use a full destination filename built by `Join-Path`, or change to the local
destination directory and copy to `.`. Avoid the old quoted destination ending
in a backslash, which previously failed on Windows OpenSSH. Check `$LASTEXITCODE`
after each transfer and `tar`, compare SHA-256 before extraction, and only
print PASS after all validations succeed. See the
[final-result guide](../FINAL_RESULTS_GUIDE.md) for the report-facing layout.

Do not delete the original CSV, prepared rows, scorer packs, exact revision
results, exclusion list or frozen specs merely because a compact export exists.
The export omits large JSONL and does not by itself support new detector fitting.
