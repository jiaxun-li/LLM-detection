# Running the sharded RAID study on NCSA Delta

The authoritative Delta launcher is `RAID/submit_raid.sh`. It creates one
dependency graph:

1. CPU-only preparation and checksum-keyed index caching;
2. four independent one-GPU Falcon shards;
3. four independent two-GPU Binoculars shards;
4. CPU-only merge, evaluation, plotting, and validation.

All 13 variants of a source stay in one shard. Final evaluation starts only
after both score arrays pass and the merge verifies exact stable-key coverage.

## 1. Checkout, environment, and storage

```bash
cd ~/LLM-detection
git branch --show-current
git pull --ff-only
git status --short --branch

readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER

source /projects/bhuc/$USER/venvs/delta-smoke/bin/activate
python -m pip install 'rapidfuzz>=3.0'
accounts
```

The default launcher accounts are `bhuc-delta-cpu` for CPU jobs and
`bhuc-delta-gpu` for GPU jobs. If `accounts` reports different project names,
export `CPU_ACCOUNT` and `GPU_ACCOUNT` before submission.

`runs` and `results` must resolve below `/work/hdd`. The labeled RAID CSV must
also remain below `/work/hdd`:

```bash
export RAID_DATA_PATH=/work/hdd/bhuc/$USER/raid/train.csv
test -s "$RAID_DATA_PATH"
ls -lh "$RAID_DATA_PATH"
sha256sum "$RAID_DATA_PATH"
```

## 2. Mandatory four-shard smoke gate

A canceled preparation index may be adopted for a bounded smoke only when the
index sits beside its original manifest and has an empty WAL. The runner checks
the original manifest's input SHA-256 and validates the SQLite schema and
relationship indexes. Omit `REUSE_INDEX_PATH` when no such index exists; the
CPU job will build the persistent checksum cache instead.

If a previous bounded run already completed preparation, prefer adopting its
prepared artifacts instead of reading the CSV or SQLite index again. Adoption
requires the same dataset provenance, source limit, split/selection protocol,
and shard count. It validates every prepared key and shard assignment, copies
the immutable preparation under the new run ID, and records its source run and
Git commit in `prepare.complete.json`.

```bash
cd ~/LLM-detection
mkdir -p logs

export NUM_SHARDS=4
export LIMIT_SOURCES=64
export BOOTSTRAP_REPETITIONS=100
export DEBUG_ONLY=1
export INDEX_CACHE_DIR=/work/hdd/bhuc/$USER/raid/index-cache
export RAID_RUN_ID="raid-sharded-smoke-$(date -u +%Y%m%dT%H%M%SZ)"

# Optional bounded-smoke salvage only:
# export REUSE_INDEX_PATH="$(readlink -f runs/raid/<old-run-id>/raid_index.sqlite3)"
# Or adopt a completed bounded preparation:
# export ADOPT_PREPARED_RUN_DIR="$(readlink -f runs/raid/<prepared-run-id>)"

bash RAID/submit_raid.sh
```

The launcher prints and saves all identifiers in:

```text
/work/hdd/bhuc/$USER/raid/last_workflow.env
```

After reconnecting:

```bash
cd ~/LLM-detection
source /work/hdd/bhuc/$USER/raid/last_workflow.env

squeue -j "$PREP_JOB_ID,$FALCON_JOB_ID,$BINOCULARS_JOB_ID,$FINALIZE_JOB_ID"
sacct -j "$PREP_JOB_ID,$FALCON_JOB_ID,$BINOCULARS_JOB_ID,$FINALIZE_JOB_ID" \
  --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
```

After the finalize job completes, independently rerun the bounded validation:

```bash
python RAID/validate_raid.py \
  --run-id "$RAID_RUN_ID" \
  --limit-sources 64 \
  --bootstrap-repetitions 100
```

Relevant logs are:

```bash
tail -n 100 "logs/raid-prepare-$PREP_JOB_ID.err"
tail -n 100 "logs/raid-finalize-$FINALIZE_JOB_ID.err"
ls logs/raid-falcon-${FALCON_JOB_ID}_*.err
ls logs/raid-binoculars-${BINOCULARS_JOB_ID}_*.err
```

The dependency graph reports success only when
`results/raid/$RAID_RUN_ID/validation_report.json` contains
`validation_status: pass`. A smoke has 64
sources, 832 prepared rows, four complete shards per scorer, fourteen detector
universal specifications (seven full and seven eligible) plus 28 fixed-bin
rate-oracle specifications, only
5% FPR, and 100 bootstrap repetitions. It remains
`debug_only` and is not a scientific result.

## 3. Preserve artifacts and freeze the development exclusion

Do not delete the RAID CSV, checksum-keyed index cache, 64-source smoke,
500-source pilot, score packs, or pilot comparison results. A new run ID keeps
the full-data artifacts separate. The only scientific cleaning step is to
exclude the 500 pilot development sources before the final split.

Locate and validate the canonical exclusion file:

```bash
cd ~/LLM-detection

export PILOT_RESULTS_ROOT="$(readlink -f ~/LLM-detection-fast-smoke/results)"
export EXCLUDE_SOURCE_IDS_PATH="$PILOT_RESULTS_ROOT/raid/raid-pilot-500-20260823T165957Z/tuning_comparison_v1/development_source_ids.json"

python - "$EXCLUDE_SOURCE_IDS_PATH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
source_ids = [str(value) for value in payload["source_ids"]]

assert payload["must_be_excluded_from_future_full_benchmark"] is True
assert payload["source_count"] == 500
assert len(source_ids) == 500
assert len(set(source_ids)) == 500

print("exclusion_path:", path)
print("source_count:", len(source_ids))
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
PY
```

Also verify storage without deleting anything:

```bash
readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER
du -sh \
  /work/hdd/bhuc/$USER/raid/train.csv \
  /work/hdd/bhuc/$USER/raid/index-cache \
  "$(readlink -f ~/LLM-detection-fast-smoke/runs)/raid/raid-pilot-500-20260823T165957Z" \
  "$(readlink -f ~/LLM-detection-fast-smoke/results)/raid/raid-pilot-500-20260823T165957Z"
```

## 4. Provisional unbounded run with 500 bootstraps

Do not explicitly reuse an interrupted run index for a full result. The first
full preparation builds a checksum-keyed completed cache on the CPU partition;
subsequent runs with the identical CSV reuse it automatically.

This first unbounded pass uses every non-development source and all GPU scoring,
but only 500 bootstrap repetitions. It is deliberately labeled `debug_only` and
is provisional. If the full-data diagnostics are satisfactory, retain its
scores. Before the frozen 2,000-repetition confirmatory evaluation, add and
audit an evaluation-only promotion path under a new result ID so model
inference is not repeated.

```bash
cd ~/LLM-detection
unset LIMIT_SOURCES REUSE_INDEX_PATH ADOPT_PREPARED_RUN_DIR

export GPU_ACCOUNT=bhuc-delta-gpu
export GPU_PARTITION=gpuA100x4
# This user currently has only a GPU-type allocation, so CPU phases use the
# same account and partition. Delta therefore requires one reserved GPU for
# prepare and finalize even though those phases perform CPU-only work.
export CPU_ACCOUNT=bhuc-delta-gpu
export CPU_PARTITION=gpuA100x4
export CPU_GPUS_PER_NODE=1

export NUM_SHARDS=4
export RAID_DATA_PATH=/work/hdd/bhuc/$USER/raid/train.csv
export INDEX_CACHE_DIR=/work/hdd/bhuc/$USER/raid/index-cache
export BOOTSTRAP_REPETITIONS=500
export DEBUG_ONLY=1
export RAID_RUN_ID="raid-full-excluded500-bootstrap500-provisional-$(date -u +%Y%m%dT%H%M%SZ)"

test -s "$RAID_DATA_PATH"
test -s "$EXCLUDE_SOURCE_IDS_PATH"
case "$(readlink -f runs)" in /work/hdd/*) ;; *) echo "runs is not on /work/hdd"; exit 2 ;; esac
case "$(readlink -f results)" in /work/hdd/*) ;; *) echo "results is not on /work/hdd"; exit 2 ;; esac

bash RAID/submit_raid.sh
```

Keep the same `RAID_RUN_ID`, shard count, input, and scientific configuration
when resubmitting after a timeout. Individual shard score packs are append-safe.
Never run multiple unsharded `run_raid.py --stage score` processes against one
run directory. Do not reuse this provisional run ID with a different bootstrap
count.

## 5. Completion contract

Scientific completion requires all of the following:

- preparation, all Falcon shards, and all Binoculars shards passed;
- merged Falcon and Binoculars keys exactly equal prepared keys;
- model provenance is identical across each scorer's shards;
- manifest reports all five logical stages complete;
- all seven raw, full-universal-clipped, and eligible-universal-clipped
  detectors are present, together with the four fixed-bin rate-oracle
  configurations per detector;
- every rate-specific bound has paired `none`, attacked-bin, and represented-
  attack trade-off rows;
- only the 5% FPR target is present;
- all 500 pilot-development sources are absent from preparation and every split;
- 2,000 bootstrap repetitions use paired source-cluster resampling in the
  confirmatory result; the initial 500-repetition run remains provisional;
- contamination and truncation audits are complete;
- `validation_report.json` reports `validation_status: pass`;
- stderr contains no traceback, CUDA failure, or offload surprise.

Retrieve the small `results/raid/<run-id>/` directory. Leave multi-gigabyte
prepared data, caches, and score packs under `/work/hdd`.

## 6. Evaluation-only tuning comparison on the bounded 500-source pilot

Do not resubmit preparation or either GPU scorer. After pulling the comparison
code into the same checkout that contains the completed pilot score packs, run:

```bash
cd ~/LLM-detection-fast-smoke
export RAID_RUN_ID=raid-pilot-500-20260823T165957Z

TUNING_JOB_ID="$(sbatch --parsable \
  --account=bhuc-delta-gpu \
  RAID/delta_raid_tuning_comparison.sbatch)"
echo "TUNING_JOB_ID=$TUNING_JOB_ID"
```

The analysis itself is CPU-only and performs no inference. The wrapper reserves
one GPU because the available `bhuc-delta-gpu` project account accepts GPU-type
jobs; no additional model is loaded or downloaded. It requests 120 GB host RAM
because JSON token-feature packs expand substantially when loaded.

The command refuses an unbounded run unless `--allow-full-run` is explicitly
provided. It reads existing score packs and writes point estimates only under:

```text
results/raid/<run-id>/tuning_comparison_v1/
```

Completion requires `comparison.complete.json`, `method_count: 11`, and the
following tables: selected specifications, candidate diagnostics, universal
attack results, contamination-rate results, rate-oracle attack-by-rate results,
counterfactual clean costs, and the compact leaderboard. It also writes
`development_source_ids.json`; those 500 sources must be excluded before the
eventual full benchmark is split. This exploratory pass uses no bootstrap and
does not modify the original pilot's manifest, validated metrics, plots, or
frozen specifications.

Check completion with:

```bash
sacct -j "$TUNING_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
tail -n 100 "logs/raid-tuning-$TUNING_JOB_ID.err"
cat "results/raid/$RAID_RUN_ID/tuning_comparison_v1/comparison.complete.json"
```

## 7. Evaluation-only trimmed-mean comparison

This reuses the completed 500-source Falcon and Binoculars score packs. It does
not repeat preparation, model loading, or inference:

```bash
cd ~/LLM-detection-fast-smoke
export RAID_RUN_ID=raid-pilot-500-20260823T165957Z
export TRIM_OUTPUT_NAME=trimmed_mean_comparison_v1

TRIM_JOB_ID="$(sbatch --parsable \
  --account=bhuc-delta-gpu \
  RAID/delta_raid_trimmed_mean.sbatch)"
echo "TRIM_JOB_ID=$TRIM_JOB_ID"
```

The wrapper reserves one A100 only because the available project account is a
GPU-type account. The analysis itself is CPU-only. It writes point estimates
under `results/raid/<run-id>/trimmed_mean_comparison_v1/` and refuses an
unbounded run by default.

Check it with:

```bash
sacct -j "$TRIM_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
tail -n 100 "logs/raid-trim-$TRIM_JOB_ID.err"
cat "results/raid/$RAID_RUN_ID/$TRIM_OUTPUT_NAME/comparison.complete.json"
column -s, -t < "results/raid/$RAID_RUN_ID/$TRIM_OUTPUT_NAME/summary.csv" | less -S
```

## 8. Evaluation-only Binoculars component-clipping comparison

Reuse the completed 500-source Falcon and Binoculars score packs:

```bash
cd ~/LLM-detection-fast-smoke
export RAID_RUN_ID=raid-pilot-500-20260823T165957Z
export BINO_COMPONENT_OUTPUT_NAME=binoculars_component_comparison_v1

BINO_COMPONENT_JOB_ID="$(sbatch --parsable \
  --account=bhuc-delta-gpu \
  RAID/delta_raid_binoculars_components.sbatch)"
echo "BINO_COMPONENT_JOB_ID=$BINO_COMPONENT_JOB_ID"
```

The job performs no inference and downloads no models. One A100 is reserved
only because the available project account is GPU-type. Results are written
under `results/raid/<run-id>/binoculars_component_comparison_v1/`.

Check completion with:

```bash
sacct -j "$BINO_COMPONENT_JOB_ID" \
  --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
tail -n 100 "logs/raid-bino-cmp-$BINO_COMPONENT_JOB_ID.err"
cat "results/raid/$RAID_RUN_ID/$BINO_COMPONENT_OUTPUT_NAME/comparison.complete.json"
column -s, -t < \
  "results/raid/$RAID_RUN_ID/$BINO_COMPONENT_OUTPUT_NAME/summary.csv" | less -S
```
