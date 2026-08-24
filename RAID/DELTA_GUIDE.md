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
sources, 832 prepared rows, four complete shards per scorer, seven detector
universal specifications plus 28 fixed-bin rate-oracle specifications, only
5% FPR, and 100 bootstrap repetitions. It remains
`debug_only` and is not a scientific result.

## 3. Full run

Do not explicitly reuse an interrupted run index for a full result. The first
full preparation builds a checksum-keyed completed cache on the CPU partition;
subsequent runs with the identical CSV reuse it automatically.

```bash
cd ~/LLM-detection
unset LIMIT_SOURCES BOOTSTRAP_REPETITIONS DEBUG_ONLY REUSE_INDEX_PATH
export NUM_SHARDS=4
export RAID_DATA_PATH=/work/hdd/bhuc/$USER/raid/train.csv
export INDEX_CACHE_DIR=/work/hdd/bhuc/$USER/raid/index-cache
export RAID_RUN_ID="raid-constrained-rate-oracle-full-$(date -u +%Y%m%dT%H%M%SZ)"

bash RAID/submit_raid.sh
```

Keep the same `RAID_RUN_ID`, shard count, input, and scientific configuration
when resubmitting after a timeout. Individual shard score packs are append-safe.
Never run multiple unsharded `run_raid.py --stage score` processes against one
run directory.

## 4. Completion contract

Scientific completion requires all of the following:

- preparation, all Falcon shards, and all Binoculars shards passed;
- merged Falcon and Binoculars keys exactly equal prepared keys;
- model provenance is identical across each scorer's shards;
- manifest reports all five logical stages complete;
- all seven raw and universal-clipped detectors are present, together with the
  four fixed-bin rate-oracle configurations per detector;
- every rate-specific bound has paired `none`, attacked-bin, and represented-
  attack trade-off rows;
- only the 5% FPR target is present;
- 2,000 bootstrap repetitions use paired source-cluster resampling;
- contamination and truncation audits are complete;
- `validation_report.json` reports `validation_status: pass`;
- stderr contains no traceback, CUDA failure, or offload surprise.

Retrieve the small `results/raid/<run-id>/` directory. Leave multi-gigabyte
prepared data, caches, and score packs under `/work/hdd`.

## 5. Evaluation-only tuning comparison on the bounded 500-source pilot

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
