# Running the RAID study on NCSA Delta

These commands are for `jli101` in `~/LLM-detection`. The user performs every
Git push and pull. Full data and outputs must remain under `/work/hdd`.

## 1. Update and check the checkout

```bash
cd ~/LLM-detection
git branch --show-current
git pull --ff-only

readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER
```

Both artifact paths must resolve below `/work/hdd`.

Install the one new CPU dependency in the existing CUDA 12.8 environment:

```bash
source /projects/bhuc/$USER/venvs/delta-smoke/bin/activate
python -m pip install 'rapidfuzz>=3.0'
```

Do not install another copy of Falcon if it is already present in the shared
Hugging Face cache.

## 2. Place the labeled RAID release on work storage

The study needs the labeled English `train.csv`, including adversarial rows.
Download it once, outside Slurm jobs:

```bash
mkdir -p /work/hdd/bhuc/$USER/raid
wget -c \
  -O /work/hdd/bhuc/$USER/raid/train.csv \
  https://dataset.raid-bench.xyz/train.csv

ls -lh /work/hdd/bhuc/$USER/raid/train.csv
sha256sum /work/hdd/bhuc/$USER/raid/train.csv
```

The paper describes 14,971 sources in the full core benchmark. The current
public labeled training release is a subset; the manifest records the actual
available and retained source counts. Never point a full run at the unlabeled
public test file.
The runner also records and checks the file's SHA-256 digest when a run is
created or resumed.

## 3. Mandatory two-GPU smoke gate

The gate keeps 64 complete source families, scores all seven detectors, and
uses 100 bootstrap repetitions. Preparation still scans the release to find
complete families, so the first preparation stage is not instantaneous.

```bash
cd ~/LLM-detection
mkdir -p logs

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="raid-universal-clipping-smoke-$STAMP"
DATA_PATH="/work/hdd/bhuc/$USER/raid/train.csv"

JOB_ID=$(sbatch --parsable \
  --account=bhuc-delta-gpu \
  --partition=gpuA100x4 \
  --gpus-per-node=2 \
  --cpus-per-task=24 \
  --mem=200G \
  --time=08:00:00 \
  --job-name=raid-smoke \
  --export=ALL,RAID_RUN_ID="$RUN_ID",RAID_DATA_PATH="$DATA_PATH",LIMIT_SOURCES=64,BOOTSTRAP_REPETITIONS=100,DEBUG_ONLY=1 \
  RAID/delta_raid.sbatch)

echo "submitted_job_id=$JOB_ID"
echo "raid_run_id=$RUN_ID"
```

Monitor and validate:

```bash
squeue -j "$JOB_ID"
tail -F "logs/raid-$JOB_ID.out"

sacct -j "$JOB_ID" \
  --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
tail -n 100 "logs/raid-$JOB_ID.err"

python RAID/validate_raid.py \
  --run-id "$RUN_ID" \
  --data-path "$DATA_PATH" \
  --limit-sources 64 \
  --bootstrap-repetitions 100
```

A passing gate has 64 sources, 832 prepared rows, 832 Falcon rows, 832
Binoculars rows, seven detector specifications, only the 5% FPR target, all
twelve machine conditions, and `validation_status: pass`. It is `debug_only`
and is not a scientific result.

## 4. Full run

After the smoke gate passes:

```bash
cd ~/LLM-detection
export RAID_DATA_PATH=/work/hdd/bhuc/$USER/raid/train.csv
bash RAID/submit_raid.sh
```

Record both printed identifiers. The 24-hour request is a ceiling, not a
runtime promise. If the job times out during scoring, submit the same `RUN_ID`,
data path, and configuration again. Score packs resume by validated row key:

```bash
export RAID_RUN_ID='<the same run id>'
export RAID_DATA_PATH=/work/hdd/bhuc/$USER/raid/train.csv
bash RAID/submit_raid.sh
```

Do not use a new run ID merely because a score-stage job reached its wall-time.
Do use a new run ID after any scientific configuration change.

## 5. Completion checks

```bash
sacct -j "$JOB_ID" \
  --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
tail -n 100 "logs/raid-$JOB_ID.err"

python RAID/validate_raid.py \
  --run-id "$RAID_RUN_ID" \
  --data-path "$RAID_DATA_PATH"
```

Then inspect the manifest and validation report:

```bash
python - "$RAID_RUN_ID" <<'PY'
import json, sys
from pathlib import Path

run_id = sys.argv[1]
manifest = json.loads((Path("runs/raid") / run_id / "manifest.json").read_text())
report = json.loads((Path("results/raid") / run_id / "validation_report.json").read_text())
print("status:", manifest["completion_status"])
print("stages:", manifest["completed_stages"])
print("validation:", report)
PY
```

Scientific completion requires `complete`, all five stages, all seven
detectors, raw and universal-clipped results, only 5% FPR, 2,000 bootstrap
repetitions, complete contamination records, the seven-row published
Binoculars comparator, and no traceback or CUDA/offload surprise in stderr.

## 6. Small artifacts to retrieve

Copy `results/raid/<run-id>/`, not the multi-gigabyte score packs in
`runs/raid/<run-id>/`. The result directory contains the metrics, frozen
specifications, contamination and attack summaries, per-row edit audit,
validation report, and two plots.
