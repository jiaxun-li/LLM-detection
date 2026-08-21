# Running Beemo on NCSA Delta

These commands are for `jli101` in `~/LLM-detection`. The user performs the Git
pull; agents do not pull or push.

## 1. Update and verify the workspace

After publishing the local changes, pull them on Delta and check the active
branch:

```bash
cd ~/LLM-detection
git branch --show-current
git pull --ff-only
```

Activate the known CUDA 12.8 environment and verify large-artifact placement:

```bash
source /projects/bhuc/$USER/venvs/delta-smoke/bin/activate

readlink -f runs
readlink -f results
df -h /work/hdd/bhuc/$USER
```

Both resolved paths must begin with `/work/hdd/`. Stop if either points into
`/u`, if the branch is wrong, or if storage is unexpectedly low.

## 2. Submit the mandatory 12-record gate

The gate runs all seven detectors, including Binoculars, but uses only 12
record groups and 100 bootstrap repetitions.

```bash
cd ~/LLM-detection
mkdir -p logs

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="beemo-output-only-4scorer-smoke-$STAMP"

JOB_ID=$(sbatch --parsable \
  --account=bhuc-delta-gpu \
  --partition=gpuA100x4 \
  --gpus-per-node=2 \
  --cpus-per-task=16 \
  --mem=160G \
  --time=06:00:00 \
  --job-name=beemo-smoke \
  --export=ALL,BEEMO_RUN_ID="$RUN_ID",LIMIT_RECORDS=12,BOOTSTRAP_REPETITIONS=100,DEBUG_ONLY=1 \
  Beemo/delta_beemo.sbatch)

echo "submitted_job_id=$JOB_ID"
echo "beemo_run_id=$RUN_ID"
```

Monitor it:

```bash
squeue -j "$JOB_ID"
tail -F "logs/beemo-$JOB_ID.out"
```

After Slurm stops:

```bash
sacct -j "$JOB_ID" \
  --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
tail -n 100 "logs/beemo-$JOB_ID.err"
```

The smoke gate should contain 12 records × 9 variants = 108 prepared rows, 108
rows in each of the four target-scorer files and the shared Binoculars file, and
1,000 metric rows across 25 scorer-detector configurations:

```bash
wc -l \
  "runs/beemo/$RUN_ID/data.jsonl" \
  "runs/beemo/$RUN_ID/target_scores/gpt2_xl.jsonl" \
  "runs/beemo/$RUN_ID/target_scores/opt_1_3b.jsonl" \
  "runs/beemo/$RUN_ID/target_scores/falcon_7b.jsonl" \
  "runs/beemo/$RUN_ID/target_scores/qwen2_7b.jsonl" \
  "runs/beemo/$RUN_ID/binoculars_scores.jsonl"

python Beemo/validate_beemo.py --run-id "$RUN_ID" \
  --limit-records 12 --bootstrap-repetitions 100
```

Do not launch the full run unless the job is `COMPLETED`, stderr has no
traceback, and validation reports `validation_status: pass`.

The validation report also records scorer-specific boundary-token and
right-truncation counts. In the full dataset, GPT2-XL is expected to add a
boundary for `144-human` and `1126-expert` and to right-truncate 11 rows at
1,024 tokens. Falcon, Qwen, and Binoculars add a boundary for the released
`1126-expert = "."` anomaly; OPT supplies its own native starting token.

## 3. Submit the full study

The helper creates a unique run ID and preserves the same two-GPU resource
request. The four single-model scorers run sequentially and Binoculars then
uses its two explicit model devices:

```bash
cd ~/LLM-detection
bash Beemo/submit_beemo.sh
```

Alternatively, choose the run ID explicitly:

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
export BEEMO_RUN_ID="beemo-output-only-4scorer-full-$STAMP"
bash Beemo/submit_beemo.sh
```

Granite is not part of the primary run. To include it as an optional fifth
scorer, choose a distinct run ID and export the explicit flag:

```bash
export BEEMO_RUN_ID="beemo-output-only-4scorer-plus-granite-full-$STAMP"
export INCLUDE_GRANITE=1
bash Beemo/submit_beemo.sh
```

That run has five target-scorer files, 31 scorer-detector configurations, and
1,240 metric rows.

Record both printed values. The full prepared file, each of the four target
score files, and the Binoculars file should each contain:

```text
2,187 records × 9 variants = 19,683 rows
```

The 24-hour request is a safety ceiling, not an estimate or promise. Beemo
is much smaller than a 78,000-row synthetic cell, but prompt/response lengths
vary and the first run may need cached model or dataset downloads. Use the
progress lines in stdout for a real ETA.

## 4. Validate completion

```bash
sacct -j "$JOB_ID" \
  --format=JobID,JobName,State,Elapsed,AllocTRES,ExitCode
tail -n 100 "logs/beemo-$JOB_ID.err"

python Beemo/validate_beemo.py --run-id "$BEEMO_RUN_ID"
```

Then inspect the manifest and result summary:

```bash
python - "$BEEMO_RUN_ID" <<'PY'
import json, sys
from pathlib import Path

run_id = sys.argv[1]
manifest = json.loads((Path("runs/beemo") / run_id / "manifest.json").read_text())
summary = json.loads((Path("results/beemo") / run_id / "summary.json").read_text())
print("status:", manifest["completion_status"])
print("stages:", manifest["completed_stages"])
print("summary:", summary)
PY
```

A valid all-stage run reports `complete` with stages `prepare`, `score`,
`evaluate`, `plot`, and `validate`; seven detector methods across 25
scorer-detector configurations; 1% and 5% FPRs; 19,683 rows per score file; and
1,000 metric rows.

## 5. Open or regenerate the plot

The primary plots are:

```text
results/beemo/<run-id>/beemo_tpr_gpt2_xl_2x7.png
results/beemo/<run-id>/beemo_tpr_opt_1_3b_2x7.png
results/beemo/<run-id>/beemo_tpr_falcon_7b_2x7.png
results/beemo/<run-id>/beemo_tpr_qwen2_7b_2x7.png
```

Each has two FPR rows and seven detector columns, including the same shared
Binoculars result. Regenerate them without GPU scoring:

```bash
python Beemo/plot_beemo.py --run-id "$BEEMO_RUN_ID"
```

## 6. Resume a failed run

Use the same run ID only when the configuration, record limit, bootstrap
override, and Binoculars choice are unchanged:

```bash
BEEMO_RUN_ID="$BEEMO_RUN_ID" bash Beemo/submit_beemo.sh
```

The JSONL scorer resumes by complete row key. If the scientific configuration
changes, use a new run ID. Never delete a partial run merely to make a command
start over; inspect its manifest and error first.

## 7. Download only small artifacts

From local PowerShell, copy the result directory—not the large token-feature
JSONL files:

```powershell
scp -r jli101@login.delta.ncsa.illinois.edu:~/LLM-detection/results/beemo/<run-id> .
```

The useful small files are `metrics.csv`, `subgroup_metrics.csv`,
`frozen_specs.json`, `summary.json`, `validation_report.json`, and the four
`beemo_tpr_<scorer>_2x7.png` files.
