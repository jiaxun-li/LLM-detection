# NCSA Delta one-GPU smoke test

This is a deliberately tiny, debug-only end-to-end check of the repository on
Delta. It uses `Qwen/Qwen2.5-0.5B`, 12 XSum validation documents, one A100 GPU,
and the six target-model detectors. It does **not** run Binoculars or Falcon and
must not be used as a scientific result.

The job is not submitted automatically. Run the commands below from an
authorized Delta login session.

## One-time environment setup

From the repository root, choose a Python module available on Delta and create
a dedicated virtual environment. The job defaults to `python/3.11`; set
`PYTHON_MODULE` and use that same module here if your Delta environment exposes
a different name.

```bash
module reset
module load python/3.11
python -m venv "$PWD/.venv-delta-smoke"
source "$PWD/.venv-delta-smoke/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import accelerate,datasets,huggingface_hub,numpy,safetensors,torch,transformers; print(torch.__version__, torch.version.cuda)'
```

Put model and dataset caches on a project or scratch filesystem with adequate
quota. For example:

```bash
export SMOKE_CACHE_ROOT="$SCRATCH/llm-detection-smoke"
mkdir -p "$SMOKE_CACHE_ROOT"
```

`SMOKE_CACHE_ROOT`, `VENV_PATH`, `PYTHON_MODULE`, `HF_HOME`,
`HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, and `TORCH_HOME` are all
configurable and are passed through by the wrapper.

## Submit exactly one smoke job

The wrapper requires an explicit Delta charge account. It contains no account
name and refuses a completed `RUN_ID` unless `FORCE=1` is deliberately set.

```bash
ACCOUNT=<delta-charge-account> \
VENV_PATH="$PWD/.venv-delta-smoke" \
SMOKE_CACHE_ROOT="$SCRATCH/llm-detection-smoke" \
bash scripts/submit_delta_smoke.sh
```

Equivalent argument form:

```bash
bash scripts/submit_delta_smoke.sh --account <delta-charge-account>
```

The wrapper prints `submitted_job_id`, resolved paths, and log names. To resume
an interrupted run, reuse the printed run ID:

```bash
ACCOUNT=<delta-charge-account> \
RUN_ID=<previous-run-id> \
bash scripts/submit_delta_smoke.sh
```

The job requests:

- partition `gpuA100x4`;
- one node and one GPU;
- 8 CPUs, 32 GB host memory, and 45 minutes;
- no exclusive node and no Falcon/Binoculars models.

The bounded protocol in
[`configs/delta_smoke_qwen_0_5b.json`](configs/delta_smoke_qwen_0_5b.json)
uses a 4/4/4 clipping/calibration/test split, a 30-token prompt, a 64-token
continuation, temperature 0.8, top-p 0.95, generation seed 101, corruption seed
99173, ratios 0/0.2/0.5, and one random corruption draw. Bootstrap repetitions
are intentionally reduced to five.

## Monitor

Use the job ID printed by the wrapper:

```bash
JOB_ID=<submitted-job-id>
squeue -j "$JOB_ID"
tail -f "logs/delta-qwen05-smoke-${JOB_ID}.out"
tail -f "logs/delta-qwen05-smoke-${JOB_ID}.err"
sstat -j "${JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,AllocTRES,ExitCode
```

The stdout log includes hostname, Slurm account/partition/resources, Python and
package versions, PyTorch/CUDA availability, selected GPU name and memory,
`nvidia-smi`, cache paths, seeds, model, split sizes, generation parameters,
contamination settings, and detector names. For an optional live GPU snapshot
while the allocation is running:

```bash
srun --jobid="$JOB_ID" --overlap --ntasks=1 --cpus-per-task=1 nvidia-smi
```

## Output layout and expected counts

Every run is isolated below:

```text
smoke_results/delta_qwen_0_5b/<run_id>/
```

The final expected artifacts and counts are:

| Artifact | Expected rows/purpose |
|---|---:|
| `selected_sources.jsonl` | 12 distinct XSum IDs, split 4/4/4 |
| `base_generations.jsonl` | 12 uncontaminated model continuations |
| `tail_candidate_cache.jsonl` | 12 once-computed white-box token orderings |
| `clean_data.jsonl` | 24 rows: one human and one LLM baseline per ID |
| `contaminated_data.jsonl` | 48 rows: random/tail at 0.2/0.5 per ID |
| `data.jsonl` | 72 unique prepared rows total |
| `target_scores.jsonl` | 72 matching scored rows |
| `metrics.csv` | 144 debug-only metric rows |
| `resume_baseline.json` | first-pass content/key snapshot |
| `validation_report.json` | strict final report with `status: PASS` |
| `smoke_validation.complete.json` | written only after the resume check passes |
| `slurm_logs.json` | exact stdout/stderr paths and job ID |

The pipeline is run twice with the same `RUN_ID`. The second pass must append
zero duplicate data/score keys, atomically regenerate the metrics, and match the
first-pass snapshot. The validator also checks all split identities, row-key
coverage, token-feature lengths, six detector names, calibration thresholds
recomputed only from the four calibration-human rows, final metrics based only
on four test human and four test LLM IDs, stage markers, model provenance, and
the visible label `SMOKE_TEST_DEBUG_ONLY_NOT_FOR_SCIENTIFIC_USE`.

After a successful job:

```bash
RUN_DIR="$(cat smoke_results/delta_qwen_0_5b/LATEST_SUCCESSFUL.txt)"
cat "$RUN_DIR/validation_report.json"
cat "$RUN_DIR/smoke_validation.complete.json"
wc -l \
  "$RUN_DIR/selected_sources.jsonl" \
  "$RUN_DIR/clean_data.jsonl" \
  "$RUN_DIR/contaminated_data.jsonl" \
  "$RUN_DIR/data.jsonl" \
  "$RUN_DIR/target_scores.jsonl"
python scripts/validate_delta_smoke.py \
  --config configs/delta_smoke_qwen_0_5b.json \
  --run-dir "$RUN_DIR" \
  --report "$RUN_DIR/validation_report.manual.json" \
  --compare-baseline "$RUN_DIR/resume_baseline.json" \
  --write-completion-marker \
  --materialize-views
```

A scheduler `COMPLETED` state alone is not a smoke-test pass. PASS requires the
strict validator report and final completion marker. Any failed assertion exits
nonzero, emits `DELTA QWEN 0.5B SMOKE: FAIL` in the Slurm log, and does not
update `LATEST_SUCCESSFUL.txt`.

## Runtime and charge estimate

With cached model/dataset files, expect roughly 5–15 minutes on one A100. A cold
cache or congested filesystem can extend that into the 15–30 minute range. That
is approximately 0.08–0.5 allocated GPU-hours; the 45-minute limit caps the job
at 0.75 one-GPU hours. Delta allocation charging may apply partition-specific
weights, so treat these as resource-time estimates rather than an account
billing guarantee.

## Local no-GPU verification

This exercises the output contract, metrics, strict validator, and duplicate-
free two-pass resume logic without downloading XSum or model weights:

```bash
python scripts/synthetic_delta_smoke.py --output-dir /tmp/delta-smoke-synthetic
python -m unittest discover -s tests -v
```

It does not validate CUDA, Hugging Face access, or Slurm behavior; those remain
the purpose of the Delta job.
