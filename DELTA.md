# NCSA Delta execution

The repository does not launch paper-scale GPU work automatically. These steps
are for an authorized Delta login session after the local test suite passes.

Run the bounded, one-GPU validation in [`DELTA_SMOKE.md`](DELTA_SMOKE.md)
before using the paper-scale scripts below.

## One-time environment

```bash
VENV_PATH="/projects/<project>/$USER/venvs/llm-detection" \
bash scripts/setup_delta_env.sh
```

Install vLLM only when it is compatible with Delta's active CUDA/PyTorch stack:

```bash
python -m pip install 'vllm>=0.7'
```

The current primary models are public, but authenticating with Hugging Face is
still recommended for higher download rate limits. Keep the cache on a project
or work filesystem with enough quota.

## Static smoke job

The Slurm job file is [`scripts/delta_experiment.sbatch`](scripts/delta_experiment.sbatch).
It defaults to Delta's `gpuA100x4` partition and four GPUs.

```bash
mkdir -p logs
sbatch \
  --account=<delta-account> \
  --export=ALL,CONFIG=configs/smoke.json,DATASET=xsum,TARGET_MODEL=Qwen/Qwen2.5-0.5B \
  scripts/delta_experiment.sbatch
```

The job logs CUDA/GPU information, records the Slurm job ID in the manifest,
uses Transformers by default, and resumes an existing `RUN_ID` without
duplicating valid rows.

To exercise optional vLLM generation:

```bash
sbatch \
  --account=<delta-account> \
  --export=ALL,CONFIG=configs/smoke.json,DATASET=xsum,TARGET_MODEL=Qwen/Qwen2.5-0.5B,GENERATION_BACKEND=vllm \
  scripts/delta_experiment.sbatch
```

## Matrices

The wrapper accepts `primary`, `replication`, `qwen-scaling`, or `all`:

```bash
ACCOUNT=<delta-account> EXPERIMENT_SET=primary \
  bash scripts/submit_delta_matrix.sh
```

Normal jobs use `gpuA100x4`. The wrapper deliberately skips Qwen 72B there.
Submit that configuration explicitly only after a smaller validation:

```bash
ACCOUNT=<delta-account> EXPERIMENT_SET=qwen-scaling \
  PARTITION=gpuH200x8 TIME=48:00:00 \
  bash scripts/submit_delta_matrix.sh
```

The wrapper rejects any partition other than `gpuA100x4` or `gpuH200x8`.

## Cluster validation gate

Before releasing paper results, verify for at least one smoke run and then every
matrix cell:

1. source, base, data, tail-cache, target-score, and Binoculars row counts match;
2. all three split-ID sets are disjoint and match the shared source manifest;
3. GPU peak memory and throughput are plausible with no CPU/model offload
   surprises;
4. an interrupted job resumes with the same `RUN_ID` and no duplicate keys;
5. the Falcon performer is `tiiuae/falcon-7b-instruct` on one GPU and the
   observer is `tiiuae/falcon-7b` on another;
6. `manifest.json` ends with `completion_status: complete`;
7. `metrics.csv` contains LRR and both 1%/5% calibrated-FPR rows.
