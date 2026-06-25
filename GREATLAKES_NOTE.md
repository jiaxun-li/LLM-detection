# Great Lakes Notes for LLM-detection

This note summarizes how to run this project on the University of Michigan Great Lakes cluster.

## 1. Log In

From your Mac:

```bash
ssh jasonli@greatlakes.arc-ts.umich.edu
```

Great Lakes usually asks for Duo/MFA. This is normal.

Then go to the project:

```bash
cd ~/LLM-detection
```

## 2. Slurm Account

Your Slurm account is:

```text
stats_dept1
```

Use it with `sbatch`:

```bash
sbatch -A stats_dept1 ...
```

## 3. One-Time Python Setup

Create the virtual environment once:

```bash
cd ~/LLM-detection
module load python/3.11 || module load python
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Create `requirements.txt` if it is missing:

```bash
printf "datasets\nmatplotlib\nnumpy\ntorch\ntransformers\n" > requirements.txt
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Important: if the job gets a V100 GPU and PyTorch fails with `no kernel image is available for execution on the device`, reinstall a V100-compatible PyTorch:

```bash
source .venv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install datasets matplotlib numpy transformers
```

You do not need to create `.venv` or install packages every time. Only redo this if `.venv` is deleted, dependencies change, or Python/CUDA compatibility breaks.

## 4. Submit a Small Test Job

First run a 5-sample smoke test:

```bash
cd ~/LLM-detection
sbatch -A stats_dept1 --export=ALL,N_SAMPLES=5 scripts/greatlakes_xsum_qwen.sbatch
```

This checks that Python, CUDA, Hugging Face downloads, and the pipeline all work.

Expected row counts for `N_SAMPLES=5`:

```text
clean.jsonl                 10
contaminated.jsonl          70
clean_scores.jsonl          10
contaminated_scores.jsonl   70
```

Check with:

```bash
wc -l real_data/xsum/Qwen__Qwen2.5-0.5B/*.jsonl
```

## 5. Submit a Real Run

After the smoke test works, run 200 samples:

```bash
cd ~/LLM-detection
sbatch -A stats_dept1 --time=12:00:00 --export=ALL,N_SAMPLES=200 scripts/greatlakes_xsum_qwen.sbatch
```

Expected row counts for `N_SAMPLES=200`:

```text
clean.jsonl                 400
contaminated.jsonl          2800
clean_scores.jsonl          400
contaminated_scores.jsonl   2800
```

The script may stay quiet for a while while loading models/datasets. It prints `prepared 1/200` only after the first example is fully generated.

## 5a. Run With Upstream Binoculars Models

To mirror Radvand et al. Section 5.2, use the target model for the single-model methods and use the upstream Binoculars model pair only for Binoculars:

```text
log_likelihood, rank, log_rank, LRR, entropy, entropy_gap:
  target model for that setup

binoculars:
  tiiuae/falcon-7b-instruct / tiiuae/falcon-7b
```

The upstream Binoculars implementation uses:

```text
observer/comparison model: tiiuae/falcon-7b
performer/numerator model: tiiuae/falcon-7b-instruct
```

To run one mirrored setup, submit:

```bash
cd ~/LLM-detection
sbatch -A stats_dept1 --time=24:00:00 \
  --export=ALL,DATASET=xsum,MODEL_ALIAS=qwen,TARGET_MODEL=Qwen/Qwen2.5-0.5B,N_SAMPLES=200 \
  scripts/greatlakes_mirror_detection.sbatch
```

This writes target-model score files to:

```text
real_data/<dataset>/<target_model_dir>/scores_target_model/
```

and Binoculars/Falcon score files to:

```text
real_data/<dataset>/<target_model_dir>/scores_binoculars_falcon/
```

The final mixed evaluation is written to:

```text
real_results/<dataset>_<model_alias>_mirror/
```

To submit the 3 dataset x 3 model matrix:

```bash
cd ~/LLM-detection
ACCOUNT=stats_dept1 N_SAMPLES=200 TIME=24:00:00 bash scripts/submit_greatlakes_mirror_matrix.sh
```

The matrix script uses these paper-family model IDs by default:

```text
llama:           meta-llama/Meta-Llama-3-8B
gpt_neox_erebus: KoboldAI/GPT-NeoX-20B-Erebus
qwen:            Qwen/Qwen2.5-32B
```

These exact paper-scale models are much larger than the Qwen-0.5B smoke test. Llama may require Hugging Face access approval and `HF_TOKEN`; GPT-NeoX-20B and Qwen-32B may require larger GPUs or model-parallel loading beyond a 16GB V100. If Great Lakes gives OOM errors, first run the mirrored script with a smaller target model to validate the pipeline.

## 6. Check Job Status

Show your active jobs:

```bash
sq
```

Check a specific job:

```bash
squeue -j <jobid>
```

Estimated start time:

```bash
squeue -j <jobid> --start
```

Final job result:

```bash
sacct -j <jobid> --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End
```

Common states:

```text
PD          pending
R           running
COMPLETED   finished successfully
FAILED      crashed
CANCELLED   canceled
TIMEOUT     hit time limit
```

If `sq` shows nothing, the job is no longer pending/running. Use `sacct` to see whether it completed or failed.

## 7. Check Logs

Logs are in:

```text
~/LLM-detection/logs/
```

Watch output:

```bash
tail -f logs/llm-detect-xsum-<jobid>.out
```

Watch errors:

```bash
tail -f logs/llm-detect-xsum-<jobid>.err
```

Stop `tail -f` with:

```text
Ctrl+C
```

## 8. Check Results

Results are written on Great Lakes, not automatically to GitHub.

Main output folders:

```text
~/LLM-detection/real_data/xsum/Qwen__Qwen2.5-0.5B/
~/LLM-detection/real_results/xsum_qwen/
~/LLM-detection/logs/
```

Check results:

```bash
ls -lh real_results/xsum_qwen
head -20 real_results/xsum_qwen/metrics.csv
ls -lh real_results/xsum_qwen/*.png
```

Check data sizes:

```bash
wc -l real_data/xsum/Qwen__Qwen2.5-0.5B/*.jsonl
```


## 10. Copy Results Back to Mac

Run this from your Mac terminal:

```bash
mkdir -p /Users/jasonli/LLM-detection/greatlake
rsync -av jasonli@greatlakes.arc-ts.umich.edu:~/LLM-detection/greatlake/ \
  /Users/jasonli/LLM-detection/greatlake/
```

Or copy only result plots/metrics:

```bash
rsync -av jasonli@greatlakes.arc-ts.umich.edu:~/LLM-detection/real_results/xsum_qwen/ \
  /Users/jasonli/LLM-detection/greatlake/
```

## 11. Common Problems

`ModuleNotFoundError: No module named 'torch'`

The virtual environment is missing or packages were not installed:

```bash
cd ~/LLM-detection
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv/bin/activate: No such file or directory`

The virtual environment does not exist yet:

```bash
cd ~/LLM-detection
module load python/3.11 || module load python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`CUDA error: no kernel image is available for execution on the device`

The installed PyTorch does not support the assigned GPU, usually V100. Reinstall compatible PyTorch:

```bash
source .venv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install datasets matplotlib numpy transformers
```

`sq` shows no job

The job is not active anymore. Check final state:

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,Start,End
```

Old 70-row files still appear during a 200-sample run

This can be normal. `prepare_real_contamination.py` writes the new files only after preparation finishes. Until then, the old 5-sample files remain.

## 12. Useful Commands

```bash
# Active jobs
sq

# Job history today
sacct -u $USER --starttime today --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End

# Specific job status
sacct -j <jobid> --format=JobID,JobName%20,State,ExitCode,Elapsed,Start,End

# Watch logs
tail -f logs/llm-detect-xsum-<jobid>.out
tail -f logs/llm-detect-xsum-<jobid>.err

# Check outputs
ls -lh real_results/xsum_qwen
wc -l real_data/xsum/Qwen__Qwen2.5-0.5B/*.jsonl
```
