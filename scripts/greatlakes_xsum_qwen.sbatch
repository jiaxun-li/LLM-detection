#!/bin/bash
#SBATCH --job-name=llm-detect-xsum
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
fi

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

python - <<'PY'
import importlib.util
import sys

missing = [
    pkg for pkg in ["datasets", "matplotlib", "numpy", "torch", "transformers"]
    if importlib.util.find_spec(pkg) is None
]
if missing:
    print("Missing Python packages:", ", ".join(missing), file=sys.stderr)
    print("Create/activate a venv and run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
PY

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export MPLBACKEND=Agg

DATASET="${DATASET:-xsum}"
N_SAMPLES="${N_SAMPLES:-200}"
GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen2.5-0.5B}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-1.5B}"
MODEL_DIR_NAME="${GENERATOR_MODEL//\//__}"
DATA_DIR="real_data/${DATASET}/${MODEL_DIR_NAME}"
OUTPUT_DIR="real_results/${DATASET}_qwen"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "dataset=${DATASET}"
echo "n_samples=${N_SAMPLES}"
echo "generator_model=${GENERATOR_MODEL}"
echo "reference_model=${REFERENCE_MODEL}"
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device=", torch.cuda.get_device_name(0))
PY

python prepare_real_contamination.py \
  --dataset "${DATASET}" \
  --n-samples "${N_SAMPLES}" \
  --generator-model "${GENERATOR_MODEL}"

python score_real_text.py \
  --input "${DATA_DIR}/clean.jsonl" \
  --output "${DATA_DIR}/clean_scores.jsonl" \
  --observer-model "${GENERATOR_MODEL}" \
  --reference-model "${REFERENCE_MODEL}"

python score_real_text.py \
  --input "${DATA_DIR}/contaminated.jsonl" \
  --output "${DATA_DIR}/contaminated_scores.jsonl" \
  --observer-model "${GENERATOR_MODEL}" \
  --reference-model "${REFERENCE_MODEL}"

python evaluate_real_clipping.py \
  --clean "${DATA_DIR}/clean_scores.jsonl" \
  --contaminated "${DATA_DIR}/contaminated_scores.jsonl" \
  --output-dir "${OUTPUT_DIR}"
