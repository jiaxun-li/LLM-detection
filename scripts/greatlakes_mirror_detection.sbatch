#!/bin/bash
#SBATCH --job-name=mirror-detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
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
MODEL_ALIAS="${MODEL_ALIAS:-qwen}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-0.5B}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
FORCE="${FORCE:-0}"

MODEL_DIR_NAME="${TARGET_MODEL//\//__}"
DATA_DIR="real_data/${DATASET}/${MODEL_DIR_NAME}"
PRIMARY_SCORE_DIR="${PRIMARY_SCORE_DIR:-${DATA_DIR}/scores_target_model}"
BINOCULARS_SCORE_DIR="${BINOCULARS_SCORE_DIR:-${DATA_DIR}/scores_binoculars_falcon}"
OUTPUT_DIR="${OUTPUT_DIR:-real_results/${DATASET}_${MODEL_ALIAS}_mirror}"

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_ARGS=(--trust-remote-code)
fi

echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "dataset=${DATASET}"
echo "n_samples=${N_SAMPLES}"
echo "model_alias=${MODEL_ALIAS}"
echo "target_model=${TARGET_MODEL}"
echo "primary_score_dir=${PRIMARY_SCORE_DIR}"
echo "binoculars_score_dir=${BINOCULARS_SCORE_DIR}"
echo "output_dir=${OUTPUT_DIR}"
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"cuda_device_{idx}=", torch.cuda.get_device_name(idx))
PY

mkdir -p "${PRIMARY_SCORE_DIR}" "${BINOCULARS_SCORE_DIR}" "${OUTPUT_DIR}"

if [[ "${FORCE}" == "1" || ! -s "${DATA_DIR}/clean.jsonl" || ! -s "${DATA_DIR}/contaminated.jsonl" ]]; then
  python prepare_real_contamination.py \
    --dataset "${DATASET}" \
    --n-samples "${N_SAMPLES}" \
    --generator-model "${TARGET_MODEL}" \
    --evaluator-model "${TARGET_MODEL}" \
    "${TRUST_ARGS[@]}"
else
  echo "skipping preparation; found ${DATA_DIR}/clean.jsonl and contaminated.jsonl"
fi

if [[ "${FORCE}" == "1" || ! -s "${PRIMARY_SCORE_DIR}/clean_scores.jsonl" ]]; then
  python score_real_text.py \
    --input "${DATA_DIR}/clean.jsonl" \
    --output "${PRIMARY_SCORE_DIR}/clean_scores.jsonl" \
    --observer-model "${TARGET_MODEL}" \
    "${TRUST_ARGS[@]}"
else
  echo "skipping primary clean scoring"
fi

if [[ "${FORCE}" == "1" || ! -s "${PRIMARY_SCORE_DIR}/contaminated_scores.jsonl" ]]; then
  python score_real_text.py \
    --input "${DATA_DIR}/contaminated.jsonl" \
    --output "${PRIMARY_SCORE_DIR}/contaminated_scores.jsonl" \
    --observer-model "${TARGET_MODEL}" \
    "${TRUST_ARGS[@]}"
else
  echo "skipping primary contaminated scoring"
fi

if [[ "${FORCE}" == "1" || ! -s "${BINOCULARS_SCORE_DIR}/clean_scores.jsonl" ]]; then
  python score_real_text.py \
    --input "${DATA_DIR}/clean.jsonl" \
    --output "${BINOCULARS_SCORE_DIR}/clean_scores.jsonl" \
    --binoculars-default-models
else
  echo "skipping Binoculars clean scoring"
fi

if [[ "${FORCE}" == "1" || ! -s "${BINOCULARS_SCORE_DIR}/contaminated_scores.jsonl" ]]; then
  python score_real_text.py \
    --input "${DATA_DIR}/contaminated.jsonl" \
    --output "${BINOCULARS_SCORE_DIR}/contaminated_scores.jsonl" \
    --binoculars-default-models
else
  echo "skipping Binoculars contaminated scoring"
fi

python evaluate_real_clipping.py \
  --clean "${PRIMARY_SCORE_DIR}/clean_scores.jsonl" \
  --contaminated "${PRIMARY_SCORE_DIR}/contaminated_scores.jsonl" \
  --binoculars-clean "${BINOCULARS_SCORE_DIR}/clean_scores.jsonl" \
  --binoculars-contaminated "${BINOCULARS_SCORE_DIR}/contaminated_scores.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --plot-title-prefix "${DATASET} ${MODEL_ALIAS} mirrored robustness"
