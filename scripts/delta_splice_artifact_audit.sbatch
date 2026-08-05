#!/bin/bash
# Paired Granite-8B x XSum splice-artifact audit. Command-line sbatch resource
# flags may override these bounded two-GPU defaults.
#SBATCH --job-name=granite-splice-audit
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/granite-splice-audit-%j.out
#SBATCH --error=logs/granite-splice-audit-%j.err

set -euo pipefail

AUDIT_ID="${AUDIT_ID:?Set a unique AUDIT_ID}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-granite8b-xsum-full-20260730T063720Z}"
SAMPLE_COUNT="${SAMPLE_COUNT:-500}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
SKIP_BINOCULARS="${SKIP_BINOCULARS:-0}"
DEBUG_ONLY="${DEBUG_ONLY:-0}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"
PYTHON_MODULE="${PYTHON_MODULE:-miniforge3-python}"

mkdir -p logs
module reset
module load "${PYTHON_MODULE}"
source "${VENV_PATH}/bin/activate"

export HF_HOME="${HF_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/torch}"
export TOKENIZERS_PARALLELISM=false

python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); assert torch.version.cuda == "12.8" and torch.cuda.is_available()'

runs_target="$(readlink -f runs)"
results_target="$(readlink -f results)"
case "${runs_target}" in
  /work/hdd/*) ;;
  *) echo "runs must resolve under /work/hdd, found ${runs_target}" >&2; exit 2 ;;
esac
case "${results_target}" in
  /work/hdd/*) ;;
  *) echo "results must resolve under /work/hdd, found ${results_target}" >&2; exit 2 ;;
esac

echo "audit_id=${AUDIT_ID}"
echo "source_run_id=${SOURCE_RUN_ID}"
echo "sample_count=${SAMPLE_COUNT}"
echo "bootstrap_repetitions=${BOOTSTRAP_REPETITIONS}"
echo "runs_target=${runs_target}"
echo "results_target=${results_target}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

ARGS=(
  --audit-id "${AUDIT_ID}"
  --source-run-id "${SOURCE_RUN_ID}"
  --sample-count "${SAMPLE_COUNT}"
  --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"
  --stage all
)
if [[ "${SKIP_BINOCULARS}" == "1" ]]; then
  ARGS+=(--skip-binoculars)
fi
if [[ "${DEBUG_ONLY}" == "1" ]]; then
  ARGS+=(--debug-only)
fi

srun python -u run_splice_artifact_audit.py "${ARGS[@]}"
echo "GRANITE SPLICE ARTIFACT AUDIT: PASS"
