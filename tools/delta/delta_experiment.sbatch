#!/bin/bash
#SBATCH --job-name=llm-detection
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/delta-%x-%j.out
#SBATCH --error=logs/delta-%x-%j.err

set -euo pipefail

if [[ -z "${DATASET:-}" || -z "${TARGET_MODEL:-}" ]]; then
  echo "DATASET and TARGET_MODEL must be exported by the submission wrapper" >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/paper.json}"
RUN_ID="${RUN_ID:-delta-${SLURM_JOB_ID}-${DATASET}-${TARGET_MODEL//\//__}}"
GENERATION_BACKEND="${GENERATION_BACKEND:-transformers}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${SLURM_GPUS_ON_NODE:-4}}"
SKIP_BINOCULARS="${SKIP_BINOCULARS:-0}"
PYTHON_MODULE="${PYTHON_MODULE:-miniforge3-python}"
VENV_PATH="${VENV_PATH:-$PWD/.venv}"

mkdir -p logs
module reset
module load "${PYTHON_MODULE}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment: ${VENV_PATH}" >&2
  echo "Create it with tools/delta/setup_delta_env.sh." >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"
python -c 'import torch; print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} cuda_available={torch.cuda.is_available()}"); assert torch.version.cuda == "12.8" and torch.cuda.is_available(), "Expected the working Delta CUDA 12.8 environment"'

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false

echo "run_id=${RUN_ID}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "partition=${SLURM_JOB_PARTITION}"
echo "dataset=${DATASET}"
echo "target_model=${TARGET_MODEL}"
echo "generation_backend=${GENERATION_BACKEND}"
echo "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

ARGS=(
  --config "${CONFIG}"
  --dataset "${DATASET}"
  --model "${TARGET_MODEL}"
  --run-id "${RUN_ID}"
  --stage all
  --set "generation.backend=\"${GENERATION_BACKEND}\""
  --set "generation.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
  --set 'generation.device_map="auto"'
  --set 'scoring.device_map="auto"'
)
if [[ "${SKIP_BINOCULARS}" == "1" ]]; then
  ARGS+=(--skip-binoculars)
fi

srun python -u run_experiment.py "${ARGS[@]}"
