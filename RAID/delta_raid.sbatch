#!/bin/bash
# RAID universal-clipping benchmark. Two GPUs are required for Binoculars.
#SBATCH --job-name=raid-clipping
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/raid-%j.out
#SBATCH --error=logs/raid-%j.err

set -euo pipefail

RAID_RUN_ID="${RAID_RUN_ID:?Set a unique RAID_RUN_ID}"
RAID_DATA_PATH="${RAID_DATA_PATH:?Set RAID_DATA_PATH to the official labeled train.csv under /work/hdd}"
RAID_STAGE="${RAID_STAGE:-all}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"
PYTHON_MODULE="${PYTHON_MODULE:-miniforge3-python}"
LIMIT_SOURCES="${LIMIT_SOURCES:-}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-}"
SKIP_BINOCULARS="${SKIP_BINOCULARS:-0}"
DEBUG_ONLY="${DEBUG_ONLY:-0}"
RESET_INDEX="${RESET_INDEX:-0}"

mkdir -p logs
module reset
module load "${PYTHON_MODULE}"
source "${VENV_PATH}/bin/activate"

export HF_HOME="${HF_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/torch}"
export TOKENIZERS_PARALLELISM=false

python -c 'import rapidfuzz; print("rapidfuzz=" + rapidfuzz.__version__)'
python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); assert torch.version.cuda == "12.8" and torch.cuda.is_available() and torch.cuda.device_count() >= 2'

if [[ ! -f "${RAID_DATA_PATH}" ]]; then
  echo "RAID_DATA_PATH is not a file: ${RAID_DATA_PATH}" >&2
  exit 2
fi
case "$(readlink -f "${RAID_DATA_PATH}")" in
  /work/hdd/*) ;;
  *) echo "RAID data must be under /work/hdd" >&2; exit 2 ;;
esac
for path in runs results; do
  target="$(readlink -f "${path}")"
  case "${target}" in
    /work/hdd/*) ;;
    *) echo "${path} must resolve under /work/hdd, found ${target}" >&2; exit 2 ;;
  esac
done

echo "raid_run_id=${RAID_RUN_ID}"
echo "raid_stage=${RAID_STAGE}"
echo "raid_data_path=$(readlink -f "${RAID_DATA_PATH}")"
echo "runs_target=$(readlink -f runs)"
echo "results_target=$(readlink -f results)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

ARGS=(
  --run-id "${RAID_RUN_ID}"
  --stage "${RAID_STAGE}"
  --data-path "${RAID_DATA_PATH}"
)
if [[ -n "${LIMIT_SOURCES}" ]]; then
  ARGS+=(--limit-sources "${LIMIT_SOURCES}")
fi
if [[ -n "${BOOTSTRAP_REPETITIONS}" ]]; then
  ARGS+=(--bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}")
fi
if [[ "${SKIP_BINOCULARS}" == "1" ]]; then
  ARGS+=(--skip-binoculars)
fi
if [[ "${DEBUG_ONLY}" == "1" ]]; then
  ARGS+=(--debug-only)
fi
if [[ "${RESET_INDEX}" == "1" ]]; then
  ARGS+=(--reset-index)
fi

srun python -u RAID/run_raid.py "${ARGS[@]}"
echo "RAID BENCHMARK STAGE: PASS"

