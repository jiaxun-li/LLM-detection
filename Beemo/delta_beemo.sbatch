#!/bin/bash
# Beemo expert-edit benchmark. Two GPUs are required when Binoculars is enabled.
#SBATCH --job-name=beemo-4scorer
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/beemo-%j.out
#SBATCH --error=logs/beemo-%j.err

set -euo pipefail

BEEMO_RUN_ID="${BEEMO_RUN_ID:?Set a unique BEEMO_RUN_ID}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"
PYTHON_MODULE="${PYTHON_MODULE:-miniforge3-python}"
LIMIT_RECORDS="${LIMIT_RECORDS:-}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-}"
SKIP_BINOCULARS="${SKIP_BINOCULARS:-0}"
INCLUDE_GRANITE="${INCLUDE_GRANITE:-0}"
DEBUG_ONLY="${DEBUG_ONLY:-0}"

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

echo "beemo_run_id=${BEEMO_RUN_ID}"
echo "runs_target=${runs_target}"
echo "results_target=${results_target}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

ARGS=(--run-id "${BEEMO_RUN_ID}" --stage all)
if [[ -n "${LIMIT_RECORDS}" ]]; then
  ARGS+=(--limit-records "${LIMIT_RECORDS}")
fi
if [[ -n "${BOOTSTRAP_REPETITIONS}" ]]; then
  ARGS+=(--bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}")
fi
if [[ "${SKIP_BINOCULARS}" == "1" ]]; then
  ARGS+=(--skip-binoculars)
fi
if [[ "${INCLUDE_GRANITE}" == "1" ]]; then
  ARGS+=(--include-granite)
fi
if [[ "${DEBUG_ONLY}" == "1" ]]; then
  ARGS+=(--debug-only)
fi

srun python -u Beemo/run_beemo.py "${ARGS[@]}"
echo "BEEMO BENCHMARK: PASS"
