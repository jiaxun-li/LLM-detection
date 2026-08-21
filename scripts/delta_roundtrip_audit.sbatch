#!/bin/bash
# Read-only audit of cached Qwen-32B contamination round-trip drift.
#SBATCH --job-name=qwen32-wr-drift-audit
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/qwen32-wr-drift-audit-%j.out
#SBATCH --error=logs/qwen32-wr-drift-audit-%j.err

set -euo pipefail

RUN_ID="${RUN_ID:?Set RUN_ID to the failed Qwen-32B WritingPrompts run}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"

mkdir -p logs
module reset
module load miniforge3-python
source "${VENV_PATH}/bin/activate"

export HF_HOME="${HF_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/torch}"
export TOKENIZERS_PARALLELISM=false

runs_target="$(readlink -f runs)"
case "${runs_target}" in
  /work/hdd/*) ;;
  *) echo "runs must resolve under /work/hdd, found ${runs_target}" >&2; exit 2 ;;
esac

echo "run_id=${RUN_ID}"
echo "runs_target=${runs_target}"
srun python -u scripts/audit_contamination_roundtrip.py --run-id "${RUN_ID}"
echo "QWEN32 WRITINGPROMPTS ROUNDTRIP AUDIT: PASS"

