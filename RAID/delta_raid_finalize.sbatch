#!/bin/bash
# CPU-only strict merge, evaluation, plotting, and validation.
#SBATCH --job-name=raid-finalize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/raid-finalize-%j.out
#SBATCH --error=logs/raid-finalize-%j.err

set -euo pipefail
RAID_RUN_ID="${RAID_RUN_ID:?Set RAID_RUN_ID}"
NUM_SHARDS="${NUM_SHARDS:-4}"
LIMIT_SOURCES="${LIMIT_SOURCES:-}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-}"
DEBUG_ONLY="${DEBUG_ONLY:-0}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"
module reset
module load miniforge3-python
source "${VENV_PATH}/bin/activate"

ARGS=(--run-id "${RAID_RUN_ID}" --stage all --num-shards "${NUM_SHARDS}")
if [[ -n "${LIMIT_SOURCES}" ]]; then ARGS+=(--limit-sources "${LIMIT_SOURCES}"); fi
if [[ -n "${BOOTSTRAP_REPETITIONS}" ]]; then ARGS+=(--bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"); fi
if [[ "${DEBUG_ONLY}" == "1" ]]; then ARGS+=(--debug-only); fi
python -u RAID/run_raid.py "${ARGS[@]}"
echo "RAID FINALIZE: PASS"
