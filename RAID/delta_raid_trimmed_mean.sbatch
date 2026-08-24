#!/bin/bash
# Evaluation-only trimmed-mean comparison for an existing bounded RAID run.
# One GPU is reserved only because this project has a GPU-type Delta account;
# the Python analysis performs no model inference.
#SBATCH --job-name=raid-trim
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/raid-trim-%j.out
#SBATCH --error=logs/raid-trim-%j.err

set -euo pipefail

RAID_RUN_ID="${RAID_RUN_ID:?Set RAID_RUN_ID to the completed bounded pilot}"
TRIM_OUTPUT_NAME="${TRIM_OUTPUT_NAME:-trimmed_mean_comparison_v1}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"

module reset
module load miniforge3-python
source "${VENV_PATH}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MALLOC_ARENA_MAX=4

case "$(readlink -f runs)" in /work/hdd/*) ;; *) exit 2 ;; esac
case "$(readlink -f results)" in /work/hdd/*) ;; *) exit 2 ;; esac
test -s "runs/raid/${RAID_RUN_ID}/falcon_scores.jsonl"
test -s "runs/raid/${RAID_RUN_ID}/binoculars_scores.jsonl"

srun python -u RAID/compare_trimmed_mean.py \
  --run-id "${RAID_RUN_ID}" \
  --output-name "${TRIM_OUTPUT_NAME}"

test -s "results/raid/${RAID_RUN_ID}/${TRIM_OUTPUT_NAME}/comparison.complete.json"
echo "RAID TRIMMED MEAN COMPARISON: PASS"
