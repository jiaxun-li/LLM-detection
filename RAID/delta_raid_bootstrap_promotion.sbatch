#!/bin/bash
# Evaluation-only 2,000-bootstrap RAID promotion from completed score packs.
# One GPU is reserved only because the available Delta allocation is GPU-type;
# the Python process performs no model inference.
#SBATCH --job-name=raid-boot2000
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=logs/raid-boot2000-%j.out
#SBATCH --error=logs/raid-boot2000-%j.err

set -euo pipefail

RAID_SOURCE_RUN_ID="${RAID_SOURCE_RUN_ID:?Set the completed 500-bootstrap source run ID}"
RAID_PROMOTION_ID="${RAID_PROMOTION_ID:?Set a new result ID}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"

module reset
module load miniforge3-python
source "${VENV_PATH}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export MALLOC_ARENA_MAX=4

case "$(readlink -f runs)" in /work/hdd/*) ;; *) exit 2 ;; esac
case "$(readlink -f results)" in /work/hdd/*) ;; *) exit 2 ;; esac
test -s "runs/raid/${RAID_SOURCE_RUN_ID}/falcon_scores.jsonl"
test -s "runs/raid/${RAID_SOURCE_RUN_ID}/binoculars_scores.jsonl"

srun python -u RAID/promote_bootstrap.py \
  --source-run-id "${RAID_SOURCE_RUN_ID}" \
  --promotion-id "${RAID_PROMOTION_ID}" \
  --bootstrap-repetitions 2000

test -s "results/raid/${RAID_PROMOTION_ID}/promotion.complete.json"
echo "RAID 2000-BOOTSTRAP PROMOTION: PASS"
