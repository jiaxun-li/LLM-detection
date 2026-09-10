#!/bin/bash
# Existing packs only. GPU reserved solely because the project has a GPU account.
#SBATCH --job-name=raid-mean-ci
#SBATCH --partition=gpuA100x4
#SBATCH --account=bhuc-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/raid-mean-ci-%j.out
#SBATCH --error=logs/raid-mean-ci-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module reset
module load miniforge3-python
source /projects/bhuc/${USER}/venvs/delta-smoke/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
case "$(readlink -f runs)" in /work/hdd/*) ;; *) echo 'runs must be on /work/hdd'; exit 2 ;; esac
case "$(readlink -f results)" in /work/hdd/*) ;; *) echo 'results must be on /work/hdd'; exit 2 ;; esac
srun python -u tools/reanalysis/raid_attack_mean_ci.py \
    --output "results/raid_aggregate_ci/accepted-v41-job-${SLURM_JOB_ID}" \
    --repetitions 2000
