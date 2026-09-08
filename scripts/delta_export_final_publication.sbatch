#!/bin/bash
#SBATCH --job-name=final-export
#SBATCH --account=bhuc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/final-export-%j.out
#SBATCH --error=logs/final-export-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is missing}"
module reset
module load miniforge3-python
source "/projects/bhuc/${USER}/venvs/delta-smoke/bin/activate"

python -c 'import csv, json, matplotlib; print("EXPORT ENVIRONMENT PREFLIGHT: PASS")'

: "${FINAL_EXPORT_ID:?FINAL_EXPORT_ID is required}"
: "${PRIMARY_BUNDLE_ID:?PRIMARY_BUNDLE_ID is required}"
: "${RAID_REVISION_ID:?RAID_REVISION_ID is required}"
: "${FINAL_EXPORT_ROOT:?FINAL_EXPORT_ROOT is required}"

python scripts/export_final_publication_bundle.py \
  --primary-bundle "results/primary_final_exports/${PRIMARY_BUNDLE_ID}" \
  --raid-results "results/raid/${RAID_REVISION_ID}" \
  --output-root "${FINAL_EXPORT_ROOT}" \
  --bundle-id "${FINAL_EXPORT_ID}"
