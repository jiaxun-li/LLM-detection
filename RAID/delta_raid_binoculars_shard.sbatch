#!/bin/bash
# One independent two-GPU Binoculars score shard.
#SBATCH --job-name=raid-binoculars
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/raid-binoculars-%A_%a.out
#SBATCH --error=logs/raid-binoculars-%A_%a.err

set -euo pipefail
RAID_RUN_ID="${RAID_RUN_ID:?Set RAID_RUN_ID}"
NUM_SHARDS="${NUM_SHARDS:-4}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"
module reset
module load miniforge3-python
source "${VENV_PATH}/bin/activate"
export HF_HOME="${HF_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/work/hdd/bhuc/${USER}/llm-detection-smoke/torch}"
export TOKENIZERS_PARALLELISM=false
python -c 'import torch; assert torch.version.cuda == "12.8" and torch.cuda.device_count() >= 2'
srun python -u RAID/run_raid_shard.py --run-id "${RAID_RUN_ID}" --scorer binoculars \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS}"
