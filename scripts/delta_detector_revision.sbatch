#!/bin/bash
# New outputs only. Override GPUs/memory/time for evaluation-only stages.
#SBATCH --job-name=detector-revision
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/detector-revision-%A_%a.out
#SBATCH --error=logs/detector-revision-%A_%a.err
set -euo pipefail
module reset
module load miniforge3-python
source /projects/bhuc/${USER}/venvs/delta-smoke/bin/activate
export HF_HOME=/work/hdd/bhuc/${USER}/llm-detection-smoke/huggingface
export TORCH_HOME=/work/hdd/bhuc/${USER}/llm-detection-smoke/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MALLOC_ARENA_MAX=4
case "$(readlink -f runs)" in /work/hdd/*) ;; *) echo 'runs must resolve under /work/hdd'; exit 2 ;; esac
case "$(readlink -f results)" in /work/hdd/*) ;; *) echo 'results must resolve under /work/hdd'; exit 2 ;; esac
: "${SOURCE_RUN_ID:?Set source run ID}"
: "${REVISION_ID:?Set new revision ID}"
: "${REVISION_STAGE:?Set primary, gate, score, or evaluate}"
if [[ "$REVISION_STAGE" == primary ]]; then
    srun python -u scripts/reevaluate_detector_revision.py \
      --source-run-id "$SOURCE_RUN_ID" --revision-id "$REVISION_ID" \
      --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS:-2000}"
else
    args=(--source-run-id "$SOURCE_RUN_ID" --revision-id "$REVISION_ID"
          --stage "$REVISION_STAGE" --num-shards "${NUM_SHARDS:-4}"
          --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS:-2000}")
    if [[ "$REVISION_STAGE" == gate ]]; then
        python -m unittest tests.test_detector_revision -v
        args+=(--reference-metrics "${REFERENCE_METRICS:?Set reviewed upstream metrics.py path}")
    elif [[ "$REVISION_STAGE" == score ]]; then
        args+=(--shard-index "${SLURM_ARRAY_TASK_ID:?Submit a scoring array}")
    fi
    srun python -u RAID/revise_detectors.py "${args[@]}"
fi
