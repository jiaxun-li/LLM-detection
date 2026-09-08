#!/bin/bash
# Layout/environment regression check only: no model inference or final-data writes.
#SBATCH --job-name=repository-smoke
#SBATCH --account=bhuc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/repository-smoke-%j.out
#SBATCH --error=logs/repository-smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
test -f experiment_core/__init__.py
test -f configs/raid.json
module reset
module load miniforge3-python
source /projects/bhuc/${USER}/venvs/delta-smoke/bin/activate
export HF_HOME=/work/hdd/bhuc/${USER}/llm-detection-smoke/huggingface
export TORCH_HOME=/work/hdd/bhuc/${USER}/llm-detection-smoke/torch
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "Repository: $PWD"
git rev-parse HEAD
srun python -B -u - <<'PY'
import sqlite3
import sys
import unittest
import torch

print("Python:", sys.executable)
print("SQLite:", sqlite3.sqlite_version)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("This smoke job requires one visible CUDA GPU")
print("GPU:", torch.cuda.get_device_name(0))
suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or result.testsRun == 0:
    raise RuntimeError("Repository tests must pass without skips")
print("ACTIVE TEST SUITE: PASS", result.testsRun)
PY
for entry in \
    run_experiment.py \
    plot_tpr_contamination.py \
    RAID/run_raid.py \
    RAID/run_raid_shard.py \
    RAID/revise_detectors.py \
    RAID/promote_bootstrap.py \
    tools/reanalysis/reevaluate_detector_revision.py \
    tools/exports/export_final_publication_bundle.py \
    tools/validation/validate_delta_smoke.py; do
    python -B "$entry" --help >/dev/null
    echo "ENTRYPOINT PASS: $entry"
done
while IFS= read -r -d '' launcher; do
    bash -n "$launcher"
done < <(find tools/delta RAID -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0)
echo 'DELTA REPOSITORY SMOKE: PASS'
