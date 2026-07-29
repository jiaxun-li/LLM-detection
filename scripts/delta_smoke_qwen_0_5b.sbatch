#!/bin/bash
# One-GPU, debug-only, end-to-end Delta smoke test. Submit through
# scripts/submit_delta_smoke.sh so the charge account is explicit.
#SBATCH --job-name=delta-qwen05-smoke
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/delta-qwen05-smoke-%j.out
#SBATCH --error=logs/delta-qwen05-smoke-%j.err

set -euo pipefail

on_exit() {
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "DELTA QWEN 0.5B SMOKE: FAIL (exit=${status})" >&2
  fi
}
trap on_exit EXIT

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPOSITORY_ROOT}"
CONFIG="${CONFIG:-configs/delta_smoke_qwen_0_5b.json}"
RUN_ID="${RUN_ID:-delta-smoke-${SLURM_JOB_ID}}"
FORCE="${FORCE:-0}"
SMOKE_ROOT="$PWD/smoke_results/delta_qwen_0_5b"
RUN_DIR="${SMOKE_ROOT}/${RUN_ID}"
VENV_PATH="${VENV_PATH:-$PWD/.venv-delta-smoke}"
PYTHON_MODULE="${PYTHON_MODULE:-miniforge3-python}"
SMOKE_CACHE_ROOT="${SMOKE_CACHE_ROOT:-${SCRATCH:-$PWD/.cache}/llm-detection-smoke}"
FINAL_MARKER="${RUN_DIR}/smoke_validation.complete.json"
BASELINE="${RUN_DIR}/resume_baseline.json"

if [[ -e "${FINAL_MARKER}" && "${FORCE}" != "1" ]]; then
  echo "Refusing to overwrite completed smoke run: ${RUN_DIR}" >&2
  echo "Choose a new RUN_ID, or explicitly set FORCE=1." >&2
  exit 2
fi
if [[ "${FORCE}" == "1" && -e "${FINAL_MARKER}" ]]; then
  echo "FORCE=1: reopening the completed run for an explicit resume validation."
  rm -f \
    "${FINAL_MARKER}" \
    "${RUN_DIR}/validation_report.json" \
    "${RUN_DIR}/validation_report.first_pass.json" \
    "${BASELINE}"
fi

mkdir -p logs "${RUN_DIR}" "${SMOKE_CACHE_ROOT}"

module reset
if ! module load "${PYTHON_MODULE}"; then
  echo "Unable to load ${PYTHON_MODULE}. Set PYTHON_MODULE to an available Delta Python module." >&2
  exit 2
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment: ${VENV_PATH}" >&2
  echo "Create it on a Delta login node and install requirements.txt; see DELTA_SMOKE.md." >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"

export HF_HOME="${HF_HOME:-$SMOKE_CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$SMOKE_CACHE_ROOT/torch}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}"

if [[ -e "${RUN_DIR}/config.snapshot.json" ]]; then
  if ! cmp -s "${CONFIG}" "${RUN_DIR}/config.snapshot.json"; then
    echo "Existing run has a different config snapshot: ${RUN_DIR}" >&2
    exit 2
  fi
else
  cp "${CONFIG}" "${RUN_DIR}/config.snapshot.json"
fi

stdout_path="${PWD}/logs/delta-qwen05-smoke-${SLURM_JOB_ID}.out"
stderr_path="${PWD}/logs/delta-qwen05-smoke-${SLURM_JOB_ID}.err"
metadata_tmp="${RUN_DIR}/slurm_logs.json.tmp.${SLURM_JOB_ID}"
printf '{\n  "job_id": "%s",\n  "stdout": "%s",\n  "stderr": "%s"\n}\n' \
  "${SLURM_JOB_ID}" "${stdout_path}" "${stderr_path}" > "${metadata_tmp}"
mv "${metadata_tmp}" "${RUN_DIR}/slurm_logs.json"

echo "=== Delta smoke diagnostics ==="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_account=${SLURM_JOB_ACCOUNT:-unknown}"
echo "slurm_partition=${SLURM_JOB_PARTITION:-unknown}"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unknown}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-unknown}"
echo "config=${CONFIG}"
echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "venv=${VENV_PATH}"
echo "hf_home=${HF_HOME}"
echo "hf_datasets_cache=${HF_DATASETS_CACHE}"
echo "transformers_cache=${TRANSFORMERS_CACHE}"
echo "torch_home=${TORCH_HOME}"
python --version
python -c 'import importlib.metadata as m; names=("accelerate","datasets","huggingface-hub","numpy","safetensors","torch","transformers"); print("package_versions=" + ", ".join(f"{name}={m.version(name)}" for name in names))'
python -c 'import torch; print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}"); assert torch.version.cuda == "12.8", "Expected the pinned CUDA 12.8 PyTorch build"; assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA GPU is unavailable"; props=torch.cuda.get_device_properties(0); print(f"selected_gpu={props.name} total_memory_gib={props.total_memory / 2**30:.1f}")'
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
python -c 'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); print("smoke_settings=" + json.dumps({"model": c["models"]["primary"], "splits": c["splits"], "generation": c["generation"], "contamination": c["contamination"], "detectors": c["scoring"]["detectors"], "evaluation": c["evaluation"]}, sort_keys=True))' "${CONFIG}"

RUN_ARGS=(
  --config "${CONFIG}"
  --dataset xsum
  --model Qwen/Qwen2.5-0.5B
  --run-id "${RUN_ID}"
  --stage all
  --skip-binoculars
)

echo "=== First pass ==="
srun python -u run_experiment.py "${RUN_ARGS[@]}"
python -u scripts/validate_delta_smoke.py \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --report "${RUN_DIR}/validation_report.first_pass.json" \
  --write-baseline "${BASELINE}" \
  --materialize-views

echo "=== Resume pass (must append zero duplicate keys) ==="
srun python -u run_experiment.py "${RUN_ARGS[@]}"
python -u scripts/validate_delta_smoke.py \
  --config "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --report "${RUN_DIR}/validation_report.json" \
  --compare-baseline "${BASELINE}" \
  --write-completion-marker \
  --materialize-views

latest_tmp="${SMOKE_ROOT}/LATEST_SUCCESSFUL.txt.tmp.${SLURM_JOB_ID}"
printf '%s\n' "${RUN_DIR}" > "${latest_tmp}"
mv "${latest_tmp}" "${SMOKE_ROOT}/LATEST_SUCCESSFUL.txt"

echo "DELTA QWEN 0.5B SMOKE: PASS"
echo "validation_report=${RUN_DIR}/validation_report.json"
echo "completion_marker=${FINAL_MARKER}"
