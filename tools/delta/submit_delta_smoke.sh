#!/bin/bash
# Guarded submission wrapper for the one-GPU Qwen 0.5B Delta smoke test.
set -euo pipefail

usage() {
  echo "Usage: ACCOUNT=<delta-account> bash tools/delta/submit_delta_smoke.sh"
  echo "   or: bash tools/delta/submit_delta_smoke.sh --account <delta-account>"
}

account="${ACCOUNT:-}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --account)
      if [[ "$#" -lt 2 ]]; then
        echo "--account requires a value" >&2
        exit 2
      fi
      account="$2"
      shift 2
      ;;
    --account=*)
      account="${1#--account=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${account}" ]]; then
  echo "A Delta charge account is required; none is hardcoded." >&2
  usage >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable. Run this wrapper from a Delta login node." >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/delta_smoke_qwen_0_5b.json}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-tools/delta/delta_smoke_qwen_0_5b.sbatch}"
SMOKE_ROOT="$PWD/smoke_results/delta_qwen_0_5b"
RUN_ID="${RUN_ID:-delta-smoke-$(date -u +%Y%m%dT%H%M%SZ)-${RANDOM}}"
FORCE="${FORCE:-0}"
RUN_DIR="${SMOKE_ROOT}/${RUN_ID}"
FINAL_MARKER="${RUN_DIR}/smoke_validation.complete.json"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing smoke config: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Missing Slurm script: ${SBATCH_SCRIPT}" >&2
  exit 2
fi
CONFIG="$(realpath "${CONFIG}")"
SBATCH_SCRIPT="$(realpath "${SBATCH_SCRIPT}")"
if [[ -e "${FINAL_MARKER}" && "${FORCE}" != "1" ]]; then
  echo "Refusing to overwrite completed smoke run: ${RUN_DIR}" >&2
  echo "Choose a new RUN_ID, or explicitly set FORCE=1." >&2
  exit 2
fi

mkdir -p logs "${SMOKE_ROOT}"
echo "Submitting debug-only Delta smoke test"
echo "account=${account}"
echo "config=${CONFIG}"
echo "dataset=xsum"
echo "model=Qwen/Qwen2.5-0.5B"
echo "protocol=splits=4/4/4 prompt=30 continuation=64 temperature=0.8 top_p=0.95 ratios=0,0.2,0.5 modes=random,tail draws=1 generation_seeds=1 bootstrap_repetitions=5"
echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "resources=partition=gpuA100x4 gpus=1 cpus=8 memory=32G time=00:45:00"
if [[ -d "${RUN_DIR}" ]]; then
  echo "resume=existing incomplete run will be resumed by stable row keys"
fi

submission="$(
  sbatch --parsable \
    --account="${account}" \
    --chdir="${PWD}" \
    --export=ALL,CONFIG="${CONFIG}",RUN_ID="${RUN_ID}",FORCE="${FORCE}" \
    "${SBATCH_SCRIPT}"
)"
job_id="${submission%%;*}"
echo "submitted_job_id=${job_id}"
echo "monitor=squeue -j ${job_id}"
echo "stdout=logs/delta-qwen05-smoke-${job_id}.out"
echo "stderr=logs/delta-qwen05-smoke-${job_id}.err"
