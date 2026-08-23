#!/bin/bash
# Submit the dependency-safe RAID prepare -> score arrays -> finalize workflow.

set -euo pipefail

GPU_ACCOUNT="${GPU_ACCOUNT:-${ACCOUNT:-bhuc-delta-gpu}}"
CPU_ACCOUNT="${CPU_ACCOUNT:-${GPU_ACCOUNT%-gpu}-cpu}"
CPU_PARTITION="${CPU_PARTITION:-cpu}"
GPU_PARTITION="${GPU_PARTITION:-gpuA100x4}"
NUM_SHARDS="${NUM_SHARDS:-4}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAID_RUN_ID="${RAID_RUN_ID:-raid-universal-clipping-full-${STAMP}}"
RAID_DATA_PATH="${RAID_DATA_PATH:-/work/hdd/bhuc/${USER}/raid/train.csv}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/work/hdd/bhuc/${USER}/raid/index-cache}"
REUSE_INDEX_PATH="${REUSE_INDEX_PATH:-}"
ADOPT_PREPARED_RUN_DIR="${ADOPT_PREPARED_RUN_DIR:-}"
LIMIT_SOURCES="${LIMIT_SOURCES:-}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-}"
DEBUG_ONLY="${DEBUG_ONLY:-0}"

if [[ ! -f "${RAID_DATA_PATH}" ]]; then
  echo "RAID data is missing: ${RAID_DATA_PATH}" >&2
  exit 2
fi
case "$(readlink -f "${RAID_DATA_PATH}")" in /work/hdd/*) ;; *) exit 2 ;; esac
for path in runs results; do
  case "$(readlink -f "${path}")" in /work/hdd/*) ;; *) exit 2 ;; esac
done
if [[ "${NUM_SHARDS}" -lt 1 ]]; then echo "NUM_SHARDS must be positive" >&2; exit 2; fi

mkdir -p logs "${INDEX_CACHE_DIR}"
EXPORTS="ALL,RAID_RUN_ID=${RAID_RUN_ID},RAID_DATA_PATH=${RAID_DATA_PATH},NUM_SHARDS=${NUM_SHARDS},INDEX_CACHE_DIR=${INDEX_CACHE_DIR},REUSE_INDEX_PATH=${REUSE_INDEX_PATH},ADOPT_PREPARED_RUN_DIR=${ADOPT_PREPARED_RUN_DIR},LIMIT_SOURCES=${LIMIT_SOURCES},BOOTSTRAP_REPETITIONS=${BOOTSTRAP_REPETITIONS},DEBUG_ONLY=${DEBUG_ONLY}"

PREP_JOB_ID="$(sbatch --parsable \
  --account="${CPU_ACCOUNT}" --partition="${CPU_PARTITION}" \
  --export="${EXPORTS}" RAID/delta_raid_prepare.sbatch)"

ARRAY_RANGE="0-$((NUM_SHARDS - 1))"
FALCON_JOB_ID="$(sbatch --parsable \
  --account="${GPU_ACCOUNT}" --partition="${GPU_PARTITION}" --gpus-per-node=1 \
  --dependency="afterok:${PREP_JOB_ID}" --array="${ARRAY_RANGE}" \
  --export="${EXPORTS}" RAID/delta_raid_falcon_shard.sbatch)"

BINOCULARS_JOB_ID="$(sbatch --parsable \
  --account="${GPU_ACCOUNT}" --partition="${GPU_PARTITION}" --gpus-per-node=2 \
  --dependency="afterok:${PREP_JOB_ID}" --array="${ARRAY_RANGE}" \
  --export="${EXPORTS}" RAID/delta_raid_binoculars_shard.sbatch)"

FINALIZE_JOB_ID="$(sbatch --parsable \
  --account="${CPU_ACCOUNT}" --partition="${CPU_PARTITION}" \
  --dependency="afterok:${FALCON_JOB_ID}:${BINOCULARS_JOB_ID}" \
  --export="${EXPORTS}" RAID/delta_raid_finalize.sbatch)"

STATE_FILE="/work/hdd/bhuc/${USER}/raid/last_workflow.env"
{
  printf 'export RAID_RUN_ID=%q\n' "${RAID_RUN_ID}"
  printf 'export RAID_DATA_PATH=%q\n' "${RAID_DATA_PATH}"
  printf 'export NUM_SHARDS=%q\n' "${NUM_SHARDS}"
  printf 'export ADOPT_PREPARED_RUN_DIR=%q\n' "${ADOPT_PREPARED_RUN_DIR}"
  printf 'export PREP_JOB_ID=%q\n' "${PREP_JOB_ID}"
  printf 'export FALCON_JOB_ID=%q\n' "${FALCON_JOB_ID}"
  printf 'export BINOCULARS_JOB_ID=%q\n' "${BINOCULARS_JOB_ID}"
  printf 'export FINALIZE_JOB_ID=%q\n' "${FINALIZE_JOB_ID}"
} > "${STATE_FILE}"

echo "raid_run_id=${RAID_RUN_ID}"
echo "prepare_job_id=${PREP_JOB_ID}"
echo "falcon_array_job_id=${FALCON_JOB_ID}"
echo "binoculars_array_job_id=${BINOCULARS_JOB_ID}"
echo "finalize_job_id=${FINALIZE_JOB_ID}"
echo "saved_state=${STATE_FILE}"
