#!/bin/bash
# Submit a full RAID run after enforcing data and artifact storage guards.

set -euo pipefail

ACCOUNT="${ACCOUNT:-bhuc-delta-gpu}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAID_RUN_ID="${RAID_RUN_ID:-raid-universal-clipping-full-${STAMP}}"
RAID_DATA_PATH="${RAID_DATA_PATH:-/work/hdd/bhuc/${USER}/raid/train.csv}"
RAID_STAGE="${RAID_STAGE:-all}"

if [[ ! -f "${RAID_DATA_PATH}" ]]; then
  echo "RAID data is missing: ${RAID_DATA_PATH}" >&2
  exit 2
fi
case "$(readlink -f "${RAID_DATA_PATH}")" in
  /work/hdd/*) ;;
  *) echo "RAID data must resolve under /work/hdd" >&2; exit 2 ;;
esac
for path in runs results; do
  target="$(readlink -f "${path}")"
  case "${target}" in
    /work/hdd/*) ;;
    *) echo "${path} must resolve under /work/hdd, found ${target}" >&2; exit 2 ;;
  esac
done

JOB_ID="$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --partition=gpuA100x4 \
  --gpus-per-node=2 \
  --cpus-per-task=24 \
  --mem=200G \
  --time=24:00:00 \
  --job-name=raid-clipping \
  --export=ALL,RAID_RUN_ID="${RAID_RUN_ID}",RAID_DATA_PATH="${RAID_DATA_PATH}",RAID_STAGE="${RAID_STAGE}" \
  RAID/delta_raid.sbatch)"

echo "submitted_job_id=${JOB_ID}"
echo "raid_run_id=${RAID_RUN_ID}"
echo "raid_stage=${RAID_STAGE}"
echo "stdout=logs/raid-${JOB_ID}.out"
echo "stderr=logs/raid-${JOB_ID}.err"

