#!/bin/bash
# Submit a Beemo full run after enforcing the Delta storage guard.

set -euo pipefail

ACCOUNT="${ACCOUNT:-bhuc-delta-gpu}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BEEMO_RUN_ID="${BEEMO_RUN_ID:-beemo-output-only-4scorer-full-${STAMP}}"
INCLUDE_GRANITE="${INCLUDE_GRANITE:-0}"

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
  --cpus-per-task=16 \
  --mem=160G \
  --time=24:00:00 \
  --job-name=beemo-4scorer \
  --export=ALL,BEEMO_RUN_ID="${BEEMO_RUN_ID}",INCLUDE_GRANITE="${INCLUDE_GRANITE}" \
  Beemo/delta_beemo.sbatch)"

echo "submitted_job_id=${JOB_ID}"
echo "beemo_run_id=${BEEMO_RUN_ID}"
echo "stdout=logs/beemo-${JOB_ID}.out"
echo "stderr=logs/beemo-${JOB_ID}.err"
