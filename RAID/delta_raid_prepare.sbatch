#!/bin/bash
# CPU-only RAID preparation and persistent index-cache population.
#SBATCH --job-name=raid-prepare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/raid-prepare-%j.out
#SBATCH --error=logs/raid-prepare-%j.err

set -euo pipefail

RAID_RUN_ID="${RAID_RUN_ID:?Set RAID_RUN_ID}"
RAID_DATA_PATH="${RAID_DATA_PATH:?Set RAID_DATA_PATH}"
NUM_SHARDS="${NUM_SHARDS:-4}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/work/hdd/bhuc/${USER}/raid/index-cache}"
REUSE_INDEX_PATH="${REUSE_INDEX_PATH:-}"
ADOPT_PREPARED_RUN_DIR="${ADOPT_PREPARED_RUN_DIR:-}"
LIMIT_SOURCES="${LIMIT_SOURCES:-}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-}"
DEBUG_ONLY="${DEBUG_ONLY:-0}"
EXCLUDE_SOURCE_IDS_PATH="${EXCLUDE_SOURCE_IDS_PATH:-}"
VENV_PATH="${VENV_PATH:-/projects/bhuc/${USER}/venvs/delta-smoke}"

module reset
module load miniforge3-python
source "${VENV_PATH}/bin/activate"
python -c 'import rapidfuzz; print("rapidfuzz=" + rapidfuzz.__version__)'

case "$(readlink -f "${RAID_DATA_PATH}")" in /work/hdd/*) ;; *) exit 2 ;; esac
case "$(readlink -f runs)" in /work/hdd/*) ;; *) exit 2 ;; esac
case "$(readlink -f results)" in /work/hdd/*) ;; *) exit 2 ;; esac

ARGS=(--run-id "${RAID_RUN_ID}" --stage prepare --num-shards "${NUM_SHARDS}")
if [[ -n "${ADOPT_PREPARED_RUN_DIR}" ]]; then
  ARGS+=(--adopt-prepared-run-dir "${ADOPT_PREPARED_RUN_DIR}")
else
  ARGS+=(--data-path "${RAID_DATA_PATH}" --index-cache-dir "${INDEX_CACHE_DIR}")
  if [[ -n "${REUSE_INDEX_PATH}" ]]; then ARGS+=(--reuse-index-path "${REUSE_INDEX_PATH}"); fi
fi
if [[ -n "${LIMIT_SOURCES}" ]]; then ARGS+=(--limit-sources "${LIMIT_SOURCES}"); fi
if [[ -n "${BOOTSTRAP_REPETITIONS}" ]]; then ARGS+=(--bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"); fi
if [[ -n "${EXCLUDE_SOURCE_IDS_PATH}" ]]; then ARGS+=(--exclude-source-ids-path "${EXCLUDE_SOURCE_IDS_PATH}"); fi
if [[ "${DEBUG_ONLY}" == "1" ]]; then ARGS+=(--debug-only); fi

python -u RAID/run_raid.py "${ARGS[@]}"
echo "RAID PREPARE: PASS"
