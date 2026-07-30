#!/bin/bash
# Submit the configured dataset/model matrix on NCSA Delta.
#
# Set EXPERIMENT_SET to primary, replication, qwen-scaling, or all.
# Set PARTITION=gpuH200x8 explicitly for the 72B or another confirmed large run.
set -euo pipefail

ACCOUNT="${ACCOUNT:?Set ACCOUNT to the Delta Slurm charge account}"
CONFIG="${CONFIG:-configs/paper.json}"
EXPERIMENT_SET="${EXPERIMENT_SET:-primary}"
PARTITION="${PARTITION:-gpuA100x4}"
TIME="${TIME:-24:00:00}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/delta_experiment.sbatch}"
mkdir -p logs

case "${PARTITION}" in
  gpuA100x4)
    GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
    ;;
  gpuH200x8)
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    ;;
  *)
    echo "Unsupported Delta partition: ${PARTITION}" >&2
    echo "Use gpuA100x4 or explicitly opt into gpuH200x8." >&2
    exit 2
    ;;
esac

DATASETS=(xsum squad writingprompts)
PRIMARY_MODELS=(
  ibm-granite/granite-3.3-8b-base
  mistralai/Mistral-Small-24B-Base-2501
  Qwen/Qwen2.5-32B
)
REPLICATION_MODELS=(KoboldAI/GPT-NeoX-20B-Erebus)
QWEN_SCALING_MODELS=(
  Qwen/Qwen2.5-7B
  Qwen/Qwen2.5-14B
  Qwen/Qwen2.5-32B
  Qwen/Qwen2.5-72B
)

case "${EXPERIMENT_SET}" in
  primary)
    MODELS=("${PRIMARY_MODELS[@]}")
    ;;
  replication)
    MODELS=("${REPLICATION_MODELS[@]}")
    ;;
  qwen-scaling)
    MODELS=("${QWEN_SCALING_MODELS[@]}")
    ;;
  all)
    MODELS=(
      "${PRIMARY_MODELS[@]}"
      "${REPLICATION_MODELS[@]}"
      "${QWEN_SCALING_MODELS[@]}"
    )
    ;;
  *)
    echo "Unknown EXPERIMENT_SET=${EXPERIMENT_SET}" >&2
    exit 2
    ;;
esac

declare -A SEEN_MODELS=()
UNIQUE_MODELS=()
for model in "${MODELS[@]}"; do
  if [[ -z "${SEEN_MODELS[${model}]:-}" ]]; then
    UNIQUE_MODELS+=("${model}")
    SEEN_MODELS["${model}"]=1
  fi
done

for dataset in "${DATASETS[@]}"; do
  for model in "${UNIQUE_MODELS[@]}"; do
    if [[ "${model}" == "Qwen/Qwen2.5-72B" && "${PARTITION}" != "gpuH200x8" ]]; then
      echo "Skipping ${model}: submit it explicitly with PARTITION=gpuH200x8." >&2
      continue
    fi
    echo "Submitting dataset=${dataset} model=${model} partition=${PARTITION}"
    sbatch \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --gpus-per-node="${GPUS_PER_NODE}" \
      --time="${TIME}" \
      --export=ALL,CONFIG="${CONFIG}",DATASET="${dataset}",TARGET_MODEL="${model}",TENSOR_PARALLEL_SIZE="${GPUS_PER_NODE}" \
      "${SBATCH_SCRIPT}"
  done
done
