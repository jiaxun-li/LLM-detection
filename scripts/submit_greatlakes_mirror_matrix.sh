#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-stats_dept1}"
N_SAMPLES="${N_SAMPLES:-200}"
TIME="${TIME:-24:00:00}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/greatlakes_mirror_detection.sbatch}"

DATASETS=(
  xsum
  squad
  writingprompts
)

# These mirror the model families in Radvand et al. Section 5.2. The exact
# paper-scale models are large and may require gated access or larger GPUs.
MODEL_SPECS=(
  "llama:meta-llama/Meta-Llama-3-8B:0"
  "gpt_neox_erebus:KoboldAI/GPT-NeoX-20B-Erebus:0"
  "qwen:Qwen/Qwen2.5-32B:0"
)

for dataset in "${DATASETS[@]}"; do
  for spec in "${MODEL_SPECS[@]}"; do
    IFS=":" read -r alias model trust_remote_code <<<"${spec}"
    echo "submitting dataset=${dataset} model_alias=${alias} target_model=${model}"
    sbatch \
      -A "${ACCOUNT}" \
      --time="${TIME}" \
      --export=ALL,DATASET="${dataset}",MODEL_ALIAS="${alias}",TARGET_MODEL="${model}",N_SAMPLES="${N_SAMPLES}",TRUST_REMOTE_CODE="${trust_remote_code}" \
      "${SBATCH_SCRIPT}"
  done
done
