#!/usr/bin/env bash
# Deliberately overfit a copy of the Synth-v1 rank-16 LoRA to the exact bank task.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

: "${MILES_LORA_ADAPTER_PATH:?Point this at a preserved copy of the Synth-v1 epoch-50 adapter.}"

export MILES_CPP_TASKS_DIR="${MILES_CPP_TASKS_DIR:-${REPO_ROOT}/benchmarks/cpp/bank-account-equivalent-v1}"
export MILES_DATA_BUILD_MODULE="glm47_posttraining.integrations.miles_aider_polyglot"
export MILES_DATA_CURRICULUM="bank-account-official-drill-v1"
export MILES_CPP_AUTO_PREPARE_DATA="1"

# Eight rows make one optimizer update. Sixteen epochs is therefore sixteen
# direct imitation updates, with resumable checkpoints every four updates.
export MILES_ROLLOUT_BATCH_SIZE="${MILES_ROLLOUT_BATCH_SIZE:-8}"
export MILES_GLOBAL_BATCH_SIZE="${MILES_GLOBAL_BATCH_SIZE:-8}"
export MILES_SFT_NUM_EPOCH="${MILES_SFT_NUM_EPOCH:-16}"
export MILES_SAVE_INTERVAL="${MILES_SAVE_INTERVAL:-4}"
export MILES_SEQ_LENGTH="${MILES_SEQ_LENGTH:-6144}"
export MILES_MAX_TOKENS_PER_GPU="${MILES_MAX_TOKENS_PER_GPU:-12288}"
export MILES_LR="${MILES_LR:-1e-5}"
export MILES_NO_REF="1"
export WANDB_TAGS="${WANDB_TAGS:-canonical,gcp,skypilot,aider,bank-account-official-imitation}"

exec "${REPO_ROOT}/examples/sft.sh" "$@"
