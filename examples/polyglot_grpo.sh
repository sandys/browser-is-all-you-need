#!/usr/bin/env bash
# 8x H100 Miles GRPO profile for Aider Polyglot C++ functional correctness.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

export MILES_RUN_ID="${MILES_RUN_ID:-glm47_h100_aider_polyglot_cpp_$(date +%Y%m%d_%H%M%S)}"
export MILES_CPP_TASKS_DIR="${MILES_CPP_TASKS_DIR:-${REPO_ROOT}/.glm47-posttraining/data/aider_tasks}"
export MILES_CPP_DATA_DIR="${MILES_CPP_DATA_DIR:-${REPO_ROOT}/.glm47-posttraining/assets/aider_data}"
export MILES_CPP_TAXONOMY_FILE="${MILES_CPP_TAXONOMY_FILE:-${REPO_ROOT}/data/task_taxonomy.json}"
export MILES_CPP_RUBRIC_RISK_FILE="${MILES_CPP_RUBRIC_RISK_FILE:-${MILES_CPP_TAXONOMY_FILE}}"
export MILES_CPP_DATA_BUILDER_MODULE="glm47_posttraining.integrations.miles_polyglot_cpp"
export MILES_CPP_REWARD_FUNC_PATH="glm47_posttraining.integrations.miles_polyglot_cpp.reward_func"
export GLM47_CPP_SANDBOX_BACKEND="${GLM47_CPP_SANDBOX_BACKEND:-docker}"
export GLM47_CPP_SANDBOX_IMAGE="${GLM47_CPP_SANDBOX_IMAGE:-glm47-cpp-perf:latest}"
export MILES_CPP_SAMPLING_STRATEGY="${MILES_CPP_SAMPLING_STRATEGY:-wire}"
export MILES_CPP_WIRE_BATCH_SIZE="${MILES_CPP_WIRE_BATCH_SIZE:-32}"
export MILES_GRPO_ROLLOUT_SHUFFLE="${MILES_GRPO_ROLLOUT_SHUFFLE:-0}"
export MILES_WANDB_PROJECT="${MILES_WANDB_PROJECT:-glm47-aider-polyglot-cpp}"
export WANDB_TAGS="${WANDB_TAGS:-canonical,aider-polyglot-cpp,grpo}"

exec "${REPO_ROOT}/examples/grpo.sh" "$@"
