#!/usr/bin/env bash
# Execution-grounded bank-account curriculum profile on the trusted GRPO runner.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

export MILES_CPP_TASKS_DIR="${MILES_CPP_TASKS_DIR:-${REPO_ROOT}/benchmarks/cpp/bank-account-equivalent-v1}"
export MILES_DATA_BUILD_MODULE="glm47_posttraining.integrations.miles_aider_polyglot"
export MILES_DATA_CURRICULUM="bank-account-v1"
export MILES_CUSTOM_RM_PATH="glm47_posttraining.integrations.miles_aider_polyglot.reward_func"
export MILES_REWARD_PREFLIGHT_MODULE="glm47_posttraining.integrations.miles_aider_polyglot"
export MILES_EXPECTED_DATASET_KIND="aider-polyglot-cpp-shadow-grpo"
export MILES_EVAL_NAME="bank_account_rl_validation"
export GLM47_CPP_REWARD_WORKERS="${GLM47_CPP_REWARD_WORKERS:-8}"
export MILES_CPP_INCLUDE_LOGS="${MILES_CPP_INCLUDE_LOGS:-1}"
export WANDB_TAGS="${WANDB_TAGS:-canonical,gcp,skypilot,8xh100,aider,bank-account-v1,grpo}"

exec "${REPO_ROOT}/examples/grpo.sh" "$@"
