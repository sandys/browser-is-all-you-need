#!/usr/bin/env bash
# Exact official fixed26 bank-account drill environment; training knobs stay in grpo.sh.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# The official curriculum is generated from pinned in-code assets; this existing
# directory only satisfies the generic build-data CLI's source-root argument.
export MILES_CPP_TASKS_DIR="${MILES_CPP_TASKS_DIR:-${REPO_ROOT}/benchmarks/cpp/bank-account-equivalent-v1}"
export MILES_DATA_BUILD_MODULE="glm47_posttraining.integrations.miles_aider_polyglot"
export MILES_DATA_CURRICULUM="bank-account-official-drill-v1"
export MILES_CUSTOM_RM_PATH="glm47_posttraining.integrations.miles_aider_polyglot.reward_func"
export MILES_REWARD_PREFLIGHT_MODULE="glm47_posttraining.integrations.miles_aider_polyglot"
export MILES_EXPECTED_DATASET_KIND="aider-polyglot-cpp-shadow-grpo"
export MILES_EXPECTED_TRAIN_COUNT="8"
export MILES_EVAL_NAME="bank_account_official_drill_monitor"
export GLM47_CPP_REWARD_WORKERS="${GLM47_CPP_REWARD_WORKERS:-8}"
export MILES_CPP_INCLUDE_LOGS="${MILES_CPP_INCLUDE_LOGS:-1}"
export WANDB_TAGS="${WANDB_TAGS:-canonical,gcp,skypilot,aider,bank-account-official-drill}"

exec "${REPO_ROOT}/examples/grpo.sh" "$@"
