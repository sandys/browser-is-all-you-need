#!/usr/bin/env bash
# Rank-16 LoRA wrapper for the Moonlight PIE C++ SLIME lane.

set -euo pipefail

STAGE="${1:-${SLIME_CPP_STAGE:-}}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
BASE_RUNNER="${REPO_ROOT}/examples/slime/moonlight_cpp_perf/moonlight_cpp_perf.sh"

PYTHON_BIN="${SLIME_CPP_PYTHON:-python3}"
if [ -n "${SLIME_ROOT:-}" ]; then
  SLIME_ROOT="${SLIME_ROOT}"
elif [ -d /root/slime/scripts/models ]; then
  SLIME_ROOT="/root/slime"
else
  SLIME_ROOT="${REPO_ROOT}/.cache/upstreams/slime"
fi
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
SLIME_TRAIN_MODULE="${SLIME_TRAIN_MODULE:-w8_biayn.integrations.slime_train_entry}"

export SLIME_RUN_ID="${SLIME_RUN_ID:-moonlight_lora16_cpp_perf}"
export SLIME_RUN_ROOT="${SLIME_RUN_ROOT:-${REPO_ROOT}/.w8-biayn/slime/moonlight-lora-cpp-perf/runs/${SLIME_RUN_ID}}"
export SLIME_CPP_PROFILE="${SLIME_CPP_PROFILE:-moonlight-lora16-cpp-perf}"
export SLIME_WANDB_PROJECT="${SLIME_WANDB_PROJECT:-slime-moonlight-lora-cpp-perf}"
export SLIME_WANDB_GROUP="${SLIME_WANDB_GROUP:-${SLIME_RUN_ID}}"
export SLIME_LORA_ENABLED="${SLIME_LORA_ENABLED:-1}"
export SLIME_LORA_RANK="${SLIME_LORA_RANK:-16}"

stage_uses_lora() {
  case "${STAGE}" in
    sft | sft-eval | grpo | grpo-eval) return 0 ;;
    *) return 1 ;;
  esac
}

append_slime_extra_args() {
  local extra="$1"
  if [ -z "${extra}" ]; then
    return
  fi
  if [ -n "${SLIME_EXTRA_ARGS:-}" ]; then
    export SLIME_EXTRA_ARGS="${SLIME_EXTRA_ARGS} ${extra}"
  else
    export SLIME_EXTRA_ARGS="${extra}"
  fi
}

resolve_lora_args() {
  if [ -z "${STAGE}" ] || ! stage_uses_lora || [ "${SLIME_LORA_ENABLED}" != "1" ]; then
    return
  fi

  if [ -n "${SLIME_LORA_EXTRA_ARGS:-}" ]; then
    export W8_RESOLVED_LORA_ARGS="${SLIME_LORA_EXTRA_ARGS}"
    append_slime_extra_args "${SLIME_LORA_EXTRA_ARGS}"
    return
  fi

  LORA_RESOLVER_ARGS=(
    -m w8_biayn.slime_integration.lora resolve
    --rank "${SLIME_LORA_RANK}"
    --help-command "${PYTHON_BIN} -m ${SLIME_TRAIN_MODULE} --help"
  )
  if [ -n "${SLIME_LORA_ALPHA:-}" ]; then
    LORA_RESOLVER_ARGS+=(--alpha "${SLIME_LORA_ALPHA}")
  fi
  if [ -n "${SLIME_LORA_DROPOUT:-}" ]; then
    LORA_RESOLVER_ARGS+=(--dropout "${SLIME_LORA_DROPOUT}")
  fi
  if [ -n "${SLIME_LORA_TARGET_MODULES:-}" ]; then
    LORA_RESOLVER_ARGS+=(--target-modules "${SLIME_LORA_TARGET_MODULES}")
  fi

  local resolved
  resolved="$(
    PYTHONPATH="${MEGATRON_DIR}:${SLIME_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" "${LORA_RESOLVER_ARGS[@]}"
  )"
  export SLIME_LORA_EXTRA_ARGS="${resolved}"
  export W8_RESOLVED_LORA_ARGS="${resolved}"
  append_slime_extra_args "${resolved}"
}

resolve_lora_args
exec "${BASE_RUNNER}" "$@"
