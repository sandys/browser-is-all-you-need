#!/usr/bin/env bash
# Prepare Aider Polyglot C++ data, run preflight verification, then launch GRPO.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
RAW_PRACTICE_DIR="${1:-${REPO_ROOT}/polyglot-benchmark/cpp/exercises/practice}"
TASKS_DIR="${2:-${REPO_ROOT}/.glm47-posttraining/data/aider_tasks}"
TAXONOMY_FILE="${3:-${REPO_ROOT}/data/task_taxonomy.json}"
DATA_DIR="${4:-${REPO_ROOT}/.glm47-posttraining/assets/aider_data}"

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python3 "${REPO_ROOT}/scripts/ingest_exercism_aider.py" \
  "${RAW_PRACTICE_DIR}" "${TASKS_DIR}"

TAXONOMY_ARGS=("${TASKS_DIR}" "${TAXONOMY_FILE}")
if [ -n "${GLM47_TOKENIZER_PATH:-}" ]; then
  TAXONOMY_ARGS+=(--tokenizer "${GLM47_TOKENIZER_PATH}")
fi
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python3 "${REPO_ROOT}/scripts/build_task_taxonomy.py" \
  "${TAXONOMY_ARGS[@]}"

PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python3 \
  -m glm47_posttraining.integrations.miles_polyglot_cpp build-data \
  --tasks-dir "${TASKS_DIR}" \
  --taxonomy-file "${TAXONOMY_FILE}" \
  --out "${DATA_DIR}" \
  --sampling-strategy wire \
  --wire-batch-size "${MILES_ROLLOUT_BATCH_SIZE:-32}" \
  --force

PREFLIGHT_ARGS=(
  --tasks-dir "${TASKS_DIR}"
  --skip-sandbox "${MILES_SKIP_AIDER_SANDBOX_PREFLIGHT:-0}"
  --backend "${GLM47_CPP_SANDBOX_BACKEND:-docker}"
)
if [ "${MILES_BUILD_CPP_SANDBOX:-1}" = "1" ]; then
  PREFLIGHT_ARGS+=(--build-image)
fi
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python3 "${REPO_ROOT}/scripts/check_aider_runtime.py" \
  "${PREFLIGHT_ARGS[@]}"

export MILES_CPP_TASKS_DIR="${TASKS_DIR}"
export MILES_CPP_DATA_DIR="${DATA_DIR}"
export MILES_CPP_TAXONOMY_FILE="${TAXONOMY_FILE}"
export MILES_CPP_RUBRIC_RISK_FILE="${MILES_CPP_RUBRIC_RISK_FILE:-${TAXONOMY_FILE}}"
exec "${REPO_ROOT}/examples/polyglot_grpo.sh"
