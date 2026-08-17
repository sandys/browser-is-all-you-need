#!/usr/bin/env bash
# SkyPilot's fail-closed SFT/GRPO dispatcher for one 8xH100 node.

set -euo pipefail

: "${GLM47_STAGE:?Pass GLM47_STAGE=sft or grpo with sky launch --env.}"
: "${MILES_RUN_ID:?Pass a unique MILES_RUN_ID with sky launch --env.}"
: "${GLM47_SOURCE_COMMIT:?Pass the synced source commit with sky launch --env.}"
: "${GLM47_RUNTIME_IMAGE:?Pass the immutable docker:...@sha256:... image with sky launch --env.}"

case "${GLM47_STAGE}" in
  sft|grpo) ;;
  *)
    echo "GLM47_STAGE must be sft or grpo, got: ${GLM47_STAGE}" >&2
    exit 2
    ;;
esac

if [[ ! "${MILES_RUN_ID}" =~ ^[a-z0-9][a-z0-9._-]{2,79}$ ]]; then
  echo "MILES_RUN_ID must be 3-80 lowercase letters, digits, dots, underscores, or hyphens." >&2
  exit 2
fi
if [[ ! "${GLM47_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GLM47_SOURCE_COMMIT must be a full 40-character Git commit SHA." >&2
  exit 2
fi
if [[ ! "${GLM47_RUNTIME_IMAGE}" =~ ^docker:ghcr\.io/.+@sha256:[0-9a-f]{64}$ ]]; then
  echo "GLM47_RUNTIME_IMAGE must be an immutable docker:ghcr.io/...@sha256:... reference." >&2
  exit 2
fi

RUN_ROOT="/workspace/runs/${MILES_RUN_ID}"
if [ -e "${RUN_ROOT}" ]; then
  echo "Refusing to reuse existing run directory: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

export MILES_RUN_ROOT="${RUN_ROOT}"
export MILES_CPP_DATA_DIR="${RUN_ROOT}/data"
export GLM47_EXPERIMENT_ID="${MILES_RUN_ID}"
export MILES_WANDB_PROJECT="glm47-pie-cpp-posttraining"
export MILES_WANDB_GROUP="${MILES_RUN_ID}"
export MILES_WANDB_RUN_ID="${MILES_RUN_ID}"
export MILES_WANDB_JOB_TYPE="${GLM47_STAGE}"
export WANDB_RUN_GROUP="${MILES_RUN_ID}"
export WANDB_JOB_TYPE="${GLM47_STAGE}"
export WANDB_TAGS="canonical,gcp,skypilot,8xh100,pie-cpp,${GLM47_STAGE}"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LAUNCH_RECEIPT="${RUN_ROOT}/skypilot_launch_receipt.txt"

write_launch_receipt() {
  local status="$1"
  local exit_code="$2"
  cat >"${LAUNCH_RECEIPT}" <<EOF
status=${status}
exit_code=${exit_code}
stage=${GLM47_STAGE}
sft_profile=${GLM47_SFT_PROFILE:-default}
grpo_profile=${GLM47_GRPO_PROFILE:-default}
run_id=${MILES_RUN_ID}
run_root=${RUN_ROOT}
source_commit=${GLM47_SOURCE_COMMIT}
runtime_image=${GLM47_RUNTIME_IMAGE}
model_revision=${GLM47_MODEL_REVISION}
started_at=${STARTED_AT}
updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
hostname=$(hostname)
EOF
}

write_launch_receipt running 0

set +e
case "${GLM47_STAGE}" in
  sft)
    case "${GLM47_SFT_PROFILE:-}" in
      "")
        export MILES_SEQ_LENGTH="${MILES_SEQ_LENGTH:-3072}"
        export MILES_GLOBAL_BATCH_SIZE="${MILES_GLOBAL_BATCH_SIZE:-20}"
        export MILES_ROLLOUT_BATCH_SIZE="${MILES_ROLLOUT_BATCH_SIZE:-20}"
        bash examples/sft.sh
        STAGE_STATUS=$?
        ;;
      bank-account-official-imitation)
        bash examples/sft_bank_account_official_drill.sh
        STAGE_STATUS=$?
        ;;
      *)
        echo "Unsupported GLM47_SFT_PROFILE: ${GLM47_SFT_PROFILE}" >&2
        STAGE_STATUS=2
        ;;
    esac
    ;;
  grpo)
    export MILES_LORA_ADAPTER_PATH="${MILES_LORA_ADAPTER_PATH:-${GLM47_GRPO_ADAPTER_PATH}}"
    if [ ! -d "${MILES_LORA_ADAPTER_PATH}" ]; then
      echo "Missing GRPO warm-start adapter: ${MILES_LORA_ADAPTER_PATH}" >&2
      STAGE_STATUS=2
    else
      case "${GLM47_GRPO_PROFILE:-}" in
        "")
          bash examples/grpo.sh
          STAGE_STATUS=$?
          ;;
        bank-account-v1)
          bash examples/grpo_bank_account.sh
          STAGE_STATUS=$?
          ;;
        bank-account-official-drill-v1)
          bash examples/grpo_bank_account_official_drill.sh
          STAGE_STATUS=$?
          ;;
        *)
          echo "Unsupported GLM47_GRPO_PROFILE: ${GLM47_GRPO_PROFILE}" >&2
          STAGE_STATUS=2
          ;;
      esac
    fi
    ;;
esac
set -e

if [ "${STAGE_STATUS}" -eq 0 ]; then
  write_launch_receipt success "${STAGE_STATUS}"
else
  write_launch_receipt failed "${STAGE_STATUS}"
fi
exit "${STAGE_STATUS}"
