#!/usr/bin/env bash
# Convert GLM-4.7-Flash HF weights for the 8x H100 Miles TP4/PP1/EP8 layout.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
MILES_ROOT="${MILES_ROOT:-/root/miles}"

PYTHON_BIN="${MILES_PYTHON:-python3}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
MODEL_ARGS_FILE="${MILES_MODEL_ARGS_FILE:-glm4.7-flash.sh}"
MODEL_ARGS_PATH="${MILES_MODEL_ARGS_PATH:-${MILES_ROOT}/scripts/models/${MODEL_ARGS_FILE}}"
HF_CHECKPOINT="${MILES_HF_CHECKPOINT:-/root/models/GLM-4.7-Flash}"
REF_LOAD_DIR="${MILES_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist_tp4_pp1_ep8}"
TP_SIZE="${MILES_TENSOR_MODEL_PARALLEL_SIZE:-4}"
PP_SIZE="${MILES_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
EP_SIZE="${MILES_EXPERT_MODEL_PARALLEL_SIZE:-8}"
ETP_SIZE="${MILES_EXPERT_TENSOR_PARALLEL_SIZE:-1}"
CONVERT_NPROC="${MILES_CONVERT_NPROC:-8}"

if [ ! -d "${MILES_ROOT}" ]; then
  echo "Missing Miles root: ${MILES_ROOT}" >&2
  exit 2
fi
if [ ! -f "${MODEL_ARGS_PATH}" ]; then
  echo "Missing model args: ${MODEL_ARGS_PATH}" >&2
  exit 2
fi
if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
  echo "Missing HF checkpoint: ${HF_CHECKPOINT}" >&2
  exit 2
fi
if [ -f "${REF_LOAD_DIR}/latest_checkpointed_iteration.txt" ] && [ "${MILES_CONVERT_FORCE:-0}" != "1" ]; then
  echo "Megatron checkpoint already exists: ${REF_LOAD_DIR}"
  exit 0
fi

cd "${MILES_ROOT}"
source "${MODEL_ARGS_PATH}"

# Keep grouped-GEMM naming aligned between conversion and training.
STRIP_GROUPED_GEMM="${GLM47_STRIP_MOE_GROUPED_GEMM:-0}"
CONVERT_MODEL_ARGS=()
for arg in "${MODEL_ARGS[@]}"; do
  if [ "${STRIP_GROUPED_GEMM}" = "1" ] && [ "${arg}" = "--moe-grouped-gemm" ]; then
    continue
  fi
  CONVERT_MODEL_ARGS+=("${arg}")
done
CONVERT_MODEL_ARGS+=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
)
if [ -n "${MILES_DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]; then
  CONVERT_MODEL_ARGS+=(--decoder-last-pipeline-num-layers "${MILES_DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi

echo "repo_root=${REPO_ROOT}"
echo "miles_root=${MILES_ROOT}"
echo "hf_checkpoint=${HF_CHECKPOINT}"
echo "ref_load=${REF_LOAD_DIR}"
echo "tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE} etp=${ETP_SIZE}"

# The converter must see the GLM-4.7 bridge (stock mbridge cannot map
# Glm4MoeLite); run it through the bridge-registering wrapper with the repo
# src on PYTHONPATH, mirroring how training uses miles_train_with_glm47_bridge.
# GLM47_KEEP_PP1 stops the converter from repurposing PP for conversion
# parallelism so the checkpoint is written in the exact training layout.
export MILES_CONVERT_PY="${MILES_ROOT}/tools/convert_hf_to_torch_dist.py"
export GLM47_KEEP_PP1="${GLM47_KEEP_PP1:-1}"
CONVERT_PYTHONPATH="${REPO_ROOT}/src:${MEGATRON_DIR}:${PYTHONPATH:-}"
if [ "${CONVERT_NPROC}" = "1" ]; then
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
    PYTHONPATH="${CONVERT_PYTHONPATH}" \
    "${PYTHON_BIN}" -m glm47_posttraining.integrations.miles_convert_with_glm47_bridge \
    "${CONVERT_MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${REF_LOAD_DIR}"
else
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
    PYTHONPATH="${CONVERT_PYTHONPATH}" \
    torchrun --nproc-per-node "${CONVERT_NPROC}" \
    -m glm47_posttraining.integrations.miles_convert_with_glm47_bridge \
    "${CONVERT_MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${REF_LOAD_DIR}"
fi
