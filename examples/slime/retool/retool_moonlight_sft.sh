#!/usr/bin/env bash
# ReTool SFT launcher for the pinned SLIME sidecar using Moonlight-16B-A3B.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
if [ -n "${SLIME_ROOT:-}" ]; then
  SLIME_ROOT="${SLIME_ROOT}"
elif [ -d /root/slime/scripts/models ]; then
  SLIME_ROOT="/root/slime"
else
  SLIME_ROOT="${REPO_ROOT}/.cache/upstreams/slime"
fi

MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
PROMPT_DATA="${SLIME_SFT_PROMPT_DATA:-${REPO_ROOT}/data/retool/ReTool-SFT.parquet}"
HF_CHECKPOINT="${SLIME_HF_CHECKPOINT:-/root/Moonlight-16B-A3B-Instruct}"
REF_LOAD_DIR="${SLIME_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist}"
ACTOR_LOAD_DIR="${SLIME_ACTOR_LOAD_DIR:-}"
ACTOR_SAVE_DIR="${SLIME_ACTOR_SAVE_DIR:-${REPO_ROOT}/.w8-biayn/slime/retool/moonlight-sft/checkpoints}"
RUN_ROOT="${SLIME_RUN_ROOT:-${REPO_ROOT}/.w8-biayn/slime/retool/moonlight-sft/runs/$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${SLIME_NUM_GPUS:-4}"
TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"
PP_SIZE="${SLIME_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CP_SIZE="${SLIME_CONTEXT_PARALLEL_SIZE:-1}"
EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"
ETP_SIZE="${SLIME_EXPERT_TENSOR_PARALLEL_SIZE:-1}"
MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-9216}"
SFT_NUM_EPOCH="${SLIME_SFT_NUM_EPOCH:-3}"
SFT_ROLLOUT_BATCH_SIZE="${SLIME_SFT_ROLLOUT_BATCH_SIZE:-128}"
SFT_GLOBAL_BATCH_SIZE="${SLIME_SFT_GLOBAL_BATCH_SIZE:-128}"
SAVE_INTERVAL="${SLIME_SAVE_INTERVAL:-1000}"

USE_EXTERNAL_RAY="${SLIME_USE_EXTERNAL_RAY:-0}"
SKIP_CLEANUP="${SLIME_SKIP_CLEANUP:-0}"
CONVERT_IF_MISSING="${SLIME_CONVERT_IF_MISSING:-0}"
CONVERT_NPROC="${SLIME_CONVERT_NPROC:-1}"

for required_path in "${SLIME_ROOT}" "${MEGATRON_DIR}" "${PROMPT_DATA}" "${HF_CHECKPOINT}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done

if [ ! -f "${SLIME_ROOT}/scripts/models/moonlight.sh" ]; then
  echo "Missing Moonlight model args: ${SLIME_ROOT}/scripts/models/moonlight.sh" >&2
  echo "Refresh the pinned SLIME checkout with: uv run w8-biayn upstreams clone slime" >&2
  exit 2
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "The SLIME runtime is not active: 'ray' is missing from PATH." >&2
  echo "Run this launcher inside the SLIME container started by .w8-biayn/slime/run-container.sh." >&2
  exit 1
fi

source "${SLIME_ROOT}/scripts/models/moonlight.sh"

if [ ! -f "${REF_LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
  if [ "${CONVERT_IF_MISSING}" != "1" ]; then
    echo "Missing Megatron torch_dist checkpoint: ${REF_LOAD_DIR}" >&2
    echo "Set SLIME_REF_LOAD_DIR to an existing converted checkpoint, or rerun with:" >&2
    echo "  SLIME_CONVERT_IF_MISSING=1 SLIME_CONVERT_NPROC=${CONVERT_NPROC} bash $0" >&2
    exit 2
  fi
  echo "Converting ${HF_CHECKPOINT} to Megatron torch_dist checkpoint at ${REF_LOAD_DIR}"
  if [ "${CONVERT_NPROC}" = "1" ]; then
    PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" \
      python3 "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --save "${REF_LOAD_DIR}"
  else
    PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" \
      torchrun --nproc-per-node "${CONVERT_NPROC}" \
      "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --save "${REF_LOAD_DIR}"
  fi
fi

if [ "${SKIP_CLEANUP}" != "1" ] && [ "${USE_EXTERNAL_RAY}" = "0" ]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 redis || true
  sleep 3
fi

if [ "${SLIME_TRACE:-1}" = "1" ]; then
  set -x
fi

export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
mkdir -p "${ACTOR_SAVE_DIR}" "${RUN_ROOT}"

TOPO_OUTPUT="$(nvidia-smi topo -m 2>/dev/null || true)"
NVLINK_COUNT="$(grep -o 'NV[0-9][0-9]*' <<<"${TOPO_OUTPUT}" | wc -l | tr -d ' ' || true)"
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} detected_nvlink_refs=${NVLINK_COUNT}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD_DIR}"
  --save "${ACTOR_SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)
if [ -n "${ACTOR_LOAD_DIR}" ]; then
  CKPT_ARGS+=(--load "${ACTOR_LOAD_DIR}")
fi

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${PROMPT_DATA}"
  --input-key messages
  --rollout-shuffle
  --num-epoch "${SFT_NUM_EPOCH}"
  --rollout-batch-size "${SFT_ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${SFT_GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${SLIME_LR:-1e-5}"
  --lr-decay-style cosine
  --min-lr "${SLIME_MIN_LR:-1e-6}"
  --lr-warmup-fraction "${SLIME_LR_WARMUP_FRACTION:-0.1}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

WANDB_ARGS=()
WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
WANDB_ALREADY_LOGGED_IN=0
if [ -f "${HOME}/.netrc" ] || [ -f "${HOME}/.config/wandb/settings" ]; then
  WANDB_ALREADY_LOGGED_IN=1
fi
if [ -n "${WANDB_KEY}" ] || [ "${WANDB_ALREADY_LOGGED_IN}" = "1" ] || [ -n "${SLIME_WANDB_PROJECT:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${SLIME_WANDB_PROJECT:-slime-retool}"
    --wandb-group "${SLIME_WANDB_GROUP:-moonlight-sft}"
    --disable-wandb-random-suffix
  )
  if [ -n "${WANDB_KEY}" ]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
  if [ -n "${SLIME_WANDB_RUN_ID:-}" ]; then
    WANDB_ARGS+=(--wandb-run-id "${SLIME_WANDB_RUN_ID}")
  fi
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
)

if [ -n "${SLIME_ATTENTION_BACKEND:-}" ]; then
  MISC_ARGS+=(--attention-backend "${SLIME_ATTENTION_BACKEND}")
fi

cd "${SLIME_ROOT}"

if [ "${USE_EXTERNAL_RAY}" = "0" ]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export no_proxy="127.0.0.1,${MASTER_ADDR}"
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="$(
  python3 - <<PY
import json
import os

env = {
    "PYTHONPATH": ":".join(
        p for p in ("${MEGATRON_DIR}", "${SCRIPT_DIR}", "${SLIME_ROOT}", os.environ.get("PYTHONPATH", "")) if p
    ),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
for key in (
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "HF_HOME",
    "NVTE_DEBUG",
    "NVTE_DEBUG_LEVEL",
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
):
    if key in os.environ:
        env[key] = os.environ[key]
print(json.dumps({"env_vars": env}))
PY
)"

LOG_FILE="${RUN_ROOT}/run.log"
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
