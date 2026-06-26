#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
if [ -n "${SLIME_ROOT:-}" ]; then
  SLIME_ROOT="${SLIME_ROOT}"
  RUNNING_IN_SLIME_CONTAINER=0
elif [ -d /root/slime/scripts/models ]; then
  SLIME_ROOT="/root/slime"
  RUNNING_IN_SLIME_CONTAINER=1
else
  SLIME_ROOT="${REPO_ROOT}/.cache/upstreams/slime"
  RUNNING_IN_SLIME_CONTAINER=0
fi

DAPO_DATA_ROOT="${DAPO_DATA_ROOT:-${REPO_ROOT}/.w8-biayn/slime/dapo-math-17k}"
PROMPT_DATA="${SLIME_PROMPT_DATA:-${DAPO_DATA_ROOT}/dapo-math-17k.jsonl}"
SUMMARY_JSON="${DAPO_SUMMARY_JSON:-${DAPO_DATA_ROOT}/summary.json}"

MODEL_NAME="${SLIME_MODEL_NAME:-Qwen3-4B}"
MODEL_HF_REPO="${SLIME_MODEL_HF_REPO:-Qwen/${MODEL_NAME}}"
MODEL_DIR_NAME="${SLIME_MODEL_DIR_NAME:-${MODEL_NAME}}"
MODEL_ARGS_FILE="${SLIME_MODEL_ARGS_FILE:-qwen3-4B.sh}"
if [ -n "${SLIME_MODEL_DIR:-}" ]; then
  MODEL_DIR="${SLIME_MODEL_DIR}"
elif [ "${RUNNING_IN_SLIME_CONTAINER}" = "1" ]; then
  MODEL_DIR="/root/models/${MODEL_DIR_NAME}"
else
  MODEL_DIR="${HOME}/models/${MODEL_DIR_NAME}"
fi
REF_LOAD_DIR="${SLIME_REF_LOAD_DIR:-${MODEL_DIR}_torch_dist}"
ACTOR_LOAD_DIR="${SLIME_ACTOR_LOAD_DIR:-${REPO_ROOT}/.w8-biayn/slime/multi-agent/checkpoints/${MODEL_DIR_NAME}}"
ACTOR_SAVE_DIR="${SLIME_ACTOR_SAVE_DIR:-${ACTOR_LOAD_DIR}}"

NUM_GPUS="${SLIME_NUM_GPUS:-4}"
ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${SLIME_N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${SLIME_GLOBAL_BATCH_SIZE:-2}"
NUM_ROLLOUT="${SLIME_NUM_ROLLOUT:-16}"
ROLLOUT_MAX_CONTEXT_LEN="${SLIME_MAX_CONTEXT_LEN:-2048}"
ROLLOUT_MAX_RESPONSE_LEN="${SLIME_MAX_RESPONSE_LEN:-512}"
ROLLOUT_TEMPERATURE="${SLIME_ROLLOUT_TEMPERATURE:-1}"
USE_EXTERNAL_RAY="${SLIME_USE_EXTERNAL_RAY:-0}"
SKIP_MODEL_DOWNLOAD="${SLIME_SKIP_MODEL_DOWNLOAD:-0}"
SGLANG_MEM_FRACTION="${SLIME_SGLANG_MEM_FRACTION:-0.65}"
SGLANG_ROLLOUT_GPUS="${SLIME_ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"
EXPERT_MODEL_PARALLEL_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${SLIME_EXPERT_TENSOR_PARALLEL_SIZE:-1}"
MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-2048}"
CONVERT_NPROC="${SLIME_CONVERT_NPROC:-1}"

if [ ! -f "${PROMPT_DATA}" ]; then
  echo "DAPO-Math JSONL is missing. Prepare it first:" >&2
  echo "  uv run python scripts/prepare_dapo_math_dataset.py --out .w8-biayn/slime/dapo-math-17k --limit 32" >&2
  exit 1
fi

if [ ! -d "${SLIME_ROOT}" ]; then
  echo "SLIME upstream checkout is missing at ${SLIME_ROOT}. Run 'uv run w8-biayn upstreams clone slime' first." >&2
  exit 1
fi

if [ ! -f "${SLIME_ROOT}/scripts/models/${MODEL_ARGS_FILE}" ]; then
  echo "Missing SLIME model args file: ${SLIME_ROOT}/scripts/models/${MODEL_ARGS_FILE}" >&2
  exit 1
fi

if ! command -v ray >/dev/null 2>&1; then
  echo "The SLIME runtime is not active: 'ray' is missing from PATH." >&2
  echo "Run this launcher inside the SLIME container started by .w8-biayn/slime/run-container.sh." >&2
  exit 1
fi

mkdir -p "$(dirname -- "${MODEL_DIR}")"
if [ "${SKIP_MODEL_DOWNLOAD}" != "1" ] && [ ! -d "${MODEL_DIR}" ]; then
  hf download "${MODEL_HF_REPO}" --local-dir "${MODEL_DIR}"
fi

if [ "${USE_EXTERNAL_RAY}" = "0" ]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 redis || true
fi

export PYTHONUNBUFFERED=1
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

echo "Using DAPO-Math data from ${PROMPT_DATA}"
if [ -f "${SUMMARY_JSON}" ]; then
  cat "${SUMMARY_JSON}"
fi

source "${SLIME_ROOT}/scripts/models/${MODEL_ARGS_FILE}"
if [ ! -f "${REF_LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
  echo "Converting ${MODEL_DIR} to Megatron torch_dist checkpoint at ${REF_LOAD_DIR}"
  if [ "${CONVERT_NPROC}" = "1" ]; then
    PYTHONPATH=/root/Megatron-LM python3 "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"       ${MODEL_ARGS[@]}       --hf-checkpoint "${MODEL_DIR}"       --save "${REF_LOAD_DIR}"
  else
    PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node "${CONVERT_NPROC}"       "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"       ${MODEL_ARGS[@]}       --hf-checkpoint "${MODEL_DIR}"       --save "${REF_LOAD_DIR}"
  fi
fi
cd "${SLIME_ROOT}"

if [ "${USE_EXTERNAL_RAY}" = "0" ]; then
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  export no_proxy="127.0.0.1,${MASTER_ADDR}"
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

read -r -d '' RUNTIME_ENV_JSON <<EOF || true
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
  }
}
EOF

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${REF_LOAD_DIR}"
  --load "${ACTOR_LOAD_DIR}"
  --save "${ACTOR_SAVE_DIR}"
  --save-interval 20
)

ROLLOUT_ARGS=(
  --custom-generate-function-path examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type deepscaler
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
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
    --wandb-project "${SLIME_WANDB_PROJECT:-slime-multi-agent}"
    --wandb-group "${SLIME_WANDB_GROUP:-${MODEL_NAME}-multi-agent}"
    --disable-wandb-random-suffix
  )
  if [ -n "${WANDB_KEY}" ]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
  if [ -n "${SLIME_WANDB_RUN_ID:-}" ]; then
    WANDB_ARGS+=(--wandb-run-id "${SLIME_WANDB_RUN_ID}")
  fi
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
)
if [ -n "${MAX_TOKENS_PER_GPU}" ]; then
  PERF_ARGS+=(--max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${SGLANG_ROLLOUT_GPUS}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
  --sglang-cuda-graph-bs 1 2 4 8 16
)

MISC_ARGS=(
  --colocate
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
