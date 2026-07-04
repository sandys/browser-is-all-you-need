#!/usr/bin/env bash
# Tiny Moonlight-16B-A3B INT4 MoE smoke for the pinned SLIME sidecar.
#
# The defaults intentionally avoid E2B, browser sandboxes, and DAPO downloads.
# Optional W&B diagnostics follow the same auto-detect pattern as the other
# repo-owned SLIME launchers.

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
PROMPT_DATA="${SLIME_PROMPT_DATA:-${REPO_ROOT}/examples/slime/moonlight_moe_smoke/moonlight_math_smoke.jsonl}"
HF_CHECKPOINT="${SLIME_HF_CHECKPOINT:-/root/Moonlight-16B-A3B-Instruct}"
REF_LOAD_DIR="${SLIME_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist}"
ACTOR_LOAD_DIR="${SLIME_ACTOR_LOAD_DIR:-${REPO_ROOT}/.w8-biayn/slime/moonlight-16b-a3b-int4-smoke/checkpoints}"
ACTOR_SAVE_DIR="${SLIME_ACTOR_SAVE_DIR:-${ACTOR_LOAD_DIR}}"
RUN_ROOT="${SLIME_RUN_ROOT:-${REPO_ROOT}/.w8-biayn/slime/moonlight-16b-a3b-int4-smoke/runs/$(date +%Y%m%d_%H%M%S)}"

NUM_GPUS="${SLIME_NUM_GPUS:-4}"
TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"
PP_SIZE="${SLIME_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CP_SIZE="${SLIME_CONTEXT_PARALLEL_SIZE:-1}"
EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"
ETP_SIZE="${SLIME_EXPERT_TENSOR_PARALLEL_SIZE:-1}"

NUM_ROLLOUT="${SLIME_NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${SLIME_N_SAMPLES_PER_PROMPT:-1}"
NUM_STEPS_PER_ROLLOUT="${SLIME_NUM_STEPS_PER_ROLLOUT:-1}"
MAX_RESPONSE_LEN="${SLIME_MAX_RESPONSE_LEN:-128}"
MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-1024}"
ROLLOUT_TEMPERATURE="${SLIME_ROLLOUT_TEMPERATURE:-0.2}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${SLIME_ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}"
SGLANG_MEM_FRACTION_STATIC="${SLIME_SGLANG_MEM_FRACTION:-0.35}"
SGLANG_MAX_RUNNING_REQUESTS="${SLIME_SGLANG_MAX_RUNNING_REQUESTS:-4}"

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
mkdir -p "${ACTOR_LOAD_DIR}" "${ACTOR_SAVE_DIR}" "${RUN_ROOT}"

VRAM_LOG="${RUN_ROOT}/vram_usage.csv"
VRAM_PEAK_FILE="${RUN_ROOT}/vram_peak.txt"
VRAM_MONITOR_PID=""

start_vram_monitor() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; VRAM monitor disabled" >&2
    return
  fi
  echo "timestamp,index,name,memory.used [MiB],memory.total [MiB]" >"${VRAM_LOG}"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,name,memory.used,memory.total \
        --format=csv,noheader,nounits >>"${VRAM_LOG}" 2>/dev/null || true
      sleep "${SLIME_VRAM_POLL_SECONDS:-2}"
    done
  ) &
  VRAM_MONITOR_PID="$!"
}

stop_vram_monitor() {
  if [ -n "${VRAM_MONITOR_PID}" ]; then
    kill "${VRAM_MONITOR_PID}" 2>/dev/null || true
    wait "${VRAM_MONITOR_PID}" 2>/dev/null || true
    VRAM_MONITOR_PID=""
  fi
  if [ -f "${VRAM_LOG}" ]; then
    python3 - <<PY
import csv
from pathlib import Path

log_path = Path("${VRAM_LOG}")
peak_path = Path("${VRAM_PEAK_FILE}")
peaks = {}
with log_path.open(newline="") as handle:
    reader = csv.reader(handle)
    next(reader, None)
    for row in reader:
        if len(row) < 5:
            continue
        index = row[1].strip()
        name = row[2].strip()
        try:
            used = int(row[3].strip())
            total = int(row[4].strip())
        except ValueError:
            continue
        current = peaks.get(index)
        if current is None or used > current["used"]:
            peaks[index] = {"name": name, "used": used, "total": total}

lines = ["peak_vram_mib_by_gpu:"]
overall = 0
for index in sorted(peaks, key=lambda value: int(value) if value.isdigit() else value):
    item = peaks[index]
    overall = max(overall, item["used"])
    lines.append(f"gpu{index} {item['name']}: {item['used']} / {item['total']} MiB")
lines.append(f"max_peak_vram_mib: {overall}")
peak_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(peak_path.read_text(encoding="utf-8"), end="")
PY
  fi
}

trap stop_vram_monitor EXIT

TOPO_OUTPUT="$(nvidia-smi topo -m 2>/dev/null || true)"
NVLINK_COUNT="$(grep -o 'NV[0-9][0-9]*' <<<"${TOPO_OUTPUT}" | wc -l | tr -d ' ' || true)"
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} detected_nvlink_refs=${NVLINK_COUNT}"
echo "Using Moonlight prompt data: ${PROMPT_DATA}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD_DIR}"
  --load "${ACTOR_LOAD_DIR}"
  --save "${ACTOR_SAVE_DIR}"
  --save-interval 1000000
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rm-type math
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
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
  --log-probs-chunk-size 256
  --train-memory-margin-bytes 268435456
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
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
    --wandb-project "${SLIME_WANDB_PROJECT:-slime-moonlight-smoke}"
    --wandb-group "${SLIME_WANDB_GROUP:-moonlight-int4-moe-smoke}"
    --disable-wandb-random-suffix
  )
  if [ -n "${WANDB_KEY}" ]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
  if [ -n "${SLIME_WANDB_RUN_ID:-}" ]; then
    WANDB_ARGS+=(--wandb-run-id "${SLIME_WANDB_RUN_ID}")
  fi
fi

SGLANG_ARGS=(
  --rollout-num-gpus "${NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-cuda-graph-bs 1 2 4
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --num-gpus-per-node "${NUM_GPUS}"
  --colocate
  --ci-test
)

if [ -n "${SLIME_ATTENTION_BACKEND:-}" ]; then
  MISC_ARGS+=(--attention-backend "${SLIME_ATTENTION_BACKEND}")
fi

if [ "${SLIME_ENABLE_DEEPEP:-0}" = "1" ]; then
  MISC_ARGS+=(--moe-enable-deepep --moe-token-dispatcher-type flex)
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
        p for p in ("${MEGATRON_DIR}", "${SLIME_ROOT}", os.environ.get("PYTHONPATH", "")) if p
    ),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NVSHMEM_DISABLE_NCCL": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
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
echo "Moonlight INT4 MoE smoke log: ${LOG_FILE}"
echo "Moonlight INT4 MoE VRAM log: ${VRAM_LOG}"
start_vram_monitor

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u train.py \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "VRAM_PEAK_FILE=${VRAM_PEAK_FILE}"
