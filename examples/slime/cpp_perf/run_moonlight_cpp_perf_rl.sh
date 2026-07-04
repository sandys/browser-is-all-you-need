#!/usr/bin/env bash
# PIE C++ performance GRPO launcher for the pinned SLIME sidecar using Moonlight-16B-A3B.
#
# This wrapper keeps the conservative 4-GPU Moonlight/AIME runtime profile but
# swaps the task layer to one-turn C++ optimization with the repo C++ reward
# harness. It does not use the ReTool Python tool trajectory.

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
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROMPT_DATA="${SLIME_PROMPT_DATA:-${REPO_ROOT}/.w8-biayn/data/slime-pie/train.jsonl}"
EVAL_PROMPT_DATA="${SLIME_EVAL_PROMPT_DATA:-/root/aime-2024/aime-2024.jsonl}"
INPUT_KEY="${SLIME_INPUT_KEY:-prompt}"
LABEL_KEY="${SLIME_LABEL_KEY:-label}"
METADATA_KEY="${SLIME_METADATA_KEY:-metadata}"
APPLY_CHAT_TEMPLATE="${SLIME_APPLY_CHAT_TEMPLATE:-1}"
REWARD_KEY="${SLIME_REWARD_KEY:-score}"
CUSTOM_GENERATE_FUNCTION_PATH="${SLIME_CUSTOM_GENERATE_FUNCTION_PATH-}"
CUSTOM_RM_PATH="${SLIME_CUSTOM_RM_PATH-generate_with_cpp_perf.reward_func}"
HF_CHECKPOINT="${SLIME_HF_CHECKPOINT:-/root/models/Moonlight-16B-A3B-Instruct}"
HF_MODEL_ID="${SLIME_HF_MODEL_ID:-moonshotai/Moonlight-16B-A3B-Instruct}"
DOWNLOAD_HF_CHECKPOINT="${SLIME_DOWNLOAD_HF_CHECKPOINT:-1}"
REF_LOAD_DIR="${SLIME_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist}"
RUN_ID="${SLIME_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
ACTOR_LOAD_DIR="${SLIME_ACTOR_LOAD_DIR:-${REF_LOAD_DIR}}"
ACTOR_SAVE_DIR="${SLIME_ACTOR_SAVE_DIR:-${REPO_ROOT}/.w8-biayn/slime/cpp-perf/moonlight-rl/checkpoints/${RUN_ID}}"
RUN_ROOT="${SLIME_RUN_ROOT:-${REPO_ROOT}/.w8-biayn/slime/cpp-perf/moonlight-rl/runs/${RUN_ID}}"

NUM_GPUS="${SLIME_NUM_GPUS:-4}"
TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"
PP_SIZE="${SLIME_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CP_SIZE="${SLIME_CONTEXT_PARALLEL_SIZE:-1}"
EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"
ETP_SIZE="${SLIME_EXPERT_TENSOR_PARALLEL_SIZE:-1}"
MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-4096}"

NUM_ROLLOUT="${SLIME_NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${SLIME_N_SAMPLES_PER_PROMPT:-1}"
MAX_RESPONSE_LEN="${SLIME_MAX_RESPONSE_LEN:-256}"
ROLLOUT_TEMPERATURE="${SLIME_ROLLOUT_TEMPERATURE:-1}"
GLOBAL_BATCH_SIZE="${SLIME_GLOBAL_BATCH_SIZE:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${SLIME_ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}"
SGLANG_MEM_FRACTION_STATIC="${SLIME_SGLANG_MEM_FRACTION:-0.45}"
SGLANG_CUDA_GRAPH_MAX_BS="${SLIME_SGLANG_CUDA_GRAPH_MAX_BS:-16}"
SAVE_INTERVAL="${SLIME_SAVE_INTERVAL:-1000}"

USE_EXTERNAL_RAY="${SLIME_USE_EXTERNAL_RAY:-0}"
SKIP_CLEANUP="${SLIME_SKIP_CLEANUP:-0}"
CONVERT_IF_MISSING="${SLIME_CONVERT_IF_MISSING:-1}"
CONVERT_NPROC="${SLIME_CONVERT_NPROC:-1}"
DISABLE_EVAL="${SLIME_DISABLE_EVAL:-1}"
RAY_MEMORY_USAGE_THRESHOLD="${SLIME_RAY_MEMORY_USAGE_THRESHOLD-0.99}"
RAY_MEMORY_MONITOR_REFRESH_MS="${SLIME_RAY_MEMORY_MONITOR_REFRESH_MS:-}"
HF_CHECKPOINT_WAS_DOWNLOADED=0

absolute_path() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

PROMPT_DATA="$(absolute_path "${PROMPT_DATA}")"
HF_CHECKPOINT="$(absolute_path "${HF_CHECKPOINT}")"
REF_LOAD_DIR="$(absolute_path "${REF_LOAD_DIR}")"
ACTOR_LOAD_DIR="$(absolute_path "${ACTOR_LOAD_DIR}")"
ACTOR_SAVE_DIR="$(absolute_path "${ACTOR_SAVE_DIR}")"
RUN_ROOT="$(absolute_path "${RUN_ROOT}")"
if [ "${DISABLE_EVAL}" != "1" ]; then
  EVAL_PROMPT_DATA="$(absolute_path "${EVAL_PROMPT_DATA}")"
fi

hf_checkpoint_is_present() {
  [ -f "${HF_CHECKPOINT}/config.json" ]
}

download_hf_checkpoint_if_missing() {
  if hf_checkpoint_is_present; then
    return
  fi

  if [ "${DOWNLOAD_HF_CHECKPOINT}" != "1" ]; then
    echo "Missing HF checkpoint: ${HF_CHECKPOINT}" >&2
    echo "Either create it yourself, set SLIME_HF_CHECKPOINT, or enable download with:" >&2
    echo "  SLIME_DOWNLOAD_HF_CHECKPOINT=1 SLIME_HF_MODEL_ID=${HF_MODEL_ID} bash $0" >&2
    exit 2
  fi

  echo "HF checkpoint missing or incomplete: ${HF_CHECKPOINT}"
  echo "Downloading ${HF_MODEL_ID} to ${HF_CHECKPOINT}"
  mkdir -p "$(dirname -- "${HF_CHECKPOINT}")"
  python3 - "${HF_MODEL_ID}" "${HF_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing huggingface_hub. Install examples/slime/cpp_perf/requirements.txt "
        "inside the SLIME container, or set SLIME_DOWNLOAD_HF_CHECKPOINT=0 and "
        "provide SLIME_HF_CHECKPOINT yourself."
    ) from exc

repo_id, local_dir = sys.argv[1], sys.argv[2]
Path(local_dir).mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_dir)
PY
  if ! hf_checkpoint_is_present; then
    echo "Downloaded HF checkpoint is missing config.json: ${HF_CHECKPOINT}" >&2
    exit 2
  fi
  HF_CHECKPOINT_WAS_DOWNLOADED=1
}

REQUIRED_PATHS=("${SLIME_ROOT}" "${MEGATRON_DIR}" "${PROMPT_DATA}")
if [ "${DISABLE_EVAL}" != "1" ]; then
  REQUIRED_PATHS+=("${EVAL_PROMPT_DATA}")
fi

for required_path in "${REQUIRED_PATHS[@]}"; do
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

download_hf_checkpoint_if_missing

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

NOFILE_SOFT_LIMIT="${SLIME_NOFILE_SOFT_LIMIT:-65536}"
if [ -n "${NOFILE_SOFT_LIMIT}" ] && [ "${NOFILE_SOFT_LIMIT}" != "0" ]; then
  ulimit -Sn "${NOFILE_SOFT_LIMIT}" 2>/dev/null || true
fi
echo "NOFILE soft=$(ulimit -Sn) hard=$(ulimit -Hn)"

export PYTHONUNBUFFERED=1
mkdir -p "${ACTOR_LOAD_DIR}" "${ACTOR_SAVE_DIR}" "${RUN_ROOT}"

VRAM_LOG="${RUN_ROOT}/vram_usage.csv"
VRAM_PEAK_FILE="${RUN_ROOT}/vram_peak.txt"
RUN_RECEIPT="${RUN_ROOT}/run_receipt.txt"
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
echo "Using Moonlight C++ Perf prompt data: ${PROMPT_DATA}"
echo "Eval disabled: ${DISABLE_EVAL}"
echo "Ray memory usage threshold: ${RAY_MEMORY_USAGE_THRESHOLD:-Ray default}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_LOAD_DIR}"
  --load "${ACTOR_LOAD_DIR}"
  --save "${ACTOR_SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key "${INPUT_KEY}"
  --label-key "${LABEL_KEY}"
  --metadata-key "${METADATA_KEY}"
  --rollout-shuffle
  --reward-key "${REWARD_KEY}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)
if [ "${APPLY_CHAT_TEMPLATE}" = "1" ]; then
  ROLLOUT_ARGS+=(--apply-chat-template)
fi

EVAL_ARGS=()
if [ "${DISABLE_EVAL}" != "1" ]; then
  EVAL_ARGS=(
    --eval-interval "${SLIME_EVAL_INTERVAL:-20}"
    --eval-prompt-data aime "${EVAL_PROMPT_DATA}"
    --n-samples-per-eval-prompt "${SLIME_N_SAMPLES_PER_EVAL_PROMPT:-16}"
    --eval-max-response-len "${SLIME_EVAL_MAX_RESPONSE_LEN:-16384}"
    --eval-top-p 1
  )
fi

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

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${SLIME_KL_LOSS_COEF:-0.00}"
  --kl-loss-type low_var_kl
  --entropy-coef "${SLIME_ENTROPY_COEF:-0.00}"
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${SLIME_LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

if [ "${SLIME_OPTIMIZER_CPU_OFFLOAD:-1}" = "1" ]; then
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)
fi

if [ -n "${SLIME_OPTIMIZER_EXTRA_ARGS:-}" ]; then
  read -r -a SLIME_OPTIMIZER_EXTRA_ARGS_ARRAY <<<"${SLIME_OPTIMIZER_EXTRA_ARGS}"
  OPTIMIZER_ARGS+=("${SLIME_OPTIMIZER_EXTRA_ARGS_ARRAY[@]}")
fi

WANDB_ARGS=()
WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
WANDB_ALREADY_LOGGED_IN=0
if [ -f "${HOME}/.netrc" ] || [ -f "${HOME}/.config/wandb/settings" ]; then
  WANDB_ALREADY_LOGGED_IN=1
fi
if [ -n "${WANDB_KEY}" ] || [ "${WANDB_ALREADY_LOGGED_IN}" = "1" ] || [ -n "${SLIME_WANDB_PROJECT:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${SLIME_WANDB_PROJECT:-slime-pie-cpp}"
    --wandb-group "${SLIME_WANDB_GROUP:-pie-cpp-rl}"
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
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

if [ -n "${SGLANG_CUDA_GRAPH_MAX_BS}" ]; then
  SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
fi

if [ "${SLIME_SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

if [ "${SLIME_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
fi

if [ -n "${SLIME_SGLANG_EXTRA_ARGS:-}" ]; then
  read -r -a SLIME_SGLANG_EXTRA_ARGS_ARRAY <<<"${SLIME_SGLANG_EXTRA_ARGS}"
  SGLANG_ARGS+=("${SLIME_SGLANG_EXTRA_ARGS_ARRAY[@]}")
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

CUSTOM_ARGS=()
if [ -n "${CUSTOM_GENERATE_FUNCTION_PATH}" ]; then
  CUSTOM_ARGS+=(--custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}")
fi
if [ -n "${CUSTOM_RM_PATH}" ]; then
  CUSTOM_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
fi

EXTRA_ARGS=()
if [ -n "${SLIME_TRAIN_MEMORY_MARGIN_BYTES:-268435456}" ]; then
  EXTRA_ARGS+=(--train-memory-margin-bytes "${SLIME_TRAIN_MEMORY_MARGIN_BYTES:-268435456}")
fi

if [ "${SLIME_SAVE_OPTIM_RNG:-0}" != "1" ]; then
  EXTRA_ARGS+=(--no-save-optim --no-save-rng)
fi

if [ -n "${SLIME_EXTRA_ARGS:-}" ]; then
  read -r -a SLIME_EXTRA_ARGS_ARRAY <<<"${SLIME_EXTRA_ARGS}"
  EXTRA_ARGS+=("${SLIME_EXTRA_ARGS_ARRAY[@]}")
fi

cd "${SLIME_ROOT}"

if [ "${USE_EXTERNAL_RAY}" = "0" ]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export no_proxy="127.0.0.1,${MASTER_ADDR}"
  if [ -n "${RAY_MEMORY_USAGE_THRESHOLD}" ]; then
    export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
  fi
  if [ -n "${RAY_MEMORY_MONITOR_REFRESH_MS}" ]; then
    export RAY_memory_monitor_refresh_ms="${RAY_MEMORY_MONITOR_REFRESH_MS}"
  fi
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="$(
  python3 - <<PY
import json
import os

env = {
    "PYTHONPATH": ":".join(
        p
        for p in (
            "${MEGATRON_DIR}",
            "${SCRIPT_DIR}",
            "${REPO_ROOT}/src",
            "${REPO_ROOT}",
            "${SLIME_ROOT}",
            os.environ.get("PYTHONPATH", ""),
        )
        if p
    ),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NVTE_UNFUSED_ATTN": os.environ.get("SLIME_NVTE_UNFUSED_ATTN", os.environ.get("NVTE_UNFUSED_ATTN", "1")),
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
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "W8_BIAYN_SLIME_TASK_ROOT",
    "SLIME_CPP_SANDBOX_IMAGE",
    "SLIME_CPP_SANDBOX_CPU",
):
    if key in os.environ:
        env[key] = os.environ[key]
print(json.dumps({"env_vars": env}))
PY
)"

LOG_FILE="${RUN_ROOT}/run.log"
RAY_ADDRESS="${SLIME_RAY_ADDRESS:-http://127.0.0.1:8265}"
RAY_JOB_ID="${SLIME_RAY_JOB_ID:-${RUN_ID}}"
RAY_JOB_ID="$(printf "%s" "${RAY_JOB_ID}" | tr -c "A-Za-z0-9_-" "-")"
RAY_JOB_TERMINAL_STATUS="unknown"
RAY_STATUS_STARTED_AT="$(date +%s)"
RAY_STATUS_TIMEOUT_SECONDS="${SLIME_RAY_STATUS_TIMEOUT_SECONDS:-7200}"
echo "Moonlight C++ Perf RL log: ${LOG_FILE}"
echo "Moonlight C++ Perf RL VRAM log: ${VRAM_LOG}"
start_vram_monitor

set +e
ray job submit --address="${RAY_ADDRESS}" \
  --submission-id="${RAY_JOB_ID}" \
  --log-style=record \
  --log-color=false \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
RAY_SUBMIT_STATUS="${PIPESTATUS[0]}"
RAY_STATUS="${RAY_SUBMIT_STATUS}"
if [ "${RAY_SUBMIT_STATUS}" = "0" ]; then
  while true; do
    STATUS_OUTPUT="$(ray job status --address="${RAY_ADDRESS}" --log-style=record --log-color=false "${RAY_JOB_ID}" 2>&1)"
    STATUS_COMMAND_STATUS="$?"
    printf '%s\n' "${STATUS_OUTPUT}" | tee -a "${LOG_FILE}"
    if [ "${STATUS_COMMAND_STATUS}" != "0" ]; then
      RAY_STATUS="${STATUS_COMMAND_STATUS}"
      break
    fi
    RAY_JOB_TERMINAL_STATUS="$(printf '%s\n' "${STATUS_OUTPUT}" | sed -n "s/.*Status for job '.*': \\([A-Z_]*\\).*/\\1/p" | tail -n 1)"
    if [ -z "${RAY_JOB_TERMINAL_STATUS}" ]; then
      if grep -qi "Job '.*' succeeded" <<<"${STATUS_OUTPUT}"; then
        RAY_JOB_TERMINAL_STATUS="SUCCEEDED"
      elif grep -qi "Job '.*' failed" <<<"${STATUS_OUTPUT}"; then
        RAY_JOB_TERMINAL_STATUS="FAILED"
      elif grep -qi "Job '.*' stopped" <<<"${STATUS_OUTPUT}"; then
        RAY_JOB_TERMINAL_STATUS="STOPPED"
      fi
    fi
    case "${RAY_JOB_TERMINAL_STATUS}" in
      SUCCEEDED)
        RAY_STATUS=0
        break
        ;;
      FAILED | STOPPED)
        RAY_STATUS=1
        break
        ;;
      PENDING | RUNNING)
        if [ "${RAY_STATUS_TIMEOUT_SECONDS}" != "0" ]; then
          NOW_SECONDS="$(date +%s)"
          if [ $((NOW_SECONDS - RAY_STATUS_STARTED_AT)) -ge "${RAY_STATUS_TIMEOUT_SECONDS}" ]; then
            echo "Ray job ${RAY_JOB_ID} timed out after ${RAY_STATUS_TIMEOUT_SECONDS}s while ${RAY_JOB_TERMINAL_STATUS}" | tee -a "${LOG_FILE}"
            RAY_JOB_TERMINAL_STATUS="TIMEOUT"
            RAY_STATUS=124
            break
          fi
        fi
        sleep "${SLIME_RAY_STATUS_POLL_SECONDS:-15}"
        ;;
      *)
        RAY_STATUS=1
        break
        ;;
    esac
  done
  ray job logs --address="${RAY_ADDRESS}" --log-style=record --log-color=false "${RAY_JOB_ID}" >>"${LOG_FILE}" 2>&1 || true
fi
set -e

stop_vram_monitor
trap - EXIT

MAX_VRAM_MIB=""
if [ -f "${VRAM_PEAK_FILE}" ]; then
  MAX_VRAM_MIB="$(awk '/max_peak_vram_mib:/ {print $2}' "${VRAM_PEAK_FILE}" | tail -n 1)"
fi

cat >"${RUN_RECEIPT}" <<EOF
status=${RAY_STATUS}
ray_submit_status=${RAY_SUBMIT_STATUS}
ray_job_id=${RAY_JOB_ID}
ray_job_terminal_status=${RAY_JOB_TERMINAL_STATUS}
ray_status_timeout_seconds=${RAY_STATUS_TIMEOUT_SECONDS}
ray_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD:-}
ray_memory_monitor_refresh_ms=${RAY_MEMORY_MONITOR_REFRESH_MS:-}
run_root=${RUN_ROOT}
log_file=${LOG_FILE}
vram_log=${VRAM_LOG}
vram_peak_file=${VRAM_PEAK_FILE}
max_vram_mib=${MAX_VRAM_MIB}
prompt_data=${PROMPT_DATA}
input_key=${INPUT_KEY}
label_key=${LABEL_KEY}
metadata_key=${METADATA_KEY}
apply_chat_template=${APPLY_CHAT_TEMPLATE}
reward_key=${REWARD_KEY}
custom_generate_function_path=${CUSTOM_GENERATE_FUNCTION_PATH}
custom_rm_path=${CUSTOM_RM_PATH}
eval_prompt_data=$([ "${DISABLE_EVAL}" = "1" ] && echo "disabled" || echo "${EVAL_PROMPT_DATA}")
model=${HF_CHECKPOINT}
hf_model_id=${HF_MODEL_ID}
download_hf_checkpoint=${DOWNLOAD_HF_CHECKPOINT}
hf_checkpoint_was_downloaded=${HF_CHECKPOINT_WAS_DOWNLOADED}
ref_load=${REF_LOAD_DIR}
num_gpus=${NUM_GPUS}
rollout_batch_size=${ROLLOUT_BATCH_SIZE}
n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}
global_batch_size=${GLOBAL_BATCH_SIZE}
max_response_len=${MAX_RESPONSE_LEN}
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}
sglang_mem_fraction=${SGLANG_MEM_FRACTION_STATIC}
sglang_cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS}
sglang_disable_custom_all_reduce=${SLIME_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}
optimizer_cpu_offload=${SLIME_OPTIMIZER_CPU_OFFLOAD:-1}
disable_eval=${DISABLE_EVAL}
save_optim_rng=${SLIME_SAVE_OPTIM_RNG:-0}
wandb_project=${SLIME_WANDB_PROJECT:-slime-pie-cpp}
wandb_group=${SLIME_WANDB_GROUP:-pie-cpp-rl}
wandb_run_id=${SLIME_WANDB_RUN_ID:-}
EOF

echo "RUN_ROOT=${RUN_ROOT}"
echo "RUN_RECEIPT=${RUN_RECEIPT}"
exit "${RAY_STATUS}"
