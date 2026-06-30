#!/usr/bin/env bash
# Moonlight-16B-A3B SLIME runner for PIE C++ base/SFT/GRPO measurement.

set -euo pipefail

STAGE="${1:-${SLIME_CPP_STAGE:-}}"
if [ -z "${STAGE}" ]; then
  echo "usage: $0 <prepare-data|base-eval|sft|sft-eval|grpo|grpo-eval|compare>" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -n "${SLIME_ROOT:-}" ]; then
  SLIME_ROOT="${SLIME_ROOT}"
elif [ -d /root/slime/scripts/models ]; then
  SLIME_ROOT="/root/slime"
else
  SLIME_ROOT="${REPO_ROOT}/.cache/upstreams/slime"
fi
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
PYTHON_BIN="${SLIME_CPP_PYTHON:-python3}"

RUN_ID="${SLIME_RUN_ID:-moonlight_cpp_perf}"
RUN_ROOT="${SLIME_RUN_ROOT:-${REPO_ROOT}/.w8-biayn/slime/moonlight-cpp-perf/runs/${RUN_ID}}"
DATA_DIR="${SLIME_CPP_DATA_DIR:-${RUN_ROOT}/data}"
TASKS_DIR="${SLIME_CPP_TASKS_DIR:-${REPO_ROOT}/.w8-biayn/data/tasks-full}"
TRAIN_LIMIT="${SLIME_CPP_TRAIN_LIMIT:-16}"
EVAL_LIMIT="${SLIME_CPP_EVAL_LIMIT:-8}"
EVAL_SPLITS="${SLIME_CPP_EVAL_SPLITS:-validation,test}"

HF_CHECKPOINT="${SLIME_HF_CHECKPOINT:-/root/models/Moonlight-16B-A3B-Instruct}"
HF_MODEL_ID="${SLIME_HF_MODEL_ID:-moonshotai/Moonlight-16B-A3B-Instruct}"
DOWNLOAD_HF_CHECKPOINT="${SLIME_DOWNLOAD_HF_CHECKPOINT:-1}"
REF_LOAD_DIR="${SLIME_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist}"
SFT_SAVE_DIR="${SLIME_SFT_SAVE_DIR:-${RUN_ROOT}/checkpoints/sft}"
GRPO_SAVE_DIR="${SLIME_GRPO_SAVE_DIR:-${RUN_ROOT}/checkpoints/grpo}"
CONVERT_IF_MISSING="${SLIME_CONVERT_IF_MISSING:-1}"
CONVERT_NPROC="${SLIME_CONVERT_NPROC:-1}"
HF_CHECKPOINT_WAS_DOWNLOADED=0

NUM_GPUS="${SLIME_NUM_GPUS:-4}"
TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"
PP_SIZE="${SLIME_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CP_SIZE="${SLIME_CONTEXT_PARALLEL_SIZE:-1}"
EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"
ETP_SIZE="${SLIME_EXPERT_TENSOR_PARALLEL_SIZE:-1}"
MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-4096}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${SLIME_ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}"
SGLANG_MEM_FRACTION_STATIC="${SLIME_SGLANG_MEM_FRACTION:-0.45}"
SGLANG_CUDA_GRAPH_MAX_BS="${SLIME_SGLANG_CUDA_GRAPH_MAX_BS:-16}"

GRPO_NUM_ROLLOUT="${SLIME_GRPO_NUM_ROLLOUT:-1}"
GRPO_ROLLOUT_BATCH_SIZE="${SLIME_GRPO_ROLLOUT_BATCH_SIZE:-4}"
GRPO_GLOBAL_BATCH_SIZE="${SLIME_GRPO_GLOBAL_BATCH_SIZE:-4}"
GRPO_N_SAMPLES_PER_PROMPT="${SLIME_GRPO_N_SAMPLES_PER_PROMPT:-1}"
GRPO_MAX_RESPONSE_LEN="${SLIME_GRPO_MAX_RESPONSE_LEN:-512}"
GRPO_TEMPERATURE="${SLIME_GRPO_TEMPERATURE:-1}"

SFT_NUM_EPOCH="${SLIME_SFT_NUM_EPOCH:-1}"
SFT_ROLLOUT_BATCH_SIZE="${SLIME_SFT_ROLLOUT_BATCH_SIZE:-4}"
SFT_GLOBAL_BATCH_SIZE="${SLIME_SFT_GLOBAL_BATCH_SIZE:-4}"
SFT_LR="${SLIME_SFT_LR:-1e-5}"

EVAL_N_SAMPLES_PER_PROMPT="${SLIME_EVAL_N_SAMPLES_PER_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${SLIME_EVAL_MAX_RESPONSE_LEN:-512}"
EVAL_TEMPERATURE="${SLIME_EVAL_TEMPERATURE:-0}"
EVAL_TOP_P="${SLIME_EVAL_TOP_P:-1}"

SAVE_INTERVAL="${SLIME_SAVE_INTERVAL:-1000}"
USE_EXTERNAL_RAY="${SLIME_USE_EXTERNAL_RAY:-0}"
SKIP_CLEANUP="${SLIME_SKIP_CLEANUP:-0}"
RAY_MEMORY_USAGE_THRESHOLD="${SLIME_RAY_MEMORY_USAGE_THRESHOLD-0.99}"
RAY_MEMORY_MONITOR_REFRESH_MS="${SLIME_RAY_MEMORY_MONITOR_REFRESH_MS:-}"

absolute_path() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

RUN_ROOT="$(absolute_path "${RUN_ROOT}")"
DATA_DIR="$(absolute_path "${DATA_DIR}")"
TASKS_DIR="$(absolute_path "${TASKS_DIR}")"
HF_CHECKPOINT="$(absolute_path "${HF_CHECKPOINT}")"
REF_LOAD_DIR="$(absolute_path "${REF_LOAD_DIR}")"
SFT_SAVE_DIR="$(absolute_path "${SFT_SAVE_DIR}")"
GRPO_SAVE_DIR="$(absolute_path "${GRPO_SAVE_DIR}")"

stage_label() {
  case "${STAGE}" in
    base-eval) echo "base" ;;
    sft-eval) echo "sft" ;;
    grpo-eval) echo "grpo" ;;
    *) echo "${STAGE}" ;;
  esac
}

STAGE_LABEL="$(stage_label)"
STAGE_ROOT="${RUN_ROOT}/stages/${STAGE}"
LOG_FILE="${STAGE_ROOT}/run.log"
VRAM_LOG="${STAGE_ROOT}/vram_usage.csv"
VRAM_PEAK_FILE="${STAGE_ROOT}/vram_peak.txt"
RUN_RECEIPT="${STAGE_ROOT}/run_receipt.txt"
ROLLOUT_DUMP_TEMPLATE="${RUN_ROOT}/rollout_dumps/${STAGE_LABEL}_{rollout_id}.pt"
EVAL_DUMP_PATH="${RUN_ROOT}/rollout_dumps/${STAGE_LABEL}_eval_0.pt"

mkdir -p "${STAGE_ROOT}" "${RUN_ROOT}/rollout_dumps" "${RUN_ROOT}/eval"

run_repo_python() {
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" "$@"
}

prepare_data() {
  if [ ! -d "${TASKS_DIR}" ]; then
    echo "Missing task JSON directory: ${TASKS_DIR}" >&2
    echo "Create it first, for example:" >&2
    echo "  uv run w8-biayn data pie download --out .w8-biayn/data/pie" >&2
    echo "  uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force" >&2
    echo "  uv run w8-biayn data pie measure-coverage --prepared-root .w8-biayn/data/pie-full --jobs 8" >&2
    echo "  uv run w8-biayn data pie build-full-tasks --prepared-root .w8-biayn/data/pie-full --out .w8-biayn/data/tasks-full --force" >&2
    exit 2
  fi
  run_repo_python -m w8_biayn.integrations.slime_cpp_perf build-data \
    --tasks-dir "${TASKS_DIR}" \
    --out "${DATA_DIR}" \
    --train-limit "${TRAIN_LIMIT}" \
    --eval-limit "${EVAL_LIMIT}" \
    --eval-splits "${EVAL_SPLITS}" \
    --profile "${SLIME_CPP_PROFILE:-moonlight-cpp-perf}" \
    --run-id "${RUN_ID}" \
    --force
}

ensure_data() {
  if [ ! -f "${DATA_DIR}/manifest.json" ]; then
    if [ "${SLIME_CPP_AUTO_PREPARE_DATA:-1}" != "1" ]; then
      echo "Missing SLIME C++ data manifest: ${DATA_DIR}/manifest.json" >&2
      echo "Run: bash ${SCRIPT_DIR}/prepare_data.sh" >&2
      exit 2
    fi
    prepare_data
  fi
}

hf_checkpoint_is_present() {
  [ -f "${HF_CHECKPOINT}/config.json" ]
}

download_hf_checkpoint_if_missing() {
  if hf_checkpoint_is_present; then
    return
  fi
  if [ "${DOWNLOAD_HF_CHECKPOINT}" != "1" ]; then
    echo "Missing HF checkpoint: ${HF_CHECKPOINT}" >&2
    echo "Set SLIME_HF_CHECKPOINT or rerun with SLIME_DOWNLOAD_HF_CHECKPOINT=1." >&2
    exit 2
  fi
  mkdir -p "$(dirname -- "${HF_CHECKPOINT}")"
  echo "Downloading ${HF_MODEL_ID} to ${HF_CHECKPOINT}"
  "${PYTHON_BIN}" - "${HF_MODEL_ID}" "${HF_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
Path(local_dir).mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_dir)
PY
  HF_CHECKPOINT_WAS_DOWNLOADED=1
}

checkpoint_ready() {
  [ -f "$1/latest_checkpointed_iteration.txt" ]
}

ensure_slime_runtime() {
  for required_path in "${SLIME_ROOT}" "${MEGATRON_DIR}" "${DATA_DIR}/manifest.json"; do
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
    echo "Run inside the SLIME container started by .w8-biayn/slime/run-container.sh." >&2
    exit 1
  fi
}

ensure_base_checkpoint() {
  download_hf_checkpoint_if_missing
  # shellcheck disable=SC1090
  source "${SLIME_ROOT}/scripts/models/moonlight.sh"
  if checkpoint_ready "${REF_LOAD_DIR}"; then
    return
  fi
  if [ "${CONVERT_IF_MISSING}" != "1" ]; then
    echo "Missing Megatron torch_dist checkpoint: ${REF_LOAD_DIR}" >&2
    echo "Set SLIME_REF_LOAD_DIR or rerun with SLIME_CONVERT_IF_MISSING=1." >&2
    exit 2
  fi
  echo "Converting ${HF_CHECKPOINT} to ${REF_LOAD_DIR}"
  if [ "${CONVERT_NPROC}" = "1" ]; then
    PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
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
}

require_checkpoint() {
  local checkpoint_dir="$1"
  local name="$2"
  if ! checkpoint_ready "${checkpoint_dir}"; then
    echo "Missing ${name} Megatron checkpoint: ${checkpoint_dir}" >&2
    echo "Expected latest_checkpointed_iteration.txt in that directory." >&2
    exit 2
  fi
}

cleanup_ray() {
  if [ "${SKIP_CLEANUP}" != "1" ] && [ "${USE_EXTERNAL_RAY}" = "0" ]; then
    pkill -9 sglang || true
    sleep 3
    ray stop --force || true
    pkill -9 ray || true
    pkill -9 redis || true
    sleep 3
  fi
}

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
    "${PYTHON_BIN}" - <<PY
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

base_model_args() {
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
    --lr "${SLIME_LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
  )
  if [ "${SLIME_OPTIMIZER_CPU_OFFLOAD:-1}" = "1" ]; then
    OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)
  fi
  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  )
  if [ -n "${SGLANG_CUDA_GRAPH_MAX_BS}" ]; then
    SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
  fi
  if [ "${SLIME_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-1}" = "1" ]; then
    SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
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
  EXTRA_ARGS=(
    --train-memory-margin-bytes "${SLIME_TRAIN_MEMORY_MARGIN_BYTES:-268435456}"
    --no-save-optim
    --no-save-rng
    --save-debug-rollout-data "${ROLLOUT_DUMP_TEMPLATE}"
  )
  if [ -n "${SLIME_EXTRA_ARGS:-}" ]; then
    read -r -a SLIME_EXTRA_ARGS_ARRAY <<<"${SLIME_EXTRA_ARGS}"
    EXTRA_ARGS+=("${SLIME_EXTRA_ARGS_ARRAY[@]}")
  fi
}

wandb_args() {
  WANDB_ARGS=()
  WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"
  WANDB_ALREADY_LOGGED_IN=0
  if [ -f "${HOME}/.netrc" ] || [ -f "${HOME}/.config/wandb/settings" ]; then
    WANDB_ALREADY_LOGGED_IN=1
  fi
  if [ -n "${WANDB_KEY}" ] || [ "${WANDB_ALREADY_LOGGED_IN}" = "1" ] || [ -n "${SLIME_WANDB_PROJECT:-}" ]; then
    WANDB_ARGS=(
      --use-wandb
      --wandb-project "${SLIME_WANDB_PROJECT:-slime-moonlight-cpp-perf}"
      --wandb-group "${SLIME_WANDB_GROUP:-${RUN_ID}}"
      --wandb-run-id "${SLIME_WANDB_RUN_ID:-${RUN_ID}-${STAGE}}"
      --disable-wandb-random-suffix
    )
    if [ -n "${WANDB_KEY}" ]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
    fi
  fi
}

stage_args() {
  CKPT_ARGS=()
  TASK_ARGS=()
  ALGO_ARGS=()
  CUSTOM_ARGS=()
  case "${STAGE}" in
    base-eval)
      CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_LOAD_DIR}" --load "${REF_LOAD_DIR}" --save "${RUN_ROOT}/checkpoints/base-eval" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --prompt-data "${DATA_DIR}/grpo/train.jsonl"
        --input-key prompt
        --label-key label
        --metadata-key metadata
        --apply-chat-template
        --reward-key score
        --num-rollout 0
        --rollout-batch-size "${GRPO_ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt 1
        --global-batch-size "${GRPO_GLOBAL_BATCH_SIZE}"
        --eval-interval 1
        --eval-prompt-data cpp_perf "${DATA_DIR}/eval/validation.jsonl"
        --n-samples-per-eval-prompt "${EVAL_N_SAMPLES_PER_PROMPT}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-temperature "${EVAL_TEMPERATURE}"
        --eval-top-p "${EVAL_TOP_P}"
      )
      CUSTOM_ARGS=(--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func)
      ;;
    sft)
      CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_LOAD_DIR}" --load "${REF_LOAD_DIR}" --save "${SFT_SAVE_DIR}" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --rollout-function-path slime.rollout.sft_rollout.generate_rollout
        --prompt-data "${DATA_DIR}/sft/train.jsonl"
        --input-key messages
        --metadata-key metadata
        --rollout-shuffle
        --num-epoch "${SFT_NUM_EPOCH}"
        --rollout-batch-size "${SFT_ROLLOUT_BATCH_SIZE}"
        --global-batch-size "${SFT_GLOBAL_BATCH_SIZE}"
        --loss-type sft_loss
        --calculate-per-token-loss
        --disable-compute-advantages-and-returns
        --debug-train-only
      )
      OPTIMIZER_ARGS=(
        --optimizer adam
        --lr "${SFT_LR}"
        --lr-decay-style cosine
        --min-lr "${SLIME_SFT_MIN_LR:-1e-6}"
        --lr-warmup-fraction "${SLIME_SFT_LR_WARMUP_FRACTION:-0.1}"
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.95
      )
      ;;
    sft-eval)
      require_checkpoint "${SFT_SAVE_DIR}" "SFT"
      CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${SFT_SAVE_DIR}" --load "${SFT_SAVE_DIR}" --save "${RUN_ROOT}/checkpoints/sft-eval" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --prompt-data "${DATA_DIR}/grpo/train.jsonl"
        --input-key prompt
        --label-key label
        --metadata-key metadata
        --apply-chat-template
        --reward-key score
        --num-rollout 0
        --rollout-batch-size "${GRPO_ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt 1
        --global-batch-size "${GRPO_GLOBAL_BATCH_SIZE}"
        --eval-interval 1
        --eval-prompt-data cpp_perf "${DATA_DIR}/eval/validation.jsonl"
        --n-samples-per-eval-prompt "${EVAL_N_SAMPLES_PER_PROMPT}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-temperature "${EVAL_TEMPERATURE}"
        --eval-top-p "${EVAL_TOP_P}"
      )
      CUSTOM_ARGS=(--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func)
      ;;
    grpo)
      require_checkpoint "${SFT_SAVE_DIR}" "SFT"
      CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${SLIME_GRPO_REF_LOAD_DIR:-${SFT_SAVE_DIR}}" --load "${SLIME_GRPO_ACTOR_LOAD_DIR:-${SFT_SAVE_DIR}}" --save "${GRPO_SAVE_DIR}" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --prompt-data "${DATA_DIR}/grpo/train.jsonl"
        --input-key prompt
        --label-key label
        --metadata-key metadata
        --apply-chat-template
        --rollout-shuffle
        --reward-key score
        --num-rollout "${GRPO_NUM_ROLLOUT}"
        --rollout-batch-size "${GRPO_ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt "${GRPO_N_SAMPLES_PER_PROMPT}"
        --rollout-max-response-len "${GRPO_MAX_RESPONSE_LEN}"
        --rollout-temperature "${GRPO_TEMPERATURE}"
        --global-batch-size "${GRPO_GLOBAL_BATCH_SIZE}"
        --balance-data
      )
      ALGO_ARGS=(
        --advantage-estimator grpo
        --use-kl-loss
        --kl-loss-coef "${SLIME_KL_LOSS_COEF:-0.00}"
        --kl-loss-type low_var_kl
        --entropy-coef "${SLIME_ENTROPY_COEF:-0.00}"
        --eps-clip 0.2
        --eps-clip-high 0.28
      )
      CUSTOM_ARGS=(--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func)
      ;;
    grpo-eval)
      require_checkpoint "${GRPO_SAVE_DIR}" "GRPO"
      CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${GRPO_SAVE_DIR}" --load "${GRPO_SAVE_DIR}" --save "${RUN_ROOT}/checkpoints/grpo-eval" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --prompt-data "${DATA_DIR}/grpo/train.jsonl"
        --input-key prompt
        --label-key label
        --metadata-key metadata
        --apply-chat-template
        --reward-key score
        --num-rollout 0
        --rollout-batch-size "${GRPO_ROLLOUT_BATCH_SIZE}"
        --n-samples-per-prompt 1
        --global-batch-size "${GRPO_GLOBAL_BATCH_SIZE}"
        --eval-interval 1
        --eval-prompt-data cpp_perf "${DATA_DIR}/eval/validation.jsonl"
        --n-samples-per-eval-prompt "${EVAL_N_SAMPLES_PER_PROMPT}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-temperature "${EVAL_TEMPERATURE}"
        --eval-top-p "${EVAL_TOP_P}"
      )
      CUSTOM_ARGS=(--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func)
      ;;
    *)
      echo "stage ${STAGE} does not submit a SLIME job" >&2
      exit 2
      ;;
  esac
}

start_ray_if_needed() {
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
}

runtime_env_json() {
  "${PYTHON_BIN}" - <<PY
import json
import os

paths = ["${MEGATRON_DIR}", "${REPO_ROOT}/src", "${SLIME_ROOT}", os.environ.get("PYTHONPATH", "")]
env = {
    "PYTHONPATH": ":".join(path for path in paths if path),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NVTE_UNFUSED_ATTN": os.environ.get("SLIME_NVTE_UNFUSED_ATTN", os.environ.get("NVTE_UNFUSED_ATTN", "1")),
    "W8_BIAYN_DATA_DIR": "${DATA_DIR}",
    "W8_CPP_SANDBOX_IMAGE": os.environ.get("W8_CPP_SANDBOX_IMAGE", "w8-biayn/cpp-sandbox:latest"),
    "W8_CPP_SANDBOX_CPU": os.environ.get("W8_CPP_SANDBOX_CPU", "3"),
    "W8_SLIME_CPP_INCLUDE_LOGS": os.environ.get("W8_SLIME_CPP_INCLUDE_LOGS", "0"),
}
for key in (
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "HF_HOME",
    "WANDB_API_KEY",
    "WANDB_KEY",
    "DOCKER_HOST",
):
    if key in os.environ:
        env[key] = os.environ[key]
print(json.dumps({"env_vars": env}))
PY
}

write_receipt() {
  local status="$1"
  local ray_submit_status="$2"
  local ray_job_id="$3"
  local ray_job_terminal_status="$4"
  local max_vram_mib=""
  if [ -f "${VRAM_PEAK_FILE}" ]; then
    max_vram_mib="$(awk '/max_peak_vram_mib:/ {print $2}' "${VRAM_PEAK_FILE}" | tail -n 1)"
  fi
  cat >"${RUN_RECEIPT}" <<EOF
status=${status}
stage=${STAGE}
stage_label=${STAGE_LABEL}
ray_submit_status=${ray_submit_status}
ray_job_id=${ray_job_id}
ray_job_terminal_status=${ray_job_terminal_status}
ray_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD:-}
run_id=${RUN_ID}
run_root=${RUN_ROOT}
stage_root=${STAGE_ROOT}
log_file=${LOG_FILE}
vram_log=${VRAM_LOG}
vram_peak_file=${VRAM_PEAK_FILE}
max_vram_mib=${max_vram_mib}
data_dir=${DATA_DIR}
tasks_dir=${TASKS_DIR}
hf_checkpoint=${HF_CHECKPOINT}
hf_model_id=${HF_MODEL_ID}
download_hf_checkpoint=${DOWNLOAD_HF_CHECKPOINT}
hf_checkpoint_was_downloaded=${HF_CHECKPOINT_WAS_DOWNLOADED}
ref_load=${REF_LOAD_DIR}
sft_save_dir=${SFT_SAVE_DIR}
grpo_save_dir=${GRPO_SAVE_DIR}
num_gpus=${NUM_GPUS}
tp_size=${TP_SIZE}
ep_size=${EP_SIZE}
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}
sglang_mem_fraction=${SGLANG_MEM_FRACTION_STATIC}
sglang_cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS}
wandb_project=${SLIME_WANDB_PROJECT:-slime-moonlight-cpp-perf}
wandb_group=${SLIME_WANDB_GROUP:-${RUN_ID}}
wandb_run_id=${SLIME_WANDB_RUN_ID:-${RUN_ID}-${STAGE}}
rollout_dump_template=${ROLLOUT_DUMP_TEMPLATE}
eval_dump_path=${EVAL_DUMP_PATH}
EOF
}

aggregate_eval() {
  if [ ! -f "${EVAL_DUMP_PATH}" ]; then
    echo "Missing eval debug rollout dump: ${EVAL_DUMP_PATH}" >&2
    exit 2
  fi
  run_repo_python -m w8_biayn.integrations.slime_cpp_perf aggregate-debug \
    --label "${STAGE_LABEL}" \
    --debug-rollout "${EVAL_DUMP_PATH}" \
    --out "${RUN_ROOT}/eval"
}

compare_eval() {
  run_repo_python -m w8_biayn.integrations.slime_cpp_perf compare \
    --summary "${RUN_ROOT}/eval/base.summary.json" \
    --summary "${RUN_ROOT}/eval/sft.summary.json" \
    --summary "${RUN_ROOT}/eval/grpo.summary.json" \
    --out "${RUN_ROOT}/eval/comparison.json"
}

submit_slime_job() {
  ensure_data
  ensure_slime_runtime
  ensure_base_checkpoint
  base_model_args
  wandb_args
  stage_args

  if [ "${SLIME_TRACE:-1}" = "1" ]; then
    set -x
  fi
  ulimit -Sn "${SLIME_NOFILE_SOFT_LIMIT:-65536}" 2>/dev/null || true
  export PYTHONUNBUFFERED=1
  cleanup_ray

  TOPO_OUTPUT="$(nvidia-smi topo -m 2>/dev/null || true)"
  NVLINK_COUNT="$(grep -o 'NV[0-9][0-9]*' <<<"${TOPO_OUTPUT}" | wc -l | tr -d ' ' || true)"
  if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
  else
    HAS_NVLINK=0
  fi
  start_ray_if_needed
  RUNTIME_ENV_JSON="$(runtime_env_json)"
  cd "${SLIME_ROOT}"

  RAY_ADDRESS="${SLIME_RAY_ADDRESS:-http://127.0.0.1:8265}"
  RAY_JOB_ID="${SLIME_RAY_JOB_ID:-${RUN_ID}-${STAGE}}"
  RAY_JOB_ID="$(printf "%s" "${RAY_JOB_ID}" | tr -c "A-Za-z0-9_-" "-")"
  RAY_JOB_TERMINAL_STATUS="unknown"
  RAY_STATUS_STARTED_AT="$(date +%s)"
  RAY_STATUS_TIMEOUT_SECONDS="${SLIME_RAY_STATUS_TIMEOUT_SECONDS:-7200}"

  echo "Moonlight C++ SLIME stage: ${STAGE}"
  echo "Run root: ${RUN_ROOT}"
  echo "Log: ${LOG_FILE}"
  echo "VRAM log: ${VRAM_LOG}"
  start_vram_monitor
  trap stop_vram_monitor EXIT

  set +e
  ray job submit --address="${RAY_ADDRESS}" \
    --submission-id="${RAY_JOB_ID}" \
    --log-style=record \
    --log-color=false \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" -u "${SLIME_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${TASK_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
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
  write_receipt "${RAY_STATUS}" "${RAY_SUBMIT_STATUS}" "${RAY_JOB_ID}" "${RAY_JOB_TERMINAL_STATUS}"
  echo "RUN_RECEIPT=${RUN_RECEIPT}"
  if [ "${RAY_STATUS}" = "0" ]; then
    case "${STAGE}" in
      base-eval | sft-eval | grpo-eval)
        aggregate_eval
        ;;
    esac
  fi
  exit "${RAY_STATUS}"
}

case "${STAGE}" in
  prepare-data)
    prepare_data
    ;;
  compare)
    compare_eval
    ;;
  base-eval | sft | sft-eval | grpo | grpo-eval)
    submit_slime_job
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
