#!/usr/bin/env bash
# GLM-4.7-Flash SLIME runner for PIE C++ base/SFT/GRPO measurement.

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
SLIME_TRAIN_MODULE="${SLIME_TRAIN_MODULE:-w8_biayn.integrations.slime_train_entry}"

RUN_ID="${SLIME_RUN_ID:-glm47_cpp_perf}"
RUN_ROOT="${SLIME_RUN_ROOT:-${REPO_ROOT}/.w8-biayn/slime/glm47-cpp-perf/runs/${RUN_ID}}"
DATA_DIR="${SLIME_CPP_DATA_DIR:-${RUN_ROOT}/data}"
TASKS_DIR="${SLIME_CPP_TASKS_DIR:-${REPO_ROOT}/.w8-biayn/data/tasks-full}"
TRAIN_LIMIT="${SLIME_CPP_TRAIN_LIMIT:-2}"
EVAL_LIMIT="${SLIME_CPP_EVAL_LIMIT:-4}"
EVAL_SPLITS="${SLIME_CPP_EVAL_SPLITS:-validation,test}"
SORT_BY_SIZE="${SLIME_CPP_SORT_BY_SIZE:-1}"

MODEL_ARGS_SCRIPT="${SLIME_MODEL_ARGS_SCRIPT:-glm4.7-30B-A3B.sh}"
HF_CHECKPOINT="${SLIME_HF_CHECKPOINT:-/root/models/GLM-4.7-Flash}"
HF_MODEL_ID="${SLIME_HF_MODEL_ID:-zai-org/GLM-4.7-Flash}"
DOWNLOAD_HF_CHECKPOINT="${SLIME_DOWNLOAD_HF_CHECKPOINT:-1}"
REF_LOAD_DIR="${SLIME_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist}"
SFT_SAVE_DIR="${SLIME_SFT_SAVE_DIR:-${RUN_ROOT}/checkpoints/sft}"
GRPO_SAVE_DIR="${SLIME_GRPO_SAVE_DIR:-${RUN_ROOT}/checkpoints/grpo}"
SAVE_HF_EXPORTS="${SLIME_SAVE_HF_EXPORTS:-1}"
INLINE_SAVE_HF_EXPORTS="${SLIME_INLINE_SAVE_HF_EXPORTS:-0}"
STANDALONE_HF_EXPORTS="${SLIME_STANDALONE_HF_EXPORTS:-1}"
SFT_HF_SAVE_TEMPLATE="${SLIME_SFT_HF_SAVE_TEMPLATE:-${RUN_ROOT}/hf/sft/rollout_{rollout_id}}"
GRPO_HF_SAVE_TEMPLATE="${SLIME_GRPO_HF_SAVE_TEMPLATE:-${RUN_ROOT}/hf/grpo/rollout_{rollout_id}}"
SFT_HF_CHECKPOINT="${SLIME_SFT_HF_CHECKPOINT:-}"
GRPO_HF_CHECKPOINT="${SLIME_GRPO_HF_CHECKPOINT:-}"
CONVERT_IF_MISSING="${SLIME_CONVERT_IF_MISSING:-1}"
CONVERT_NPROC="${SLIME_CONVERT_NPROC:-8}"
HF_CHECKPOINT_WAS_DOWNLOADED=0

NUM_GPUS="${SLIME_NUM_GPUS:-8}"
TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"
PP_SIZE="${SLIME_PIPELINE_MODEL_PARALLEL_SIZE:-2}"
CP_SIZE="${SLIME_CONTEXT_PARALLEL_SIZE:-2}"
# Megatron requires ETP*EP*PP to divide world_size: with the default
# ETP1/PP2 on 8 GPUs, EP must be 4 (EP8 needs 16 ranks and fails init).
EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"
ETP_SIZE="${SLIME_EXPERT_TENSOR_PARALLEL_SIZE:-1}"
DECODER_LAST_PIPELINE_NUM_LAYERS="${SLIME_DECODER_LAST_PIPELINE_NUM_LAYERS:-23}"
MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-8192}"
MICRO_BATCH_SIZE="${SLIME_MICRO_BATCH_SIZE:-1}"
# Prompt (visible tests + v0 program) plus a complete reasoning+C++ response
# must fit one training sample; 2048 truncated 100% of eval generations.
SEQ_LENGTH="${SLIME_SEQ_LENGTH:-4096}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${SLIME_ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}"
SGLANG_MEM_FRACTION_STATIC="${SLIME_SGLANG_MEM_FRACTION:-0.70}"
SGLANG_CUDA_GRAPH_MAX_BS="${SLIME_SGLANG_CUDA_GRAPH_MAX_BS:-16}"
SGLANG_MAX_RUNNING_REQUESTS="${SLIME_SGLANG_MAX_RUNNING_REQUESTS:-64}"
SGLANG_DP_SIZE="${SLIME_SGLANG_DP_SIZE:-${NUM_GPUS}}"
SGLANG_MOE_DENSE_TP_SIZE="${SLIME_SGLANG_MOE_DENSE_TP_SIZE:-1}"
SGLANG_ENABLE_DP_ATTENTION="${SLIME_SGLANG_ENABLE_DP_ATTENTION:-1}"
SGLANG_ENABLE_DP_LM_HEAD="${SLIME_SGLANG_ENABLE_DP_LM_HEAD:-1}"
SGLANG_SPECULATIVE="${SLIME_SGLANG_SPECULATIVE:-1}"
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SLIME_SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK:-1}"
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK="${SLIME_SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-0}"
NVSHMEM_DISABLE_NCCL="${SLIME_NVSHMEM_DISABLE_NCCL:-${NVSHMEM_DISABLE_NCCL:-1}}"
UPDATE_WEIGHT_BUFFER_SIZE="${SLIME_UPDATE_WEIGHT_BUFFER_SIZE:-}"
ATTENTION_BACKEND="${SLIME_ATTENTION_BACKEND:-flash}"
SEQUENCE_PARALLEL="${SLIME_SEQUENCE_PARALLEL:-1}"
MOE_GROUPED_GEMM="${SLIME_MOE_GROUPED_GEMM:-1}"
USE_DYNAMIC_BATCH_SIZE="${SLIME_USE_DYNAMIC_BATCH_SIZE:-1}"
GLM_MOE_TOKEN_DISPATCHER_TYPE="${SLIME_GLM_MOE_TOKEN_DISPATCHER_TYPE:-flex}"
GLM_ENABLE_DEEPEP="${SLIME_GLM_ENABLE_DEEPEP:-0}"
GLM_ENABLE_MTP_TRAINING="${SLIME_GLM_ENABLE_MTP_TRAINING:-1}"
GLM_MTP_NUM_LAYERS="${SLIME_GLM_MTP_NUM_LAYERS:-1}"
GLM_MTP_LOSS_SCALING_FACTOR="${SLIME_GLM_MTP_LOSS_SCALING_FACTOR:-0.2}"

GRPO_NUM_ROLLOUT="${SLIME_GRPO_NUM_ROLLOUT:-1}"
GRPO_ROLLOUT_BATCH_SIZE="${SLIME_GRPO_ROLLOUT_BATCH_SIZE:-2}"
GRPO_GLOBAL_BATCH_SIZE="${SLIME_GRPO_GLOBAL_BATCH_SIZE:-2}"
GRPO_N_SAMPLES_PER_PROMPT="${SLIME_GRPO_N_SAMPLES_PER_PROMPT:-1}"
GRPO_MAX_RESPONSE_LEN="${SLIME_GRPO_MAX_RESPONSE_LEN:-2048}"
GRPO_TEMPERATURE="${SLIME_GRPO_TEMPERATURE:-1}"
GRPO_SKIP_WEIGHT_UPDATE="${SLIME_GRPO_SKIP_WEIGHT_UPDATE:-1}"
GRPO_LOAD_WEIGHTS_ONLY="${SLIME_GRPO_LOAD_WEIGHTS_ONLY:-1}"
# The converted torch_dist base checkpoint is weights-only and saved under a
# different parallel layout; loading optimizer/RNG object shards across that
# resharding crashes Megatron dist-ckpt (BytesIO has no len()).
SFT_LOAD_WEIGHTS_ONLY="${SLIME_SFT_LOAD_WEIGHTS_ONLY:-1}"
GRPO_SKIP_FINAL_TRAIN_SLEEP="${SLIME_GRPO_SKIP_FINAL_TRAIN_SLEEP:-1}"
FINAL_TRAIN_SLEEP_SKIP="${SLIME_FINAL_TRAIN_SLEEP_SKIP:-0}"

SFT_NUM_EPOCH="${SLIME_SFT_NUM_EPOCH:-1}"
SFT_ROLLOUT_BATCH_SIZE="${SLIME_SFT_ROLLOUT_BATCH_SIZE:-2}"
SFT_GLOBAL_BATCH_SIZE="${SLIME_SFT_GLOBAL_BATCH_SIZE:-2}"
SFT_LR="${SLIME_SFT_LR:-1e-5}"
SFT_SKIP_FINAL_TRAIN_SLEEP="${SLIME_SFT_SKIP_FINAL_TRAIN_SLEEP:-1}"
SFT_START_ROLLOUT_ID="${SLIME_SFT_START_ROLLOUT_ID:-0}"
GRPO_START_ROLLOUT_ID="${SLIME_GRPO_START_ROLLOUT_ID:-0}"

EVAL_N_SAMPLES_PER_PROMPT="${SLIME_EVAL_N_SAMPLES_PER_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${SLIME_EVAL_MAX_RESPONSE_LEN:-2048}"
EVAL_TEMPERATURE="${SLIME_EVAL_TEMPERATURE:-0}"
EVAL_TOP_P="${SLIME_EVAL_TOP_P:-1}"

SAVE_INTERVAL="${SLIME_SAVE_INTERVAL:-1000}"
EVAL_LR_DECAY_ITERS="${SLIME_EVAL_LR_DECAY_ITERS:-1}"
DISTRIBUTED_TIMEOUT_MINUTES="${SLIME_DISTRIBUTED_TIMEOUT_MINUTES:-90}"
MEGATRON_TO_HF_MODE="${SLIME_MEGATRON_TO_HF_MODE:-raw}"
USE_EXTERNAL_RAY="${SLIME_USE_EXTERNAL_RAY:-0}"
SKIP_CLEANUP="${SLIME_SKIP_CLEANUP:-0}"
RAY_MEMORY_USAGE_THRESHOLD="${SLIME_RAY_MEMORY_USAGE_THRESHOLD-0.999}"
RAY_MEMORY_MONITOR_REFRESH_MS="${SLIME_RAY_MEMORY_MONITOR_REFRESH_MS:-}"
WANDB_DIR_ROOT="${SLIME_WANDB_DIR:-${RUN_ROOT}/wandb}"
COLOCATE="${SLIME_COLOCATE:-1}"
SFT_COLOCATE="${SLIME_SFT_COLOCATE:-0}"
if [ -n "${SLIME_OPTIMIZER_CPU_OFFLOAD:-}" ]; then
  OPTIMIZER_CPU_OFFLOAD="${SLIME_OPTIMIZER_CPU_OFFLOAD}"
else
  case "${STAGE}" in
    sft | grpo) OPTIMIZER_CPU_OFFLOAD=1 ;;
    *) OPTIMIZER_CPU_OFFLOAD=0 ;;
  esac
fi
HF_EXPORT_CHUNK_SIZE="${SLIME_HF_EXPORT_CHUNK_SIZE:-5368709120}"
HF_EXPORT_FORCE="${SLIME_HF_EXPORT_FORCE:-1}"
HF_EXPORT_ADD_MISSING_FROM_ORIGIN="${SLIME_HF_EXPORT_ADD_MISSING_FROM_ORIGIN:-1}"
HF_EXPORT_MODEL_NAME="${SLIME_HF_EXPORT_MODEL_NAME:-}"
HF_EXPORT_VOCAB_SIZE="${SLIME_HF_EXPORT_VOCAB_SIZE:-}"
HF_EXPORT_LOG=""
HF_EXPORT_STATUS=""

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
WANDB_DIR_ROOT="$(absolute_path "${WANDB_DIR_ROOT}")"

format_rollout_template() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys

print(sys.argv[1].format(rollout_id=sys.argv[2]))
PY
}

if [ -z "${SFT_HF_CHECKPOINT}" ]; then
  SFT_HF_CHECKPOINT="$(format_rollout_template "${SFT_HF_SAVE_TEMPLATE}" "${SLIME_SFT_EVAL_ROLLOUT_ID:-${SFT_START_ROLLOUT_ID:-0}}")"
fi
if [ -z "${GRPO_HF_CHECKPOINT}" ]; then
  GRPO_HF_CHECKPOINT="$(format_rollout_template "${GRPO_HF_SAVE_TEMPLATE}" "${SLIME_GRPO_EVAL_ROLLOUT_ID:-$((GRPO_NUM_ROLLOUT - 1))}")"
fi
SFT_HF_CHECKPOINT="$(absolute_path "${SFT_HF_CHECKPOINT}")"
GRPO_HF_CHECKPOINT="$(absolute_path "${GRPO_HF_CHECKPOINT}")"

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

mkdir -p "${STAGE_ROOT}" "${RUN_ROOT}/rollout_dumps" "${RUN_ROOT}/eval" "${WANDB_DIR_ROOT}/${STAGE}"

run_repo_python() {
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" "$@"
}

# Best-effort pipeline milestones into the <run-id>-pipeline W&B run.
w8_milestone() {
  if [ -n "${WANDB_API_KEY:-}${WANDB_KEY:-}" ]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/src/w8_biayn/wandb_milestones.py" \
      --project "${SLIME_WANDB_PROJECT:-slime-glm47-cpp-perf}" \
      --run-id "${RUN_ID}" --event "$1" >/dev/null 2>&1 || true
  fi
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
  BUILD_DATA_ARGS=(
    -m w8_biayn.integrations.slime_cpp_perf build-data
    --tasks-dir "${TASKS_DIR}"
    --out "${DATA_DIR}"
    --train-limit "${TRAIN_LIMIT}"
    --eval-limit "${EVAL_LIMIT}"
    --eval-splits "${EVAL_SPLITS}"
    --profile "${SLIME_CPP_PROFILE:-glm47-cpp-perf}"
    --run-id "${RUN_ID}"
    --force
  )
  if [ "${SORT_BY_SIZE}" = "1" ]; then
    BUILD_DATA_ARGS+=(--sort-by-size)
  fi
  run_repo_python "${BUILD_DATA_ARGS[@]}"
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
  w8_milestone model_download_started
  "${PYTHON_BIN}" - "${HF_MODEL_ID}" "${HF_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
Path(local_dir).mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, local_dir=local_dir)
PY
  w8_milestone model_download_finished
  HF_CHECKPOINT_WAS_DOWNLOADED=1
}

checkpoint_ready() {
  [ -f "$1/latest_checkpointed_iteration.txt" ]
}

hf_export_ready() {
  local checkpoint_dir="$1"
  [ -f "${checkpoint_dir}/config.json" ] && find "${checkpoint_dir}" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "model.safetensors.index.json" -o -name "pytorch_model.bin.index.json" \) -print -quit | grep -q .
}

ensure_slime_runtime() {
  for required_path in "${SLIME_ROOT}" "${MEGATRON_DIR}" "${DATA_DIR}/manifest.json"; do
    if [ ! -e "${required_path}" ]; then
      echo "Missing required path: ${required_path}" >&2
      exit 2
    fi
  done
  if [ ! -f "${SLIME_ROOT}/scripts/models/${MODEL_ARGS_SCRIPT}" ]; then
    echo "Missing GLM model args: ${SLIME_ROOT}/scripts/models/${MODEL_ARGS_SCRIPT}" >&2
    echo "Refresh the pinned SLIME checkout with: uv run w8-biayn upstreams clone slime" >&2
    exit 2
  fi
  if ! command -v ray >/dev/null 2>&1; then
    echo "The SLIME runtime is not active: 'ray' is missing from PATH." >&2
    echo "Run inside the SLIME container started by .w8-biayn/slime/run-container.sh." >&2
    exit 1
  fi
}

filter_model_args() {
  if [ "${MOE_GROUPED_GEMM}" = "1" ]; then
    return
  fi
  local filtered_args=()
  local arg
  for arg in "${MODEL_ARGS[@]}"; do
    if [ "${arg}" = "--moe-grouped-gemm" ]; then
      continue
    fi
    filtered_args+=("${arg}")
  done
  MODEL_ARGS=("${filtered_args[@]}")
}

ensure_base_checkpoint() {
  download_hf_checkpoint_if_missing
  # shellcheck disable=SC1090
  source "${SLIME_ROOT}/scripts/models/${MODEL_ARGS_SCRIPT}"
  CONVERT_MODEL_ARGS=("${MODEL_ARGS[@]}")
  filter_model_args
  if checkpoint_ready "${REF_LOAD_DIR}"; then
    return
  fi
  if [ "${CONVERT_IF_MISSING}" != "1" ]; then
    echo "Missing Megatron torch_dist checkpoint: ${REF_LOAD_DIR}" >&2
    echo "Set SLIME_REF_LOAD_DIR or rerun with SLIME_CONVERT_IF_MISSING=1." >&2
    exit 2
  fi
  echo "Converting ${HF_CHECKPOINT} to ${REF_LOAD_DIR}"
  w8_milestone checkpoint_convert_started
  if [ "${CONVERT_NPROC}" = "1" ]; then
    PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
      "${CONVERT_MODEL_ARGS[@]}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --save "${REF_LOAD_DIR}"
  else
    PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" \
      torchrun --nproc-per-node "${CONVERT_NPROC}" \
      "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
      "${CONVERT_MODEL_ARGS[@]}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --save "${REF_LOAD_DIR}"
  fi
  w8_milestone checkpoint_convert_finished
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

require_hf_export() {
  local checkpoint_dir="$1"
  local name="$2"
  if ! hf_export_ready "${checkpoint_dir}" "${name}"; then
    echo "Missing ${name} HuggingFace export: ${checkpoint_dir}" >&2
    echo "Expected config.json and HF weight files in that directory." >&2
    exit 2
  fi
}

latest_checkpoint_iteration_dir() {
  local checkpoint_dir="$1"
  local latest
  latest="$(tr -d "[:space:]" <"${checkpoint_dir}/latest_checkpointed_iteration.txt")"
  if [ -z "${latest}" ]; then
    echo "Empty latest checkpoint marker: ${checkpoint_dir}/latest_checkpointed_iteration.txt" >&2
    exit 2
  fi
  case "${latest}" in
    iter_*)
      printf "%s/%s\n" "${checkpoint_dir}" "${latest}"
      ;;
    *)
      printf "%s/iter_%07d\n" "${checkpoint_dir}" "${latest}"
      ;;
  esac
}

export_hf_checkpoint() {
  local checkpoint_dir="$1"
  local output_dir="$2"
  local name="$3"

  HF_EXPORT_LOG="${STAGE_ROOT}/hf_export.log"
  HF_EXPORT_STATUS="skipped"
  if [ "${SAVE_HF_EXPORTS}" != "1" ] || [ "${STANDALONE_HF_EXPORTS}" != "1" ]; then
    return 0
  fi
  if hf_export_ready "${output_dir}" && [ "${HF_EXPORT_FORCE}" != "1" ]; then
    HF_EXPORT_STATUS="already_present"
    return 0
  fi

  local input_dir
  input_dir="$(latest_checkpoint_iteration_dir "${checkpoint_dir}")"
  echo "Exporting ${name} Megatron checkpoint to HuggingFace: ${output_dir}" | tee "${HF_EXPORT_LOG}"
  set +e
  HF_EXPORT_ARGS=(
    "${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py"
    --input-dir "${input_dir}"
    --output-dir "${output_dir}"
    --origin-hf-dir "${HF_CHECKPOINT}"
    --chunk-size "${HF_EXPORT_CHUNK_SIZE}"
  )
  if [ -n "${HF_EXPORT_MODEL_NAME}" ]; then
    HF_EXPORT_ARGS+=(--model-name "${HF_EXPORT_MODEL_NAME}")
  fi
  if [ -n "${HF_EXPORT_VOCAB_SIZE}" ]; then
    HF_EXPORT_ARGS+=(--vocab-size "${HF_EXPORT_VOCAB_SIZE}")
  fi
  if [ "${HF_EXPORT_FORCE}" = "1" ]; then
    HF_EXPORT_ARGS+=(--force)
  fi
  if [ "${HF_EXPORT_ADD_MISSING_FROM_ORIGIN}" = "1" ]; then
    HF_EXPORT_ARGS+=(--add-missing-from-origin-hf)
  fi
  PYTHONPATH="${MEGATRON_DIR}:${SLIME_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${HF_EXPORT_ARGS[@]}" >>"${HF_EXPORT_LOG}" 2>&1
  local export_status="$?"
  set -e
  if [ "${export_status}" = "0" ]; then
    HF_EXPORT_STATUS="succeeded"
  else
    HF_EXPORT_STATUS="failed"
    tail -80 "${HF_EXPORT_LOG}" >&2 || true
  fi
  return "${export_status}"
}

write_frozen_sglang_config() {
  local model_path="$1"
  local config_path="${RUN_ROOT}/sglang/${STAGE}-frozen-sglang.yaml"
  mkdir -p "$(dirname -- "${config_path}")"
  cat >"${config_path}" <<EOF
sglang:
  - name: default
    model_path: "${model_path}"
    update_weights: false
    num_gpus_per_engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}
    server_groups:
      - worker_type: regular
        num_gpus: ${NUM_GPUS}
        num_gpus_per_engine: ${ROLLOUT_NUM_GPUS_PER_ENGINE}
EOF
  printf "%s" "${config_path}"
}

cleanup_ray() {
  if [ "${SKIP_CLEANUP}" != "1" ] && [ "${USE_EXTERNAL_RAY}" = "0" ]; then
    pkill -9 sglang || true
    sleep 3
    if command -v timeout >/dev/null 2>&1; then
      timeout "${SLIME_RAY_STOP_TIMEOUT_SECONDS:-60}" ray stop --force || true
    else
      ray stop --force || true
    fi
    pkill -9 ray || true
    pkill -9 raylet || true
    pkill -9 gcs_server || true
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
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size "${ETP_SIZE}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --seq-length "${SEQ_LENGTH}"
  )
  if [ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]; then
    PERF_ARGS+=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
  fi
  if [ "${USE_DYNAMIC_BATCH_SIZE}" = "1" ] || { [ "${USE_DYNAMIC_BATCH_SIZE}" = "auto" ] && [ "${ATTENTION_BACKEND}" != "local" ]; }; then
    PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
  else
    PERF_ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE}")
  fi
  if [ "${SEQUENCE_PARALLEL}" = "1" ] || { [ "${SEQUENCE_PARALLEL}" = "auto" ] && [ "${ATTENTION_BACKEND}" != "local" ]; }; then
    PERF_ARGS+=(--sequence-parallel)
  fi
  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${SLIME_LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
  )
  SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  )
  if [ "${SGLANG_ENABLE_DP_ATTENTION}" = "1" ]; then
    SGLANG_ARGS+=(--sglang-enable-dp-attention --sglang-dp-size "${SGLANG_DP_SIZE}")
  fi
  if [ "${SGLANG_ENABLE_DP_LM_HEAD}" = "1" ]; then
    SGLANG_ARGS+=(--sglang-enable-dp-lm-head)
  fi
  if [ -n "${SGLANG_MOE_DENSE_TP_SIZE}" ]; then
    SGLANG_ARGS+=(--sglang-moe-dense-tp-size "${SGLANG_MOE_DENSE_TP_SIZE}")
  fi
  if [ "${SGLANG_SPECULATIVE}" = "1" ]; then
    SGLANG_ARGS+=(
      --sglang-speculative-algorithm EAGLE
      --sglang-speculative-num-steps "${SLIME_SGLANG_SPECULATIVE_NUM_STEPS:-3}"
      --sglang-speculative-eagle-topk "${SLIME_SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
      --sglang-speculative-num-draft-tokens "${SLIME_SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
    )
  fi
  if [ -n "${SGLANG_CUDA_GRAPH_MAX_BS}" ]; then
    SGLANG_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
  fi
  if [ -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]; then
    SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
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
  if [ -n "${ATTENTION_BACKEND}" ]; then
    MISC_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
  fi
  if [ -n "${GLM_MOE_TOKEN_DISPATCHER_TYPE}" ]; then
    MISC_ARGS+=(--moe-token-dispatcher-type "${GLM_MOE_TOKEN_DISPATCHER_TYPE}")
  fi
  if [ "${GLM_ENABLE_DEEPEP}" = "1" ]; then
    MISC_ARGS+=(--moe-enable-deepep)
  fi
  MTP_ARGS=()
  if [ "${GLM_ENABLE_MTP_TRAINING}" = "1" ]; then
    MTP_ARGS=(
      --mtp-num-layers "${GLM_MTP_NUM_LAYERS}"
      --enable-mtp-training
      --mtp-loss-scaling-factor "${GLM_MTP_LOSS_SCALING_FACTOR}"
    )
  fi
  EXTRA_ARGS=(
    --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}"
    --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
    --train-memory-margin-bytes "${SLIME_TRAIN_MEMORY_MARGIN_BYTES:-268435456}"
    --no-save-optim
    --no-save-rng
    --save-debug-rollout-data "${ROLLOUT_DUMP_TEMPLATE}"
  )
  if [ -n "${UPDATE_WEIGHT_BUFFER_SIZE}" ]; then
    EXTRA_ARGS+=(--update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE}")
  fi
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
      --wandb-project "${SLIME_WANDB_PROJECT:-slime-glm47-cpp-perf}"
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
  SGLANG_CONFIG_ARGS=()
  case "${STAGE}" in
    base-eval)
      SGLANG_CONFIG_ARGS=(--sglang-config "$(write_frozen_sglang_config "${HF_CHECKPOINT}")")
      CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_LOAD_DIR}" --load "${REF_LOAD_DIR}" --save "${RUN_ROOT}/checkpoints/base-eval" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --debug-rollout-only
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
      OPTIMIZER_ARGS+=(--lr-decay-iters "${EVAL_LR_DECAY_ITERS}")
      CUSTOM_ARGS=(--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func)
      ;;
    sft)
      if [ "${SFT_SKIP_FINAL_TRAIN_SLEEP}" = "1" ]; then
        FINAL_TRAIN_SLEEP_SKIP=1
      fi
      CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${REF_LOAD_DIR}" --load "${REF_LOAD_DIR}" --save "${SFT_SAVE_DIR}" --save-interval "${SAVE_INTERVAL}")
      if [ "${SFT_LOAD_WEIGHTS_ONLY}" = "1" ]; then
        CKPT_ARGS+=(--no-load-optim --no-load-rng)
      fi
      if [ "${SAVE_HF_EXPORTS}" = "1" ] && [ "${INLINE_SAVE_HF_EXPORTS}" = "1" ]; then
        CKPT_ARGS+=(--save-hf "${SFT_HF_SAVE_TEMPLATE}")
      fi
      TASK_ARGS=(
        --rollout-function-path slime.rollout.sft_rollout.generate_rollout
        --prompt-data "${DATA_DIR}/sft/train.jsonl"
        --input-key messages
        --metadata-key metadata
        --rollout-shuffle
        --num-epoch "${SFT_NUM_EPOCH}"
        --start-rollout-id "${SFT_START_ROLLOUT_ID}"
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
      require_hf_export "${SFT_HF_CHECKPOINT}" "SFT"
      SGLANG_CONFIG_ARGS=(--sglang-config "$(write_frozen_sglang_config "${SFT_HF_CHECKPOINT}")")
      CKPT_ARGS=(--hf-checkpoint "${SFT_HF_CHECKPOINT}" --ref-load "${SFT_SAVE_DIR}" --load "${SFT_SAVE_DIR}" --save "${RUN_ROOT}/checkpoints/sft-eval" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --debug-rollout-only
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
      OPTIMIZER_ARGS+=(--lr-decay-iters "${EVAL_LR_DECAY_ITERS}")
      CUSTOM_ARGS=(--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func)
      ;;
    grpo)
      if [ "${GRPO_SKIP_FINAL_TRAIN_SLEEP}" = "1" ]; then
        FINAL_TRAIN_SLEEP_SKIP=1
      fi
      require_checkpoint "${SFT_SAVE_DIR}" "SFT"
      if [ "${GRPO_SKIP_WEIGHT_UPDATE}" = "1" ]; then
        if [ "${GRPO_NUM_ROLLOUT}" != "1" ]; then
          echo "SLIME_GRPO_SKIP_WEIGHT_UPDATE=1 is only valid for the one-rollout measurement smoke." >&2
          echo "Set SLIME_GRPO_NUM_ROLLOUT=1 or SLIME_GRPO_SKIP_WEIGHT_UPDATE=0." >&2
          exit 2
        fi
        require_hf_export "${SFT_HF_CHECKPOINT}" "SFT"
        SGLANG_CONFIG_ARGS=(--sglang-config "$(write_frozen_sglang_config "${SFT_HF_CHECKPOINT}")")
        CKPT_ARGS=(--hf-checkpoint "${SFT_HF_CHECKPOINT}" --ref-load "${SLIME_GRPO_REF_LOAD_DIR:-${SFT_SAVE_DIR}}" --load "${SLIME_GRPO_ACTOR_LOAD_DIR:-${SFT_SAVE_DIR}}" --save "${GRPO_SAVE_DIR}" --save-interval "${SAVE_INTERVAL}")
      else
        CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}" --ref-load "${SLIME_GRPO_REF_LOAD_DIR:-${SFT_SAVE_DIR}}" --load "${SLIME_GRPO_ACTOR_LOAD_DIR:-${SFT_SAVE_DIR}}" --save "${GRPO_SAVE_DIR}" --save-interval "${SAVE_INTERVAL}")
      fi
      if [ "${GRPO_LOAD_WEIGHTS_ONLY}" = "1" ]; then
        CKPT_ARGS+=(--no-load-optim --no-load-rng)
      fi
      if [ "${SAVE_HF_EXPORTS}" = "1" ] && [ "${INLINE_SAVE_HF_EXPORTS}" = "1" ]; then
        CKPT_ARGS+=(--save-hf "${GRPO_HF_SAVE_TEMPLATE}")
      fi
      TASK_ARGS=(
        --prompt-data "${DATA_DIR}/grpo/train.jsonl"
        --input-key prompt
        --label-key label
        --metadata-key metadata
        --apply-chat-template
        --rollout-shuffle
        --reward-key score
        --num-rollout "${GRPO_NUM_ROLLOUT}"
        --start-rollout-id "${GRPO_START_ROLLOUT_ID}"
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
      require_hf_export "${GRPO_HF_CHECKPOINT}" "GRPO"
      SGLANG_CONFIG_ARGS=(--sglang-config "$(write_frozen_sglang_config "${GRPO_HF_CHECKPOINT}")")
      CKPT_ARGS=(--hf-checkpoint "${GRPO_HF_CHECKPOINT}" --ref-load "${GRPO_SAVE_DIR}" --load "${GRPO_SAVE_DIR}" --save "${RUN_ROOT}/checkpoints/grpo-eval" --save-interval "${SAVE_INTERVAL}")
      TASK_ARGS=(
        --debug-rollout-only
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
      OPTIMIZER_ARGS+=(--lr-decay-iters "${EVAL_LR_DECAY_ITERS}")
      CUSTOM_ARGS=(--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func)
      ;;
    *)
      echo "stage ${STAGE} does not submit a SLIME job" >&2
      exit 2
      ;;
  esac
}

append_optimizer_offload_args() {
  if [ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
    OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)
  fi
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
    "W8_BIAYN_DATA_DIR": "${DATA_DIR}",
    "W8_CPP_SANDBOX_IMAGE": os.environ.get("W8_CPP_SANDBOX_IMAGE", "w8-biayn-cpp-perf:latest"),
    "W8_CPP_SANDBOX_CPU": os.environ.get("W8_CPP_SANDBOX_CPU", "3"),
    "W8_SLIME_CPP_INCLUDE_LOGS": os.environ.get("W8_SLIME_CPP_INCLUDE_LOGS", "0"),
    "W8_SLIME_SKIP_FINAL_TRAIN_SLEEP": "${FINAL_TRAIN_SLEEP_SKIP}",
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK}",
    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}",
    "NVSHMEM_DISABLE_NCCL": "${NVSHMEM_DISABLE_NCCL}",
}
attention_backend = "${ATTENTION_BACKEND}".strip().lower()
nvte_flags_by_backend = {
    "flash": {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"},
    "fused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"},
    "unfused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "1"},
    "local": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"},
}
for key, value in nvte_flags_by_backend.get(attention_backend, {}).items():
    env[key] = os.environ.get(f"SLIME_{key}", value)
if attention_backend not in nvte_flags_by_backend:
    for key in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        if f"SLIME_{key}" in os.environ:
            env[key] = os.environ[f"SLIME_{key}"]
for key in (
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "HF_HOME",
    "WANDB_API_KEY",
    "WANDB_KEY",
    "WANDB_DIR",
    "WANDB_MODE",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
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
train_limit=${TRAIN_LIMIT}
eval_limit=${EVAL_LIMIT}
sort_by_size=${SORT_BY_SIZE}
model_args_script=${MODEL_ARGS_SCRIPT}
hf_checkpoint=${HF_CHECKPOINT}
hf_model_id=${HF_MODEL_ID}
download_hf_checkpoint=${DOWNLOAD_HF_CHECKPOINT}
hf_checkpoint_was_downloaded=${HF_CHECKPOINT_WAS_DOWNLOADED}
ref_load=${REF_LOAD_DIR}
sft_save_dir=${SFT_SAVE_DIR}
grpo_save_dir=${GRPO_SAVE_DIR}
save_hf_exports=${SAVE_HF_EXPORTS}
inline_save_hf_exports=${INLINE_SAVE_HF_EXPORTS}
standalone_hf_exports=${STANDALONE_HF_EXPORTS}
hf_export_force=${HF_EXPORT_FORCE}
hf_export_add_missing_from_origin=${HF_EXPORT_ADD_MISSING_FROM_ORIGIN}
hf_export_chunk_size=${HF_EXPORT_CHUNK_SIZE}
hf_export_model_name=${HF_EXPORT_MODEL_NAME}
hf_export_vocab_size=${HF_EXPORT_VOCAB_SIZE}
hf_export_status=${HF_EXPORT_STATUS}
hf_export_log=${HF_EXPORT_LOG}
sft_hf_save_template=${SFT_HF_SAVE_TEMPLATE}
grpo_hf_save_template=${GRPO_HF_SAVE_TEMPLATE}
sft_hf_checkpoint=${SFT_HF_CHECKPOINT}
grpo_hf_checkpoint=${GRPO_HF_CHECKPOINT}
sft_skip_final_train_sleep=${SFT_SKIP_FINAL_TRAIN_SLEEP}
grpo_skip_weight_update=${GRPO_SKIP_WEIGHT_UPDATE}
grpo_load_weights_only=${GRPO_LOAD_WEIGHTS_ONLY}
grpo_skip_final_train_sleep=${GRPO_SKIP_FINAL_TRAIN_SLEEP}
final_train_sleep_skip=${FINAL_TRAIN_SLEEP_SKIP}
colocate=${COLOCATE}
sft_colocate=${SFT_COLOCATE}
distributed_timeout_minutes=${DISTRIBUTED_TIMEOUT_MINUTES}
megatron_to_hf_mode=${MEGATRON_TO_HF_MODE}
num_gpus=${NUM_GPUS}
tp_size=${TP_SIZE}
pp_size=${PP_SIZE}
cp_size=${CP_SIZE}
ep_size=${EP_SIZE}
decoder_last_pipeline_num_layers=${DECODER_LAST_PIPELINE_NUM_LAYERS}
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}
micro_batch_size=${MICRO_BATCH_SIZE}
seq_length=${SEQ_LENGTH}
optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD}
nvshmem_disable_nccl=${NVSHMEM_DISABLE_NCCL}
glm_enable_deepep=${GLM_ENABLE_DEEPEP}
glm_enable_mtp_training=${GLM_ENABLE_MTP_TRAINING}
update_weight_buffer_size=${UPDATE_WEIGHT_BUFFER_SIZE}
sglang_mem_fraction=${SGLANG_MEM_FRACTION_STATIC}
sglang_cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS}
sglang_dp_size=${SGLANG_DP_SIZE}
sglang_enable_dp_attention=${SGLANG_ENABLE_DP_ATTENTION}
sglang_enable_dp_lm_head=${SGLANG_ENABLE_DP_LM_HEAD}
sglang_speculative=${SGLANG_SPECULATIVE}
sglang_max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}
sglang_disable_tp_memory_inbalance_check=${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}
sglang_enable_tp_memory_inbalance_check=${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK}
wandb_project=${SLIME_WANDB_PROJECT:-slime-glm47-cpp-perf}
wandb_group=${SLIME_WANDB_GROUP:-${RUN_ID}}
wandb_run_id=${SLIME_WANDB_RUN_ID:-${RUN_ID}-${STAGE}}
wandb_dir=${WANDB_DIR:-}
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
  append_optimizer_offload_args

  local colocate_for_stage="${COLOCATE}"
  if [ "${STAGE}" = "sft" ]; then
    colocate_for_stage="${SFT_COLOCATE}"
  fi
  COLOCATE_ARGS=()
  if [ "${colocate_for_stage}" = "1" ]; then
    COLOCATE_ARGS=(--colocate)
  fi

  if [ "${SLIME_TRACE:-1}" = "1" ]; then
    set -x
  fi
  ulimit -Sn "${SLIME_NOFILE_SOFT_LIMIT:-65536}" 2>/dev/null || true
  export PYTHONUNBUFFERED=1
  export WANDB_DIR="${WANDB_DIR_ROOT}/${STAGE}"
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
  RAY_STATUS_TIMEOUT_SECONDS="${SLIME_RAY_STATUS_TIMEOUT_SECONDS:-43200}"

  echo "GLM-4.7-Flash C++ SLIME stage: ${STAGE}"
  w8_milestone "stage_${STAGE}_started"
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
    -- "${PYTHON_BIN}" -u -m "${SLIME_TRAIN_MODULE}" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    "${COLOCATE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${TASK_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${SGLANG_CONFIG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
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
  if [ "${RAY_STATUS}" = "0" ]; then
    case "${STAGE}" in
      sft)
        if ! export_hf_checkpoint "${SFT_SAVE_DIR}" "${SFT_HF_CHECKPOINT}" "SFT"; then
          RAY_STATUS=1
          RAY_JOB_TERMINAL_STATUS="${RAY_JOB_TERMINAL_STATUS}+HF_EXPORT_FAILED"
        fi
        ;;
      grpo)
        if ! export_hf_checkpoint "${GRPO_SAVE_DIR}" "${GRPO_HF_CHECKPOINT}" "GRPO"; then
          RAY_STATUS=1
          RAY_JOB_TERMINAL_STATUS="${RAY_JOB_TERMINAL_STATUS}+HF_EXPORT_FAILED"
        fi
        ;;
    esac
  fi
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
