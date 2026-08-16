#!/usr/bin/env bash
# Miles GRPO LoRA rank-16 runner for GLM-4.7-Flash on the PIE C++ task.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
MILES_ROOT="${MILES_ROOT:-/root/miles}"
PYTHON_BIN="${MILES_PYTHON:-python3}"
MODEL_ARGS_FILE="${MILES_MODEL_ARGS_FILE:-glm4.7-flash.sh}"
MODEL_ARGS_PATH="${MILES_MODEL_ARGS_PATH:-${MILES_ROOT}/scripts/models/${MODEL_ARGS_FILE}}"

RUN_ID="${MILES_RUN_ID:-glm47_h100_pie_cpp_lora_r16_$(date +%Y%m%d_%H%M%S)}"
STAGE_STARTED_AT="${SECONDS}"
RUN_ROOT="${MILES_RUN_ROOT:-${REPO_ROOT}/.glm47-posttraining/miles/glm47-h100-cpp-perf/runs/${RUN_ID}}"
DATA_DIR="${MILES_CPP_DATA_DIR:-${RUN_ROOT}/data}"
TASKS_DIR="${MILES_CPP_TASKS_DIR:-${REPO_ROOT}/.glm47-posttraining/data/tasks-small}"
DATA_BUILD_MODULE="${MILES_DATA_BUILD_MODULE:-glm47_posttraining.integrations.miles_cpp_perf}"
DATA_CURRICULUM="${MILES_DATA_CURRICULUM:-}"
ROLLOUT_ONLY="${MILES_ROLLOUT_ONLY:-0}"
CUSTOM_RM_PATH="${MILES_CUSTOM_RM_PATH:-glm47_posttraining.integrations.miles_cpp_perf.reward_func}"
REWARD_PREFLIGHT_MODULE="${MILES_REWARD_PREFLIGHT_MODULE:-}"
EXPECTED_DATASET_KIND="${MILES_EXPECTED_DATASET_KIND:-}"
EVAL_NAME="${MILES_EVAL_NAME:-pie_cpp}"
TRAIN_LIMIT="${MILES_CPP_TRAIN_LIMIT:-}"
EVAL_LIMIT="${MILES_CPP_EVAL_LIMIT:-}"
EVAL_SPLITS="${MILES_CPP_EVAL_SPLITS:-validation,test}"
SORT_BY_SIZE="${MILES_CPP_SORT_BY_SIZE:-1}"
FILTER_TRAIN_ORACLE_FULL_MARKS="${MILES_CPP_FILTER_TRAIN_ORACLE_FULL_MARKS:-0}"
ORACLE_FILTER_WORKERS="${MILES_CPP_ORACLE_FILTER_WORKERS:-8}"

HF_CHECKPOINT="${MILES_HF_CHECKPOINT:-/root/models/GLM-4.7-Flash}"
REF_LOAD_DIR="${MILES_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist_tp4_pp1_ep8}"
SAVE_DIR="${MILES_SAVE_DIR:-${RUN_ROOT}/checkpoints/grpo_lora_r16}"

GPUS_PER_NODE="${MILES_GPUS_PER_NODE:-8}"
TP_SIZE="${MILES_TENSOR_MODEL_PARALLEL_SIZE:-4}"
PP_SIZE="${MILES_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CP_SIZE="${MILES_CONTEXT_PARALLEL_SIZE:-1}"
EP_SIZE="${MILES_EXPERT_MODEL_PARALLEL_SIZE:-8}"
ETP_SIZE="${MILES_EXPERT_TENSOR_PARALLEL_SIZE:-1}"
MOE_TOKEN_DISPATCHER_TYPE="${MILES_MOE_TOKEN_DISPATCHER_TYPE:-}"
MOE_ENABLE_DEEPEP="${MILES_MOE_ENABLE_DEEPEP:-0}"
RECOMPUTE_GRANULARITY="${MILES_RECOMPUTE_GRANULARITY:-selective}"
ATTENTION_BACKEND="${MILES_ATTENTION_BACKEND:-flash}"
SEQ_LENGTH="${MILES_SEQ_LENGTH:-4096}"
MAX_TOKENS_PER_GPU="${MILES_MAX_TOKENS_PER_GPU:-16384}"
MICRO_BATCH_SIZE="${MILES_MICRO_BATCH_SIZE:-1}"
USE_DYNAMIC_BATCH_SIZE="${MILES_USE_DYNAMIC_BATCH_SIZE:-1}"
BALANCE_DATA="${MILES_BALANCE_DATA:-1}"

NUM_ROLLOUT="${MILES_NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${MILES_ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${MILES_N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${MILES_GLOBAL_BATCH_SIZE:-256}"
GRPO_ROLLOUT_SHUFFLE="${MILES_GRPO_ROLLOUT_SHUFFLE:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${MILES_ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_TEMPERATURE="${MILES_ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_SKIP_SPECIAL_TOKENS="${MILES_ROLLOUT_SKIP_SPECIAL_TOKENS:-0}"
ROLLOUT_STOP_TOKEN_IDS="${MILES_ROLLOUT_STOP_TOKEN_IDS:-}"
read -r -a ROLLOUT_STOP_TOKEN_ID_ARGS <<< "${ROLLOUT_STOP_TOKEN_IDS}"
APPLY_CHAT_TEMPLATE_KWARGS="${MILES_APPLY_CHAT_TEMPLATE_KWARGS:-}"
TRAIN_MODULE="${MILES_TRAIN_MODULE:-}"
EVAL_INTERVAL="${MILES_EVAL_INTERVAL:-1}"
EVAL_N_SAMPLES_PER_PROMPT="${MILES_EVAL_N_SAMPLES_PER_PROMPT:-1}"
EVAL_MAX_RESPONSE_LEN="${MILES_EVAL_MAX_RESPONSE_LEN:-1536}"
EVAL_PROMPT_DATA="${MILES_EVAL_PROMPT_DATA:-}"
KL_LOSS_COEF="${MILES_KL_LOSS_COEF:-0.00}"

LORA_RANK="${MILES_LORA_RANK:-16}"
LORA_ALPHA="${MILES_LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${MILES_LORA_TARGET_MODULES:-q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj}"
SGLANG_LORA_TARGET_MODULES="${MILES_SGLANG_LORA_TARGET_MODULES:-${LORA_TARGET_MODULES}}"
read -r -a SGLANG_LORA_TARGET_MODULE_ARGS <<< "${SGLANG_LORA_TARGET_MODULES//,/ }"
EXPERTS_SHARED_OUTER_LORAS="${MILES_EXPERTS_SHARED_OUTER_LORAS:-1}"
LORA_BASE_CPU_BACKUP="${MILES_LORA_BASE_CPU_BACKUP:-0}"
NO_GRADIENT_ACCUMULATION_FUSION="${MILES_NO_GRADIENT_ACCUMULATION_FUSION:-0}"
CUDA_DEVICE_MAX_CONNECTIONS="${MILES_CUDA_DEVICE_MAX_CONNECTIONS:-1}"
SGLANG_LORA_USE_VIRTUAL_EXPERTS="${MILES_SGLANG_LORA_USE_VIRTUAL_EXPERTS:-1}"
SGLANG_MEM_FRACTION_STATIC="${MILES_SGLANG_MEM_FRACTION_STATIC:-0.25}"
SGLANG_SERVER_CONCURRENCY="${MILES_SGLANG_SERVER_CONCURRENCY:-512}"
SGLANG_CUDA_GRAPH_MAX_BS="${MILES_SGLANG_CUDA_GRAPH_MAX_BS:-4}"
SGLANG_MAX_RUNNING_REQUESTS="${MILES_SGLANG_MAX_RUNNING_REQUESTS:-}"
SGLANG_DP_SIZE="${MILES_SGLANG_DP_SIZE:-${GPUS_PER_NODE}}"
SGLANG_ENABLE_DP_ATTENTION="${MILES_SGLANG_ENABLE_DP_ATTENTION:-0}"
SGLANG_ENABLE_DP_LM_HEAD="${MILES_SGLANG_ENABLE_DP_LM_HEAD:-0}"
SGLANG_MOE_DENSE_TP_SIZE="${MILES_SGLANG_MOE_DENSE_TP_SIZE:-}"
SGLANG_SPECULATIVE="${MILES_SGLANG_SPECULATIVE:-0}"
SGLANG_SPECULATIVE_NUM_STEPS="${MILES_SGLANG_SPECULATIVE_NUM_STEPS:-3}"
SGLANG_SPECULATIVE_EAGLE_TOPK="${MILES_SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS="${MILES_SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${MILES_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}"
SGLANG_ATTENTION_BACKEND="${MILES_SGLANG_ATTENTION_BACKEND:-}"

WANDB_PROJECT="${MILES_WANDB_PROJECT:-miles-glm47-h100-cpp-perf}"
WANDB_GROUP="${MILES_WANDB_GROUP:-glm47-h100-pie-cpp-lora-r16}"
WANDB_RUN_ID="${MILES_WANDB_RUN_ID:-${RUN_ID}}"
WANDB_JOB_TYPE="${MILES_WANDB_JOB_TYPE:-${WANDB_JOB_TYPE:-grpo}}"
EXPERIMENT_ID="${GLM47_EXPERIMENT_ID:-${WANDB_GROUP}}"

STAGE_ROOT="${RUN_ROOT}/grpo_lora_r16"
LOG_FILE="${STAGE_ROOT}/run.log"
VRAM_LOG="${STAGE_ROOT}/vram_usage.csv"
VRAM_PEAK_FILE="${STAGE_ROOT}/vram_peak.txt"
RUN_RECEIPT="${STAGE_ROOT}/run_receipt.txt"
TRAINING_GATE="${STAGE_ROOT}/grpo_training_gate.json"
ROLLOUT_DUMP_TEMPLATE="${RUN_ROOT}/rollout_dumps/grpo_{rollout_id}.pt"

mkdir -p "${STAGE_ROOT}" "${RUN_ROOT}/rollout_dumps" "${SAVE_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "run_id=${RUN_ID}"
echo "run_root=${RUN_ROOT}"
echo "tasks_dir=${TASKS_DIR}"
echo "data_build_module=${DATA_BUILD_MODULE}"
echo "data_curriculum=${DATA_CURRICULUM}"
echo "rollout_only=${ROLLOUT_ONLY}"
echo "custom_rm_path=${CUSTOM_RM_PATH}"
echo "expected_dataset_kind=${EXPECTED_DATASET_KIND}"
echo "eval_name=${EVAL_NAME}"
echo "hf_checkpoint=${HF_CHECKPOINT}"
echo "model_args_path=${MODEL_ARGS_PATH}"
echo "ref_load=${REF_LOAD_DIR}"
echo "save_dir=${SAVE_DIR}"
echo "seq_length=${SEQ_LENGTH}"
echo "rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}"
echo "eval_max_response_len=${EVAL_MAX_RESPONSE_LEN}"

if [ ! -d "${MILES_ROOT}" ]; then
  echo "Missing Miles root: ${MILES_ROOT}" >&2
  exit 2
fi
if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
  echo "Missing HF checkpoint: ${HF_CHECKPOINT}" >&2
  exit 2
fi
if [ ! -f "${REF_LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
  echo "Missing Megatron checkpoint: ${REF_LOAD_DIR}" >&2
  exit 2
fi
if [ ! -f "${MODEL_ARGS_PATH}" ]; then
  echo "Missing model args: ${MODEL_ARGS_PATH}" >&2
  exit 2
fi
# Validate the selected C++ reward backend.
if [ "${GLM47_CPP_SANDBOX_BACKEND:-docker}" != "local" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Missing docker CLI inside container. Mount it with -v /usr/bin/docker:/usr/bin/docker:ro." >&2
    exit 2
  fi
  if ! docker image inspect "${GLM47_CPP_SANDBOX_IMAGE:-glm47-cpp-perf:latest}" >/dev/null 2>&1; then
    echo "Missing PIE C++ sandbox image: ${GLM47_CPP_SANDBOX_IMAGE:-glm47-cpp-perf:latest}" >&2
    exit 2
  fi
fi
# Local backend needs a working compiler toolchain instead.
if [ "${GLM47_CPP_SANDBOX_BACKEND:-docker}" = "local" ] && ! command -v g++ >/dev/null 2>&1; then
  echo "GLM47_CPP_SANDBOX_BACKEND=local but g++ is missing in this container." >&2
  exit 2
fi

BUILD_DATA_ARGS=(
  -m "${DATA_BUILD_MODULE}" build-data
  --tasks-dir "${TASKS_DIR}"
  --out "${DATA_DIR}"
  --eval-splits "${EVAL_SPLITS}"
  --profile "glm47-h100-grpo"
  --run-id "${RUN_ID}"
  --force
)
if [ -n "${DATA_CURRICULUM}" ]; then
  BUILD_DATA_ARGS+=(--curriculum "${DATA_CURRICULUM}")
fi
if [ "${MILES_ALLOW_NON_GCC_CURRICULUM:-0}" = "1" ]; then
  BUILD_DATA_ARGS+=(--allow-non-gcc-curriculum)
fi
if [ "${ROLLOUT_ONLY}" != "0" ] && [ "${ROLLOUT_ONLY}" != "1" ]; then
  echo "MILES_ROLLOUT_ONLY must be 0 or 1, got: ${ROLLOUT_ONLY}" >&2
  exit 2
fi
if [ -n "${TRAIN_LIMIT}" ]; then
  BUILD_DATA_ARGS+=(--train-limit "${TRAIN_LIMIT}")
fi
if [ -n "${EVAL_LIMIT}" ]; then
  BUILD_DATA_ARGS+=(--eval-limit "${EVAL_LIMIT}")
fi
if [ "${SORT_BY_SIZE}" = "1" ]; then
  BUILD_DATA_ARGS+=(--sort-by-size)
fi
if [ "${FILTER_TRAIN_ORACLE_FULL_MARKS}" = "1" ]; then
  BUILD_DATA_ARGS+=(--filter-train-oracle-full-marks --oracle-filter-workers "${ORACLE_FILTER_WORKERS}")
fi

# Reuse a prepared dataset when available.
if [ ! -f "${DATA_DIR}/grpo/train.jsonl" ]; then
  export GLM47_DATA_DIR="${GLM47_DATA_DIR:-${DATA_DIR}}"
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" "${BUILD_DATA_ARGS[@]}"
fi
export GLM47_DATA_DIR="${GLM47_DATA_DIR:-${DATA_DIR}}"
if [ ! -f "${DATA_DIR}/grpo/train.jsonl" ]; then
  echo "Missing GRPO train data: ${DATA_DIR}/grpo/train.jsonl" >&2
  exit 2
fi
if [ -n "${EXPECTED_DATASET_KIND}" ]; then
  DATA_MANIFEST_PATH="${DATA_DIR}/manifest.json" EXPECTED_DATASET_KIND="${EXPECTED_DATASET_KIND}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DATA_MANIFEST_PATH"])
if not path.is_file():
    raise SystemExit(f"missing dataset manifest: {path}")
manifest = json.loads(path.read_text(encoding="utf-8"))
expected = os.environ["EXPECTED_DATASET_KIND"]
if manifest.get("kind") != expected:
    raise SystemExit(f"dataset kind mismatch: {manifest.get('kind')!r} != {expected!r}")
print(f"DATASET_KIND_VERIFIED={expected}")
PY
fi
if [ -n "${REWARD_PREFLIGHT_MODULE}" ]; then
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m "${REWARD_PREFLIGHT_MODULE}" preflight
fi

monitor_vram() {
  echo "timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw" > "${VRAM_LOG}"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits >> "${VRAM_LOG}" || true
    sleep 2
  done
}

monitor_vram &
VRAM_MONITOR_PID=$!
cleanup() {
  if [ -n "${VRAM_MONITOR_PID:-}" ]; then
    kill "${VRAM_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${VRAM_MONITOR_PID}" >/dev/null 2>&1 || true
    VRAM_MONITOR_PID=""
  fi
  if [ -s "${VRAM_LOG}" ]; then
    awk -F, 'NR>1 {gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($3+0 > max) max=$3+0} END {print "max_memory_used_mib=" max}' "${VRAM_LOG}" > "${VRAM_PEAK_FILE}" || true
    cat "${VRAM_PEAK_FILE}" || true
  fi
}
trap cleanup EXIT

write_receipt() {
  local status="$1"
  local ray_status="$2"
  local max_memory_used_mib=""
  if [ -s "${VRAM_LOG}" ]; then
    awk -F, 'NR>1 {gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($3+0 > max) max=$3+0} END {print "max_memory_used_mib=" max}' "${VRAM_LOG}" > "${VRAM_PEAK_FILE}" || true
  fi
  if [ -s "${VRAM_PEAK_FILE}" ]; then
    max_memory_used_mib="$(awk -F= '/max_memory_used_mib/ {print $2}' "${VRAM_PEAK_FILE}" | tail -n 1)"
  fi
  cat >"${RUN_RECEIPT}" <<EOF
status=${status}
ray_status=${ray_status}
wall_s=$((SECONDS - STAGE_STARTED_AT))
run_id=${RUN_ID}
run_root=${RUN_ROOT}
stage_root=${STAGE_ROOT}
log_file=${LOG_FILE}
vram_log=${VRAM_LOG}
vram_peak_file=${VRAM_PEAK_FILE}
max_memory_used_mib=${max_memory_used_mib}
data_dir=${DATA_DIR}
tasks_dir=${TASKS_DIR}
data_build_module=${DATA_BUILD_MODULE}
data_curriculum=${DATA_CURRICULUM}
rollout_only=${ROLLOUT_ONLY}
post_update_eval=${POST_UPDATE_EVAL:-unknown}
post_update_eval_max_id=${POST_UPDATE_EVAL_MAX_ID:-}
custom_rm_path=${CUSTOM_RM_PATH}
expected_dataset_kind=${EXPECTED_DATASET_KIND}
data_manifest_sha256=$(sha256sum "${DATA_DIR}/manifest.json" | awk '{print $1}')
hf_checkpoint=${HF_CHECKPOINT}
model_args_path=${MODEL_ARGS_PATH}
ref_load=${REF_LOAD_DIR}
save_dir=${SAVE_DIR}
lora_source_adapter_path=${MILES_LORA_SOURCE_ADAPTER_PATH:-}
lora_adapter_path=${MILES_LORA_ADAPTER_PATH:-}
expected_source_adapter_sha256=${MILES_EXPECTED_SOURCE_ADAPTER_SHA256:-}
seq_length=${SEQ_LENGTH}
gpus_per_node=${GPUS_PER_NODE}
tensor_model_parallel_size=${TP_SIZE}
pipeline_model_parallel_size=${PP_SIZE}
context_parallel_size=${CP_SIZE}
expert_model_parallel_size=${EP_SIZE}
expert_tensor_parallel_size=${ETP_SIZE}
rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}
rollout_skip_special_tokens=${ROLLOUT_SKIP_SPECIAL_TOKENS}
rollout_stop_token_ids=${ROLLOUT_STOP_TOKEN_IDS}
eval_max_response_len=${EVAL_MAX_RESPONSE_LEN}
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}
micro_batch_size=${MICRO_BATCH_SIZE}
use_dynamic_batch_size=${USE_DYNAMIC_BATCH_SIZE}
balance_data=${BALANCE_DATA}
num_rollout=${NUM_ROLLOUT}
grpo_rollout_shuffle=${GRPO_ROLLOUT_SHUFFLE}
rollout_batch_size=${ROLLOUT_BATCH_SIZE}
n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}
global_batch_size=${GLOBAL_BATCH_SIZE}
train_module=${TRAIN_MODULE}
lora_rank=${LORA_RANK}
lora_alpha=${LORA_ALPHA}
no_gradient_accumulation_fusion=${NO_GRADIENT_ACCUMULATION_FUSION}
moe_token_dispatcher_type=${MOE_TOKEN_DISPATCHER_TYPE}
moe_enable_deepep=${MOE_ENABLE_DEEPEP}
recompute_granularity=${RECOMPUTE_GRANULARITY}
cuda_device_max_connections=${CUDA_DEVICE_MAX_CONNECTIONS}
sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}
sglang_cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS}
sglang_server_concurrency=${SGLANG_SERVER_CONCURRENCY}
sglang_max_running_requests=${SGLANG_MAX_RUNNING_REQUESTS}
sglang_dp_size=${SGLANG_DP_SIZE}
sglang_enable_dp_attention=${SGLANG_ENABLE_DP_ATTENTION}
sglang_enable_dp_lm_head=${SGLANG_ENABLE_DP_LM_HEAD}
sglang_moe_dense_tp_size=${SGLANG_MOE_DENSE_TP_SIZE}
sglang_speculative=${SGLANG_SPECULATIVE}
wandb_project=${WANDB_PROJECT}
wandb_group=${WANDB_GROUP}
wandb_run_id=${WANDB_RUN_ID}
wandb_job_type=${WANDB_JOB_TYPE}
experiment_id=${EXPERIMENT_ID}
eval_name=${EVAL_NAME}
kl_loss_coef=${KL_LOSS_COEF}
timing_status=${GLM47_TIMING_STATUS:-unverified}
training_gate=${TRAINING_GATE}
training_gate_status=${TRAINING_GATE_STATUS:-not_requested}
EOF
  cat "${RUN_RECEIPT}"
}

finalize_wandb() {
  local status="$1"
  local rollout_dump_dir
  local -a finalize_args
  rollout_dump_dir="$(dirname -- "${ROLLOUT_DUMP_TEMPLATE}")"
  finalize_args=(
    finalize-stage
    --project "${WANDB_PROJECT}"
    --experiment-id "${EXPERIMENT_ID}"
    --run-id "${WANDB_RUN_ID}"
    --group "${WANDB_GROUP}"
    --stage "${WANDB_JOB_TYPE}"
    --status "${status}"
    --receipt "${RUN_RECEIPT}"
    --artifact-path "${LOG_FILE}"
    --artifact-path "${VRAM_LOG}"
    --artifact-path "${VRAM_PEAK_FILE}"
    --run-log "${LOG_FILE}"
    --rollout-dump-dir "${rollout_dump_dir}"
    --checkpoint-dir "${SAVE_DIR}"
    --timing-status "${GLM47_TIMING_STATUS:-unverified}"
    --output-dir "${STAGE_ROOT}"
  )
  if [ -n "${GLM47_SYNC_METRICS_DIR:-}" ] && [ -d "${GLM47_SYNC_METRICS_DIR}" ]; then
    finalize_args+=(--sync-metrics-dir "${GLM47_SYNC_METRICS_DIR}")
  fi
  if [ -f "${TRAINING_GATE}" ]; then
    finalize_args+=(--artifact-path "${TRAINING_GATE}")
  fi
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${REPO_ROOT}/scripts/publish_results.py" "${finalize_args[@]}"
}

pkill -9 sglang >/dev/null 2>&1 || true
ray stop --force >/dev/null 2>&1 || true
pkill -9 ray >/dev/null 2>&1 || true
pkill -9 redis >/dev/null 2>&1 || true

export PYTHONBUFFERED=16
export MASTER_ADDR="${MILES_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
RAY_NODE_IP_ADDRESS="${MILES_RAY_NODE_IP_ADDRESS:-127.0.0.1}"
RAY_DASHBOARD_HOST="${MILES_RAY_DASHBOARD_HOST:-127.0.0.1}"
RAY_DASHBOARD_PORT="${MILES_RAY_DASHBOARD_PORT:-8265}"
export no_proxy="127.0.0.1,${MASTER_ADDR},${RAY_NODE_IP_ADDRESS},${RAY_DASHBOARD_HOST}"
export GLM47_DATA_DIR="${DATA_DIR}"
export GLM47_CPP_SANDBOX_IMAGE="${GLM47_CPP_SANDBOX_IMAGE:-glm47-cpp-perf:latest}"

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} detected_nvlink_refs=${NVLINK_COUNT}"

cd "${MILES_ROOT}"
source "${MODEL_ARGS_PATH}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${REF_LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval "${MILES_SAVE_INTERVAL:-1}"
  --megatron-to-hf-mode bridge
)
# The canonical GRPO profile uses the policy model directly.
if [ "${MILES_NO_REF:-0}" != "1" ]; then
  CKPT_ARGS+=(--ref-load "${REF_LOAD_DIR}")
fi

LORA_ARGS=(
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout 0.0
  --target-modules "${LORA_TARGET_MODULES}"
  --sglang-lora-backend triton
  --sglang-enable-lora
  --sglang-max-lora-rank "${LORA_RANK}"
  --sglang-lora-target-modules "${SGLANG_LORA_TARGET_MODULE_ARGS[@]}"
)
# Initialize adapter weights from a prior run (e.g. GRPO warm-started from the
# SFT adapter): point at an iter_*/adapter dir with Megatron-native shards.
LORA_ADAPTER_PATH="${MILES_LORA_ADAPTER_PATH:-}"
if [ -n "${LORA_ADAPTER_PATH}" ]; then
  # Three shard namings exist across Miles generations: legacy tp{t}_pp0.pt,
  # the synth-v1 era's tp{t}_pp0_ep{e}.pt, and mainline rank{r}.pt. The r3 run
  # staged ep-suffixed shards into a loader that knew neither newer naming and
  # silently trained from a fresh LoRA init. The bridge now resolves all three
  # per rank; refuse to launch only on names nothing is known to load.
  "${PYTHON_BIN}" - "${LORA_ADAPTER_PATH}" <<'PY'
import pathlib, re, sys

adapter = pathlib.Path(sys.argv[1])
names = sorted(p.name for p in adapter.glob("adapter_megatron_*.pt"))
if not names:
    sys.exit(f"warm-start adapter has no Megatron-native shards: {adapter}")
known = re.compile(r"adapter_megatron_(rank\d+|tp\d+_pp0(_ep\d+)?)\.pt")
bad = [n for n in names if not known.fullmatch(n)]
if bad:
    sys.exit(
        f"warm-start adapter {adapter} contains native shards no known Miles "
        f"loader or bridge shim resolves: {bad}. Refusing to launch a warm "
        "start that would silently fall back to fresh init."
    )
PY
  LORA_ARGS+=(--lora-adapter-path "${LORA_ADAPTER_PATH}")
fi
if [ "${EXPERTS_SHARED_OUTER_LORAS}" = "1" ]; then
  LORA_ARGS+=(--experts-shared-outer-loras)
fi
if [ "${LORA_BASE_CPU_BACKUP}" = "1" ]; then
  LORA_ARGS+=(--lora-base-cpu-backup)
fi
if [ "${NO_GRADIENT_ACCUMULATION_FUSION}" = "1" ]; then
  LORA_ARGS+=(--no-gradient-accumulation-fusion)
fi
if [ "${SGLANG_LORA_USE_VIRTUAL_EXPERTS}" = "1" ]; then
  LORA_ARGS+=(--sglang-lora-use-virtual-experts)
fi

ROLLOUT_ARGS=(
  --prompt-data "${DATA_DIR}/grpo/train.jsonl"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --apply-chat-template
  --custom-rm-path "${CUSTOM_RM_PATH}"
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
)
if [ "${ROLLOUT_SKIP_SPECIAL_TOKENS}" = "1" ]; then
  ROLLOUT_ARGS+=(--rollout-skip-special-tokens)
elif [ "${ROLLOUT_SKIP_SPECIAL_TOKENS}" != "0" ]; then
  echo "MILES_ROLLOUT_SKIP_SPECIAL_TOKENS must be 0 or 1, got: ${ROLLOUT_SKIP_SPECIAL_TOKENS}" >&2
  exit 2
fi
if [ "${#ROLLOUT_STOP_TOKEN_ID_ARGS[@]}" -gt 0 ]; then
  ROLLOUT_ARGS+=(--rollout-stop-token-ids "${ROLLOUT_STOP_TOKEN_ID_ARGS[@]}")
fi
if [ -n "${APPLY_CHAT_TEMPLATE_KWARGS}" ]; then
  ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi
if [ "${GRPO_ROLLOUT_SHUFFLE}" = "1" ]; then
  ROLLOUT_ARGS+=(--rollout-shuffle)
fi

# Prefer the stratified mini eval when the caller did not pick one: the full
# validation set is a standalone gate, not an in-training trend eval, and it
# costs ~10x the wall-clock per eval interval.
if [ -z "${EVAL_PROMPT_DATA}" ] && [ -f "${DATA_DIR}/eval/validation_mini126.jsonl" ]; then
  EVAL_PROMPT_DATA="${DATA_DIR}/eval/validation_mini126.jsonl"
fi

EVAL_ARGS=(
  --eval-interval "${EVAL_INTERVAL}"
  --eval-prompt-data "${EVAL_NAME}" "${EVAL_PROMPT_DATA:-${DATA_DIR}/eval/validation.jsonl}"
  --eval-input-key prompt
  --eval-label-key label
  --n-samples-per-eval-prompt "${EVAL_N_SAMPLES_PER_PROMPT}"
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  --eval-top-p 1
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --seq-length "${SEQ_LENGTH}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [ "${USE_DYNAMIC_BATCH_SIZE}" = "1" ]; then
  PERF_ARGS+=(--use-dynamic-batch-size)
fi
if [ "${BALANCE_DATA}" = "1" ]; then
  PERF_ARGS+=(--balance-data)
fi
# Select the activation recompute policy.
case "${RECOMPUTE_GRANULARITY}" in
  full)
    PERF_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
    ;;
  selective)
    PERF_ARGS+=(--recompute-granularity selective)
    ;;
  none) ;;
  *)
    echo "MILES_RECOMPUTE_GRANULARITY must be full|selective|none, got: ${RECOMPUTE_GRANULARITY}" >&2
    exit 2
    ;;
esac
if [ -n "${MOE_TOKEN_DISPATCHER_TYPE}" ]; then
  PERF_ARGS+=(--moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}")
fi
if [ "${MOE_ENABLE_DEEPEP}" = "1" ]; then
  PERF_ARGS+=(--moe-enable-deepep)
fi
if [ -n "${ATTENTION_BACKEND}" ]; then
  PERF_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)
# The KL penalty coefficient above is inert unless --use-kl-loss is also set; the
# canonical PIE path leaves it off, so gate it behind an opt-in env var. Requires a
# reference model (MILES_NO_REF must not be 1).
if [ "${MILES_USE_KL_LOSS:-0}" = "1" ]; then
  if [ "${MILES_NO_REF:-0}" = "1" ]; then
    echo "MILES_USE_KL_LOSS=1 requires a reference model (MILES_NO_REF must not be 1)" >&2
    exit 2
  fi
  GRPO_ARGS+=(--use-kl-loss)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${MILES_LR:-1e-5}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

WANDB_ARGS=(
  --use-wandb
  --wandb-dir "${RUN_ROOT}/wandb"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${WANDB_RUN_ID}"
  --log-passrate
  --log-correct-samples
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${GPUS_PER_NODE}"
  --sglang-dtype bfloat16
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-moe-runner-backend triton
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
    --sglang-speculative-num-steps "${SGLANG_SPECULATIVE_NUM_STEPS}"
    --sglang-speculative-eagle-topk "${SGLANG_SPECULATIVE_EAGLE_TOPK}"
    --sglang-speculative-num-draft-tokens "${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS}"
  )
fi
if [ -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]; then
  SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
fi
if [ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)
fi
if [ -n "${SGLANG_ATTENTION_BACKEND}" ]; then
  SGLANG_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --save-debug-rollout-data "${ROLLOUT_DUMP_TEMPLATE}"
)
if [ "${ROLLOUT_ONLY}" = "1" ]; then
  MISC_ARGS+=(--debug-rollout-only)
fi
# Raw passthrough for experiments (e.g. --sglang-disable-cuda-graph); appended last.
if [ -n "${MILES_EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS <<< "${MILES_EXTRA_ARGS}"
  MISC_ARGS+=("${EXTRA_ARGS[@]}")
fi

ray start --head \
  --node-ip-address "${RAY_NODE_IP_ADDRESS}" \
  --num-gpus "${GPUS_PER_NODE}" \
  --disable-usage-stats \
  --dashboard-host="${RAY_DASHBOARD_HOST}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${REPO_ROOT}/src:${MILES_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"${CUDA_DEVICE_MAX_CONNECTIONS}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"GLM47_DATA_DIR\": \"${DATA_DIR}\",
    \"GLM47_CPP_SANDBOX_IMAGE\": \"${GLM47_CPP_SANDBOX_IMAGE}\",
    \"GLM47_CPP_SANDBOX_BACKEND\": \"${GLM47_CPP_SANDBOX_BACKEND:-docker}\",
    \"GLM47_CPP_SANDBOX_UNSHARE_NET\": \"${GLM47_CPP_SANDBOX_UNSHARE_NET:-1}\",
    \"GLM47_ROUTER_READY_TIMEOUT_S\": \"${GLM47_ROUTER_READY_TIMEOUT_S:-}\",
    \"GLM47_CPP_SANDBOX_CPU\": \"${GLM47_CPP_SANDBOX_CPU:-1}\",
    \"GLM47_CPP_REWARD_WORKERS\": \"${GLM47_CPP_REWARD_WORKERS:-8}\",
    \"NVSHMEM_DISABLE_NCCL\": \"${NVSHMEM_DISABLE_NCCL:-}\",
    \"WANDB_RUN_ID\": \"${WANDB_RUN_ID}\",
    \"WANDB_JOB_TYPE\": \"${WANDB_JOB_TYPE}\",
    \"WANDB_MODE\": \"${WANDB_MODE:-online}\",
    \"WANDB_RUN_GROUP\": \"${WANDB_RUN_GROUP:-${WANDB_GROUP}}\",
    \"WANDB_TAGS\": \"${WANDB_TAGS:-}\",
    \"GLM47_EXPERIMENT_ID\": \"${EXPERIMENT_ID}\",
    \"GLM47_TIMING_STATUS\": \"${GLM47_TIMING_STATUS:-unverified}\",
    \"GLM47_REGISTER_BRIDGE\": \"${GLM47_REGISTER_BRIDGE:-}\",
    \"GLM47_DISABLE_SHARED_LORA_CKPT_PATCH\": \"${GLM47_DISABLE_SHARED_LORA_CKPT_PATCH:-}\",
    \"GLM47_SYNC_METRICS_DIR\": \"${GLM47_SYNC_METRICS_DIR:-}\"
  }
}"

set +e
TRAIN_ENTRYPOINT=(python3 train.py)
if [ -n "${TRAIN_MODULE}" ]; then
  TRAIN_ENTRYPOINT=(python3 -m "${TRAIN_MODULE}")
fi
ray job submit --address="http://${RAY_DASHBOARD_HOST}:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_ENTRYPOINT[@]}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
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
  "${LORA_ARGS[@]}"
RAY_STATUS=$?
set -e

cleanup

# An eval dump at rollout id k reflects the policy after k optimizer updates,
# so the trained final state is only ever measured by an eval with id >=
# NUM_ROLLOUT. Issue #110 r3 shipped eval_0 (the frozen pre-update policy) as
# "post-update validation"; record the truth in the receipt so that mislabel
# cannot recur, and let launch profiles hard-require the post-update eval.
POST_UPDATE_EVAL="absent"
POST_UPDATE_EVAL_MAX_ID=""
ROLLOUT_DUMP_DIR="$(dirname -- "${ROLLOUT_DUMP_TEMPLATE}")"
if [ -d "${ROLLOUT_DUMP_DIR}" ]; then
  POST_UPDATE_EVAL_MAX_ID="$(find "${ROLLOUT_DUMP_DIR}" -maxdepth 1 -name 'grpo_eval_*.pt' 2>/dev/null \
    | sed -E 's/.*grpo_eval_([0-9]+)\.pt$/\1/' | sort -n | tail -n 1)"
  if [ -n "${POST_UPDATE_EVAL_MAX_ID}" ] && [ "${POST_UPDATE_EVAL_MAX_ID}" -ge "${NUM_ROLLOUT}" ]; then
    POST_UPDATE_EVAL="present"
  fi
fi
EVAL_REQUIREMENT_STATUS=0
if [ "${MILES_REQUIRE_POST_UPDATE_EVAL:-0}" = "1" ] \
  && [ "${ROLLOUT_ONLY}" = "0" ] \
  && [ "${RAY_STATUS}" -eq 0 ] \
  && [ "${POST_UPDATE_EVAL}" != "present" ]; then
  echo "MILES_REQUIRE_POST_UPDATE_EVAL=1 but no eval dump with rollout id >= ${NUM_ROLLOUT} exists in ${ROLLOUT_DUMP_DIR}; the trained policy was never evaluated" >&2
  EVAL_REQUIREMENT_STATUS=1
fi

TRAINING_GATE_STATUS="not_requested"
GATE_STATUS=0
if [ "${RAY_STATUS}" -eq 0 ] && [ "${ROLLOUT_ONLY}" = "1" ]; then
  TRAINING_GATE_STATUS="not_applicable_rollout_only"
elif [ "${RAY_STATUS}" -eq 0 ] && [ "${EXPECTED_DATASET_KIND}" = "aider-polyglot-cpp-shadow-grpo" ]; then
  TRAINING_GATE_STATUS="failed"
  set +e
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${REPO_ROOT}/scripts/create_grpo_training_gate.py" \
    --run-id "${RUN_ID}" \
    --run-root "${RUN_ROOT}" \
    --save-dir "${SAVE_DIR}" \
    --data-manifest "${DATA_DIR}/manifest.json" \
    --source-adapter "${MILES_LORA_SOURCE_ADAPTER_PATH:-}" \
    --expected-source-adapter-sha256 "${MILES_EXPECTED_SOURCE_ADAPTER_SHA256:-}" \
    --hybrid-manifest "${MILES_LORA_ADAPTER_PATH:-}/mtp_strip_manifest.json" \
    --source-commit "${GLM47_SOURCE_COMMIT:-unbound}" \
    --phase "${GLM47_TIMING_STATUS:-full}" \
    --num-rollout "${NUM_ROLLOUT}" \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --expected-native-shards "${MILES_EXPECTED_NATIVE_SHARDS:-${TP_SIZE}}" \
    --expected-source-native-shards "${MILES_EXPECTED_SOURCE_NATIVE_SHARDS:-${MILES_EXPECTED_NATIVE_SHARDS:-${TP_SIZE}}}" \
    --expected-train-count "${MILES_EXPECTED_TRAIN_COUNT:-253}" \
    --output "${TRAINING_GATE}"
  GATE_STATUS=$?
  set -e
  if [ "${GATE_STATUS}" -eq 0 ]; then
    TRAINING_GATE_STATUS="passed"
  fi
fi
if [ "${RAY_STATUS}" -eq 0 ] && [ "${GATE_STATUS}" -eq 0 ] && [ "${EVAL_REQUIREMENT_STATUS}" -eq 0 ]; then
  STAGE_STATUS="success"
else
  STAGE_STATUS="failed"
fi
write_receipt "${STAGE_STATUS}" "${RAY_STATUS}"
set +e
finalize_wandb "${STAGE_STATUS}"
FINALIZE_STATUS=$?
set -e
if [ "${RAY_STATUS}" -ne 0 ]; then
  exit "${RAY_STATUS}"
fi
if [ "${GATE_STATUS}" -ne 0 ]; then
  exit "${GATE_STATUS}"
fi
if [ "${EVAL_REQUIREMENT_STATUS}" -ne 0 ]; then
  exit "${EVAL_REQUIREMENT_STATUS}"
fi
exit "${FINALIZE_STATUS}"
