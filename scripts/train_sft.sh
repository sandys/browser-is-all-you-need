#!/usr/bin/env bash
# Miles SFT LoRA rank-16 runner for GLM-4.7-Flash on the PIE C++ task.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
MILES_ROOT="${MILES_ROOT:-/root/miles}"
PYTHON_BIN="${MILES_PYTHON:-python3}"
MODEL_ARGS_FILE="${MILES_MODEL_ARGS_FILE:-glm4.7-flash.sh}"
MODEL_ARGS_PATH="${MILES_MODEL_ARGS_PATH:-${MILES_ROOT}/scripts/models/${MODEL_ARGS_FILE}}"

RUN_ID="${MILES_RUN_ID:-glm47_h100_pie_cpp_lora_r16_sft_$(date +%Y%m%d_%H%M%S)}"
STAGE_STARTED_AT="${SECONDS}"
RUN_ROOT="${MILES_RUN_ROOT:-${REPO_ROOT}/.glm47-posttraining/miles/glm47-h100-cpp-perf/runs/${RUN_ID}}"
DATA_DIR="${MILES_CPP_DATA_DIR:-${RUN_ROOT}/data}"
TASKS_DIR="${MILES_CPP_TASKS_DIR:-${REPO_ROOT}/.glm47-posttraining/data/tasks-small}"
TRAIN_LIMIT="${MILES_CPP_TRAIN_LIMIT:-}"
EVAL_LIMIT="${MILES_CPP_EVAL_LIMIT:-}"
EVAL_SPLITS="${MILES_CPP_EVAL_SPLITS:-validation,test}"
SORT_BY_SIZE="${MILES_CPP_SORT_BY_SIZE:-1}"
AUTO_PREPARE_DATA="${MILES_CPP_AUTO_PREPARE_DATA:-1}"
FILTER_TRAIN_ORACLE_FULL_MARKS="${MILES_CPP_FILTER_TRAIN_ORACLE_FULL_MARKS:-0}"
ORACLE_FILTER_WORKERS="${MILES_CPP_ORACLE_FILTER_WORKERS:-8}"

HF_CHECKPOINT="${MILES_HF_CHECKPOINT:-/root/models/GLM-4.7-Flash}"
REF_LOAD_DIR="${MILES_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist_tp4_pp1_ep8}"
SAVE_DIR="${MILES_SAVE_DIR:-${RUN_ROOT}/checkpoints/sft_lora_r16}"
SAVE_INTERVAL="${MILES_SAVE_INTERVAL:-1000}"

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

SFT_NUM_EPOCH="${MILES_SFT_NUM_EPOCH:-1}"
START_ROLLOUT_ID="${MILES_START_ROLLOUT_ID:-0}"
ROLLOUT_BATCH_SIZE="${MILES_ROLLOUT_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${MILES_GLOBAL_BATCH_SIZE:-32}"
SFT_ROLLOUT_SHUFFLE="${MILES_SFT_ROLLOUT_SHUFFLE:-1}"
SFT_ROLLOUT_FUNCTION_PATH="${MILES_SFT_ROLLOUT_FUNCTION_PATH:-miles.rollout.sft_rollout.generate_rollout}"
TRAIN_MODULE="${MILES_TRAIN_MODULE:-}"

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
SGLANG_MEM_FRACTION_STATIC="${MILES_SGLANG_MEM_FRACTION_STATIC:-0.20}"
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

WANDB_PROJECT="${MILES_WANDB_PROJECT:-glm47-h100-cpp-sft}"
WANDB_GROUP="${MILES_WANDB_GROUP:-glm47-h100-pie-cpp-lora-r16-sft}"
WANDB_RUN_ID="${MILES_WANDB_RUN_ID:-${RUN_ID}}"
WANDB_JOB_TYPE="${MILES_WANDB_JOB_TYPE:-${WANDB_JOB_TYPE:-sft}}"
EXPERIMENT_ID="${GLM47_EXPERIMENT_ID:-${WANDB_GROUP}}"

STAGE_ROOT="${RUN_ROOT}/sft_lora_r16"
LOG_FILE="${STAGE_ROOT}/run.log"
VRAM_LOG="${STAGE_ROOT}/vram_usage.csv"
VRAM_PEAK_FILE="${STAGE_ROOT}/vram_peak.txt"
RUN_RECEIPT="${STAGE_ROOT}/run_receipt.txt"
ROLLOUT_DUMP_TEMPLATE="${RUN_ROOT}/rollout_dumps/sft_{rollout_id}.pt"

mkdir -p "${STAGE_ROOT}" "${RUN_ROOT}/rollout_dumps" "${SAVE_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "run_id=${RUN_ID}"
echo "run_root=${RUN_ROOT}"
echo "tasks_dir=${TASKS_DIR}"
echo "data_dir=${DATA_DIR}"
echo "hf_checkpoint=${HF_CHECKPOINT}"
echo "model_args_path=${MODEL_ARGS_PATH}"
echo "ref_load=${REF_LOAD_DIR}"
echo "save_dir=${SAVE_DIR}"
echo "sft_num_epoch=${SFT_NUM_EPOCH}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
echo "sft_rollout_shuffle=${SFT_ROLLOUT_SHUFFLE}"
echo "lora_rank=${LORA_RANK}"

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
if ! command -v ray >/dev/null 2>&1; then
  echo "Missing ray CLI. Run inside the Miles runtime container." >&2
  exit 2
fi
# Validate the selected C++ reward backend.
if [ "${GLM47_CPP_SANDBOX_BACKEND:-docker}" != "local" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Missing docker CLI inside container. Mount it with -v /usr/bin/docker:/usr/bin/docker:ro." >&2
    echo "GLM47_CPP_SANDBOX_BACKEND=${GLM47_CPP_SANDBOX_BACKEND:-<unset>}" >&2
    exit 2
  fi
fi

prepare_data() {
  if [ ! -d "${TASKS_DIR}" ]; then
    echo "Missing task JSON directory: ${TASKS_DIR}" >&2
    exit 2
  fi

  BUILD_DATA_ARGS=(
    -m glm47_posttraining.integrations.miles_cpp_perf build-data
    --tasks-dir "${TASKS_DIR}"
    --out "${DATA_DIR}"
    --eval-splits "${EVAL_SPLITS}"
    --profile "glm47-h100-sft"
    --run-id "${RUN_ID}"
    --force
  )
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

  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" "${BUILD_DATA_ARGS[@]}"
}

if [ ! -f "${DATA_DIR}/manifest.json" ]; then
  if [ "${AUTO_PREPARE_DATA}" != "1" ]; then
    echo "Missing Miles C++ data manifest: ${DATA_DIR}/manifest.json" >&2
    exit 2
  fi
  prepare_data
fi
if [ ! -f "${DATA_DIR}/sft/train.jsonl" ]; then
  echo "Missing SFT train data: ${DATA_DIR}/sft/train.jsonl" >&2
  exit 2
fi

monitor_vram() {
  echo "timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw" > "${VRAM_LOG}"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits >> "${VRAM_LOG}" || true
    sleep 2
  done
}

write_receipt() {
  local status="$1"
  local ray_status="$2"
  local max_memory_used_mib=""
  if [ -s "${VRAM_LOG}" ]; then
    awk -F, 'NR>1 {gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($3+0 > max) max=$3+0} END {print "max_memory_used_mib=" max}' "${VRAM_LOG}" > "${VRAM_PEAK_FILE}" || true
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
hf_checkpoint=${HF_CHECKPOINT}
model_args_path=${MODEL_ARGS_PATH}
ref_load=${REF_LOAD_DIR}
save_dir=${SAVE_DIR}
seq_length=${SEQ_LENGTH}
gpus_per_node=${GPUS_PER_NODE}
tensor_model_parallel_size=${TP_SIZE}
pipeline_model_parallel_size=${PP_SIZE}
context_parallel_size=${CP_SIZE}
expert_model_parallel_size=${EP_SIZE}
expert_tensor_parallel_size=${ETP_SIZE}
max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}
micro_batch_size=${MICRO_BATCH_SIZE}
use_dynamic_batch_size=${USE_DYNAMIC_BATCH_SIZE}
balance_data=${BALANCE_DATA}
sft_num_epoch=${SFT_NUM_EPOCH}
sft_rollout_shuffle=${SFT_ROLLOUT_SHUFFLE}
rollout_batch_size=${ROLLOUT_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
lora_rank=${LORA_RANK}
lora_alpha=${LORA_ALPHA}
lora_base_cpu_backup=${LORA_BASE_CPU_BACKUP}
no_gradient_accumulation_fusion=${NO_GRADIENT_ACCUMULATION_FUSION}
sft_rollout_function_path=${SFT_ROLLOUT_FUNCTION_PATH}
train_module=${TRAIN_MODULE}
moe_token_dispatcher_type=${MOE_TOKEN_DISPATCHER_TYPE}
moe_enable_deepep=${MOE_ENABLE_DEEPEP}
recompute_granularity=${RECOMPUTE_GRANULARITY}
cuda_device_max_connections=${CUDA_DEVICE_MAX_CONNECTIONS}
sglang_mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC}
sglang_cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS}
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
timing_status=${GLM47_TIMING_STATUS:-unverified}
extra_args=${MILES_EXTRA_ARGS:-}
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
  PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${REPO_ROOT}/scripts/publish_results.py" "${finalize_args[@]}"
}

monitor_vram &
VRAM_MONITOR_PID=$!
cleanup() {
  if [ -n "${VRAM_MONITOR_PID:-}" ]; then
    kill "${VRAM_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${VRAM_MONITOR_PID}" >/dev/null 2>&1 || true
    VRAM_MONITOR_PID=""
  fi
}
trap cleanup EXIT

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
  --save-interval "${SAVE_INTERVAL}"
  --megatron-to-hf-mode bridge
)
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

SFT_ARGS=(
  --rollout-function-path "${SFT_ROLLOUT_FUNCTION_PATH}"
	  --prompt-data "${DATA_DIR}/sft/train.jsonl"
	  --input-key messages
	  --metadata-key metadata
	  --num-epoch "${SFT_NUM_EPOCH}"
	  --start-rollout-id "${START_ROLLOUT_ID}"
	  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
	  --debug-train-only
	)
if [ "${SFT_ROLLOUT_SHUFFLE}" = "1" ]; then
  SFT_ARGS+=(--rollout-shuffle)
fi

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

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${MILES_LR:-1e-5}"
  --lr-decay-style cosine
  --min-lr "${MILES_MIN_LR:-1e-6}"
  --lr-warmup-fraction "${MILES_LR_WARMUP_FRACTION:-0.1}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

WANDB_ARGS=(
  --use-wandb
  --wandb-dir "${RUN_ROOT}/wandb"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${WANDB_RUN_ID}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${GPUS_PER_NODE}"
  --sglang-dtype bfloat16
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
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

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --save-debug-rollout-data "${ROLLOUT_DUMP_TEMPLATE}"
)
# Raw passthrough for measured runtime experiments; appended last so an
# explicit setting can override Miles' colocated defaults.
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

RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<PY
import json
import os

paths = ["/root/Megatron-LM", "${REPO_ROOT}/src", "${MILES_ROOT}", os.environ.get("PYTHONPATH", "")]
env = {
    "PYTHONPATH": ":".join(path for path in paths if path),
    "CUDA_DEVICE_MAX_CONNECTIONS": "${CUDA_DEVICE_MAX_CONNECTIONS}",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "GLM47_DATA_DIR": "${DATA_DIR}",
    "GLM47_CPP_SANDBOX_IMAGE": os.environ.get("GLM47_CPP_SANDBOX_IMAGE", "glm47-cpp-perf:latest"),
    "GLM47_CPP_SANDBOX_BACKEND": os.environ.get("GLM47_CPP_SANDBOX_BACKEND", "docker"),
    "GLM47_CPP_SANDBOX_CPU": os.environ.get("GLM47_CPP_SANDBOX_CPU", "1"),
    "GLM47_CPP_REWARD_WORKERS": os.environ.get("GLM47_CPP_REWARD_WORKERS", "8"),
    "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", ""),
    "WANDB_RUN_ID": "${WANDB_RUN_ID}",
}
for key in (
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "HF_HOME",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_BASE_URL",
    "WANDB_MODE",
    "WANDB_JOB_TYPE",
    "WANDB_RUN_GROUP",
    "WANDB_TAGS",
    "GLM47_EXPERIMENT_ID",
    "GLM47_TIMING_STATUS",
    "GLM47_REGISTER_BRIDGE",
    "GLM47_DISABLE_SHARED_LORA_CKPT_PATCH",
    "GLM47_SYNC_METRICS_DIR",
):
    if key in os.environ:
        env[key] = os.environ[key]
print(json.dumps({"env_vars": env}))
PY
)"

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
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${LORA_ARGS[@]}"
RAY_STATUS=$?
set -e

cleanup
if [ "${RAY_STATUS}" -eq 0 ]; then
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
exit "${FINALIZE_STATUS}"
