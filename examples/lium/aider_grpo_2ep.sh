#!/usr/bin/env bash
set -euo pipefail

repo_root="${GLM47_REPRO_REPO_ROOT:-/workspace/glm47}"
assets_root="${GLM47_REPRO_ASSETS_ROOT:-/workspace/assets}"
runs_root="${GLM47_REPRO_RUNS_ROOT:-/workspace/runs}"
run_id="${GLM47_REPRO_RUN_ID:-glm47-aider-grpo169-merge1211-530-r32-2ep-fixed-repro-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${runs_root}/${run_id}"

expected_manifest_sha256="a7e54c0245b97ae78f9b2fa57ff5278844585cf03004254137b6cfc8e91ef157"
expected_train_sha256="b72394ab603b4b6faf22370ea70605446f112ab50c883eb61e308e2dd9ab4dd2"
expected_adapter_sha256="dbea7d3e2d6603f278b94c6be134bca83bb5f0ebdc4840eb53898ec5b3affb91"
expected_shadow_manifest_sha256="002993b94ddf85e23863e22484459df4b724d91204e5e48c37904a1f34748f00"
data_root="${assets_root}/prepared-aider-169"
adapter_root="${assets_root}/merged-1211-530-r32"
aider_tasks_root="${GLM47_AIDER_TASKS_DIR:-${assets_root}/aider-shadow/tasks/aider_polyglot_cpp_shadow}"

verify_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  if [[ ! -f "${path}" ]]; then
    echo "missing required reproduction input: ${path}" >&2
    exit 1
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${path}" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "${path}" | awk '{print $1}')"
  fi
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: ${actual} != ${expected}" >&2
    exit 1
  fi
}

verify_sha256 "${expected_manifest_sha256}" "${data_root}/manifest.json"
verify_sha256 "${expected_train_sha256}" "${data_root}/grpo/train.jsonl"
verify_sha256 "${expected_adapter_sha256}" "${adapter_root}/adapter_model.bin"
verify_sha256 "${expected_shadow_manifest_sha256}" "${aider_tasks_root}/manifest.json"

source_commit="$(git -C "${repo_root}" rev-parse HEAD)"

export GLM47_CPP_REWARD_WORKERS=32
export GLM47_CPP_SANDBOX_BACKEND=docker
export GLM47_CPP_SANDBOX_CPU=1
export GLM47_CPP_SANDBOX_IMAGE=glm47-aider-polyglot-cpp:latest
export GLM47_EXPERIMENT_ID="${run_id}"
export GLM47_MODEL_REVISION=7dd20894a642a0aa287e9827cb1a1f7f91386b67
export GLM47_REGISTER_BRIDGE=1
export GLM47_SOURCE_COMMIT="${source_commit}"
export GLM47_SYNC_METRICS_DIR="${run_root}/sync_metrics"
export GLM47_TIMING_STATUS=full
export GLM47_TRAINING_IMAGE='radixark/miles:latest-cu12@sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37'

# examples/grpo.sh appends its parameter-expansion closing brace to this value.
export MILES_APPLY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false'
export MILES_ATTENTION_BACKEND=flash
export MILES_BALANCE_DATA=1
export MILES_CONTEXT_PARALLEL_SIZE=1
export MILES_CPP_DATA_DIR="${data_root}"
export MILES_CPP_TASKS_DIR="${aider_tasks_root}"
export MILES_CUSTOM_RM_PATH=glm47_posttraining.integrations.miles_aider_polyglot.reward_func
export MILES_DATA_BUILD_MODULE=glm47_posttraining.integrations.miles_aider_polyglot
export MILES_EVAL_INTERVAL=1
export MILES_EVAL_MAX_RESPONSE_LEN=4096
export MILES_EVAL_NAME=aider_shadow_train_monitor
export MILES_EVAL_N_SAMPLES_PER_PROMPT=1
export MILES_EVAL_PROMPT_DATA="${data_root}/eval/train_monitor.jsonl"
export MILES_EXPECTED_DATASET_KIND=aider-polyglot-cpp-shadow-grpo
export MILES_EXPECTED_NATIVE_SHARDS=4
export MILES_EXPECTED_SOURCE_ADAPTER_SHA256="${expected_adapter_sha256}"
export MILES_EXPECTED_SOURCE_TENSORS=9741
export MILES_EXPECTED_STRIPPED_TENSORS=207
export MILES_EXPECTED_TRAIN_COUNT=169
export MILES_EXPERTS_SHARED_OUTER_LORAS=1
export MILES_EXPERT_MODEL_PARALLEL_SIZE=8
export MILES_EXPERT_TENSOR_PARALLEL_SIZE=1
export MILES_GLOBAL_BATCH_SIZE=256
export MILES_GPUS_PER_NODE=8
export MILES_GRPO_ROLLOUT_SHUFFLE=1
export MILES_HF_CHECKPOINT="${GLM47_REPRO_MODEL_PATH:-/workspace/models/GLM-4.7-Flash}"
export MILES_KL_LOSS_COEF=0.02
export MILES_LORA_ADAPTER_PATH="${adapter_root}"
export MILES_LORA_ALPHA=32
export MILES_LORA_BASE_CPU_BACKUP=1
export MILES_LORA_RANK=32
export MILES_LORA_TARGET_MODULES=q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj
export MILES_LR=5e-7
export MILES_MAX_TOKENS_PER_GPU=12288
export MILES_MICRO_BATCH_SIZE=1
export MILES_MODEL_ARGS_FILE=glm4.7-flash.sh
export MILES_MOE_ENABLE_DEEPEP=1
export MILES_MOE_TOKEN_DISPATCHER_TYPE=flex
export MILES_NO_GRADIENT_ACCUMULATION_FUSION=1
export MILES_NO_REF=0
export MILES_NUM_ROLLOUT=11
export MILES_N_SAMPLES_PER_PROMPT=8
export MILES_PIPELINE_MODEL_PARALLEL_SIZE=1
export MILES_RECOMPUTE_GRANULARITY=full
export MILES_REF_LOAD_DIR="${GLM47_REPRO_REF_LOAD_DIR:-/workspace/models/GLM-4.7-Flash_torch_dist_tp4_pp1_ep8}"
export MILES_REWARD_PREFLIGHT_MODULE=glm47_posttraining.integrations.miles_aider_polyglot
export MILES_ROLLOUT_BATCH_SIZE=32
export MILES_ROLLOUT_MAX_RESPONSE_LEN=4096
export MILES_ROLLOUT_SKIP_SPECIAL_TOKENS=1
export MILES_ROLLOUT_STOP_TOKEN_IDS='154820 154827 154829'
export MILES_ROLLOUT_TEMPERATURE=0.7
export MILES_RUN_ID="${run_id}"
export MILES_RUN_ROOT="${run_root}"
export MILES_SAVE_INTERVAL=1
export MILES_SEQ_LENGTH=6144
export MILES_SGLANG_ATTENTION_BACKEND=flashinfer
export MILES_SGLANG_CUDA_GRAPH_MAX_BS=64
export MILES_SGLANG_DISABLE_CUSTOM_ALL_REDUCE=0
export MILES_SGLANG_DP_SIZE=8
export MILES_SGLANG_ENABLE_DP_ATTENTION=1
export MILES_SGLANG_ENABLE_DP_LM_HEAD=1
export MILES_SGLANG_LORA_TARGET_MODULES=q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj
export MILES_SGLANG_LORA_USE_VIRTUAL_EXPERTS=1
export MILES_SGLANG_MAX_RUNNING_REQUESTS=256
export MILES_SGLANG_MEM_FRACTION_STATIC=0.75
export MILES_SGLANG_MOE_DENSE_TP_SIZE=1
export MILES_SGLANG_SERVER_CONCURRENCY=1024
export MILES_SGLANG_SPECULATIVE=0
export MILES_TENSOR_MODEL_PARALLEL_SIZE=4
export MILES_TRAIN_MODULE=glm47_posttraining.integrations.miles_train_with_glm47_bridge
export MILES_USE_DYNAMIC_BATCH_SIZE=1
export MILES_USE_KL_LOSS=1
export MILES_WANDB_GROUP="${run_id}"
export MILES_WANDB_JOB_TYPE=grpo
export MILES_WANDB_PROJECT=glm47-aider-polyglot-cpp-grpo
export MILES_WANDB_RUN_ID="${run_id}"

export NCCL_DEBUG=WARN
export NVSHMEM_DISABLE_NCCL=1
export WANDB_MODE=offline
export WANDB_TAGS=canonical,aider-polyglot-cpp,grpo,2epoch,8xh100,lium,parser-fixed

mkdir -p "${run_root}"
cd "${repo_root}"
exec bash examples/grpo.sh
