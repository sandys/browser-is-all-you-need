#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GLM47_EVAL_TAG=synthmem-50ep \
GLM47_EXPECTED_TRAINING_PHASE=sft \
GLM47_EXPECTED_TRAINING_TASK_COUNT=260 \
GLM47_EVAL_LORA_RANK=16 \
GLM47_EVAL_DISABLE_THINKING=0 \
modal run --detach "$SCRIPT_DIR/aider_eval_app.py"::launch_fixed26_eval_plan \
  --plan-json "$SCRIPT_DIR/plan.json"
