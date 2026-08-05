#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="glm47-aider-sft-v5-aiderfmt-1117-3ep-20260726T211943Z"
CHECKPOINT="iter_0000164"
MODEL_REVISION="5d06951941a30939920fb2b7558aa95085531d52"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/tmp/glm47-aiderfmt-1117}"
EVAL_RUN_ID="${EVAL_RUN_ID:-glm47-aider-sft-v5-four-trial-repro-$(date -u +%Y%m%dT%H%M%SZ)}"

for command in hf modal; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 1
  }
done

hf download TokenBender/glm47-aider-sft-v5-aiderfmt-1117-3ep \
  --revision "$MODEL_REVISION" \
  --local-dir "$DOWNLOAD_ROOT"

modal volume put glm47-runs "$DOWNLOAD_ROOT/final/adapter" \
  "$RUN_ID/checkpoints/sft_lora_r16/$CHECKPOINT/adapter"
modal volume put glm47-runs "$DOWNLOAD_ROOT/training" \
  "$RUN_ID/sft_lora_r16"

GLM47_EVAL_TAG=sft-v5-aiderfmt-1117 \
GLM47_TRAINING_GATE_KIND=glm47-aider-sft-training-gate \
GLM47_EXPECTED_TRAINING_PHASE=sft \
GLM47_EXPECTED_TRAINING_TASK_COUNT=1117 \
GLM47_EVAL_LORA_RANK=16 \
modal run "$SCRIPT_DIR/aider_eval_app.py"::pass_at_k \
  --adapter-path "/runs/$RUN_ID/checkpoints/sft_lora_r16/$CHECKPOINT/adapter" \
  --expected-adapter-sha256 \
    9608cfe476b3bd2573ffd02f334f55a18e80a9de00a0581d1d6aacc54ad8972e \
  --expected-data-manifest-sha256 \
    db1df88ed80bf3b8db8c857ec2cec61228323609547a95082d5d53e0ed60ec4b \
  --expected-training-phase sft \
  --run-id "$EVAL_RUN_ID" \
  --samples 4 \
  --tries 2
