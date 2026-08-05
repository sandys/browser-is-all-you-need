#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${LUNA_REPRO_ROOT:-$SCRIPT_DIR/work/$(date -u +%Y%m%dT%H%M%SZ)}"
AIDER_ROOT="$RUN_ROOT/aider"
POLYGLOT_ROOT="$RUN_ROOT/polyglot-benchmark"
SOURCE_ROOT="$RUN_ROOT/source"
ARTIFACT_ROOT="$RUN_ROOT/output"
VENV=/private/tmp/aider-luna-venv

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"

mkdir -p "$RUN_ROOT" "$SOURCE_ROOT/contract" "$ARTIFACT_ROOT/provenance"
git clone https://github.com/Aider-AI/aider.git "$AIDER_ROOT"
git -C "$AIDER_ROOT" checkout --detach 5dc9490bb35f9729ef2c95d00a19ccd30c26339c
cp "$SCRIPT_DIR/benchmark.py" "$AIDER_ROOT/benchmark/benchmark.py"
git clone https://github.com/Aider-AI/polyglot-benchmark.git "$POLYGLOT_ROOT"
git -C "$POLYGLOT_ROOT" checkout --detach 7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
mkdir -p "$SOURCE_ROOT/provenance/polyglot-benchmark/cpp/exercises"
cp -R "$POLYGLOT_ROOT/cpp/exercises/practice" \
  "$SOURCE_ROOT/provenance/polyglot-benchmark/cpp/exercises/practice"
cp "$SCRIPT_DIR/source-audit.json" "$SOURCE_ROOT/contract/audit.json"
cp "$SCRIPT_DIR/model-settings.yml" "$ARTIFACT_ROOT/provenance/model-settings.yml"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$AIDER_ROOT[dev]"

export LUNA_EXPECTED_REASONING_EFFORT=medium
export LUNA_ALLOW_AUTHORIZED_FEEDBACK=1
export LUNA_EXPECTED_TREE_SHA256=f8fc8bcea4f17297f57c911912ac44118371c696f7f46b15b6969f1e7985e5b3
export LUNA_EVAL_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export LUNA_EVAL_SOURCE_ARTIFACT="$SOURCE_ROOT"
export LUNA_EVAL_AIDER_ROOT="$AIDER_ROOT"
export LUNA_EVAL_EXERCISES_ROOT="$SOURCE_ROOT/provenance/polyglot-benchmark"
export LUNA_EVAL_OVERLAY_AUDIT="$SOURCE_ROOT/contract/audit.json"
export LUNA_EVAL_TEST_COMMAND="$SCRIPT_DIR/luna_fixed26_macos_test_command.py"
export LUNA_EVAL_LABEL_PREFIX=gpt56-luna-openrouter-fixed26-pristine-v1-repro
export LUNA_EVAL_EXPECTED_BENCHMARK_SHA256=d5a8900a834e22afa991948aa54dc2047073357e5f24dd58547fa078c052ad8f
export LUNA_EVAL_FEEDBACK_ONLY=1
export LUNA_EVAL_SINGLE_TRIALS=4
export LUNA_EXPECTED_OVERLAY_VERSION=pristine-original-v1
export LUNA_MODAL_SCORER_APP_NAME=luna-fixed26-pristine-original-v1-macos-scorer
export LUNA_FIXED26_SCORER_VARIANT=pristine-original-v1
export PYTHONHASHSEED=0

python3 "$SCRIPT_DIR/luna_low_openrouter_chat_proxy.py" \
  --artifact-dir "$ARTIFACT_ROOT/transport" \
  --practice-root "$SOURCE_ROOT/provenance/polyglot-benchmark/cpp/exercises/practice" \
  --feedback-receipts-root "$ARTIFACT_ROOT/modal-receipts" \
  --max-concurrency 8 \
  --timeout-seconds 900 &
proxy_pid=$!
trap 'kill "$proxy_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl --fail --silent http://127.0.0.1:8765/health >/dev/null 2>&1 && break
  sleep 1
done

curl --fail --silent --show-error http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"openai/gpt-5.6-luna","stream":false,"temperature":0.7,"top_p":1.0,"messages":[{"role":"user","content":"Return exactly LUNA_IDENTITY_OK and nothing else."}]}' \
  > "$ARTIFACT_ROOT/identity-smoke.json"

python3 "$SCRIPT_DIR/run_verified_luna_fixed26_matrix.py"

export LUNA_VERIFY_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export LUNA_VERIFY_SCORER_PROVIDER='Local Mac'
export LUNA_VERIFY_LABEL_PREFIX="$LUNA_EVAL_LABEL_PREFIX"
export LUNA_VERIFY_STRICT_ONE_SHOT=1
export LUNA_VERIFY_FEEDBACK_ONLY=1
export LUNA_VERIFY_REQUIRE_AUTHORIZED_FEEDBACK=1
export LUNA_VERIFY_SINGLE_TRIALS=4
export LUNA_VERIFY_TRANSPORT=openrouter
export LUNA_VERIFY_EXPECTED_LOCAL_REJECTED_PREFLIGHTS=0
python3 "$SCRIPT_DIR/verify_completed_luna_fixed26_matrix.py"
