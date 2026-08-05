#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIDER_COMMIT="5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
POLYGLOT_COMMIT="7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
MODEL_SETTINGS_SHA256="3d716929098f9d6e1abbccb2ed1a4d37f749eaf549d7ccb685c01a6b17199170"
RUN_LABEL="${RUN_LABEL:-base-fixed26-repro-$(date -u +%Y%m%dT%H%M%SZ)}"
REPRO_WORKDIR="${REPRO_WORKDIR:-$SCRIPT_DIR/work/$RUN_LABEL}"
TRIALS="${TRIALS:-4}"

: "${OPENAI_API_BASE:?Set OPENAI_API_BASE to the OpenAI-compatible base-model endpoint}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for that endpoint}"

for command in git cmake make g++ python3 shasum; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 1
  }
done

if [[ -e "$REPRO_WORKDIR" ]]; then
  echo "refusing to reuse existing REPRO_WORKDIR: $REPRO_WORKDIR" >&2
  exit 1
fi

mkdir -p "$REPRO_WORKDIR"
AIDER_ROOT="$REPRO_WORKDIR/aider"
POLYGLOT_ROOT="$AIDER_ROOT/tmp.benchmarks/polyglot-benchmark"
VENV="$REPRO_WORKDIR/venv"
MODEL_SETTINGS="$SCRIPT_DIR/model-settings.yml"

observed_settings_sha="$(shasum -a 256 "$MODEL_SETTINGS" | awk '{print $1}')"
if [[ "$observed_settings_sha" != "$MODEL_SETTINGS_SHA256" ]]; then
  echo "model-settings.yml SHA mismatch" >&2
  echo "got:  $observed_settings_sha" >&2
  echo "want: $MODEL_SETTINGS_SHA256" >&2
  exit 1
fi

git clone https://github.com/Aider-AI/aider.git "$AIDER_ROOT"
git -C "$AIDER_ROOT" checkout --detach "$AIDER_COMMIT"
git clone https://github.com/Aider-AI/polyglot-benchmark.git "$POLYGLOT_ROOT"
git -C "$POLYGLOT_ROOT" checkout --detach "$POLYGLOT_COMMIT"

observed_aider="$(git -C "$AIDER_ROOT" rev-parse HEAD)"
observed_polyglot="$(git -C "$POLYGLOT_ROOT" rev-parse HEAD)"
[[ "$observed_aider" == "$AIDER_COMMIT" ]]
[[ "$observed_polyglot" == "$POLYGLOT_COMMIT" ]]

exercise_root="$POLYGLOT_ROOT/cpp/exercises/practice"
task_count="$(find "$exercise_root" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
if [[ "$task_count" != "26" ]]; then
  echo "fixed-26 task-count mismatch: got $task_count, want 26" >&2
  exit 1
fi

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$AIDER_ROOT[dev]"

cd "$AIDER_ROOT"
for trial in $(seq 1 "$TRIALS"); do
  trial_label="${RUN_LABEL}-a${trial}"
  "$VENV/bin/python" benchmark/benchmark.py \
    "$trial_label" \
    --model openai/glm-4.7-flash \
    --edit-format whole \
    --languages cpp \
    --tries 2 \
    --threads 8 \
    --exercises-dir polyglot-benchmark \
    --read-model-settings "$MODEL_SETTINGS" \
    --cont

  output_dir="$(find "$AIDER_ROOT/tmp.benchmarks" -mindepth 1 -maxdepth 1 -type d -name "*--$trial_label" | sort | tail -1)"
  if [[ -z "$output_dir" ]]; then
    echo "benchmark completed without a discoverable output directory for trial $trial" >&2
    exit 1
  fi
  "$VENV/bin/python" benchmark/benchmark.py --stats "$output_dir"
done
echo "Reproduction artifacts: $REPRO_WORKDIR"
echo "Completed trials: $TRIALS"
