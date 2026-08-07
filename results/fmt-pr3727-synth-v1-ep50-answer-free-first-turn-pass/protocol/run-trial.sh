#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "usage: $0 TRIAL_LABEL EDIT_FORMAT [--first-turn-only]" >&2
  exit 64
fi
TRIAL_LABEL=$1
EDIT_FORMAT=$2
MODE=${3:-}
[[ -z "$MODE" || "$MODE" == "--first-turn-only" ]] || {
  echo "unknown mode: $MODE" >&2
  exit 64
}
[[ "$EDIT_FORMAT" == "diff" || "$EDIT_FORMAT" == "udiff" || "$EDIT_FORMAT" == "whole" ]] || {
  echo "EDIT_FORMAT must be diff, udiff, or whole" >&2
  exit 64
}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRIAL="$ROOT/trials/$TRIAL_LABEL"
AIDER_IMAGE="aider-benchmark:latest"
EXPECTED_AIDER_IMAGE_ID="sha256:77cd719f97d4493db95721998c2651dd443d785a961fd54e5fa3e50832ea0b04"
CONTAINER="fmt-pr3727-answerfree-aider-$TRIAL_LABEL"

[[ "$(docker image inspect "$AIDER_IMAGE" --format '{{.Id}}')" == "$EXPECTED_AIDER_IMAGE_ID" ]]
[[ -f "$ROOT/lora-activation-receipt.json" ]]
[[ -f "$ROOT/workspace-receipt.json" ]]
[[ -d "$ROOT/workspace-template" ]]
[[ ! -e "$TRIAL" ]]
python3 - "$ROOT/lora-activation-receipt.json" <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
assert receipt["status"] == "diverged"
assert receipt["with_lora"] != receipt["without_lora"]
PY
! docker container inspect "$CONTAINER" >/dev/null 2>&1

docker run -d \
  --name "$CONTAINER" \
  --network host \
  --mount "type=bind,src=$ROOT,dst=$ROOT" \
  --workdir "$ROOT" \
  --entrypoint python3 \
  "$AIDER_IMAGE" \
  "$ROOT/run-aider.py" "$TRIAL_LABEL" --root "$ROOT" --edit-format "$EDIT_FORMAT" >/dev/null

fail_if_aider_exited() {
  if [[ "$(docker inspect "$CONTAINER" --format '{{.State.Running}}')" != "true" ]]; then
    docker logs "$CONTAINER" > "$TRIAL.aider-failed.log" 2>&1 || true
    echo "Aider container exited before expected handoff" >&2
    exit 70
  fi
}
while [[ ! -f "$TRIAL/turn1.ready" ]]; do
  fail_if_aider_exited
  sleep 2
done
sudo chown -R "$(id -u):$(id -g)" "$TRIAL"

set +e
"$ROOT/grade-candidate.sh" "$TRIAL/workspace" "$TRIAL/grades" turn1 \
  >"$TRIAL/grader-driver-turn1.log" 2>&1
turn1_rc=$?
set -e
if [[ $turn1_rc -eq 0 ]]; then
  printf 'stop\n' > "$TRIAL/stop-after-turn1"
elif [[ "$MODE" == "--first-turn-only" ]]; then
  printf 'stop\n' > "$TRIAL/stop-after-turn1"
else
  {
    if [[ -f "$TRIAL/grades/turn1/grader.log" ]]; then
      cat "$TRIAL/grades/turn1/grader.log"
    else
      cat "$TRIAL/grader-driver-turn1.log"
    fi
    cat <<'EOF'

####

See the testing errors above.
The tests are correct, don't try and change them.
Fix the code in include/fmt/chrono.h to resolve the errors.
EOF
  } > "$TRIAL/feedback-turn1.txt"
fi

container_rc=$(docker wait "$CONTAINER")
docker logs "$CONTAINER" > "$TRIAL/aider-container.log" 2>&1 || true
docker rm "$CONTAINER" >/dev/null
sudo chown -R "$(id -u):$(id -g)" "$TRIAL"
[[ "$container_rc" == "0" ]] || {
  echo "Aider container failed with status $container_rc" >&2
  exit 71
}

turn2_rc=99
if [[ $turn1_rc -ne 0 && "$MODE" != "--first-turn-only" ]]; then
  [[ -f "$TRIAL/turn2.ready" ]]
  set +e
  "$ROOT/grade-candidate.sh" "$TRIAL/workspace" "$TRIAL/grades" turn2 \
    >"$TRIAL/grader-driver-turn2.log" 2>&1
  turn2_rc=$?
  set -e
fi

python3 - "$ROOT" "$TRIAL" "$turn1_rc" "$turn2_rc" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root, trial = map(Path, sys.argv[1:3])
turn1_rc, turn2_rc = map(int, sys.argv[3:5])
def load(path):
    return json.loads(path.read_text()) if path.exists() else None
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "schema_version": 1,
    "generation": load(trial / "generation-receipt.json"),
    "workspace": load(root / "workspace-receipt.json"),
    "adapter_conversion": load(root / "serving-adapter/conversion_receipt.json"),
    "lora_activation": load(root / "lora-activation-receipt.json"),
    "turn1_grade": load(trial / "grades/turn1/grade-receipt.json"),
    "turn2_grade": load(trial / "grades/turn2/grade-receipt.json"),
    "turn1_driver_return_code": turn1_rc,
    "turn2_driver_return_code": None if turn2_rc == 99 else turn2_rc,
    "first_turn_passed": turn1_rc == 0,
    "final_passed": turn1_rc == 0 or turn2_rc == 0,
    "aider_container_log_sha256": digest(trial / "aider-container.log"),
}
(trial / "run-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "first_turn_passed": receipt["first_turn_passed"],
    "final_passed": receipt["final_passed"],
}, indent=2))
PY

[[ $turn1_rc -eq 0 || $turn2_rc -eq 0 ]]
