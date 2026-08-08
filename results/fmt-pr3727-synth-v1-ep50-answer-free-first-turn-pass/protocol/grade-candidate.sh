#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 WORKSPACE RESULTS_DIR TURN_LABEL" >&2
  exit 64
fi
WORKSPACE=$(cd "$1" && pwd)
RESULTS_DIR=$2
TURN_LABEL=$3
IMAGE="mswebench/fmtlib_m_fmt:pr-3727"
EXPECTED_IMAGE_ID="sha256:855d772196ea6554f59c98415689d73f91dcaf47df484bcf0fbcce9aad884a6a"
EXPECTED_BASE_COMMIT="06f1c0d725855861535e9e65cd4d502aca7c61ed"
TARGET="include/fmt/chrono.h"
TURN_DIR="$RESULTS_DIR/$TURN_LABEL"
GIT=(git -c "safe.directory=$WORKSPACE" -C "$WORKSPACE")

[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$EXPECTED_IMAGE_ID" ]]
[[ "$("${GIT[@]}" rev-parse HEAD)" == "$EXPECTED_BASE_COMMIT" ]]
mapfile -t changed < <("${GIT[@]}" status --porcelain=v1 | sed -E 's/^...//')
[[ ${#changed[@]} -eq 1 && "${changed[0]}" == "$TARGET" ]] || {
  printf 'scope violation; changed paths:\n' >&2
  printf '%s\n' "${changed[@]}" >&2
  exit 65
}
"${GIT[@]}" diff --quiet -- "$TARGET" && {
  echo "no production change" >&2
  exit 66
}
"${GIT[@]}" diff --check
[[ ! -e "$TURN_DIR" ]] || {
  echo "refusing to overwrite grader result: $TURN_DIR" >&2
  exit 67
}
mkdir -p "$TURN_DIR"
"${GIT[@]}" diff --binary -- "$TARGET" > "$TURN_DIR/candidate.patch"
cp "$WORKSPACE/$TARGET" "$TURN_DIR/chrono.h"

set +e
docker run --rm \
  --mount "type=bind,src=$WORKSPACE/$TARGET,dst=/home/fmt/$TARGET,readonly" \
  "$IMAGE" bash -lc '
    set +e
    bash /home/test-run.sh
    rc=$?
    if [[ $rc -ne 0 && -d /home/fmt/build ]]; then
      printf "\n--- failing test details ---\n"
      ctest --test-dir /home/fmt/build --rerun-failed --output-on-failure || true
    fi
    exit "$rc"
  ' >"$TURN_DIR/grader.log" 2>&1
grade_rc=$?
set -e

python3 - "$TURN_DIR" "$grade_rc" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

turn_dir = Path(sys.argv[1])
return_code = int(sys.argv[2])
def digest(name: str) -> str:
    return hashlib.sha256((turn_dir / name).read_bytes()).hexdigest()
receipt = {
    "schema_version": 1,
    "official_image_id": "sha256:855d772196ea6554f59c98415689d73f91dcaf47df484bcf0fbcce9aad884a6a",
    "base_commit": "06f1c0d725855861535e9e65cd4d502aca7c61ed",
    "test_patch_sha256": "76eac48ef70cf13f40a9d99390ec7decc840451c0bae1aedaa93dd2ba7b77002",
    "return_code": return_code,
    "passed": return_code == 0,
    "candidate_patch_sha256": digest("candidate.patch"),
    "candidate_header_sha256": digest("chrono.h"),
    "grader_log_sha256": digest("grader.log"),
}
(turn_dir / "grade-receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
exit "$grade_rc"
