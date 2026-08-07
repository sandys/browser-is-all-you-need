#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE="mswebench/fmtlib_m_fmt:pr-3727"
EXPECTED_IMAGE_ID="sha256:855d772196ea6554f59c98415689d73f91dcaf47df484bcf0fbcce9aad884a6a"
EXPECTED_BASE_COMMIT="06f1c0d725855861535e9e65cd4d502aca7c61ed"
DEST="$ROOT/workspace-template"

actual_image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[[ "$actual_image_id" == "$EXPECTED_IMAGE_ID" ]] || {
  echo "official task image mismatch: $actual_image_id" >&2
  exit 1
}
[[ ! -e "$DEST" ]] || {
  echo "refusing to overwrite existing workspace template: $DEST" >&2
  exit 1
}

container_id=$(docker create "$IMAGE")
trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT
mkdir "$DEST"
docker cp "$container_id:/home/fmt/." "$DEST"
docker rm -f "$container_id" >/dev/null
trap - EXIT

git -C "$DEST" reset --hard "$EXPECTED_BASE_COMMIT"
git -C "$DEST" clean -fdx
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$EXPECTED_BASE_COMMIT" ]]
[[ -z "$(git -C "$DEST" status --porcelain)" ]]

python3 - "$DEST" "$ROOT/workspace-receipt.json" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
receipt_path = Path(sys.argv[2])
target = workspace / "include/fmt/chrono.h"
receipt = {
    "schema_version": 1,
    "task": "fmtlib/fmt issue 3725 answer-free evaluation",
    "official_image": "mswebench/fmtlib_m_fmt:pr-3727",
    "official_image_id": "sha256:855d772196ea6554f59c98415689d73f91dcaf47df484bcf0fbcce9aad884a6a",
    "base_commit": subprocess.check_output(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True
    ).strip(),
    "editable_file": "include/fmt/chrono.h",
    "editable_file_bytes": target.stat().st_size,
    "editable_file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "hidden_test_patch_sha256": "76eac48ef70cf13f40a9d99390ec7decc840451c0bae1aedaa93dd2ba7b77002",
    "accepted_implementation_present_in_model_root": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
