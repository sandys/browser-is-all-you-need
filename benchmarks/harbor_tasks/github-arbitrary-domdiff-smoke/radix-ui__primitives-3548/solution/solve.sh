#!/bin/bash
set -euo pipefail

cd /testbed
PATCH_PATH="/task/solution/fix.patch"

if [ ! -f "$PATCH_PATH" ]; then
  echo "Missing patch: $PATCH_PATH"
  exit 1
fi

git apply "$PATCH_PATH" || patch -p1 < "$PATCH_PATH"
echo "Patch applied successfully"
