#!/usr/bin/env bash
set -euo pipefail
CONTAINER="fmt-pr3727-synth-v1-ep50-answer-free-sglang"
docker logs "$CONTAINER" > "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sglang.log" 2>&1
docker rm -f "$CONTAINER"
