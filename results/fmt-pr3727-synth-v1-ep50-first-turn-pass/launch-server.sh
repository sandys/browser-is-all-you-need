#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-}
[[ -z "$MODE" || "$MODE" == "--preflight-only" ]] || {
  echo "usage: $0 [--preflight-only]" >&2
  exit 64
}


ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_MODEL="/home/pipeshift/models/zai-org--GLM-4.7-Flash/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
BASE_RECEIPT="$ROOT/base-model-receipt.json"
SERVING_ADAPTER="$ROOT/serving-adapter"
LORA_MEM_POOL_HOTFIX="$ROOT/sglang-mem_pool.py"
EXPECTED_LORA_MEM_POOL_HOTFIX_SHA256="c367ed3c912dff448c8448ec3ebca60a2816c7cc198c177b6e6ca0dc414b3b40"
IMAGE="radixark/miles:latest"
EXPECTED_IMAGE_ID="sha256:926f671a9da56d96eec1e81af8b3eec58063ebaa13fba44c160b4810ce1301fa"
CONTAINER="fmt-pr3727-synth-v1-ep50-sglang"

[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$EXPECTED_IMAGE_ID" ]]
[[ -f "$BASE_MODEL/config.json" ]]
[[ -f "$BASE_RECEIPT" ]]
[[ -f "$LORA_MEM_POOL_HOTFIX" ]]
[[ "$(sha256sum "$LORA_MEM_POOL_HOTFIX" | cut -d' ' -f1)" == "$EXPECTED_LORA_MEM_POOL_HOTFIX_SHA256" ]]
[[ -f "$SERVING_ADAPTER/adapter_model.bin" ]]
[[ -f "$SERVING_ADAPTER/adapter_config.json" ]]
[[ -f "$SERVING_ADAPTER/conversion_receipt.json" ]]
[[ "$(nvidia-smi -L | wc -l)" -ge 4 ]]
! docker container inspect "$CONTAINER" >/dev/null 2>&1 || {
  echo "refusing to reuse existing server container: $CONTAINER" >&2
  exit 1
}

python3 - "$BASE_MODEL" "$BASE_RECEIPT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
model = Path(sys.argv[1])
receipt = json.loads(Path(sys.argv[2]).read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert receipt["repository"] == "zai-org/GLM-4.7-Flash"
assert receipt["revision"] == "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
assert receipt["model_type"] == "glm4_moe_lite"
assert receipt["config_sha256"] == sha(model / "config.json")
assert receipt["index_sha256"] == sha(model / "model.safetensors.index.json")
assert receipt["shard_count"] == 48
assert all((model / name).stat().st_size == size for name, size in receipt["shard_bytes"].items())
PY

python3 - "$SERVING_ADAPTER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
def sha(name):
    return hashlib.sha256((root / name).read_bytes()).hexdigest()
receipt = json.loads((root / "conversion_receipt.json").read_text())
assert receipt["source_adapter_sha256"] == "4acb7f23c295f45380155c5d9ee6bc59422262f0cb51f0c02f7e550d405b575a"
assert receipt["source_tensor_count"] == 9741
assert receipt["removed_layer_47_tensor_count"] == 207
assert receipt["serving_tensor_count"] == 9534
assert receipt["serving_adapter_sha256"] == sha("adapter_model.bin")
assert receipt["serving_adapter_config_sha256"] == sha("adapter_config.json")
PY
if [[ "$MODE" == "--preflight-only" ]]; then
  echo "server preflight passed; model not launched"
  exit 0
fi


docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --network host \
  --ipc host \
  --mount "type=bind,src=$BASE_MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$SERVING_ADAPTER,dst=/adapter,readonly" \
  --mount "type=bind,src=$LORA_MEM_POOL_HOTFIX,dst=/sgl-workspace/sglang/python/sglang/srt/lora/mem_pool.py,readonly" \
  --entrypoint python3 \
  "$IMAGE" \
  -m sglang.launch_server \
  --model-path /model \
  --tp-size 4 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --mem-fraction-static 0.8 \
  --max-running-requests 16 \
  --served-model-name glm-4.7-flash-grpo \
  --api-key local-eval \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-backend triton \
  --lora-target-modules q_a_proj kv_a_proj_with_mqa o_proj gate_proj up_proj down_proj \
  --experts-shared-outer-loras \
  --lora-use-virtual-experts

trap 'docker logs "$CONTAINER" > "$ROOT/sglang-launch-failure.log" 2>&1 || true; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' ERR
python3 "$ROOT/verify-server.py" \
  --port 8000 \
  --adapter-path /adapter \
  --receipt "$ROOT/lora-activation-receipt.json"
trap - ERR
printf 'server ready: %s\n' "$CONTAINER"
