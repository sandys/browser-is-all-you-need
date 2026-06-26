#!/usr/bin/env bash
set -euo pipefail

# Repo-owned scaffold derived from the upstream 8-node launcher.
# This file is intentionally not wired yet; the next change switches the
# custom generate path to examples.slime.coding_agent_rl.generate.generate
# and enables the Docker sandbox backend.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SLIME_DIR="${SLIME_DIR:-${REPO_ROOT}/.cache/upstreams/slime}"

export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/Qwen3.6-35B-A3B}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/path/to/Qwen3.6-35B-A3B_torch_dist}"
export PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/.w8-biayn/slime/data/coding_agent_smoke.jsonl}"
export SWE_AGENT="${SWE_AGENT:-claude_code}"
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-127.0.0.1}"

echo "scaffold_only=1"
echo "repo_root=${REPO_ROOT}"
echo "slime_dir=${SLIME_DIR}"
echo "hf_checkpoint=${HF_CHECKPOINT}"
echo "ref_model_path=${REF_MODEL_PATH}"
echo "prompt_data=${PROMPT_DATA}"
echo "sandbox_backend=${SLIME_AGENT_SANDBOX_BACKEND}"
echo "next_step=wire_custom_generate_and_docker_backend"
