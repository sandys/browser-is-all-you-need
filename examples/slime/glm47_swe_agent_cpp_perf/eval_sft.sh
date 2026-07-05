#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec "${SCRIPT_DIR}/glm47_swe_agent_cpp_perf.sh" sft-eval "$@"
