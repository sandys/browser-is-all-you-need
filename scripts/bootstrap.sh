#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [--no-sky]

Bootstraps a fresh machine for w8-biayn development and operation.

Options:
  --no-sky   Skip SkyPilot installation. Useful for local unit-test-only setup.
EOF
}

install_sky=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-sky)
      install_sky=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was installed but is still not on PATH. Add \$HOME/.local/bin to PATH and rerun." >&2
  exit 1
fi

echo "Syncing Python environment..."
uv sync --extra dev --extra browser

if [ "$install_sky" -eq 1 ]; then
  echo "Installing SkyPilot with GCP support..."
  if ! uv tool install --with pip "skypilot-nightly[gcp]"; then
    uv tool upgrade --with pip "skypilot-nightly[gcp]"
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v gcloud >/dev/null 2>&1; then
  cat >&2 <<'EOF'

gcloud is not installed or not on PATH.
Install Google Cloud CLI before running real GCP smoke tests:
  https://cloud.google.com/sdk/docs/install

EOF
fi

echo
echo "Bootstrap complete."
echo
echo "Next commands:"
echo "  cp /secure/path/service-account.json .gcp-service-account.json"
echo "  uv run w8-biayn doctor --cloud"
echo "  uv run w8-biayn launch miniwob --dry-run"
echo
echo "Real smoke launch:"
echo "  uv run w8-biayn launch miniwob"

