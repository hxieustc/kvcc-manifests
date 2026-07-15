#!/usr/bin/env bash
# build-all.sh — build all components in dependency order
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$WORKSPACE_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found — run scripts/bootstrap.sh first" >&2
  exit 1
fi

PYTHON="$VENV_DIR/bin/python"

# ---------------------------------------------------------------------------
# 1. Build Dynamo (Rust + Python)
# ---------------------------------------------------------------------------
if [[ -d "$WORKSPACE_ROOT/dynamo" ]]; then
  echo "==> Building Dynamo"
  # TODO: replace with the correct Dynamo build command (e.g. cargo build --release)
  pushd "$WORKSPACE_ROOT/dynamo" >/dev/null
  # cargo build --release
  popd >/dev/null
fi

# ---------------------------------------------------------------------------
# 2. Build vLLM (C/C++ extensions)
# ---------------------------------------------------------------------------
if [[ -d "$WORKSPACE_ROOT/vllm" ]]; then
  echo "==> Building vLLM"
  pushd "$WORKSPACE_ROOT/vllm" >/dev/null
  "$VENV_DIR/bin/pip" install -e . --torch-backend=auto
  popd >/dev/null
fi

echo "==> Build complete"
