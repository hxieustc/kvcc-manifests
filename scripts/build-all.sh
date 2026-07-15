#!/usr/bin/env bash
# build-all.sh — build and install Dynamo then vLLM into the workspace venv
#
# Dynamo must be installed before vLLM: the Dynamo [vllm] extras would
# otherwise overwrite the editable vLLM installation.
#
# Run bootstrap.sh first to create the venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$WORKSPACE_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found — run scripts/bootstrap.sh first" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Dynamo (must come before vLLM)
# ---------------------------------------------------------------------------
DYNAMO_DIR="$WORKSPACE_ROOT/dynamo"
if [[ -d "$DYNAMO_DIR" ]]; then
  echo "==> Building and installing Dynamo"

  uv pip install pip 'maturin[patchelf]'

  pushd "$DYNAMO_DIR/lib/bindings/python" >/dev/null
  maturin develop --uv
  popd >/dev/null

  pushd "$DYNAMO_DIR" >/dev/null
  uv pip install -e .
  uv pip install -e '.[vllm]'
  popd >/dev/null
else
  echo "WARNING: dynamo directory not found at $DYNAMO_DIR — skipping" >&2
fi

# ---------------------------------------------------------------------------
# 2. vLLM (after Dynamo so editable install is not overwritten)
# ---------------------------------------------------------------------------
VLLM_DIR="$WORKSPACE_ROOT/vllm"
if [[ -d "$VLLM_DIR" ]]; then
  echo "==> Building and installing vLLM"

  uv pip install pip pandas

  pushd "$VLLM_DIR" >/dev/null
  VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
  popd >/dev/null
else
  echo "WARNING: vllm directory not found at $VLLM_DIR — skipping" >&2
fi

echo ""
echo "==> Build complete. Activate the venv with:"
echo "    source $VENV_DIR/bin/activate"
