#!/usr/bin/env bash
# bootstrap.sh — initialise the repo workspace and install dependencies
#
# Install order matters: Dynamo must be installed before vLLM because
# the Dynamo install would otherwise overwrite the editable vLLM installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Workspace: $WORKSPACE_ROOT"

# ---------------------------------------------------------------------------
# Verify prerequisites
# ---------------------------------------------------------------------------
if ! command -v repo &>/dev/null; then
  echo "ERROR: 'repo' not found. Install it from https://source.android.com/setup/develop/repo" >&2
  exit 1
fi
if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Python venv
# ---------------------------------------------------------------------------
VENV_DIR="$WORKSPACE_ROOT/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating Python venv at $VENV_DIR"
  uv venv --python 3.12 "$VENV_DIR"
fi

echo "==> Python: $("$VENV_DIR/bin/python" --version)"

# Run all uv pip commands inside the venv without activating the shell
UV="uv"
UV_PIP="$UV pip"

# ---------------------------------------------------------------------------
# 1. Dynamo (must come before vLLM)
# ---------------------------------------------------------------------------
DYNAMO_DIR="$WORKSPACE_ROOT/dynamo"
if [[ -d "$DYNAMO_DIR" ]]; then
  echo "==> Building and installing Dynamo"

  # Bootstrap pip and maturin inside the venv
  "$UV_PIP" install pip 'maturin[patchelf]'

  # Build the Rust Python bindings
  pushd "$DYNAMO_DIR/lib/bindings/python" >/dev/null
  maturin develop --uv
  popd >/dev/null

  # Install Dynamo core and the vLLM extras
  pushd "$DYNAMO_DIR" >/dev/null
  "$UV_PIP" install -e .
  "$UV_PIP" install -e '.[vllm]'
  popd >/dev/null
else
  echo "WARNING: dynamo directory not found at $DYNAMO_DIR — skipping" >&2
fi

# ---------------------------------------------------------------------------
# 2. vLLM (after Dynamo so it wins the editable-install race)
# ---------------------------------------------------------------------------
VLLM_DIR="$WORKSPACE_ROOT/vllm"
if [[ -d "$VLLM_DIR" ]]; then
  echo "==> Building and installing vLLM"
  "$UV_PIP" install pip pandas
  pushd "$VLLM_DIR" >/dev/null
  VLLM_USE_PRECOMPILED=1 "$UV_PIP" install --editable . --torch-backend=auto
  popd >/dev/null
else
  echo "WARNING: vllm directory not found at $VLLM_DIR — skipping" >&2
fi

# ---------------------------------------------------------------------------
# 3. Dev tooling
# ---------------------------------------------------------------------------
echo "==> Installing pre-commit"
"$UV_PIP" install pre-commit
pushd "$VLLM_DIR" >/dev/null
"$VENV_DIR/bin/pre-commit" install
popd >/dev/null

echo ""
echo "==> Bootstrap complete. Activate the venv with:"
echo "    source $VENV_DIR/bin/activate"
