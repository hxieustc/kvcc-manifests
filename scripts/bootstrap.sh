#!/usr/bin/env bash
# bootstrap.sh — initialise the repo workspace and install dependencies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Workspace: $WORKSPACE_ROOT"

# ---------------------------------------------------------------------------
# 1. Verify repo tool is available
# ---------------------------------------------------------------------------
if ! command -v repo &>/dev/null; then
  echo "ERROR: 'repo' not found. Install it from https://source.android.com/setup/develop/repo" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Python venv for vLLM
# ---------------------------------------------------------------------------
VENV_DIR="$WORKSPACE_ROOT/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating Python venv at $VENV_DIR"
  if command -v uv &>/dev/null; then
    uv venv --python 3.12 "$VENV_DIR"
  else
    python3 -m venv "$VENV_DIR"
  fi
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "==> Python: $($PYTHON --version)"

# ---------------------------------------------------------------------------
# 3. Install vLLM (pre-compiled wheel path; adjust for source builds)
# ---------------------------------------------------------------------------
if [[ -d "$WORKSPACE_ROOT/vllm" ]]; then
  echo "==> Installing vLLM in editable mode"
  VLLM_USE_PRECOMPILED=1 "$VENV_DIR/bin/pip" install -e "$WORKSPACE_ROOT/vllm" --torch-backend=auto
fi

# ---------------------------------------------------------------------------
# 4. Install Dynamo components
# ---------------------------------------------------------------------------
if [[ -d "$WORKSPACE_ROOT/dynamo" ]]; then
  echo "==> Installing Dynamo components"
  # TODO: replace with the correct Dynamo install command
  "$PIP" install -e "$WORKSPACE_ROOT/dynamo/components" 2>/dev/null || true
fi

echo "==> Bootstrap complete. Activate the venv with:"
echo "    source $VENV_DIR/bin/activate"
