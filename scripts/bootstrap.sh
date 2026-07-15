#!/usr/bin/env bash
# bootstrap.sh — verify prerequisites and create the Python venv
#
# Run this once after `repo sync`. Then run build-all.sh to build and install
# Dynamo and vLLM into the venv.
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
if ! command -v rustc &>/dev/null; then
  echo "ERROR: 'rustc' not found. Install it with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
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

# ---------------------------------------------------------------------------
# Dev tooling
# ---------------------------------------------------------------------------
echo "==> Installing pre-commit"
uv pip install pre-commit

VLLM_DIR="$WORKSPACE_ROOT/vllm"
if [[ -d "$VLLM_DIR" ]]; then
  "$VENV_DIR/bin/pre-commit" install --work-tree "$VLLM_DIR" --git-dir "$VLLM_DIR/.git"
else
  echo "WARNING: vllm directory not found at $VLLM_DIR — skipping pre-commit install" >&2
fi

echo ""
echo "==> Bootstrap complete. Next step:"
echo "    bash $SCRIPT_DIR/build-all.sh"
