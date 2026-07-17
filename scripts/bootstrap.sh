#!/usr/bin/env bash
# bootstrap.sh — verify prerequisites and create the Python venv
#
# Run this once after `repo sync`. Then run build-all.sh to build and install
# Dynamo and vLLM into the venv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ is at manifests/scripts/ inside the workspace; go up two levels
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "==> Workspace: $WORKSPACE_ROOT"

# ---------------------------------------------------------------------------
# System dev libraries (apt)
# ---------------------------------------------------------------------------
APT_PACKAGES=(
  build-essential
  libhwloc-dev
  libudev-dev
  pkg-config
  libclang-dev
  protobuf-compiler
  python3-dev
  cmake
)

missing_pkgs=()
for pkg in "${APT_PACKAGES[@]}"; do
  if ! dpkg -s "$pkg" &>/dev/null; then
    missing_pkgs+=("$pkg")
  fi
done

if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
  echo "ERROR: missing system packages: ${missing_pkgs[*]}" >&2
  echo "       Install with:" >&2
  echo "       sudo apt-get install -y ${missing_pkgs[*]}" >&2
  exit 1
fi

echo "==> System dev libraries: OK"

# ---------------------------------------------------------------------------
# CLI tools
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

# Explicitly target the workspace venv for all subsequent uv pip calls,
# overriding any venv that may already be active in the caller's shell.
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

echo "==> Python: $(python --version)"

# ---------------------------------------------------------------------------
# Build-environment Python packages
# ---------------------------------------------------------------------------
echo "==> Installing build-environment packages"
uv pip install pip 'maturin[patchelf]' pandas pre-commit pytest

# ---------------------------------------------------------------------------
# NIXL
# ---------------------------------------------------------------------------
echo "==> Installing NIXL"
uv pip install pip nixl

# ---------------------------------------------------------------------------
# Dev tooling
# ---------------------------------------------------------------------------
VLLM_DIR="$WORKSPACE_ROOT/vllm"
if [[ -d "$VLLM_DIR" ]]; then
  echo "==> Installing pre-commit hooks"
  pushd "$VLLM_DIR" >/dev/null
  pre-commit install
  popd >/dev/null
else
  echo "WARNING: vllm directory not found at $VLLM_DIR — skipping pre-commit install" >&2
fi

echo ""
echo "==> Bootstrap complete. Next step:"
echo "    bash $SCRIPT_DIR/build-all.sh"
