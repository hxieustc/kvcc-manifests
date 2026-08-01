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

KVCC_INSTALL_SYSTEM_DEPS="${KVCC_INSTALL_SYSTEM_DEPS:-1}"
KVCC_INSTALL_PRECOMMIT="${KVCC_INSTALL_PRECOMMIT:-0}"
EFFECTIVE_UID="${KVCC_EFFECTIVE_UID:-$(id -u)}"

export HOME="${HOME:-/root}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/uv/bin:$PATH"

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
  git-lfs
  git
  less
  jq
  curl
)

missing_pkgs=()
for pkg in "${APT_PACKAGES[@]}"; do
  if ! dpkg -s "$pkg" &>/dev/null; then
    missing_pkgs+=("$pkg")
  fi
done

if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
  if [[ "$KVCC_INSTALL_SYSTEM_DEPS" != "1" ]]; then
    echo "ERROR: missing system packages: ${missing_pkgs[*]}" >&2
    echo "       Install with:" >&2
    echo "       apt-get install -y ${missing_pkgs[*]}" >&2
    exit 1
  fi
  if [[ "$EFFECTIVE_UID" != "0" ]]; then
    echo "ERROR: root is required to install: ${missing_pkgs[*]}" >&2
    echo "       Rerun as root or install them before bootstrap." >&2
    exit 1
  fi

  echo "==> Installing missing system packages: ${missing_pkgs[*]}"
  apt-get update
  apt-get install -y --no-install-recommends "${missing_pkgs[@]}"
fi

echo "==> System dev libraries: OK"

# ---------------------------------------------------------------------------
# CLI tools
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "==> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  hash -r
fi
if ! command -v rustc &>/dev/null; then
  echo "==> Installing Rust"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  hash -r
fi
if ! command -v repo &>/dev/null; then
  echo "ERROR: 'repo' is required before repository synchronization." >&2
  echo "       Use kubernetes/dev-pod.yaml or install repo manually." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Python venv
# ---------------------------------------------------------------------------
VENV_DIR="$WORKSPACE_ROOT/.venv"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "==> Creating Python venv at $VENV_DIR"
  uv venv --python 3.12 "$VENV_DIR"
fi

# Explicitly target the workspace venv for all subsequent uv pip calls,
# overriding any venv that may already be active in the caller's shell.
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

PYTHON="$VENV_DIR/bin/python"
echo "==> Python: $($PYTHON --version)"

# ---------------------------------------------------------------------------
# Build-environment Python packages
# ---------------------------------------------------------------------------
echo "==> Installing build-environment packages"
uv pip install --python "$PYTHON" \
  pip \
  'maturin[patchelf]' \
  pandas \
  pre-commit \
  pytest

# ---------------------------------------------------------------------------
# NIXL
# ---------------------------------------------------------------------------
echo "==> Installing NIXL"
uv pip install --python "$PYTHON" nixl

# ---------------------------------------------------------------------------
# Dev tooling
# ---------------------------------------------------------------------------
VLLM_DIR="$WORKSPACE_ROOT/vllm"
if [[ "$KVCC_INSTALL_PRECOMMIT" == "1" && -d "$VLLM_DIR" ]]; then
  echo "==> Installing pre-commit hooks"
  pushd "$VLLM_DIR" >/dev/null
  pre-commit install
  popd >/dev/null
elif [[ "$KVCC_INSTALL_PRECOMMIT" == "1" ]]; then
  echo "WARNING: vllm directory not found at $VLLM_DIR — skipping pre-commit install" >&2
else
  echo "==> Pre-commit hook installation disabled"
fi

echo ""
echo "==> Bootstrap complete. Next step:"
echo "    bash $SCRIPT_DIR/build-all.sh"
