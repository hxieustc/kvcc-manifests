#!/usr/bin/env bash
# build-all.sh — build and install Dynamo then vLLM into the workspace venv
#
# Dynamo must be installed before vLLM: the Dynamo [vllm] extras would
# otherwise overwrite the editable vLLM installation.
#
# Run bootstrap.sh first to create the venv and install build tools.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ is at manifests/scripts/ inside the workspace; go up two levels
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$WORKSPACE_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found — run manifests/scripts/bootstrap.sh first" >&2
  exit 1
fi

# Explicitly target the workspace venv for all subsequent uv pip / maturin
# calls, overriding any venv that may already be active in the caller's shell.
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

# ---------------------------------------------------------------------------
# 1. Dynamo (must come before vLLM)
# ---------------------------------------------------------------------------
DYNAMO_DIR="$WORKSPACE_ROOT/dynamo"
if [[ -d "$DYNAMO_DIR" ]]; then
  echo "==> Building and installing Dynamo"

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

  pushd "$VLLM_DIR" >/dev/null
  # kvcc_p2p branch tag can't be parsed as semver by setuptools-scm; pin a
  # dummy version so the build doesn't fail on version resolution.
  VLLM_USE_PRECOMPILED=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.3 \
    uv pip install --editable . --torch-backend=auto
  uv pip install -r requirements/test/cuda.in
  popd >/dev/null

  # vllm/requirements/cuda.txt pins torch==2.11.0, but the precompiled FA2/FA3
  # CUDA extensions in the kvcc_repo branch were built against torch==2.13.0.
  # Restore 2.13.0 from the uv local cache (no network needed).
  TORCH_CACHE=$(find "$HOME/.cache/uv/archive-v0" -maxdepth 2 \
    -name "torch-2.13.0+cu130.dist-info" -type d 2>/dev/null | \
    head -1 | xargs -r dirname)
  if [[ -z "$TORCH_CACHE" ]]; then
    echo "ERROR: torch 2.13.0+cu130 not found in uv cache; run 'uv pip install torch==2.13.0' from a working environment first" >&2
    exit 1
  fi
  uv pip install "torch==2.13.0" --find-links "$TORCH_CACHE"

  # torchvision 0.26.0 (pinned by vllm alongside torch 2.11.0) is incompatible
  # with torch 2.13.0; upgrade to matching 0.28.0 from cache.
  TV_CACHE=$(find "$HOME/.cache/uv/archive-v0" -maxdepth 2 \
    -name "torchvision-0.28.0.dist-info" -type d 2>/dev/null | \
    head -1 | xargs -r dirname)
  if [[ -n "$TV_CACHE" ]]; then
    uv pip install "torchvision==0.28.0" --find-links "$TV_CACHE"
  fi

  # dynamo[vllm] pulls vllm==0.24.0 which pins flashinfer-cubin==0.6.12 (PyPI).
  # The vllm editable install then upgrades flashinfer-python to 0.6.14 but
  # setup.py deliberately skips flashinfer-cubin (not on PyPI since 0.6.14).
  # Re-pin cubin from flashinfer.ai to match python.
  FI_VER=$(python3 -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
  uv pip install "flashinfer-cubin==${FI_VER}" --extra-index-url https://flashinfer.ai/whl/
else
  echo "WARNING: vllm directory not found at $VLLM_DIR — skipping" >&2
fi

# ---------------------------------------------------------------------------
# 3. KVCC (install into workspace venv, editable, after vLLM)
# ---------------------------------------------------------------------------
KVCC_DIR="$WORKSPACE_ROOT/kvcc"
if [[ -d "$KVCC_DIR" ]]; then
  echo "==> Installing KVCC (editable) into venv"
  pushd "$KVCC_DIR" >/dev/null
  uv pip install --python "$VENV_DIR/bin/python" --editable .
  popd >/dev/null
else
  echo "WARNING: kvcc directory not found at $KVCC_DIR — skipping" >&2
fi

echo ""
echo "==> Build complete. Activate the venv with:"
echo "    source $VENV_DIR/bin/activate"
