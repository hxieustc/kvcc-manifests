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
KVCC_TORCH_VERSION="${KVCC_TORCH_VERSION:-2.13.0+cu130}"
KVCC_TORCHVISION_VERSION="${KVCC_TORCHVISION_VERSION:-0.28.0+cu130}"
KVCC_TORCH_INDEX_URL="${KVCC_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
KVCC_FLASHINFER_INDEX_URL="${KVCC_FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/}"
KVCC_FLASHINFER_CUDA_INDEX_URL="${KVCC_FLASHINFER_CUDA_INDEX_URL:-https://flashinfer.ai/whl/cu130}"

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

  # pushd "$VLLM_DIR" >/dev/null
  # # kvcc_p2p branch tag can't be parsed as semver by setuptools-scm; pin a
  # # dummy version so the build doesn't fail on version resolution.
  # VLLM_USE_PRECOMPILED=1 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.3 \
  #   uv pip install --editable . --torch-backend=auto
  # uv pip install -r requirements/test/cuda.in
  # popd >/dev/null


  pushd "$VLLM_DIR" >/dev/null

  # Find the merge-base between this checkout and current upstream vLLM main.
  #
  # This mirrors the logic vLLM uses when VLLM_PRECOMPILED_WHEEL_COMMIT
  # is not explicitly specified.
  git fetch -q https://github.com/vllm-project/vllm.git main
  
  VLLM_UPSTREAM_MAIN_COMMIT="$(git rev-parse FETCH_HEAD)"
  VLLM_MERGE_BASE_COMMIT="$(
    git merge-base HEAD "$VLLM_UPSTREAM_MAIN_COMMIT"
  )"
  
  echo "vLLM HEAD:          $(git rev-parse HEAD)"
  echo "Upstream main:      $VLLM_UPSTREAM_MAIN_COMMIT"
  echo "Merge-base commit:  $VLLM_MERGE_BASE_COMMIT"
  
  # Normalize the machine architecture to the names used by wheel tags.
  case "$(uname -m)" in
    x86_64|amd64)
      VLLM_WHEEL_ARCH="x86_64"
      ;;
    aarch64|arm64)
      VLLM_WHEEL_ARCH="aarch64"
      ;;
    *)
      echo "ERROR: unsupported architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
  
  # Determine the wheel variant.
  #
  # Allow the caller to override it explicitly:
  #   VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 ./install.sh
  #
  # Otherwise infer it from CUDA.
  if [[ -n "${VLLM_PRECOMPILED_WHEEL_VARIANT:-}" ]]; then
    VLLM_WHEEL_VARIANT="$VLLM_PRECOMPILED_WHEEL_VARIANT"
  else
    CUDA_VERSION="$(
      nvidia-smi \
        --query-gpu=driver_version \
        --format=csv,noheader 2>/dev/null |
        head -1
    )"
  
    # Prefer torch's CUDA version when available, because this is what matters
    # for the vLLM/PyTorch build environment.
    TORCH_CUDA_VERSION="$(
      python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.version.cuda or "")
except Exception:
    pass
PY
    )"
  
    if [[ -n "$TORCH_CUDA_VERSION" ]]; then
      CUDA_MAJOR="${TORCH_CUDA_VERSION%%.*}"
    else
      # CUDA 13 was already detected by vLLM in this environment.
      #
      # If torch is unavailable here, use nvidia-smi's reported CUDA version.
      CUDA_MAJOR="$(
        nvidia-smi 2>/dev/null |
          sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p' |
          head -1
      )"
    fi
  
    case "$CUDA_MAJOR" in
      13)
        VLLM_WHEEL_VARIANT="cu130"
        ;;
      12)
        VLLM_WHEEL_VARIANT="cu129"
        ;;
      *)
        echo "ERROR: cannot determine vLLM wheel variant for CUDA major: ${CUDA_MAJOR:-unknown}" >&2
        echo "Set VLLM_PRECOMPILED_WHEEL_VARIANT explicitly." >&2
        exit 1
        ;;
    esac
  fi
  
  echo "Wheel architecture: $VLLM_WHEEL_ARCH"
  echo "Wheel variant:      $VLLM_WHEEL_VARIANT"
  
  # Search backward from the merge-base on the main-branch first-parent history.
  #
  # We deliberately search BACKWARD from the merge-base instead of using the
  # newest nightly wheel. The closest older wheel is less likely to be
  # incompatible with this source checkout.
  VLLM_PRECOMPILED_WHEEL_COMMIT=""
  
  tmp_metadata="$(mktemp)"
  trap 'rm -f "$tmp_metadata"' EXIT
  
  for commit in $(git rev-list --first-parent "$VLLM_MERGE_BASE_COMMIT"); do
    metadata_url="https://wheels.vllm.ai/${commit}/${VLLM_WHEEL_VARIANT}/vllm/metadata.json"
  
    if ! curl -fsSL "$metadata_url" -o "$tmp_metadata" 2>/dev/null; then
      continue
    fi
  
    if python - "$tmp_metadata" "$VLLM_WHEEL_ARCH" <<'PY'
import json
import sys

metadata_path = sys.argv[1]
wanted_arch = sys.argv[2]

with open(metadata_path) as f:
    data = json.load(f)

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value)

for item in walk(data):
    platform_tag = item.get("platform_tag")
    if isinstance(platform_tag, str) and platform_tag.endswith("_" + wanted_arch):
        print(item.get("filename", platform_tag))
        sys.exit(0)

sys.exit(1)
PY
    then
      VLLM_PRECOMPILED_WHEEL_COMMIT="$commit"
      break
    fi
  done
  
  rm -f "$tmp_metadata"
  trap - EXIT
  
  if [[ -z "$VLLM_PRECOMPILED_WHEEL_COMMIT" ]]; then
    echo "ERROR: no ${VLLM_WHEEL_ARCH} wheel found for ${VLLM_WHEEL_VARIANT}" >&2
    echo "       at or before merge-base $VLLM_MERGE_BASE_COMMIT" >&2
    exit 1
  fi
  
  echo "Using precompiled wheel commit: $VLLM_PRECOMPILED_WHEEL_COMMIT"
  
  VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_PRECOMPILED_WHEEL_COMMIT" \
  VLLM_PRECOMPILED_WHEEL_VARIANT="$VLLM_WHEEL_VARIANT" \
    uv pip install --editable . --torch-backend=auto
  
  uv pip install -r requirements/test/cuda.in

  # fix the decord incompatibility problem
  uv pip uninstall decord
  uv pip install --no-cache decord==0.6.0
  
  popd >/dev/null

  # Dynamo's backend extra and vLLM's test requirements can temporarily select
  # another torch build. Reconcile from the authoritative CUDA 13.0 index;
  # never depend on uv's private cache layout or pre-populated HOME state.
  uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url "$KVCC_TORCH_INDEX_URL" \
    "torch==${KVCC_TORCH_VERSION}" \
    "torchvision==${KVCC_TORCHVISION_VERSION}"

  # dynamo[vllm] pulls vllm==0.24.0 which pins flashinfer-cubin==0.6.12 (PyPI).
  # The vllm editable install then upgrades flashinfer-python to 0.6.14 but
  # setup.py deliberately skips flashinfer-cubin (not on PyPI since 0.6.14).
  # Re-pin cubin from flashinfer.ai to match python.
  FI_VER=$("$VENV_DIR/bin/python" -c \
    "import importlib.metadata as m; print(m.version('flashinfer-python'))")
  uv pip install \
    --extra-index-url "$KVCC_FLASHINFER_INDEX_URL" \
    --extra-index-url "$KVCC_FLASHINFER_CUDA_INDEX_URL" \
    "flashinfer-cubin==${FI_VER}" \
    "flashinfer-jit-cache==${FI_VER}+cu130"
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

echo "==> Verifying the resolved runtime"
"$SCRIPT_DIR/verify-env.sh"

echo ""
echo "==> Build complete. Activate the venv with:"
echo "    source $VENV_DIR/bin/activate"
