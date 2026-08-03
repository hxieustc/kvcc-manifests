#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

KVCC_WORKSPACE="${KVCC_WORKSPACE:-/opt/kvcc-workspace}"
KVCC_MANIFEST_URL="${KVCC_MANIFEST_URL:-https://github.com/hxieustc/kvcc-manifests.git}"
KVCC_MANIFEST_BRANCH="${KVCC_MANIFEST_BRANCH:-cuda-env-fixes}"
KVCC_MANIFEST_FILE="${KVCC_MANIFEST_FILE:-manifests/develop.xml}"
REPO_SYNC_JOBS="${REPO_SYNC_JOBS:-8}"
KVCC_VLLM_OVERLAY="${KVCC_VLLM_OVERLAY:-/opt/kvcc-vllm-overlay/kvcc}"

export CUDARC_CUDA_VERSION
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export GITHUB_USER GITHUB_EMAIL

require_env GITHUB_USER
require_env GITHUB_EMAIL
require_env GITHUB_TOKEN
require_command base64
require_command git
require_command repo
require_command uv

mkdir -p "$KVCC_WORKSPACE"

log "Synchronizing the manifest workspace at $KVCC_WORKSPACE"
setup_github_git_config
trap clear_github_git_config EXIT

pushd "$KVCC_WORKSPACE" >/dev/null
repo init \
  -u "$KVCC_MANIFEST_URL" \
  -b "$KVCC_MANIFEST_BRANCH" \
  -m "$KVCC_MANIFEST_FILE"
repo sync "-j${REPO_SYNC_JOBS}"

repo forall -c \
  'git config user.name "$GITHUB_USER"; git config user.email "$GITHUB_EMAIL"'
popd >/dev/null

if [[ "${KVCC_SKIP_BUILD:-0}" == "1" ]]; then
  log "Repository synchronization complete; build deferred"
  clear_github_git_config
  trap - EXIT
  exit 0
fi

clear_github_git_config
trap - EXIT

DYNAMO_DIR="$KVCC_WORKSPACE/dynamo"
VLLM_DIR="$KVCC_WORKSPACE/vllm"
KVCC_DIR="$KVCC_WORKSPACE/kvcc"
VENV_DIR="$KVCC_WORKSPACE/.venv"

[[ -d "$DYNAMO_DIR" ]] || die "Dynamo checkout is missing: $DYNAMO_DIR"
[[ -d "$VLLM_DIR" ]] || die "vLLM checkout is missing: $VLLM_DIR"
[[ -d "$KVCC_DIR" ]] || die "KVCC checkout is missing: $KVCC_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "Creating Python 3.12 venv that reuses base-image packages"
  uv venv --python 3.12 --system-site-packages "$VENV_DIR"
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
PYTHON="$VENV_DIR/bin/python"

log "Installing build tools"
uv pip install --python "$PYTHON" \
  pip \
  'maturin[patchelf]'

CONSTRAINTS_FILE="$KVCC_WORKSPACE/kvcc-runtime-constraints.txt"
BASE_VERSIONS_FILE="$KVCC_WORKSPACE/base-runtime-versions.json"
"$PYTHON" - "$CONSTRAINTS_FILE" "$BASE_VERSIONS_FILE" <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path
import sys

constraints_path = Path(sys.argv[1])
versions_path = Path(sys.argv[2])
protected = (
    "torch",
    "torchvision",
    "vllm",
    "triton",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)
versions = {}
for name in protected:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass

required = {"torch", "vllm"}
missing = sorted(required.difference(versions))
if missing:
    raise SystemExit(f"base image is missing required packages: {missing}")

constraints_path.write_text(
    "".join(f"{name}=={version}\n" for name, version in versions.items()),
    encoding="utf-8",
)
versions_path.write_text(
    json.dumps(versions, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

log "Building Dynamo Rust/Python bindings"
pushd "$DYNAMO_DIR/lib/bindings/python" >/dev/null
maturin develop --uv
popd >/dev/null

log "Installing customized Dynamo without backend extras"
uv pip install --python "$PYTHON" \
  --constraint "$CONSTRAINTS_FILE" \
  --editable "$DYNAMO_DIR"

SITE_PACKAGES="$(
  "$PYTHON" -c 'import site; print(site.getsitepackages()[0])'
)"
printf '%s\n' "$DYNAMO_DIR/components/src" \
  >"$SITE_PACKAGES/00-kvcc-dynamo-source.pth"

log "Staging the customized vLLM KVCC Python backend"
VLLM_KVCC_SOURCE="$VLLM_DIR/vllm/v1/kv_offload/tiering/kvcc"
[[ -d "$VLLM_KVCC_SOURCE" ]] || \
  die "customized vLLM KVCC source is missing: $VLLM_KVCC_SOURCE"
rm -rf "$KVCC_VLLM_OVERLAY"
mkdir -p "$KVCC_VLLM_OVERLAY"
cp -a "$VLLM_KVCC_SOURCE/." "$KVCC_VLLM_OVERLAY/"

log "Installing KVCC after customized vLLM"
uv pip install --python "$PYTHON" \
  --constraint "$CONSTRAINTS_FILE" \
  --editable "$KVCC_DIR"

log "Confirming that protected base runtime packages were not replaced"
"$PYTHON" - "$BASE_VERSIONS_FILE" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(
        "protected base packages changed:\n"
        f"expected={expected}\nactual={actual}"
    )
PY

log "Customized Dynamo and KVCC are installed; vLLM overlay is staged"
