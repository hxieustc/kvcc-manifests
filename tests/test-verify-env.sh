#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_312="${PYTHON_312:-/Users/harryx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
if [[ ! -x "$PYTHON_312" ]]; then
  PYTHON_312="$(command -v python3.12 || true)"
fi
[[ -x "$PYTHON_312" ]] || {
  echo 'Python 3.12 is required for verifier tests' >&2
  exit 1
}

FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT
WORKSPACE="$FIXTURE/workspace"
SITE="$FIXTURE/site"
EXTERNAL="$FIXTURE/external"
mkdir -p \
  "$WORKSPACE/.venv/bin" \
  "$WORKSPACE/dynamo/dynamo/vllm" \
  "$WORKSPACE/vllm/vllm" \
  "$WORKSPACE/kvcc/kvcc" \
  "$SITE/torch" \
  "$SITE/nixl" \
  "$EXTERNAL/vllm" \
  "$FIXTURE/bin"
ln -s "$PYTHON_312" "$WORKSPACE/.venv/bin/python"

cat >"$SITE/torch/__init__.py" <<'PY'
import os
from types import SimpleNamespace
__version__ = os.environ.get("FAKE_TORCH_VERSION", "2.13.0+cu130")
version = SimpleNamespace(cuda=os.environ.get("FAKE_TORCH_CUDA", "13.0"))
PY
printf '' >"$SITE/nixl/__init__.py"
printf '' >"$WORKSPACE/dynamo/dynamo/__init__.py"
printf '' >"$WORKSPACE/dynamo/dynamo/vllm/__init__.py"
printf '' >"$WORKSPACE/vllm/vllm/__init__.py"
printf '' >"$WORKSPACE/vllm/vllm/_C_stable_libtorch.py"
printf '' >"$WORKSPACE/kvcc/kvcc/__init__.py"
printf '' >"$EXTERNAL/vllm/__init__.py"
printf '' >"$EXTERNAL/vllm/_C_stable_libtorch.py"

make_dist() {
  local directory="$1"
  local project="$2"
  local version="$3"
  mkdir -p "$SITE/${directory}-${version}.dist-info"
  printf 'Metadata-Version: 2.1\nName: %s\nVersion: %s\n' \
    "$project" "$version" \
    >"$SITE/${directory}-${version}.dist-info/METADATA"
}

make_dist torchvision torchvision 0.28.0+cu130
make_dist flashinfer_python flashinfer-python 0.6.15.post1
make_dist flashinfer_cubin flashinfer-cubin 0.6.15.post1
make_dist flashinfer_jit_cache flashinfer-jit-cache 0.6.15.post1+cu130

cat >"$FIXTURE/bin/uv" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'uv %s\n' "$*" >>"$UV_CHECK_LOG"
STUB
chmod +x "$FIXTURE/bin/uv"
export UV_CHECK_LOG="$FIXTURE/uv-check.log"

run_verify() {
  local prefix="${1:-}"
  env \
    PATH="$FIXTURE/bin:/usr/bin:/bin" \
    PYTHONPATH="$prefix:$WORKSPACE/dynamo:$WORKSPACE/vllm:$WORKSPACE/kvcc:$SITE" \
    KVCC_WORKSPACE="$WORKSPACE" \
    "$REPO_ROOT/scripts/verify-env.sh"
}

run_verify
grep -F 'uv pip check --python' "$UV_CHECK_LOG" >/dev/null

expect_failure() {
  local expected="$1"
  shift
  local output
  if output=$("$@" 2>&1); then
    echo "expected verifier failure containing: $expected" >&2
    exit 1
  fi
  grep -F "$expected" <<<"$output" >/dev/null || {
    printf 'missing expected failure %s in:\n%s\n' "$expected" "$output" >&2
    exit 1
  }
}

run_bad_torch() {
  FAKE_TORCH_VERSION=2.13.0+cu132 run_verify
}
expect_failure 'torch 2.13.0+cu130 required' run_bad_torch

mv "$SITE/flashinfer_jit_cache-0.6.15.post1+cu130.dist-info" \
  "$SITE/flashinfer_jit_cache-0.6.14+cu130.dist-info"
sed -i.bak 's/0\.6\.15\.post1+cu130/0.6.14+cu130/' \
  "$SITE/flashinfer_jit_cache-0.6.14+cu130.dist-info/METADATA"
expect_failure 'FlashInfer JIT cache mismatch' run_verify
mv "$SITE/flashinfer_jit_cache-0.6.14+cu130.dist-info" \
  "$SITE/flashinfer_jit_cache-0.6.15.post1+cu130.dist-info"
sed -i.bak 's/0\.6\.14+cu130/0.6.15.post1+cu130/' \
  "$SITE/flashinfer_jit_cache-0.6.15.post1+cu130.dist-info/METADATA"

expect_failure 'resolved outside workspace' run_verify "$EXTERNAL"

echo 'PASS: verify-env rejects incompatible versions and import provenance'
