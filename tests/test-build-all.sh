#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT

WORKSPACE="$FIXTURE/workspace"
mkdir -p \
  "$WORKSPACE/manifests/scripts" \
  "$WORKSPACE/dynamo/lib/bindings/python" \
  "$WORKSPACE/vllm/requirements/test" \
  "$WORKSPACE/kvcc" \
  "$WORKSPACE/.venv/bin" \
  "$FIXTURE/empty-home"
cp "$REPO_ROOT/scripts/build-all.sh" \
  "$WORKSPACE/manifests/scripts/build-all.sh"
printf '#!/usr/bin/env bash\nexit 0\n' \
  >"$WORKSPACE/manifests/scripts/verify-env.sh"
chmod +x "$WORKSPACE/manifests/scripts/verify-env.sh"
: >"$WORKSPACE/vllm/requirements/test/cuda.in"

UV_LOG="$FIXTURE/uv.log"
export UV_LOG

cat >"$WORKSPACE/.venv/bin/uv" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'cwd=%s uv ' "$PWD" >>"$UV_LOG"
printf '%q ' "$@" >>"$UV_LOG"
printf '\n' >>"$UV_LOG"
STUB
cat >"$WORKSPACE/.venv/bin/maturin" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'cwd=%s maturin ' "$PWD" >>"$UV_LOG"
printf '%q ' "$@" >>"$UV_LOG"
printf '\n' >>"$UV_LOG"
STUB
cat >"$WORKSPACE/.venv/bin/python3" <<'STUB'
#!/usr/bin/env bash
printf '0.6.15.post1\n'
STUB
chmod +x \
  "$WORKSPACE/.venv/bin/uv" \
  "$WORKSPACE/.venv/bin/maturin" \
  "$WORKSPACE/.venv/bin/python3"
cp "$WORKSPACE/.venv/bin/python3" "$WORKSPACE/.venv/bin/python"

HOME="$FIXTURE/empty-home" \
CUDARC_CUDA_VERSION=13000 \
  bash "$WORKSPACE/manifests/scripts/build-all.sh"

grep -F -- \
  '--extra-index-url https://download.pytorch.org/whl/cu130' \
  "$UV_LOG" >/dev/null
grep -F -- 'torch==2.13.0+cu130' "$UV_LOG" >/dev/null
grep -F -- 'torchvision==0.28.0+cu130' "$UV_LOG" >/dev/null
grep -F -- 'flashinfer-cubin==0.6.15.post1' "$UV_LOG" >/dev/null
grep -F -- 'flashinfer-jit-cache==0.6.15.post1+cu130' \
  "$UV_LOG" >/dev/null
grep -F -- 'https://flashinfer.ai/whl/cu130' "$UV_LOG" >/dev/null

if grep -F -- '.cache/uv/archive-v0' \
  "$WORKSPACE/manifests/scripts/build-all.sh" >/dev/null; then
  echo 'build-all.sh must not depend on uv internal cache layout' >&2
  exit 1
fi

dynamo_line="$(grep -n 'cwd=.*/dynamo uv pip install -e' "$UV_LOG" | head -1 | cut -d: -f1)"
vllm_line="$(grep -n 'cwd=.*/vllm uv pip install --editable' "$UV_LOG" | head -1 | cut -d: -f1)"
kvcc_line="$(grep -n 'cwd=.*/kvcc uv pip install' "$UV_LOG" | head -1 | cut -d: -f1)"
if ! (( dynamo_line < vllm_line && vllm_line < kvcc_line )); then
  echo 'expected build order: Dynamo, customized vLLM, KVCC' >&2
  exit 1
fi

echo 'PASS: build-all reconciles CUDA 13 dependencies without a warm cache'
