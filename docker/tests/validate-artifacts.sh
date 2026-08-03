#!/usr/bin/env bash
set -euo pipefail

DOCKER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DOCKER_ROOT"

failures=0

pass() {
  printf 'PASS: %s\n' "$1"
}

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  failures=$((failures + 1))
}

assert_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    pass "$path exists"
  else
    fail "$path is missing"
  fi
}

assert_contains() {
  local path="$1"
  local pattern="$2"
  if [[ ! -f "$path" ]]; then
    fail "$path is missing (wanted pattern: $pattern)"
  elif grep -Eq -- "$pattern" "$path"; then
    pass "$path contains: $pattern"
  else
    fail "$path lacks: $pattern"
  fi
}

assert_not_contains() {
  local path="$1"
  local pattern="$2"
  if [[ ! -f "$path" ]]; then
    fail "$path is missing (cannot check: $pattern)"
  elif grep -Eq -- "$pattern" "$path"; then
    fail "$path contains forbidden pattern: $pattern"
  else
    pass "$path omits: $pattern"
  fi
}

assert_file Dockerfile
assert_contains Dockerfile \
  'ARG[[:space:]]+DYNAMO_BUILDER_IMAGE=rust:1\.93-bookworm'
assert_contains Dockerfile \
  'ARG[[:space:]]+VLLM_RUNTIME_IMAGE=vllm/vllm-openai:nightly-6f91edf96d3f3272945809c04702380053bff4de@sha256:674c5aa666d38c07a0dc779f8c77a05c2b859617410d16dcdd8a8776166c92b9'
assert_contains Dockerfile \
  'FROM[[:space:]]+\$\{DYNAMO_BUILDER_IMAGE\} AS dynamo-builder'
assert_contains Dockerfile \
  'FROM[[:space:]]+\$\{VLLM_RUNTIME_IMAGE\} AS runtime'
assert_contains Dockerfile 'mount=type=secret,id=github_token'
assert_contains Dockerfile 'GIT_CONFIG_COUNT'
assert_contains Dockerfile 'repo init'
assert_contains Dockerfile 'https://github\.com/hxieustc/kvcc-manifests\.git'
assert_contains Dockerfile 'cuda-env-fixes'
assert_contains Dockerfile 'manifests/develop\.xml'
assert_contains Dockerfile 'repo sync'
assert_contains Dockerfile \
  'maturin build --release --features select-service,kv-indexer --out /wheels'
assert_contains Dockerfile 'pip wheel --no-deps'
assert_contains Dockerfile 'ai_dynamo_runtime-\*\.whl'
assert_contains Dockerfile 'ai_dynamo-\*\.whl'
assert_contains Dockerfile 'flashinfer-jit-cache'
assert_contains Dockerfile 'flashinfer-cubin'
assert_contains Dockerfile 'python3 -m pip install --no-cache-dir /src/nvidia-kvcc'
assert_contains Dockerfile 'vllm/v1/kv_offload/tiering/kvcc'
assert_contains Dockerfile 'vllm/tests/v1/kv_offload/tiering'
assert_contains Dockerfile 'vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e'
assert_contains Dockerfile 'vllm\.v1\.kv_offload\.tiering\.kvcc\.manager'
assert_contains Dockerfile 'vllm\._C_stable_libtorch'
assert_contains Dockerfile 'dynamo\.vllm\.router_hints'
assert_contains Dockerfile 'openai\.types\.responses import NamespaceTool'
assert_not_contains Dockerfile 'DYNAMO_BASE_IMAGE'
assert_not_contains Dockerfile 'ai-dynamo/vllm-runtime-nightly'
assert_not_contains Dockerfile 'VLLM_USE_PRECOMPILED'
assert_not_contains Dockerfile 'COPY[[:space:]]+scripts/'
assert_not_contains Dockerfile 'hf download'
assert_not_contains Dockerfile 'build-all\.sh'
assert_not_contains Dockerfile 'test-all\.sh'
assert_not_contains Dockerfile 'ARG[[:space:]]+GITHUB_TOKEN'
assert_not_contains Dockerfile 'ENV[[:space:]]+GITHUB_TOKEN'

assert_file scripts/build-image.sh
assert_contains scripts/build-image.sh \
  '--secret id=github_token,env=GITHUB_TOKEN'
assert_contains scripts/build-image.sh '--build-arg GITHUB_USER'
assert_contains scripts/build-image.sh '--build-arg GITHUB_EMAIL'
assert_contains scripts/build-image.sh '--build-arg.*DYNAMO_BUILDER_IMAGE'
assert_contains scripts/build-image.sh '--build-arg.*VLLM_RUNTIME_IMAGE'
assert_contains scripts/build-image.sh 'KVCC_PLATFORM:-linux/amd64'
assert_not_contains scripts/build-image.sh 'source .*lib\.sh'

assert_file README.md
assert_contains README.md 'export GITHUB_USER=hxieustc'
assert_contains README.md 'export GITHUB_EMAIL=.*harryx@nvidia\.com'
assert_contains README.md 'GITHUB_TOKEN'
assert_contains README.md 'scripts/build-image\.sh'

if bash -n scripts/build-image.sh; then
  pass 'scripts/build-image.sh passes bash -n'
else
  fail 'scripts/build-image.sh fails bash -n'
fi

if [[ "$failures" -ne 0 ]]; then
  printf '\n%d Docker artifact validation failure(s)\n' "$failures" >&2
  exit 1
fi

printf '\nAll Docker artifact validation checks passed\n'
