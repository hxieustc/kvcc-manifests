#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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
    fail "$path is missing (wanted: $pattern)"
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
    fail "$path is missing (cannot inspect: $pattern)"
  elif grep -Eq -- "$pattern" "$path"; then
    fail "$path contains forbidden pattern: $pattern"
  else
    pass "$path omits: $pattern"
  fi
}

assert_file kubernetes/dev-pod.yaml
assert_contains kubernetes/dev-pod.yaml \
  'nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest'
assert_contains kubernetes/dev-pod.yaml \
  'nvidia.com/gpu:[[:space:]]*"?2"?'
assert_contains kubernetes/dev-pod.yaml \
  'nvidia.com/gpu.product:[[:space:]]*NVIDIA-B200'
assert_contains kubernetes/dev-pod.yaml \
  'kubernetes.io/arch:[[:space:]]*amd64'
assert_contains kubernetes/dev-pod.yaml \
  'mountPath:[[:space:]]*/dev/shm'
assert_contains kubernetes/dev-pod.yaml \
  'name:[[:space:]]*kvcc-github-auth'
assert_contains kubernetes/dev-pod.yaml \
  'name:[[:space:]]*repo-installer'
assert_contains kubernetes/dev-pod.yaml \
  'storageClassName:[[:space:]]*vast'
assert_contains kubernetes/dev-pod.yaml 'storage:[[:space:]]*200Gi'
assert_contains kubernetes/dev-pod.yaml 'sizeLimit:[[:space:]]*64Gi'
assert_contains kubernetes/dev-pod.yaml 'credential\.helper'
assert_contains kubernetes/dev-pod.yaml 'password=%s'
assert_contains kubernetes/dev-pod.yaml 'GITHUB_USER.*GITHUB_TOKEN'
assert_not_contains kubernetes/dev-pod.yaml 'ghp_|github_pat_'
assert_not_contains manifests/develop.xml 'github-ssh|ssh://'
assert_contains manifests/develop.xml 'revision="cuda-env-fixes"'
assert_file scripts/verify-env.sh
assert_contains scripts/build-all.sh 'verify-env\.sh'
assert_contains scripts/verify-env.sh '2\.13\.0\+cu130'
assert_contains scripts/verify-env.sh 'flashinfer-jit-cache'
assert_contains scripts/verify-env.sh 'resolved outside workspace'
assert_contains scripts/test-all.sh 'VLLM_DEEP_GEMM_WARMUP'
assert_contains scripts/test-all.sh 'test_router_hints\.py'
assert_contains README.md 'export GITHUB_USER=hxieustc'
assert_contains README.md 'export GITHUB_EMAIL=.*harryx@nvidia\.com'
assert_contains README.md 'create secret generic kvcc-github-auth'
assert_contains README.md '--from-env-file'
assert_contains README.md \
  'tsh kubectl apply -f kubernetes/dev-pod\.yaml'
assert_contains README.md \
  'repo init.*https://github\.com/hxieustc/kvcc-manifests\.git'
assert_contains README.md '-b cuda-env-fixes'
assert_contains README.md 'repo sync'
assert_contains README.md 'bash manifests/scripts/bootstrap\.sh'
assert_contains README.md 'bash manifests/scripts/build-all\.sh'
assert_contains README.md 'bash manifests/scripts/test-all\.sh'
assert_contains README.md 'delete pod kvcc-dev'
assert_contains README.md 'PVC.*retained|retains.*PVC'
assert_not_contains README.md 'http\..*\.extraHeader|Authorization: Basic'

if [[ "$failures" -ne 0 ]]; then
  printf '\n%d repository contract failure(s)\n' "$failures" >&2
  exit 1
fi

printf '\nRepository contract checks passed\n'
