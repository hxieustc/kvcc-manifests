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
  'mountPath:[[:space:]]*/models-shared'
assert_contains kubernetes/dev-pod.yaml \
  '^[[:space:]]{8}claimName:[[:space:]]*shared-model-cache'
assert_contains kubernetes/dev-pod.yaml \
  'name:[[:space:]]*kvcc-github-auth'
assert_contains kubernetes/dev-pod.yaml 'sizeLimit:[[:space:]]*64Gi'
assert_contains kubernetes/dev-pod.yaml 'apt-get install'
assert_contains kubernetes/dev-pod.yaml 'git-lfs'
assert_contains kubernetes/dev-pod.yaml '/usr/local/bin/repo'
assert_contains kubernetes/dev-pod.yaml \
  'credential\.https://github\.com/\.username'
assert_contains kubernetes/dev-pod.yaml \
  'credential\.https://github\.com/\.email'
assert_contains kubernetes/dev-pod.yaml \
  'http\.https://github\.com/\.extraHeader'
assert_contains kubernetes/dev-pod.yaml 'Authorization: Basic'
assert_contains kubernetes/dev-pod.yaml 'readinessProbe:'
assert_contains kubernetes/dev-pod.yaml \
  'https://github\.com/hxieustc/kvcc-manifests\.git'
assert_contains kubernetes/dev-pod.yaml 'repo sync -j8'
assert_contains kubernetes/dev-pod.yaml \
  'bash manifests/scripts/bootstrap\.sh'
assert_contains kubernetes/dev-pod.yaml \
  'bash manifests/scripts/build-all\.sh'
assert_contains kubernetes/dev-pod.yaml \
  'bash manifests/scripts/test-all\.sh'
assert_contains kubernetes/dev-pod.yaml '/tmp/kvcc-workflow-ready'
assert_contains kubernetes/dev-pod.yaml 'sleep infinity'
assert_not_contains kubernetes/dev-pod.yaml 'kind:[[:space:]]*PersistentVolumeClaim'
assert_not_contains kubernetes/dev-pod.yaml 'kind:[[:space:]]*ConfigMap'
assert_not_contains kubernetes/dev-pod.yaml 'initContainers:'
assert_not_contains kubernetes/dev-pod.yaml 'kvcc-pod-tools|kvcc-bin'
assert_not_contains kubernetes/dev-pod.yaml 'credential\.helper'
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
assert_contains README.md '--from-env-file=/dev/stdin'
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
assert_contains README.md 'Manual.*without Kubernetes|without a Pod'
assert_contains README.md 'Automated Kubernetes|automated Pod'
assert_contains README.md 'get pvc shared-model-cache'
assert_contains README.md '--ignore-not-found'
assert_contains README.md 'logs.*kvcc-dev'
assert_contains README.md 'exec -it kvcc-dev'
assert_contains README.md 'ephemeral|container-local'
assert_not_contains README.md 'PVC.*retained|retains.*PVC'
assert_not_contains README.md 'credential helper|credential-helper'

if [[ "$failures" -ne 0 ]]; then
  printf '\n%d repository contract failure(s)\n' "$failures" >&2
  exit 1
fi

printf '\nRepository contract checks passed\n'
