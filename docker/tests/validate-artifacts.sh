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
    fail "$path is missing (wanted pattern: $pattern)"
  elif grep -Eq -- "$pattern" "$path"; then
    pass "$path contains required pattern: $pattern"
  else
    fail "$path lacks required pattern: $pattern"
  fi
}

assert_not_contains() {
  local path="$1"
  local pattern="$2"
  if [[ ! -f "$path" ]]; then
    fail "$path is missing (cannot check forbidden pattern: $pattern)"
  elif grep -Eq -- "$pattern" "$path"; then
    fail "$path contains forbidden pattern: $pattern"
  else
    pass "$path omits forbidden pattern: $pattern"
  fi
}

assert_executable() {
  local path="$1"
  if [[ -x "$path" ]]; then
    pass "$path is executable"
  else
    fail "$path is not executable"
  fi
}

assert_file Dockerfile
assert_contains Dockerfile 'AS builder'
assert_contains Dockerfile 'AS runtime'
assert_contains Dockerfile 'mount=type=secret,id=github_token'
assert_contains Dockerfile 'target=/root/\.cache/uv'
assert_contains Dockerfile 'target=/root/\.cache/ccache'
assert_contains Dockerfile 'target=/root/\.cargo'
assert_contains Dockerfile 'FLASHINFER_DISABLE_VERSION_CHECK=1'
assert_contains Dockerfile 'git lfs install --system'
assert_not_contains Dockerfile 'COPY patches/'
assert_contains Dockerfile 'install-vllm-overlay\.py'
assert_not_contains Dockerfile 'ARG[[:space:]]+GITHUB_TOKEN'
assert_not_contains Dockerfile 'ENV[[:space:]]+GITHUB_TOKEN'
assert_contains scripts/build-image.sh \
  '--secret id=github_token,env=GITHUB_TOKEN'
assert_contains scripts/build-image.sh '--build-arg GITHUB_USER'
assert_contains scripts/build-image.sh '--build-arg GITHUB_EMAIL'
assert_contains scripts/build-image.sh 'KVCC_PLATFORM:-linux/amd64'

assert_file kubernetes/dev-pod.yaml
assert_contains kubernetes/dev-pod.yaml 'namespace:[[:space:]]+kvcc-mbench'
assert_contains kubernetes/dev-pod.yaml 'storage:[[:space:]]+200Gi'
assert_contains kubernetes/dev-pod.yaml 'storageClassName:[[:space:]]+vast'
assert_contains kubernetes/dev-pod.yaml 'ReadWriteMany'
assert_contains kubernetes/dev-pod.yaml 'kubernetes.io/arch:[[:space:]]+amd64'
assert_contains kubernetes/dev-pod.yaml \
  'nvidia.com/gpu.product:[[:space:]]+NVIDIA-B200'
assert_contains kubernetes/dev-pod.yaml 'nvidia.com/gpu:[[:space:]]+"?2"?'
assert_contains kubernetes/dev-pod.yaml 'key:[[:space:]]+nvidia.com/gpu'
assert_contains kubernetes/dev-pod.yaml 'value:[[:space:]]+"true"'
assert_contains kubernetes/dev-pod.yaml 'effect:[[:space:]]+NoSchedule'
assert_contains kubernetes/dev-pod.yaml 'claimName:[[:space:]]+kvcc-workspace'
assert_contains kubernetes/dev-pod.yaml 'name:[[:space:]]+kvcc-tools'
assert_contains kubernetes/dev-pod.yaml 'name:[[:space:]]+kvcc-patches'
assert_contains kubernetes/dev-pod.yaml 'name:[[:space:]]+KVCC_WHOLE_TEST'
assert_contains kubernetes/dev-pod.yaml 'secretRef:'
assert_contains kubernetes/dev-pod.yaml 'name:[[:space:]]+kvcc-github-auth'
assert_contains kubernetes/dev-pod.yaml 'mountPath:[[:space:]]+/dev/shm'
assert_contains kubernetes/dev-pod.yaml 'medium:[[:space:]]+Memory'
assert_contains kubernetes/dev-pod.yaml 'restartPolicy:[[:space:]]+Never'
assert_contains kubernetes/dev-pod.yaml 'value:[[:space:]]+"0"'
assert_contains kubernetes/dev-pod.yaml \
  'name:[[:space:]]+FLASHINFER_DISABLE_VERSION_CHECK'
assert_contains kubernetes/dev-pod.yaml \
  '^[[:space:]]+- "trap : TERM INT; sleep infinity & wait"$'
assert_not_contains kubernetes/dev-pod.yaml 'harryx@nvidia.com'
assert_not_contains kubernetes/dev-pod.yaml 'GITHUB_TOKEN:'

assert_contains scripts/sync-build.sh \
  'https://github.com/hxieustc/kvcc-manifests.git'
assert_contains scripts/sync-build.sh 'manifests/develop.xml'
assert_contains scripts/lib.sh 'GIT_CONFIG_COUNT'
assert_contains scripts/sync-build.sh 'repo init'
assert_contains scripts/sync-build.sh 'repo sync'
assert_contains scripts/sync-build.sh 'cuda-env-fixes'
assert_not_contains scripts/sync-build.sh 'apply-custom-patches\.sh'
assert_contains scripts/sync-build.sh 'maturin develop'
assert_contains scripts/sync-build.sh '--editable "\$KVCC_DIR"'
assert_contains scripts/sync-build.sh 'KVCC_VLLM_OVERLAY'
assert_contains scripts/sync-build.sh 'v1/kv_offload/tiering/kvcc'
assert_not_contains scripts/sync-build.sh 'VLLM_USE_PRECOMPILED'
assert_not_contains scripts/sync-build.sh '--editable "\$VLLM_DIR"'
assert_not_contains scripts/sync-build.sh '00-kvcc-workspace\.pth'
assert_not_contains scripts/sync-build.sh 'build-all\.sh'
assert_not_contains scripts/sync-build.sh 'test-all\.sh'
assert_contains scripts/build-in-pod.sh 'export HOME=/root'
assert_not_contains scripts/sync-build.sh \
  'credential\.https://github\.com'
assert_not_contains scripts/sync-build.sh \
  'git config.*extraHeader'
assert_contains scripts/verify-environment.sh '2\.13\.0'
assert_contains scripts/verify-environment.sh '0\.28\.0'
assert_contains scripts/verify-environment.sh 'flashinfer-jit-cache'
assert_contains scripts/verify-environment.sh \
  'decord.*built for a different platform'
assert_contains scripts/verify-environment.sh 'dynamo\.vllm'
assert_contains scripts/verify-environment.sh 'vllm\._C_stable_libtorch'
assert_contains scripts/verify-environment.sh 'assert_from_workspace\("kvcc"\)'
assert_not_contains scripts/verify-environment.sh '\["git",[[:space:]]+"-C"'
assert_contains scripts/apply-pod.sh 'tsh kubectl'
assert_contains scripts/apply-pod.sh 'chmod 600'
assert_contains scripts/apply-pod.sh 'trap cleanup EXIT'
assert_contains scripts/apply-pod.sh 'rm -f'
assert_contains scripts/apply-pod.sh 'create configmap kvcc-tools'
assert_contains scripts/apply-pod.sh 'create configmap.*PATCH_CONFIGMAP_NAME'
assert_contains scripts/apply-pod.sh 'create secret generic kvcc-github-auth'
assert_not_contains scripts/apply-pod.sh \
  'kubectl create secret generic.*--from-literal'
assert_contains scripts/delete-pod.sh 'delete pod'
assert_not_contains scripts/delete-pod.sh 'delete pvc'

assert_file scripts/apply-custom-patches.sh
assert_executable scripts/apply-custom-patches.sh
assert_contains scripts/apply-custom-patches.sh 'git -C.*apply --check'
assert_contains scripts/apply-custom-patches.sh 'apply --reverse --check'
assert_file scripts/test-all-in-pod.sh
assert_executable scripts/test-all-in-pod.sh
assert_contains scripts/test-all-in-pod.sh 'bootstrap\.sh'
assert_contains scripts/test-all-in-pod.sh 'build-all\.sh'
assert_contains scripts/test-all-in-pod.sh 'test-all\.sh'
assert_contains scripts/test-all-in-pod.sh 'test_router_hints\.py'

for patch in patches/*.patch; do
  assert_file "$patch"
done

assert_file README.md
assert_contains README.md 'export GITHUB_USER=hxieustc'
assert_contains README.md 'export GITHUB_EMAIL=.*harryx@nvidia\.com'
assert_contains README.md 'GITHUB_TOKEN'
assert_contains README.md 'scripts/build-image\.sh'
assert_contains README.md 'scripts/apply-pod\.sh'
assert_contains README.md 'scripts/exec-pod\.sh'
assert_contains README.md 'tsh kubectl -n kvcc-mbench'
assert_contains README.md 'must not be committed'
assert_contains README.md 'PVC.*retained'

assert_file scripts/install-vllm-overlay.py
if bash tests/test-install-vllm-overlay.sh; then
  pass 'selective vLLM overlay replaces only the KVCC backend'
else
  fail 'selective vLLM overlay behavior is incorrect'
fi

assert_file scripts/verify-selective-runtime.py
if bash tests/test-verify-selective-runtime.sh; then
  pass 'selective runtime keeps vLLM native code in the base package'
else
  fail 'selective runtime import provenance is incorrect'
fi

if [[ -d scripts ]]; then
  while IFS= read -r script; do
    if bash -n "$script"; then
      pass "$script passes bash -n"
    else
      fail "$script fails bash -n"
    fi
  done < <(find scripts -maxdepth 1 -type f -name '*.sh' -print | sort)
fi

if [[ "$failures" -ne 0 ]]; then
  printf '\n%d artifact validation failure(s)\n' "$failures" >&2
  exit 1
fi

printf '\nAll artifact validation checks passed\n'
