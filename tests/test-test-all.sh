#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE="$(mktemp -d)"
trap 'rm -rf "$FIXTURE"' EXIT
WORKSPACE="$FIXTURE/workspace"

mkdir -p \
  "$WORKSPACE/manifests/scripts" \
  "$WORKSPACE/.venv/bin" \
  "$WORKSPACE/kvcc/tests" \
  "$WORKSPACE/vllm/tests" \
  "$WORKSPACE/dynamo/components/src/dynamo/vllm/tests"
cp "$REPO_ROOT/scripts/test-all.sh" \
  "$WORKSPACE/manifests/scripts/test-all.sh"

PYTEST_LOG="$FIXTURE/pytest.log"
export PYTEST_LOG
cat >"$WORKSPACE/.venv/bin/python" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-" ]]; then
  requested="${2:-}"
  count="${FAKE_GPU_COUNT:-2}"
  IFS=, read -ra indices <<<"$requested"
  for index in "${indices[@]}"; do
    [[ "$index" =~ ^[0-9]+$ ]] || exit 2
    (( index < count )) || exit 2
  done
  exit 0
fi
printf 'warmup=%s python ' "${VLLM_DEEP_GEMM_WARMUP:-unset}" >>"$PYTEST_LOG"
printf '%s ' "$@" >>"$PYTEST_LOG"
printf '\n' >>"$PYTEST_LOG"
STUB
chmod +x "$WORKSPACE/.venv/bin/python"

KVCC_RUN_E2E=1 KVCC_TEST_GPUS=0,1 \
  bash "$WORKSPACE/manifests/scripts/test-all.sh"

grep -F 'python -m pytest tests/ -q' "$PYTEST_LOG" >/dev/null
grep -F 'test_config.py' "$PYTEST_LOG" >/dev/null
grep -F 'test_harness.py' "$PYTEST_LOG" >/dev/null
grep -F 'test_kvcc_tier.py' "$PYTEST_LOG" >/dev/null
grep -F \
  'dynamo/components/src/dynamo/vllm/tests/test_router_hints.py' \
  "$PYTEST_LOG" >/dev/null
grep -F 'validate_kvcc_e2e.py' "$PYTEST_LOG" >/dev/null
grep -F -- '--gpus 0,1' "$PYTEST_LOG" >/dev/null
grep -F 'warmup=skip' "$PYTEST_LOG" >/dev/null

: >"$PYTEST_LOG"
if FAKE_GPU_COUNT=2 KVCC_RUN_E2E=1 KVCC_TEST_GPUS=0,2 \
  bash "$WORKSPACE/manifests/scripts/test-all.sh"; then
  echo 'unavailable GPU index must fail test-all' >&2
  exit 1
fi
if grep -F 'validate_kvcc_e2e.py' "$PYTEST_LOG" >/dev/null; then
  echo 'E2E must not start after GPU preflight failure' >&2
  exit 1
fi

: >"$PYTEST_LOG"
KVCC_RUN_E2E=0 bash "$WORKSPACE/manifests/scripts/test-all.sh"
grep -F 'test_router_hints.py' "$PYTEST_LOG" >/dev/null
if grep -F 'validate_kvcc_e2e.py' "$PYTEST_LOG" >/dev/null; then
  echo 'KVCC_RUN_E2E=0 must omit E2E' >&2
  exit 1
fi

echo 'PASS: test-all runs the supported deterministic test scope'
