#!/usr/bin/env bash
# test-all.sh — run unit and integration tests across all components
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$WORKSPACE_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found — run scripts/bootstrap.sh first" >&2
  exit 1
fi

PYTHON="$VENV_DIR/bin/python"

# Options (override via env)
GPUS="${KVCC_TEST_GPUS:-0,1}"
KVCC_E2E_TEST="${KVCC_RUN_E2E:-1}"

FAILED=0

# ---------------------------------------------------------------------------
# 1. KVCC standalone unit tests (CPU only)
# ---------------------------------------------------------------------------
if [[ -d "$WORKSPACE_ROOT/kvcc" ]]; then
  echo "==> KVCC standalone unit tests"
  pushd "$WORKSPACE_ROOT/kvcc" >/dev/null
  "$PYTHON" -m pytest tests/ -q || FAILED=1
  popd >/dev/null
fi

# ---------------------------------------------------------------------------
# 2. vLLM KVCC unit tests (CPU only)
# ---------------------------------------------------------------------------
echo "==> vLLM KVCC unit tests"
"$PYTHON" -m pytest \
  "$WORKSPACE_ROOT/vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e/test_config.py" \
  "$WORKSPACE_ROOT/vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e/test_harness.py" \
  "$WORKSPACE_ROOT/vllm/tests/v1/kv_offload/tiering/test_kvcc_tier.py" \
  -q || FAILED=1

# ---------------------------------------------------------------------------
# 3. KVCC end-to-end GPU test
# ---------------------------------------------------------------------------
if [[ "$KVCC_E2E_TEST" == "1" ]]; then
  echo "==> KVCC E2E test (GPUs $GPUS)"
  "$PYTHON" -m pytest \
    "$WORKSPACE_ROOT/vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e/validate_kvcc_e2e.py" \
    --gpus "$GPUS" \
    --eager-ctrl-connect=true \
    -vv -s --log-cli-level=INFO || FAILED=1
fi

# ---------------------------------------------------------------------------
# 4. Dynamo tests
# ---------------------------------------------------------------------------
if [[ -d "$WORKSPACE_ROOT/dynamo" ]]; then
  echo "==> Dynamo tests"
  # TODO: add Dynamo test commands
  # cargo test --manifest-path "$WORKSPACE_ROOT/dynamo/Cargo.toml" || FAILED=1
  true
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo "==> One or more test suites FAILED" >&2
  exit 1
fi

echo "==> All tests passed"
