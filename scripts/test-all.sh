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
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

FAILED=0

cleanup_children() {
  local pid
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done < <(jobs -pr)
}
trap cleanup_children EXIT INT TERM

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
# 3. Focused Dynamo/vLLM router integration tests
# ---------------------------------------------------------------------------
if [[ -d "$WORKSPACE_ROOT/dynamo" ]]; then
  echo "==> Dynamo/vLLM router integration tests"
  "$PYTHON" -m pytest \
    -c /dev/null \
    "$WORKSPACE_ROOT/dynamo/components/src/dynamo/vllm/tests/test_router_hints.py" \
    -q || FAILED=1
fi

# ---------------------------------------------------------------------------
# 4. KVCC end-to-end GPU test
# ---------------------------------------------------------------------------
if [[ "$KVCC_E2E_TEST" == "1" ]]; then
  if ! "$PYTHON" - "$GPUS" <<'PY'
import sys

import torch

requested = sys.argv[1].split(",")
count = torch.cuda.device_count()
invalid = [item for item in requested if not item.isdigit() or int(item) >= count]
if invalid:
    raise SystemExit(
        f"requested GPU indices {invalid} are unavailable; visible GPU count={count}"
    )
PY
  then
    echo "ERROR: GPU preflight failed for KVCC_TEST_GPUS=$GPUS" >&2
    FAILED=1
  else
    echo "==> KVCC E2E test (GPUs $GPUS)"
    "$PYTHON" -m pytest \
      "$WORKSPACE_ROOT/vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e/validate_kvcc_e2e.py" \
      --gpus "$GPUS" \
      --eager-ctrl-connect=true \
      -vv -s --log-cli-level=INFO || FAILED=1
  fi
fi

if [[ "$FAILED" -ne 0 ]]; then
  echo "==> One or more test suites FAILED" >&2
  exit 1
fi

echo "==> All supported KVCC, vLLM/KVCC, Dynamo router, and E2E tests passed"
