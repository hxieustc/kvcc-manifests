#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

FAKE_SITE_PACKAGES="$TEST_ROOT/site-packages"
VLLM_PACKAGE="$FAKE_SITE_PACKAGES/vllm"
OVERLAY="$TEST_ROOT/overlay/kvcc"

mkdir -p \
  "$VLLM_PACKAGE/v1/kv_offload/tiering/kvcc" \
  "$VLLM_PACKAGE/v1/kv_offload/tiering/other_backend" \
  "$OVERLAY"

touch "$VLLM_PACKAGE/__init__.py"
printf '%s\n' 'stale base implementation' \
  >"$VLLM_PACKAGE/v1/kv_offload/tiering/kvcc/stale.py"
printf '%s\n' 'must survive' \
  >"$VLLM_PACKAGE/v1/kv_offload/tiering/other_backend/keep.py"
printf '%s\n' 'customized implementation' >"$OVERLAY/custom.py"

PYTHONPATH="$FAKE_SITE_PACKAGES" \
  python3 "$REPO_ROOT/scripts/install-vllm-overlay.py" \
    --source "$OVERLAY"

test -f "$VLLM_PACKAGE/v1/kv_offload/tiering/kvcc/custom.py"
test ! -e "$VLLM_PACKAGE/v1/kv_offload/tiering/kvcc/stale.py"
test -f "$VLLM_PACKAGE/v1/kv_offload/tiering/other_backend/keep.py"

printf '%s\n' 'vLLM KVCC overlay test passed'
