#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf "$TEST_ROOT"' EXIT

WORKSPACE="$TEST_ROOT/workspace"
DYNAMO_SOURCE="$WORKSPACE/dynamo/components/src"
KVCC_SOURCE="$WORKSPACE/kvcc/src"
BASE_SITE="$TEST_ROOT/base-site"

mkdir -p \
  "$DYNAMO_SOURCE/dynamo/vllm" \
  "$KVCC_SOURCE/kvcc" \
  "$BASE_SITE/vllm/v1/kv_offload/tiering/kvcc"

touch \
  "$DYNAMO_SOURCE/dynamo/__init__.py" \
  "$DYNAMO_SOURCE/dynamo/vllm/__init__.py" \
  "$KVCC_SOURCE/kvcc/__init__.py" \
  "$BASE_SITE/vllm/__init__.py" \
  "$BASE_SITE/vllm/_C_stable_libtorch.py" \
  "$BASE_SITE/vllm/v1/kv_offload/tiering/kvcc/__init__.py"

PYTHONPATH="$DYNAMO_SOURCE:$KVCC_SOURCE:$BASE_SITE" \
  python3 "$REPO_ROOT/scripts/verify-selective-runtime.py" \
    --workspace "$WORKSPACE"

printf '%s\n' 'selective runtime provenance test passed'
