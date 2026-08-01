#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/test-repository-contract.sh"
bash "$SCRIPT_DIR/test-bootstrap.sh"
bash "$SCRIPT_DIR/test-build-all.sh"
bash "$SCRIPT_DIR/test-verify-env.sh"
