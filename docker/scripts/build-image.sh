#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

require_env GITHUB_USER
require_env GITHUB_EMAIL
require_env GITHUB_TOKEN
require_command docker

DYNAMO_BASE_IMAGE="${DYNAMO_BASE_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest}"
KVCC_IMAGE="${KVCC_IMAGE:-kvcc-custom-runtime:dev}"
KVCC_PLATFORM="${KVCC_PLATFORM:-linux/amd64}"
REPO_SYNC_JOBS="${REPO_SYNC_JOBS:-8}"

log "Building $KVCC_IMAGE for $KVCC_PLATFORM"
docker buildx build \
  --platform "$KVCC_PLATFORM" \
  --load \
  --secret id=github_token,env=GITHUB_TOKEN \
  --build-arg GITHUB_USER \
  --build-arg GITHUB_EMAIL \
  --build-arg "DYNAMO_BASE_IMAGE=$DYNAMO_BASE_IMAGE" \
  --build-arg "REPO_SYNC_JOBS=$REPO_SYNC_JOBS" \
  --tag "$KVCC_IMAGE" \
  --file "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT"

log "Built $KVCC_IMAGE"
