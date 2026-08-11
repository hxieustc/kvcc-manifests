#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

for name in GITHUB_USER GITHUB_EMAIL GITHUB_TOKEN; do
  if [[ -z "${!name:-}" ]]; then
    printf 'ERROR: required environment variable is not set: %s\n' "$name" >&2
    exit 1
  fi
done

command -v docker >/dev/null 2>&1 || {
  printf 'ERROR: docker is required\n' >&2
  exit 1
}

DYNAMO_BUILDER_IMAGE="${DYNAMO_BUILDER_IMAGE:-rust:1.93-bookworm}"
VLLM_RUNTIME_IMAGE="${VLLM_RUNTIME_IMAGE:-vllm/vllm-openai:dsv4-megamoe-mxfp4-x86_64-cu130-4ba0a72}"
KVCC_IMAGE="${KVCC_IMAGE:-kvcc-custom-runtime:dev}"
KVCC_PLATFORM="${KVCC_PLATFORM:-linux/amd64}"
REPO_SYNC_JOBS="${REPO_SYNC_JOBS:-8}"

printf '==> Building %s for %s\n' "$KVCC_IMAGE" "$KVCC_PLATFORM"
docker buildx build \
  --platform "$KVCC_PLATFORM" \
  --load \
  --secret id=github_token,env=GITHUB_TOKEN \
  --build-arg GITHUB_USER \
  --build-arg GITHUB_EMAIL \
  --build-arg "DYNAMO_BUILDER_IMAGE=$DYNAMO_BUILDER_IMAGE" \
  --build-arg "VLLM_RUNTIME_IMAGE=$VLLM_RUNTIME_IMAGE" \
  --build-arg "REPO_SYNC_JOBS=$REPO_SYNC_JOBS" \
  --tag "$KVCC_IMAGE" \
  --file "$DOCKER_ROOT/Dockerfile" \
  "$DOCKER_ROOT"

printf '==> Built %s\n' "$KVCC_IMAGE"
