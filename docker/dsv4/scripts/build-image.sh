#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSV4_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

for command_name in docker git mktemp; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'ERROR: required command is unavailable: %s\n' "$command_name" >&2
    exit 1
  }
done

VLLM_REPOSITORY="${VLLM_REPOSITORY:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-6e448d0ea9bf3d88d898b65449ca6dc2aec170ac}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.27.1}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm-openai:v0.27.1-dsv4-amxf4-cu130}"
VLLM_PLATFORM="${VLLM_PLATFORM:-linux/arm64}"
CUDA_VERSION="${CUDA_VERSION:-13.0.3}"
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-8}"
RUN_WHEEL_CHECK="${RUN_WHEEL_CHECK:-false}"
VLLM_OUTPUT="${VLLM_OUTPUT:---load}"

case "$VLLM_PLATFORM" in
  linux/amd64|linux/arm64) ;;
  *)
    printf 'ERROR: VLLM_PLATFORM must be linux/amd64 or linux/arm64, got %s\n' \
      "$VLLM_PLATFORM" >&2
    exit 1
    ;;
esac

case "$VLLM_OUTPUT" in
  --load|--push) ;;
  *)
    printf 'ERROR: VLLM_OUTPUT must be --load or --push, got %s\n' \
      "$VLLM_OUTPUT" >&2
    exit 1
    ;;
esac

case "$RUN_WHEEL_CHECK" in
  true|false) ;;
  *)
    printf 'ERROR: RUN_WHEEL_CHECK must be true or false, got %s\n' \
      "$RUN_WHEEL_CHECK" >&2
    exit 1
    ;;
esac

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vllm-dsv4-amxf4.XXXXXX")"
SOURCE_DIR="$BUILD_ROOT/vllm"
cleanup() {
  rm -rf -- "$BUILD_ROOT"
}
trap cleanup EXIT

printf '==> Fetching vLLM %s\n' "$VLLM_REF"
git init --quiet "$SOURCE_DIR"
git -C "$SOURCE_DIR" remote add origin "$VLLM_REPOSITORY"
git -C "$SOURCE_DIR" fetch --quiet --depth=1 origin "$VLLM_REF"
git -C "$SOURCE_DIR" checkout --quiet --detach FETCH_HEAD

actual_ref="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [[ "$actual_ref" != "$VLLM_REF" ]]; then
  printf 'ERROR: fetched vLLM commit %s, expected %s\n' \
    "$actual_ref" "$VLLM_REF" >&2
  exit 1
fi

printf '==> Applying vLLM v0.27.1 AMXF4 forward-port\n'
git -C "$SOURCE_DIR" apply --check \
  "$DSV4_DIR/patch/vllm-v0.27.1-amxf4.patch"
git -C "$SOURCE_DIR" apply \
  "$DSV4_DIR/patch/vllm-v0.27.1-amxf4.patch"
git -C "$SOURCE_DIR" diff --check

grep -Fq 'deep_gemm_amxf4_mega_moe' "$SOURCE_DIR/vllm/config/kernel.py"
grep -Fq 'e1e5123e469eaeaf0aa4115ea97106b35935802a' \
  "$SOURCE_DIR/cmake/external_projects/deepgemm.cmake"
grep -Fq 'e1e5123e469eaeaf0aa4115ea97106b35935802a' \
  "$SOURCE_DIR/tools/install_deepgemm.sh"

printf '==> Building %s for %s with CUDA %s\n' \
  "$VLLM_IMAGE" "$VLLM_PLATFORM" "$CUDA_VERSION"
docker buildx build \
  --platform "$VLLM_PLATFORM" \
  "$VLLM_OUTPUT" \
  --progress=plain \
  --target vllm-openai \
  --build-arg "CUDA_VERSION=$CUDA_VERSION" \
  --build-arg "VLLM_VERSION_OVERRIDE=$VLLM_VERSION_OVERRIDE" \
  --build-arg "max_jobs=$MAX_JOBS" \
  --build-arg "nvcc_threads=$NVCC_THREADS" \
  --build-arg "RUN_WHEEL_CHECK=$RUN_WHEEL_CHECK" \
  --build-arg "VLLM_BUILD_COMMIT=$VLLM_REF+dsv4-amxf4-e1e5123" \
  --build-arg "VLLM_BUILD_PIPELINE=kvcc-manifests/docker/dsv4" \
  --build-arg "VLLM_IMAGE_TAG=$VLLM_IMAGE" \
  --tag "$VLLM_IMAGE" \
  --file "$DSV4_DIR/Dockerfile" \
  "$SOURCE_DIR"

printf '==> Built %s\n' "$VLLM_IMAGE"
