#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSV4_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DSV4_DIR/../.." && pwd)"

required=(
  "$DSV4_DIR/Dockerfile"
  "$DSV4_DIR/README.md"
  "$DSV4_DIR/scripts/build-image.sh"
  "$SCRIPT_DIR/test-image.sh"
  "$SCRIPT_DIR/test-megamoe-buffer.py"
  "$SCRIPT_DIR/test-mxfp4-staging.py"
  "$DSV4_DIR/patch/vllm-v0.27.1-amxf4.patch"
)

for name in "${required[@]}"; do
  [[ -f "$name" ]] || {
    printf 'ERROR: missing required artifact: %s\n' "$name" >&2
    exit 1
  }
done

for name in "$DSV4_DIR/scripts/build-image.sh" \
  "$SCRIPT_DIR/test-image.sh" "$SCRIPT_DIR/validate-artifacts.sh"; do
  [[ -x "$name" ]] || {
    printf 'ERROR: script is not executable: %s\n' "$name" >&2
    exit 1
  }
  bash -n "$name"
done

grep -Fq 'ARG CUDA_VERSION=13.0.3' "$DSV4_DIR/Dockerfile"
grep -Fq 'FROM vllm-openai-base AS vllm-openai' "$DSV4_DIR/Dockerfile"
test "$(grep -Fc 'ARG VLLM_VERSION_OVERRIDE=0.27.1' \
  "$DSV4_DIR/Dockerfile")" -eq 2
grep -Fq 'RUN_WHEEL_CHECK="${RUN_WHEEL_CHECK:-false}"' \
  "$DSV4_DIR/scripts/build-image.sh"

patch_file="$DSV4_DIR/patch/vllm-v0.27.1-amxf4.patch"
grep -Fq 'deep_gemm_amxf4_mega_moe' "$patch_file"
grep -Fq 'e1e5123e469eaeaf0aa4115ea97106b35935802a' "$patch_file"
grep -Fq 'act_format=self.act_format' "$patch_file"
grep -Fq 'combine_dtype=self.combine_dtype' "$patch_file"
grep -Fq 'tools/install_deepgemm.sh' "$patch_file"
grep -Fq 'VLLM_USE_DEEP_GEMM=1' "$REPO_ROOT/docker/Dockerfile"
grep -Fq 'uv pip install --system /src/nvidia-kvcc' \
  "$REPO_ROOT/docker/Dockerfile"

printf 'Artifact validation: PASS\n'
