#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_IMAGE="${VLLM_IMAGE:-vllm-openai:v0.27.1-dsv4-amxf4-cu130}"

command -v docker >/dev/null 2>&1 || {
  printf 'ERROR: docker is required\n' >&2
  exit 1
}

printf '==> Static backend and DeepGEMM API checks\n'
docker run --rm --gpus all --ipc=host \
  --entrypoint python3 \
  "$VLLM_IMAGE" -c '
import inspect
from typing import get_args

import torch
import vllm
from vllm.config.kernel import MoEBackend
from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_supported
from vllm.utils.import_utils import has_deep_gemm

backends = get_args(MoEBackend)
assert "deep_gemm_amxf4_mega_moe" in backends, backends

assert has_deep_gemm()
deep_gemm = _import_deep_gemm()
assert deep_gemm is not None
signature = inspect.signature(deep_gemm.get_symm_buffer_for_mega_moe)
assert "act_format" in signature.parameters, signature
assert "combine_dtype" in signature.parameters, signature

assert torch.cuda.is_available()
capability = torch.cuda.get_device_capability()
assert capability >= (10, 0), capability
assert is_deep_gemm_supported()
assert vllm.__version__ == "0.27.1", vllm.__version__
print("vLLM", vllm.__version__)
print("DeepGEMM", deep_gemm.__file__)
print("MegaMoE signature", signature)
print("GPU", torch.cuda.get_device_name(), capability)
print("Static AMXF4 checks: PASS")
'

printf '==> MXFP4 E2M1 staging correctness at H=2048,4096,7168\n'
docker run --rm --gpus all --ipc=host \
  --entrypoint python3 \
  --volume "$SCRIPT_DIR/test-mxfp4-staging.py:/opt/dsv4/test-mxfp4-staging.py:ro" \
  "$VLLM_IMAGE" /opt/dsv4/test-mxfp4-staging.py

printf '==> Two-rank DeepGEMM MXFP4 MegaMoE symmetric-buffer check\n'
docker run --rm --gpus all --ipc=host \
  --entrypoint torchrun \
  --volume "$SCRIPT_DIR/test-megamoe-buffer.py:/opt/dsv4/test-megamoe-buffer.py:ro" \
  "$VLLM_IMAGE" --standalone --nproc-per-node=2 \
  /opt/dsv4/test-megamoe-buffer.py

if [[ -n "${MODEL_PATH:-}" ]]; then
  printf '==> Optional two-GPU model initialization check\n'
  docker run --rm --gpus all --ipc=host --shm-size=32g \
    --volume "$MODEL_PATH:/model:ro" \
    "$VLLM_IMAGE" \
    --model /model \
    --moe-backend deep_gemm_amxf4_mega_moe \
    --enable-expert-parallel \
    --tensor-parallel-size 2 \
    --max-model-len "${MAX_MODEL_LEN:-1024}" \
    --enforce-eager
else
  printf '==> MODEL_PATH is unset; skipping full model initialization\n'
fi

printf 'Image validation: PASS\n'
