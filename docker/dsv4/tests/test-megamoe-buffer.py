# SPDX-License-Identifier: Apache-2.0
"""Two-rank smoke test for DeepGEMM's MXFP4 MegaMoE buffer."""

import os

import torch
import torch.distributed as dist

from vllm.utils.deep_gemm import _import_deep_gemm


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    deep_gemm = _import_deep_gemm()
    assert deep_gemm is not None
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        num_experts=256,
        num_max_tokens_per_rank=64,
        num_topk=8,
        hidden=7168,
        intermediate_hidden=2048,
        use_fp8_dispatch=True,
        activation="swiglu",
        combine_dtype=torch.bfloat16,
        act_format="mxfp4",
    )
    assert buffer is not None
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"MegaMoE buffer: {type(buffer).__module__}.{type(buffer).__name__}")
        print("MXFP4_MEGAMOE_BUFFER_PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
