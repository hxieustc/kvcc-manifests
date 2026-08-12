# vLLM v0.27.1 DeepSeek-V4 AMXF4 MegaMoE image

This directory builds vLLM v0.27.1 with the experimental
`deep_gemm_amxf4_mega_moe` backend used by the historical images:

- `vllm/vllm-openai:dsv4-megamoe-mxfp4-x86_64-cu130-4ba0a72`
- `vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72`

The result keeps the v0.27.1 application/runtime code while pinning the
DeepGEMM data plane to the known DSV4-Pro AMXF4 snapshot:

```text
vLLM v0.27.1 @ 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
  + AMXF4 backend/model/input-staging forward-port
  + optional FP8 combine support (disabled by default)
  + zyongye/DeepGEMM @ e1e5123e469eaeaf0aa4115ea97106b35935802a
```

The build intentionally does not use DeepGEMM PR #382. That PR is a closed,
unmerged rewrite with different small-batch heuristics. It also does not
install a custom wheel over the stock v0.27.1 image; vLLM and DeepGEMM are
compiled together from the patched source context.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | vLLM v0.27.1 Dockerfile plus an explicit package-version override |
| `patch/vllm-v0.27.1-amxf4.patch` | v0.27.1 source forward-port and DeepGEMM pin |
| `scripts/build-image.sh` | Fetch, verify, patch, and build the image |
| `tests/test-image.sh` | Static API, SM100, staging, and optional model checks |
| `tests/test-mxfp4-staging.py` | Packed E2M1/UE8M0 numerical test, including H=7168 |
| `tests/test-megamoe-buffer.py` | Two-rank MXFP4 MegaMoE symmetric-buffer smoke test |
| `tests/validate-artifacts.sh` | Fast shell and artifact consistency checks |

## Requirements

- Docker BuildKit/buildx;
- network access to GitHub, Docker Hub, NVIDIA CUDA repositories, and Python
  package indexes;
- enough disk for a full vLLM CUDA build;
- an SM100 GPU for runtime validation; and
- a native ARM64 or AMD64 builder matching `VLLM_PLATFORM` when using
  `VLLM_OUTPUT=--load`.

## Build

ARM64/GB200:

```bash
cd docker/dsv4
VLLM_PLATFORM=linux/arm64 \
VLLM_IMAGE=your-registry/vllm-openai:v0.27.1-dsv4-amxf4-arm64-cu130 \
./scripts/build-image.sh
```

AMD64/B200 or B300:

```bash
VLLM_PLATFORM=linux/amd64 \
VLLM_IMAGE=your-registry/vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
./scripts/build-image.sh
```

Defaults are:

```text
VLLM_REF=6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
VLLM_VERSION_OVERRIDE=0.27.1
CUDA_VERSION=13.0.3
VLLM_PLATFORM=linux/arm64
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-cu130
MAX_JOBS=2
NVCC_THREADS=8
RUN_WHEEL_CHECK=false
VLLM_OUTPUT=--load
```

`RUN_WHEEL_CHECK=false` disables vLLM's CI wheel-size policy gate. The full
ARM64 CUDA 13 wheel is larger than that gate's 500 MB limit because it bundles
the complete FA2/FA3 compatibility matrix; this does not remove kernels from
the image. Set it to `true` if enforcing the upstream size limit is desired.

`VLLM_VERSION_OVERRIDE=0.27.1` is passed into both wheel-build stages so
`vllm.__version__` remains `0.27.1`. The source checkout is intentionally
fetched by immutable SHA and then patched, so relying on setuptools-scm alone
would otherwise produce a development version string.

Use `VLLM_OUTPUT=--push` with a registry tag to publish directly. The build
script accepts only `--load` and `--push` to avoid accidental arbitrary buildx
argument injection.

## Validate

Check the repository artifacts without building:

```bash
./tests/validate-artifacts.sh
```

After building on an SM100 host:

```bash
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-cu130 \
./tests/test-image.sh
```

The default test proves:

1. the backend enum includes `deep_gemm_amxf4_mega_moe`;
2. the bundled DeepGEMM API accepts `act_format` and `combine_dtype`;
3. CUDA is available on an SM100-or-newer GPU; and
4. the vLLM staging kernel produces correct packed E2M1 activations and
   UE8M0 scales for hidden sizes 2048, 4096, and 7168; and
5. two NCCL ranks can construct DeepGEMM's symmetric MegaMoE buffer with
   `act_format="mxfp4"`.

For an optional two-GPU model initialization test:

```bash
MODEL_PATH=/path/to/DeepSeek-V4-Pro \
MAX_MODEL_LEN=1024 \
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-cu130 \
./tests/test-image.sh
```

This mounts the model read-only and starts vLLM with:

```text
--moe-backend deep_gemm_amxf4_mega_moe
--enable-expert-parallel
--tensor-parallel-size 2
```

The two-GPU check validates integration but cannot qualify rack-scale
performance. Final deployment acceptance should use the intended topology,
including the 8-GPU DSV4-Pro shape for which the DeepGEMM branch was tuned.

## Recorded validation

Validated on 2026-08-12 in namespace `harryx-kvcc`, using retained pod
`dsv4-amxf4-builder` with two NVIDIA GB200 GPUs:

```text
architecture: linux/arm64
image: vllm-openai:v0.27.1-dsv4-amxf4-arm64-cu130
image ID: sha256:81730b8c824a31b13d1cbdf3333fa2aaacf7e0372f662ad44cc25aa9df0af622
vLLM: 0.27.1
GPU capability: 10.0
backend enum: deep_gemm_amxf4_mega_moe
DeepGEMM act_format/combine_dtype API: PASS
MXFP4 staging H=2048,4096,7168: PASS
two-rank MXFP4 MegaMoE symmetric buffer: PASS
```

The full DeepSeek-V4-Pro model initialization was not run because model
weights were not present in the retained pod. Supply `MODEL_PATH` to run that
optional acceptance check.

## Updating

Do not change `VLLM_REF`, the stored Dockerfile, or the DeepGEMM SHA
independently. For a newer vLLM release:

1. regenerate the source patch against that exact release;
2. update the stored Dockerfile from the same release;
3. run `tests/validate-artifacts.sh`;
4. build both architectures; and
5. repeat the SM100 numerical and full model checks.

This image is DSV4-specific. The older DeepGEMM pin may omit newer features
used by unrelated models, so it should not replace the stock general-purpose
vLLM image.
