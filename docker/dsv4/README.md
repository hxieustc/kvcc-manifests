# DeepSeek-V4 AMXF4 MegaMoE image (vLLM v0.27.1 + Dynamo + KVCC)

## Background

**Problem**: We would like to have an image for optimized baseline, following the Pareto baseline's historical image supporting `deep_gemm_amxf4_mega_moe` MoE backend, used in [Agentic trace detail | InferenceX | InferenceX by SemiAnalysis](https://inferencex.semianalysis.com/inference/agentic/436494) (deprecated due to bug fixes in aiperf):

- `vllm/vllm-openai:dsv4-megamoe-mxfp4-x86_64-cu130-4ba0a72`
- `vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72`

**Solution**: create a container image, **`KVCC-DSV4`**, for 

- vLLM v0.27.1 application/runtime code while pinning the
  DeepGEMM data plane to the known DSV4-Pro AMXF4 snapshot

```text
vLLM v0.27.1 @ 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
  + AMXF4 backend/model/input-staging forward-port
  + optional FP8 combine support (disabled by default)
  + zyongye/DeepGEMM @ e1e5123e469eaeaf0aa4115ea97106b35935802a
```

- Dynamo

- KVCC

The build intentionally does not use DeepGEMM PR #382. That PR is a closed,
unmerged rewrite with different small-batch heuristics. It also does not
install a custom wheel over the stock v0.27.1 image; vLLM and DeepGEMM are
compiled together from the patched source context.

```textile
Keep constant:
  DeepGEMM fork + e1e5123 kernel chain
  AMXF4 staging contract
  backend name
  optional FP8 combine behavior

Modernize:
  vLLM baseline
  CUDA patch level
  PyTorch / FlashInfer / NCCL / DeepEP stack
  build provenance

Add separately:
  Dynamo
  KVCC
  NIXL and tiering adapter
```

## Requirements

- Docker BuildKit/buildx;
- network access to GitHub, Docker Hub, NVIDIA CUDA repositories, and Python
  package indexes;
- enough disk for a full vLLM CUDA build;
- an SM100 GPU for runtime validation; and
- a native ARM64 or AMD64 builder matching `VLLM_PLATFORM` when using
  `VLLM_OUTPUT=--load`.

## Build

### ARM64/GB200

```bash
cd docker/dsv4
VLLM_PLATFORM=linux/arm64 \
VLLM_IMAGE=your-registry/vllm-openai:v0.27.1-dsv4-amxf4-arm64-cu130 \
./scripts/build-image.sh
```

### x86_64/B200 or B300

Run the build on a native AMD64 builder. From the repository root, use
`--load` to retain the result in the builder's local Docker daemon:

```bash
cd docker/dsv4

VLLM_PLATFORM=linux/amd64 \
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
NVCC_THREADS=8 \
VLLM_OUTPUT=--load \
./scripts/build-image.sh
```

`MAX_JOBS` controls host compilation parallelism; reduce it if the builder is
memory-constrained. The build script accepts `linux/amd64` explicitly and
passes it to BuildKit, but `--load` still requires an AMD64 Docker daemon.

To push directly instead of loading the image locally:

```bash
VLLM_PLATFORM=linux/amd64 \
VLLM_IMAGE=your-registry/vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
NVCC_THREADS=8 \
VLLM_OUTPUT=--push \
./scripts/build-image.sh
```

The x86_64 build follows the same source path as ARM64: the script checks out
vLLM at the immutable `VLLM_REF`, applies
`patch/vllm-v0.27.1-amxf4.patch`, verifies the DeepGEMM `e1e5123` pin, and
builds the `vllm-openai` target from this directory's Dockerfile. It does not
install a custom wheel on top of a stock vLLM image.

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
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
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
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
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

## What's new in `KVCC-DSV4` image

| Component                | Historical `dsv4-…-x86_64-cu130-4ba0a72`                                           | New v0.27.1 DSV4 image                                                       | Assessment                                      |
| ------------------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------- |
| Architecture             | `linux/amd64`                                                                      | Validated artifact is `linux/arm64`; build supports `linux/amd64`            | Architecture differs in the validated artifacts |
| vLLM baseline            | Development tree ending at `4ba0a72`                                               | vLLM v0.27.1 at `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`                   | Deliberately modernized                         |
| Reported vLLM version    | Development/local build; image labels did not identify a release                   | Exactly `0.27.1`                                                             | Improved release identity                       |
| AMXF4 vLLM integration   | Direct ancestry `ea4780e → f2331da → 4ba0a72`                                      | Semantically forward-ported onto v0.27.1 with the checked-in patch           | Same functionality, adapted to newer APIs       |
| MoE backend name         | `deep_gemm_amxf4_mega_moe`                                                         | `deep_gemm_amxf4_mega_moe`                                                   | Same                                            |
| MXFP4 activation staging | Added by `ea4780e`                                                                 | Forward-ported implementation, including H=7168 handling                     | Same intended contract                          |
| FP8 combine              | Added by `f2331da`; opt-in                                                         | Forward-ported as `VLLM_DSV4_MEGA_FP8_COMBINE`, default off                  | Same behavior                                   |
| DeepGEMM repository      | `zyongye/DeepGEMM`                                                                 | `zyongye/DeepGEMM`                                                           | Same                                            |
| DeepGEMM revision        | `e1e5123e469eaeaf0aa4115ea97106b35935802a`                                         | Exactly the same `e1e5123`                                                   | Kernel plane held constant                      |
| DeepGEMM install paths   | CMake pin changed by `4ba0a72`                                                     | Both CMake and `tools/install_deepgemm.sh` pin `e1e5123`                     | New build is more internally consistent         |
| CUDA                     | 13.0.1                                                                             | 13.0.3                                                                       | Patch-level update                              |
| Ubuntu                   | 22.04                                                                              | 22.04                                                                        | Same                                            |
| Python                   | 3.12                                                                               | 3.12                                                                         | Same                                            |
| PyTorch                  | From historical branch’s dependency lock; exact installed version not established  | Observed `2.13.0+cu130`                                                      | Modernized and verified                         |
| FlashInfer               | Historical branch’s dependency set                                                 | Observed `flashinfer-python 0.6.16.post3`                                    | Modernized                                      |
| NCCL                     | Historical branch’s CUDA/NCCL stack                                                | Dockerfile pins NCCL 2.30.7                                                  | Modernized                                      |
| DeepEP/EP kernels        | Source-built wheels included                                                       | Source-built through the v0.27.1 Dockerfile and installed in the runtime     | Same role, newer source/build system            |
| KV connector bundle      | Built with `INSTALL_KV_CONNECTORS=true`; included LMCache, NIXL, Mooncake and CuPy | Standalone DSV4 image uses the v0.27.1 default `INSTALL_KV_CONNECTORS=false` | Intentionally omitted from base image           |
| Dynamo                   | No evidence it was included                                                        | Not in standalone DSV4 image; added by the planned outer image               | Separate composition layer                      |
| KVCC                     | No evidence it was included                                                        | Not in standalone DSV4 image; added by the planned outer image               | Separate composition layer                      |
| Entrypoint               | `vllm serve`                                                                       | `vllm serve`                                                                 | Same for standalone DSV4 image                  |
| Build provenance         | Labels reported local/unknown provenance                                           | Labels record vLLM SHA, AMXF4 port, DeepGEMM snapshot and build pipeline     | Improved reproducibility                        |

## Recommended near-term image composition

The best near-term solution is a dedicated vLLM v0.27.1 source build that
forward-ports the vLLM integration from `ea4780e` and `f2331da`, while
continuing to use the proven DeepGEMM snapshot `e1e5123`. The resulting vLLM
runtime should then be used as the base of the existing Dynamo/KVCC image,
rather than installing a second vLLM wheel into either image.

### What the historical `4ba0a72` image contains

The published
[`dsv4-megamoe-mxfp4-x86_64-cu130-4ba0a72`](https://hub.docker.com/layers/vllm/vllm-openai/dsv4-megamoe-mxfp4-x86_64-cu130-4ba0a72/images/sha256-a49ab39f08d103494ab61a909db88e67956acbae4c7d1a5bf6bda5ea009ab1d2)
image is a CUDA 13.0.1, Ubuntu 22.04, Python 3.12 vLLM development image. Its
published digest is
`sha256:a49ab39f08d103494ab61a909db88e67956acbae4c7d1a5bf6bda5ea009ab1d2`.
It was built from the following source lineage, not from the v0.27.1 release:

| Layer                   | Revision                                                                                          | Contribution                                                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| vLLM integration        | [`ea4780e`](https://github.com/vllm-project/vllm/commit/ea4780e18cda189eca015ae8be7c56f6404d28bb) | Adds `deep_gemm_amxf4_mega_moe` to the backend configuration, DeepSeek-V4 model routing, MXFP4 activation packing/staging, EPLB handling, and tests. |
| vLLM follow-up          | [`f2331da`](https://github.com/vllm-project/vllm/commit/f2331da0d23eb921da15493ae3dd73b1c4cdb545) | Adds the optional FP8 MegaMoE combine path and its environment control.                                                                              |
| vLLM build pin          | [`4ba0a72`](https://github.com/vllm-project/vllm/commit/4ba0a72a4089cfa23d9664093427ae6b1b71e6fe) | Changes the DeepGEMM CMake pin to `e1e5123`; it is a one-line change whose parent already contains the two integration commits above.                |
| DeepGEMM implementation | [`e1e5123`](https://github.com/zyongye/DeepGEMM/commit/e1e5123e469eaeaf0aa4115ea97106b35935802a)  | Contains the SM100 MegaMoE MXFP4/FP8 implementation and the relevant `d4079d0`, `4ec3457`, `65b3085`, and `e1e5123` fix chain.                       |

This explains why the one-line `4ba0a72` change appears to add an entire MoE
backend: it does not do so in isolation. The parent tree already understands
the backend name, prepares packed MXFP4 activations, and calls the extended
DeepGEMM API. The changed CMake pin supplies the matching native kernels and
Python API. Applying only the `e1e5123` pin to an unmodified vLLM v0.27.1 tree
would provide kernel code that the vLLM control path neither selects nor calls.

The historical image also builds vLLM and its CUDA extensions from source,
builds DeepEP wheels, and installs the CUDA 13 runtime dependency set and KV
connector packages, including LMCache, NIXL, Mooncake transfer engine, and
CuPy. Those supporting packages are useful runtime plumbing, but the AMXF4
backend itself comes from the matched vLLM integration and DeepGEMM source
pair above. Image labels report a local/development build and do not identify
an official vLLM release, so the tag and verified commit ancestry are the
relevant provenance.

### How the replacement image should be built

Use two explicit image boundaries:

1. **DSV4 vLLM runtime.** Check out the immutable vLLM v0.27.1 source revision
   `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, apply
   `patch/vllm-v0.27.1-amxf4.patch`, and compile vLLM and DeepGEMM together.
   The patch forward-ports the `ea4780e`/`f2331da` integration and pins the
   bundled DeepGEMM build to `e1e5123`. This is what `scripts/build-image.sh`
   and this directory's `Dockerfile` implement.
2. **Dynamo/KVCC runtime.** Pass the first image, preferably by digest, as
   `VLLM_RUNTIME_IMAGE` to `docker/scripts/build-image.sh`. The top-level
   `docker/Dockerfile` builds Dynamo wheels, installs Dynamo and KVCC, and
   overlays the synchronized KVCC vLLM tiering adapter onto that runtime.

For example, from the repository root:

```bash
VLLM_PLATFORM=linux/amd64 \
VLLM_IMAGE=your-registry/vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
VLLM_OUTPUT=--push \
./docker/dsv4/scripts/build-image.sh

# Resolve the pushed image to an immutable digest before the combined build.
export VLLM_RUNTIME_IMAGE=your-registry/vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130@sha256:DIGEST
export KVCC_IMAGE=your-registry/kvcc:v0.27.1-dsv4-amxf4-x86_64-cu130
export KVCC_PLATFORM=linux/amd64
export GITHUB_USER="xxx"
export GITHUB_EMAIL="xxx@nvidia.com"
export GITHUB_TOKEN="..."
./docker/scripts/build-image.sh
```

Use `linux/arm64` and the corresponding ARM64 tag for GB200. Keeping vLLM and
DeepGEMM in the first image makes their compiled ABI and source revisions
coherent; keeping Dynamo and KVCC in the second layer lets their code continue
to come from the KVCC manifest workspace. Do not install a stock v0.27.1 vLLM
wheel after the first stage, because doing so would replace the forward-ported
Python control path and potentially its compiled extensions.

There are two compatibility requirements for the combined image:

- The synchronized KVCC tiering adapter copied by the top-level Dockerfile
  must match the vLLM v0.27.1 APIs. If it does not, port that adapter to
  v0.27.1; do not solve the mismatch by replacing the DSV4 vLLM package.
- The top-level `docker/Dockerfile` must retain `VLLM_USE_DEEP_GEMM=1` so it
  does not disable the integration inherited from the DSV4 vLLM base. This
  environment setting enables an existing compiled integration; it cannot add
  the backend to a stock vLLM image. Select
  `--moe-backend deep_gemm_amxf4_mega_moe` and enable expert parallelism at
  runtime.

Validation should preserve the same boundary. First run this directory's
artifact, API, numerical staging, and two-rank symmetric-buffer tests against
the DSV4 vLLM image. Then run the Dynamo/KVCC import and tiering integration
tests against the combined image. Finally, initialize DeepSeek-V4-Pro with the
target tensor/expert-parallel topology on SM100 hardware. Passing import or
backend-enum checks alone proves that the feature is exposed, not that the
native kernel executes correctly or that the KVCC adapter is compatible.

## Files

| File                             | Purpose                                                           |
| -------------------------------- | ----------------------------------------------------------------- |
| `Dockerfile`                     | vLLM v0.27.1 Dockerfile plus an explicit package-version override |
| `patch/vllm-v0.27.1-amxf4.patch` | v0.27.1 source forward-port and DeepGEMM pin                      |
| `scripts/build-image.sh`         | Fetch, verify, patch, and build the image                         |
| `tests/test-image.sh`            | Static API, SM100, staging, and optional model checks             |
| `tests/test-mxfp4-staging.py`    | Packed E2M1/UE8M0 numerical test, including H=7168                |
| `tests/test-megamoe-buffer.py`   | Two-rank MXFP4 MegaMoE symmetric-buffer smoke test                |
| `tests/validate-artifacts.sh`    | Fast shell and artifact consistency checks                        |
