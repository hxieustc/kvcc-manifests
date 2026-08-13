# Customized Dynamo + vLLM + KVCC Image

This directory builds one runtime image from the source revisions selected by
the `cuda-env-fixes` Repo manifest:

| Workspace path | Repository | Revision |
|---|---|---|
| `dynamo/` | `ai-dynamo/dynamo` | `oandreeva/router_hints` |
| `vllm/` | `mkhazraee/vllm-priv` | `kvcc_repo` |
| `kvcc/` | `NVIDIA-dev/kvcc` | `main` |

The Docker build runs `repo init` and `repo sync`; source checkouts do not need
to be prepared in the host build context.

## What the image contains

- `ai-dynamo-runtime`, built with the `select-service,kv-indexer` features;
- `ai-dynamo`, built as a pure-Python wheel;
- the vLLM Python package and native CUDA extensions inherited from a pinned
  `vllm/vllm-openai` nightly image;
- standalone `nvidia-kvcc`, installed from synchronized source;
- the synchronized `vllm/v1/kv_offload/tiering/kvcc` adapter copied over the
  corresponding directory in the base vLLM package; and
- customized tiering and KVCC E2E test sources under
  `/opt/kvcc/vllm-tests`, available for optional later execution.

The Docker build does not compile or install synchronized vLLM, apply legacy
patch files, download a model, install test requirements, or run
`bootstrap.sh`, `build-all.sh`, `test-all.sh`, unit tests, or E2E tests.

## Credentials

Export the GitHub identity and token in the shell that launches the build:

```bash
export GITHUB_USER=xxx
export GITHUB_EMAIL="xxx@nvidia.com"
test -n "${GITHUB_TOKEN:-}"
```

`GITHUB_TOKEN` is passed as a BuildKit secret. It is not a Docker build
argument, image environment variable, Git URL, or committed file.

## Build

Requirements:

- Docker with BuildKit/buildx;
- access to Docker Hub, GitHub, PyPI, and the FlashInfer package sources; and
- access to the private repositories selected by the manifest.

From this `docker/` directory, run:

```bash
scripts/build-image.sh
```

Defaults:

```text
builder:  rust:1.93-bookworm
runtime:  vllm/vllm-openai:nightly-6f91edf96...@sha256:674c5aa...
platform: linux/amd64
output:   kvcc-custom-runtime:dev
```

The builder and runtime bases are configurable while retaining pinned defaults:

```bash
export DYNAMO_BUILDER_IMAGE="rust:1.93-bookworm"
export VLLM_RUNTIME_IMAGE="vllm/vllm-openai:nightly-6f91edf96d3f3272945809c04702380053bff4de@sha256:674c5aa666d38c07a0dc779f8c77a05c2b859617410d16dcdd8a8776166c92b9"
export KVCC_PLATFORM="linux/amd64"
export KVCC_IMAGE="my-registry.example/kvcc-custom-runtime:dev"
export REPO_SYNC_JOBS=8
scripts/build-image.sh
```

Changing `VLLM_RUNTIME_IMAGE` requires verifying that its vLLM revision is
compatible with the overlaid `kvcc_repo` adapter.

## Build KVCC with DeepSeek-V4 AMXF4 MegaMoE

The `deep_gemm_amxf4_mega_moe` backend cannot be added to a stock vLLM image
by setting an environment variable or installing Dynamo and KVCC. It requires
rebuilding vLLM and its vendored DeepGEMM extension from a coherent patched
source tree. Build the final DSV4 image in two incremental steps:

```text
docker/dsv4/Dockerfile
  vLLM v0.27.1 source + AMXF4 forward-port + DeepGEMM e1e5123
                           |
                           v
  vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130
                           |
                           | VLLM_RUNTIME_IMAGE
                           v
docker/Dockerfile
  inherited AMXF4-enabled vLLM + Dynamo + KVCC + KVCC tiering overlay
                           |
                           v
  kvcc:vllm-0.27.1-dsv4-amxf4-x86_64-cu130
```

### Step 1: build the AMXF4-enabled vLLM runtime

From the repository root, build on a native AMD64 machine for B200/B300:

```bash
cd docker/dsv4

VLLM_PLATFORM=linux/amd64 \
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
NVCC_THREADS=8 \
VLLM_OUTPUT=--load \
./scripts/build-image.sh
```

This script fetches the immutable vLLM v0.27.1 source commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, applies
`patch/vllm-v0.27.1-amxf4.patch`, pins the vendored DeepGEMM source to
`e1e5123e469eaeaf0aa4115ea97106b35935802a`, and uses
`docker/dsv4/Dockerfile` to compile the vLLM wheel and CUDA extensions. The
patch supplies the backend registration, DeepSeek-V4 routing, MXFP4 input
staging, and the MegaMoE `act_format`/`combine_dtype` integration.

Validate this boundary before adding Dynamo or KVCC:

```bash
VLLM_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130 \
./tests/test-image.sh
```

`tests/test-image.sh` requires an SM100-or-newer host with two GPUs for its
two-rank symmetric-buffer test. See `dsv4/README.md` for direct registry push
and optional full-model validation commands.

### Step 2: add Dynamo, KVCC, and the KVCC vLLM overlay

Return to the top-level Docker directory and use the image from Step 1 as the
runtime base:

```bash
cd ..

export GITHUB_USER="xxx"
export GITHUB_EMAIL="xxx@nvidia.com"
export GITHUB_TOKEN="..."
export VLLM_RUNTIME_IMAGE=vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130
export KVCC_PLATFORM=linux/amd64
export KVCC_IMAGE=nvcr.io/nvidian/dynamo-dev/kvcc:vllm-0.27.1-dsv4-amxf4-x86_64-cu130

./scripts/build-image.sh
```

When the two steps run on different builders, push the Step 1 image first and
set `VLLM_RUNTIME_IMAGE` to its immutable registry digest:

```bash
export VLLM_RUNTIME_IMAGE=your-registry/vllm-openai:v0.27.1-dsv4-amxf4-x86_64-cu130@sha256:DIGEST
```

The top-level `docker/Dockerfile` does not compile or implement the AMXF4
backend. Its `FROM ${VLLM_RUNTIME_IMAGE}` inherits the customized vLLM package
and native extensions, then installs Dynamo and KVCC and overlays the
manifest-selected KVCC tiering adapter. It also keeps
`VLLM_USE_DEEP_GEMM=1`; that setting enables the already-built integration but
cannot add it to a stock vLLM image.

Do not install a stock vLLM v0.27.1 wheel after Step 1. Doing so can replace
the patched Python control path or native extensions and remove or break the
AMXF4 MegaMoE integration. The KVCC tiering adapter synchronized in Step 2
must also remain API-compatible with the vLLM v0.27.1 package inherited from
Step 1.

After the combined build, rerun the DSV4 tests against the final image to
prove the backend survived the Dynamo/KVCC assembly:

```bash
cd dsv4
VLLM_IMAGE=nvcr.io/nvidian/dynamo-dev/kvcc:vllm-0.27.1-dsv4-amxf4-x86_64-cu130 \
./tests/test-image.sh
```

## Validation

Check the DSV4 build artifacts without building either image:

```bash
bash dsv4/tests/validate-artifacts.sh
```

The Dockerfile also runs lightweight build-time canaries for FlashInfer,
OpenAI's required response type, Dynamo router hints, vLLM native code, KVCC,
the vLLM KVCC adapter, and NIXL.

After building, `tests/test-docker.sh` runs the KVCC unit and E2E tests against
`kvcc-custom-runtime:dev`. Adjust that hard-coded image tag and model mount in
the script when using a different output image or model location.
