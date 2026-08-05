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

## Validation

Run the Docker artifact and shell checks without building the image:

```bash
bash tests/validate-artifacts.sh
```

The Dockerfile also runs lightweight build-time canaries for FlashInfer,
OpenAI's required response type, Dynamo router hints, vLLM native code, KVCC,
the vLLM KVCC adapter, and NIXL.

Only one operational shell script is required: `scripts/build-image.sh`.
`tests/validate-artifacts.sh` is a development-time validator, not part of the
image.
