# syntax=docker/dockerfile:1.7

ARG DYNAMO_BASE_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest

FROM ${DYNAMO_BASE_IMAGE} AS builder

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GITHUB_USER
ARG GITHUB_EMAIL
ARG REPO_SYNC_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    CUDARC_CUDA_VERSION=13000 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    KVCC_WORKSPACE=/opt/kvcc-workspace \
    PATH=/root/.local/bin:/root/.cargo/bin:${PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ccache \
      clang \
      cmake \
      curl \
      git \
      git-lfs \
      jq \
      less \
      libclang-dev \
      libhwloc-dev \
      libudev-dev \
      pkg-config \
      protobuf-compiler \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.local/bin \
 && git lfs install --system \
 && if ! command -v repo >/dev/null 2>&1; then \
      curl -fsSL \
        https://storage.googleapis.com/git-repo-downloads/repo \
        -o /root/.local/bin/repo; \
      chmod 0755 /root/.local/bin/repo; \
    fi \
 && if ! command -v uv >/dev/null 2>&1; then \
      curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi \
 && if ! command -v rustc >/dev/null 2>&1; then \
      curl --proto '=https' --tlsv1.2 -sSf \
        https://sh.rustup.rs | sh -s -- -y --profile minimal; \
    fi

COPY scripts/ /opt/kvcc-tools/

RUN --mount=type=secret,id=github_token,required=true \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/opt/kvcc-workspace/dynamo/target \
    export GITHUB_TOKEN="$(< /run/secrets/github_token)" \
 && export GITHUB_USER GITHUB_EMAIL \
 && export REPO_SYNC_JOBS \
 && /opt/kvcc-tools/sync-build.sh \
 && find /opt/kvcc-workspace -type d -name __pycache__ -prune \
      -exec rm -rf {} +

FROM ${DYNAMO_BASE_IMAGE} AS runtime

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DYNAMO_BASE_IMAGE

ENV CUDARC_CUDA_VERSION=13000 \
    DYNAMO_BASE_IMAGE=${DYNAMO_BASE_IMAGE} \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    KVCC_WORKSPACE=/opt/kvcc-workspace \
    VIRTUAL_ENV=/opt/kvcc-workspace/.venv \
    PATH=/opt/kvcc-workspace/.venv/bin:${PATH}

COPY --from=builder /opt/kvcc-workspace /opt/kvcc-workspace
COPY --from=builder /opt/kvcc-vllm-overlay /opt/kvcc-vllm-overlay
COPY scripts/install-vllm-overlay.py \
     scripts/verify-selective-runtime.py \
     /opt/kvcc-tools/

RUN /opt/kvcc-workspace/.venv/bin/python \
      /opt/kvcc-tools/install-vllm-overlay.py \
 && /opt/kvcc-workspace/.venv/bin/python \
      /opt/kvcc-tools/verify-selective-runtime.py

WORKDIR /opt/kvcc-workspace
CMD ["bash"]
