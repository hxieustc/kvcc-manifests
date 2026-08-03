# syntax=docker/dockerfile:1.7

ARG DYNAMO_BUILDER_IMAGE=rust:1.93-bookworm
ARG VLLM_RUNTIME_IMAGE=vllm/vllm-openai:nightly-6f91edf96d3f3272945809c04702380053bff4de@sha256:674c5aa666d38c07a0dc779f8c77a05c2b859617410d16dcdd8a8776166c92b9

############################
# Stage 1: synchronize sources and build Dynamo wheels
############################
FROM ${DYNAMO_BUILDER_IMAGE} AS dynamo-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GITHUB_USER
ARG GITHUB_EMAIL
ARG REPO_SYNC_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      cmake \
      curl \
      git \
      git-lfs \
      libclang-dev \
      libhwloc-dev \
      libudev-dev \
      pkg-config \
      protobuf-compiler \
      python3-dev \
      python3-pip \
      python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://storage.googleapis.com/git-repo-downloads/repo \
      -o /usr/local/bin/repo \
 && chmod 0755 /usr/local/bin/repo \
 && git lfs install --system \
 && python3 -m venv /venv \
 && /venv/bin/pip install --no-cache-dir "maturin[patchelf]"

RUN --mount=type=secret,id=github_token,required=true \
    GITHUB_TOKEN="$(< /run/secrets/github_token)" \
 && BASIC_AUTH="$(printf '%s:%s' "$GITHUB_USER" "$GITHUB_TOKEN" | base64 | tr -d '\r\n')" \
 && export GIT_CONFIG_COUNT=3 \
 && export GIT_CONFIG_KEY_0=user.name \
 && export GIT_CONFIG_VALUE_0="$GITHUB_USER" \
 && export GIT_CONFIG_KEY_1=user.email \
 && export GIT_CONFIG_VALUE_1="$GITHUB_EMAIL" \
 && export GIT_CONFIG_KEY_2=http.https://github.com/.extraHeader \
 && export GIT_CONFIG_VALUE_2="Authorization: Basic ${BASIC_AUTH}" \
 && mkdir -p /src/workspace \
 && cd /src/workspace \
 && repo init \
      -u https://github.com/hxieustc/kvcc-manifests.git \
      -b cuda-env-fixes \
      -m manifests/develop.xml \
 && repo sync "-j${REPO_SYNC_JOBS}"

RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/workspace/dynamo/target \
    mkdir -p /wheels \
 && cd /src/workspace/dynamo/lib/bindings/python \
 && /venv/bin/maturin build --release --features select-service,kv-indexer --out /wheels \
 && cd /src/workspace/dynamo \
 && /venv/bin/pip wheel --no-deps . -w /wheels \
 && ls -la /wheels

############################
# Stage 2: install Dynamo/KVCC and overlay the base vLLM package
############################
FROM ${VLLM_RUNTIME_IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1 \
    VLLM_USE_DEEP_GEMM=0

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      jq \
      numactl \
 && rm -rf /var/lib/apt/lists/*

RUN FI_VER="$(python3 -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")" \
 && python3 -m pip install --no-cache-dir \
      "flashinfer-jit-cache==${FI_VER}" \
      "flashinfer-cubin @ https://github.com/flashinfer-ai/flashinfer/releases/download/v${FI_VER}/flashinfer_cubin-${FI_VER}-py3-none-any.whl" \
      --index-url https://flashinfer.ai/whl/cu130 \
 && python3 -c "import flashinfer; print('flashinfer OK', flashinfer.__version__)"

RUN python3 -m pip show nixl >/dev/null 2>&1 \
 || python3 -m pip install --no-cache-dir nixl

COPY --from=dynamo-builder /wheels /tmp/dynamo-wheels
RUN python3 -m pip install --no-cache-dir \
      /tmp/dynamo-wheels/ai_dynamo_runtime-*.whl \
 && python3 -m pip install --no-cache-dir \
      /tmp/dynamo-wheels/ai_dynamo-*.whl \
 && rm -rf /tmp/dynamo-wheels \
 && python3 -m pip install --no-cache-dir --upgrade openai \
 && python3 -c "from openai.types.responses import NamespaceTool; print('openai floor OK')"

COPY --from=dynamo-builder /src/workspace/kvcc /src/nvidia-kvcc
RUN python3 -m pip install --no-cache-dir /src/nvidia-kvcc \
 && python3 -c "import importlib.metadata as m, kvcc; print('nvidia-kvcc OK', m.version('nvidia-kvcc'))"

COPY --from=dynamo-builder \
     /src/workspace/vllm/vllm/v1/kv_offload/tiering/kvcc \
     /overlay/vllm/v1/kv_offload/tiering/kvcc
RUN VLLM_DIR="$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")" \
 && rm -rf "${VLLM_DIR}/v1/kv_offload/tiering/kvcc" \
 && mkdir -p "${VLLM_DIR}/v1/kv_offload/tiering" \
 && cp -a /overlay/vllm/v1/kv_offload/tiering/kvcc \
      "${VLLM_DIR}/v1/kv_offload/tiering/kvcc" \
 && python3 -m compileall -q "${VLLM_DIR}/v1/kv_offload" \
 && python3 -c "import vllm.v1.kv_offload.tiering.kvcc.manager as m; print('KVCC overlay OK', m.__file__)"

COPY --from=dynamo-builder \
     /src/workspace/vllm/tests/v1/kv_offload/tiering \
     /opt/kvcc/vllm-tests/tiering
COPY --from=dynamo-builder \
     /src/workspace/vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e \
     /opt/kvcc/vllm-tests/kvcc-e2e

RUN python3 - <<'PY'
import importlib
import vllm

importlib.import_module("vllm._C_stable_libtorch")
importlib.import_module("vllm.v1.kv_offload.tiering.kvcc.manager")
importlib.import_module("dynamo.vllm.router_hints")
importlib.import_module("kvcc")
importlib.import_module("nixl")
print("runtime sanity OK: vllm", vllm.__version__)
PY

WORKDIR /opt/kvcc
ENTRYPOINT []
CMD ["/bin/bash"]
