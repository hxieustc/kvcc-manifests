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
      -b develop \
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

RUN rm -rf /src/workspace/.repo \
 && find /src/workspace -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true \
 && find /src/workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
 && find /src/workspace -name "*.pyc" -o -name "*.pyo" | xargs rm -f 2>/dev/null || true

############################
# Stage 2: install Dynamo/KVCC and overlay the base vLLM package
############################
FROM ${VLLM_RUNTIME_IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=1 \
    VLLM_USE_DEEP_GEMM=1 \
    PYTHONPATH=/opt/kvcc

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      jq \
      numactl \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --no-cache-dir pandas pytest tblib

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
RUN uv pip install --system /src/nvidia-kvcc \
 && KVCC_DIR="$(python3 -c "import kvcc, os; print(os.path.dirname(kvcc.__file__))")" \
 && sed -i 's/^_JOIN_TIMEOUT_SECONDS = .*/_JOIN_TIMEOUT_SECONDS = 60.0/' "${KVCC_DIR}/progress.py" \
 && grep -q "_JOIN_TIMEOUT_SECONDS = 60.0" "${KVCC_DIR}/progress.py" \
 && python3 - <<'EOF'
import importlib.metadata as m
import kvcc

expected = {"KVCC", "DURATION_METRIC", "STATE_METRIC",
            "TRANSFER_BLOCKS_METRIC", "TRANSFER_BYTES_METRIC"}
missing = expected - set(getattr(kvcc, "__all__", []))
assert not missing, f"kvcc.__all__ is missing {missing}"
for name in expected:
    assert hasattr(kvcc, name), f"kvcc.{name} not importable from package root"

print("nvidia-kvcc OK", m.version("nvidia-kvcc"), "| exports:", sorted(expected))
EOF

COPY --from=dynamo-builder \
     /src/workspace/vllm/vllm/v1/kv_offload/tiering \
     /overlay/vllm/v1/kv_offload/tiering
RUN VLLM_DIR="$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")" \
 && rm -rf "${VLLM_DIR}/v1/kv_offload/tiering" \
 && mkdir -p "${VLLM_DIR}/v1/kv_offload" \
 && cp -a /overlay/vllm/v1/kv_offload/tiering \
      "${VLLM_DIR}/v1/kv_offload/tiering" \
 && python3 -m compileall -q "${VLLM_DIR}/v1/kv_offload" \
 && python3 -c "import vllm.v1.kv_offload.tiering.kvcc.manager as m; print('KVCC overlay OK', m.__file__)"

COPY --from=dynamo-builder /src/workspace/vllm/tests /opt/kvcc/vllm-tests
RUN sed -i \
      -e 's/kvcc_progress\.nixl_utils,/kvcc_progress,/' \
      -e 's/NixlWrapper=lambda \*_: object(),/nixl_agent=lambda *_: object(),/' \
      /opt/kvcc/vllm-tests/v1/kv_offload/kvcc-tests/kvcc-e2e/test_harness.py
RUN VLLM_PKG="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" \
 && ln -sfn /opt/kvcc/vllm-tests /opt/kvcc/tests \
 && ln -sfn "$VLLM_PKG" /opt/kvcc/vllm \
 && ln -sfn "$VLLM_PKG" /opt/kvcc/vllm-tests/vllm
COPY --from=dynamo-builder /src/workspace /src/workspace

RUN python3 - <<'EOF'
import vllm, dynamo, nixl
from dynamo._core import ModelRuntimeConfig
rc = ModelRuntimeConfig()
if hasattr(rc, "kv_control_endpoints"):
    rc.kv_control_endpoints = {0: "tcp://127.0.0.1:17777"}
    assert rc.kv_control_endpoints == {0: "tcp://127.0.0.1:17777"}
    import dynamo.select_service
    flavor = "change72 (kv_control_endpoints + select_service)"
else:
    import dynamo.vllm.router_hints as rh
    assert callable(rh.enable_router_hint_support)
    flavor = "mainline router_hints"
print("sanity OK: vllm", vllm.__version__, "| dynamo", flavor, "| nixl")
EOF

WORKDIR /opt/kvcc
ENTRYPOINT []
CMD ["/bin/bash"]
