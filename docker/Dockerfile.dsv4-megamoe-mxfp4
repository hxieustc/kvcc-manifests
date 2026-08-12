# syntax=docker/dockerfile:1.7

ARG DYNAMO_BUILDER_IMAGE=rust:1.93-bookworm
ARG VLLM_RUNTIME_IMAGE=vllm/vllm-openai:dsv4-megamoe-mxfp4-x86_64-cu130-4ba0a72

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
    mkdir -p /wheels /nvcc-tmp \
 && cd /src/workspace/dynamo/lib/bindings/python \
 && TMPDIR=/nvcc-tmp /venv/bin/maturin build --release --features select-service,kv-indexer --out /wheels \
 && cd /src/workspace/dynamo \
 && TMPDIR=/nvcc-tmp /venv/bin/pip wheel --no-deps . -w /wheels \
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
    VLLM_USE_DEEP_GEMM=0 \
    PYTHONPATH=/opt/kvcc

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      cmake \
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
RUN python3 -m pip install --no-cache-dir /src/nvidia-kvcc \
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

# Install kvcc_repo vLLM from the synced workspace.
# This replaces the dsv4-megamoe base vLLM with a version whose connector/tiering
# code is consistent with the KVCC secondary tier (tiering/kvcc/manager.py etc.).
# --no-deps: reuse dsv4-megamoe's torch, flashinfer, and other system packages.
COPY --from=dynamo-builder /src/workspace/vllm /opt/kvcc-vllm
RUN pip install --no-cache-dir setuptools-rust setuptools_scm uv \
 && SETUPTOOLS_SCM_PRETEND_VERSION=0.1.3 \
    VLLM_USE_PRECOMPILED=1 \
    uv pip install --system --torch-backend=cu130 \
      --no-deps --editable /opt/kvcc-vllm \
 && python3 -c "import vllm; print('vLLM OK', vllm.__version__, vllm.__file__)"

# Add deep_gemm_amxf4_mega_moe CLI support.
# Routes to the same mega_moe kernel path; full MXFP4 _C.so parity is separate work.
# enable_ep_weight_filter is already in kvcc_repo's config/parallel.py.
RUN python3 - <<'EOF'
import pathlib, os
vd = os.path.dirname(__import__("vllm").__file__)
# 0. kv_offload/config.py: add data_parallel_size to OffloadingParallelConfig
#    (manager.py accesses it, but it's missing from the dataclass)
fc = pathlib.Path(f"{vd}/v1/kv_offload/config.py")
tc = fc.read_text()
if "data_parallel_size" not in tc:
    # Add data_parallel_size AFTER is_parallelism_agnostic (last field, no ordering issue)
    tc = tc.replace(
        "    is_parallelism_agnostic: bool\n",
        "    is_parallelism_agnostic: bool\n    # Number of data parallel replicas.\n    data_parallel_size: int = 1\n"
    )
    fc.write_text(tc)
    print("config.py: added data_parallel_size to OffloadingParallelConfig")
# Also patch build_offloading_config to pass data_parallel_size
fo = pathlib.Path(f"{vd}/distributed/kv_transfer/kv_connector/v1/offloading/config.py")
to = fo.read_text()
if "data_parallel_size=" not in to and "data_parallel_index=parallel_config.data_parallel_index" in to:
    to = to.replace(
        "data_parallel_index=parallel_config.data_parallel_index,",
        "data_parallel_index=parallel_config.data_parallel_index,\n            data_parallel_size=getattr(parallel_config, 'data_parallel_size', 1),"
    )
    fo.write_text(to)
    print("offloading/config.py: added data_parallel_size to OffloadingParallelConfig build")
# 1. config/kernel.py: register the backend name
f = pathlib.Path(f"{vd}/config/kernel.py")
t = f.read_text()
if "deep_gemm_amxf4_mega_moe" not in t:
    t = t.replace('"deep_gemm_mega_moe",',
                  '"deep_gemm_mega_moe",\n    "deep_gemm_amxf4_mega_moe",')
    t = t.replace(
        '- "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels',
        '- "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels\n'
        '    - "deep_gemm_amxf4_mega_moe": Use DeepGEMM mega MoE kernels with MXFP4 activations'
        ' (requires matching DeepGEMM MXFP4 build)')
    f.write_text(t)
    print("kernel.py: registered deep_gemm_amxf4_mega_moe")
# 2. models/deepseek_v4/nvidia/model.py: route amxf4 to mega_moe code path
for pattern in [
    'vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"',
]:
    g = pathlib.Path(f"{vd}/models/deepseek_v4/nvidia/model.py")
    if g.exists():
        u = g.read_text()
        repl = pattern.replace(
            '== "deep_gemm_mega_moe"',
            'in ("deep_gemm_mega_moe", "deep_gemm_amxf4_mega_moe")'
        )
        if pattern in u:
            g.write_text(u.replace(pattern, repl))
            print(f"model.py: amxf4 routed via mega_moe path")
print("deep_gemm_amxf4_mega_moe patches applied")
EOF

COPY --from=dynamo-builder /src/workspace/vllm/tests /opt/kvcc/vllm-tests
RUN sed -i \
      -e 's/kvcc_progress\.nixl_utils,/kvcc_progress,/' \
      -e 's/NixlWrapper=lambda \*_: object(),/nixl_agent=lambda *_: object(),/' \
      /opt/kvcc/vllm-tests/v1/kv_offload/kvcc-tests/kvcc-e2e/test_harness.py
# Dynamo 1.4.0: enable_router_hint_support() only fires for Prefill workers;
# kvcc uses kv_both -> Aggregated, so hints are never registered, and the
# multi_worker tracker aggressively cleans up worker slots after each request.
# Make _is_prefill_worker always return True so hints are enabled for all types.
RUN python3 - <<'EOF'
import pathlib
f = pathlib.Path("/usr/local/lib/python3.12/dist-packages/dynamo/vllm/router_hints.py")
if not f.exists():
    print("router_hints.py not found, skipping")
else:
    t = f.read_text()
    # Patch _router_hint_worker_type to also enable hints for Aggregated/kv_both workers.
    # Without this, kv_both workers (Aggregated type) get None and hints are disabled,
    # causing Dynamo's multi_worker to aggressively clean up worker slots after each request.
    # 1. Enable hints for Aggregated/kv_both workers
    old = '    if role not in {"aggregated", "prefill", "decode"}:\n        return None\n    return role'
    new = '    if role not in {"aggregated", "prefill", "decode", "kv_both"}:\n        role = "aggregated"  # patched: enable hints for all worker types\n    return role'
    if old in t:
        t = t.replace(old, new)
        print("router_hints.py patched (_router_hint_worker_type)")
    else:
        old2 = "    if router_hint_worker_type is None:\n        return\n"
        new2 = "    # patched: enable hints regardless of worker type\n    router_hint_worker_type = router_hint_worker_type or 'aggregated'\n"
        if old2 in t:
            t = t.replace(old2, new2)
            print("router_hints.py patched (enable_router_hint_support fallback)")
    # 2. Support control_ports list in addition to control_port scalar
    #    (kvcc_repo vLLM uses "control_ports": [port] instead of "control_port": port)
    old3 = '    configured_port = tier.get("control_port")\n    if not isinstance(configured_port, (int, str)):\n        return None'
    new3 = ('    configured_port = tier.get("control_port")\n'
            '    if configured_port is None:  # patched: also accept control_ports list\n'
            '        ports = tier.get("control_ports", [])\n'
            '        configured_port = ports[0] if ports else None\n'
            '    if not isinstance(configured_port, (int, str)):\n        return None')
    if old3 in t:
        t = t.replace(old3, new3)
        print("router_hints.py patched (control_ports list support)")
    f.write_text(t)
EOF
# Dynamo 1.4.0 compatibility: after each generation the multi_worker tracker
# removes the worker, making pinned 2nd-fill requests fail. Unpin fill_worker
# requests (router routes to the only registered worker anyway) and add retry
# for transient routing errors. Also pass dp_rank=0 for pinned requests.
RUN python3 - <<'EOF'
import pathlib, re
h = pathlib.Path("/opt/kvcc/vllm-tests/v1/kv_offload/kvcc-tests/kvcc-e2e/helpers.py")
t = h.read_text()
# 1. Unpin fill_worker so router picks freely (Dynamo 1.4.0 removes worker after each gen)
t = re.sub(
    r"(def fill_worker\b.*?for token_ids in token_lists:\s+output_ids, _ = generate\(\s+config,\s+router,\s+token_ids,\s+)worker_id,",
    r"\1None,", t, flags=re.DOTALL)
# 2. Pass dp_rank=0 when worker_id is pinned (required by Dynamo 1.4.0)
t = t.replace(
    "                worker_id=worker_id,\n"
    "                response_buffer_size=0,\n"
    "            )",
    "                worker_id=worker_id,\n"
    "                **({'dp_rank': 0} if worker_id is not None else {}),\n"
    "                response_buffer_size=0,\n"
    "            )")
# 3. Retry router.run(run_generation()) for transient routing/stream errors
#    "Disconnected" happens during stream consumption, so must retry at this level.
t = t.replace(
    "    return router.run(run_generation(), timeout=180)",
    "    import time\n"
    "    for _retry in range(10):\n"
    "        try:\n"
    "            return router.run(run_generation(), timeout=180)\n"
    "        except Exception as _e:\n"
    "            if any(s in str(_e) for s in ('no endpoints', 'Disconnected')) and _retry < 9:\n"
    "                time.sleep(2)\n"
    "                continue\n"
    "            raise")
h.write_text(t)
print("helpers.py patched")
EOF
RUN VLLM_PKG="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')" \
 && ln -sfn /opt/kvcc/vllm-tests /opt/kvcc/tests \
 && ln -sfnT "$VLLM_PKG" /opt/kvcc/vllm \
 && ln -sfnT "$VLLM_PKG" /opt/kvcc/vllm-tests/vllm
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
