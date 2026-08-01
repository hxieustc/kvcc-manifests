# CUDA Environment Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `kvcc-manifests` provide a self-contained Dynamo-image Kubernetes pod in which a user can run `repo sync`, `bootstrap.sh`, `build-all.sh`, and `test-all.sh` without source patches or manual dependency repair.

**Architecture:** A Kubernetes manifest prepares only the capabilities needed before repository synchronization: authenticated HTTPS Git access and the Google repo executable. The synchronized manifest scripts then own system/bootstrap preparation, ordered editable installs, CUDA 13.0 dependency reconciliation, runtime verification, and the documented unit/E2E scope. Shell fixture tests stub external tools so dependency and orchestration behavior can be verified without GPUs or package downloads.

**Tech Stack:** Bash, Kubernetes YAML, Google repo manifests, uv, Python 3.12, PyTorch CUDA 13.0, FlashInfer, pytest, tsh kubectl.

## Global Constraints

- Work only on branch `cuda-env-fixes`, created from `develop`.
- Do not patch customized Dynamo, vLLM, or KVCC source code.
- Use `nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest` as the pod image.
- Target AMD64 and two NVIDIA B200 GPUs.
- Use only HTTPS Git remotes and a Secret-backed credential helper.
- Never embed or print `GITHUB_TOKEN`.
- Require torch `2.13.0+cu130`, torchvision `0.28.0+cu130`, and CUDA `13.0`.
- Require FlashInfer cubin to equal the Python package version and JIT cache to equal that version plus `+cu130`.
- Keep the manual sequence `repo sync -> bootstrap.sh -> build-all.sh -> test-all.sh`.
- “All tests” means KVCC unit, vLLM/KVCC unit, focused Dynamo/vLLM router integration, and two-GPU KVCC E2E.
- Retain the PVC during ordinary pod deletion.

---

### Task 1: Establish the repository contract tests

**Files:**
- Create: `tests/test-repository-contract.sh`
- Create: `tests/run.sh`
- Modify: `.github/workflows/integration.yml`

**Interfaces:**
- Consumes: repository files at the checkout root.
- Produces: `bash tests/run.sh`, the single local/CI entry point for manifest tests.

- [ ] **Step 1: Write the failing repository contract test**

Create `tests/test-repository-contract.sh` with helpers `assert_file`, `assert_contains`, and `assert_not_contains`. Assert:

```bash
assert_file kubernetes/dev-pod.yaml
assert_contains kubernetes/dev-pod.yaml   'nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest'
assert_contains kubernetes/dev-pod.yaml 'nvidia.com/gpu:[[:space:]]*"?2"?'
assert_contains kubernetes/dev-pod.yaml 'nvidia.com/gpu.product:[[:space:]]*NVIDIA-B200'
assert_contains kubernetes/dev-pod.yaml 'kubernetes.io/arch:[[:space:]]*amd64'
assert_contains kubernetes/dev-pod.yaml 'mountPath:[[:space:]]*/dev/shm'
assert_contains kubernetes/dev-pod.yaml 'name:[[:space:]]*kvcc-github-auth'
assert_contains kubernetes/dev-pod.yaml 'name:[[:space:]]*repo-installer'
assert_not_contains kubernetes/dev-pod.yaml 'GITHUB_TOKEN:[[:space:]]*[^$]'
assert_not_contains manifests/develop.xml 'github-ssh|ssh://'
assert_contains manifests/develop.xml 'revision="cuda-env-fixes"'
assert_file scripts/verify-env.sh
assert_contains scripts/build-all.sh 'verify-env.sh'
assert_contains scripts/test-all.sh 'VLLM_DEEP_GEMM_WARMUP'
assert_contains scripts/test-all.sh 'test_router_hints.py'
assert_contains README.md 'repo init.*cuda-env-fixes'
assert_contains README.md 'tsh kubectl'
```

Create `tests/run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/test-repository-contract.sh"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
bash tests/run.sh
```

Expected: FAIL because `kubernetes/dev-pod.yaml` and `scripts/verify-env.sh` do not exist.

- [ ] **Step 3: Add the test runner to CI**

Modify `.github/workflows/integration.yml` to execute:

```yaml
- name: Validate manifest workflow
  run: bash tests/run.sh
```

Extend the pull-request path filter to include `tests/**`, `kubernetes/**`, and
`README.md` so changes to the manual workflow cannot bypass this validation.

Do not add production files yet.

- [ ] **Step 4: Commit the red contract**

```bash
git add tests .github/workflows/integration.yml
git commit -m "test: define manual CUDA pod workflow contract"
```

### Task 2: Add the authenticated Dynamo development pod

**Files:**
- Create: `kubernetes/dev-pod.yaml`
- Modify: `manifests/develop.xml`
- Modify: `.github/workflows/integration.yml`
- Test: `tests/test-repository-contract.sh`

**Interfaces:**
- Consumes: Kubernetes Secret `kvcc-github-auth` with keys `GITHUB_USER`, `GITHUB_EMAIL`, and `GITHUB_TOKEN`.
- Produces: pod `kvcc-mbench/kvcc-dev`, PVC `kvcc-workspace`, `repo` on `PATH`, and an HTTPS Git credential helper.

- [ ] **Step 1: Narrow the contract test to pod security and topology**

Add assertions that the manifest contains:

```bash
assert_contains kubernetes/dev-pod.yaml 'storageClassName:[[:space:]]*vast'
assert_contains kubernetes/dev-pod.yaml 'storage:[[:space:]]*200Gi'
assert_contains kubernetes/dev-pod.yaml 'sizeLimit:[[:space:]]*64Gi'
assert_contains kubernetes/dev-pod.yaml 'credential.helper'
assert_contains kubernetes/dev-pod.yaml 'password=.*GITHUB_TOKEN'
assert_not_contains kubernetes/dev-pod.yaml 'ghp_|github_pat_'
```

- [ ] **Step 2: Run and verify RED**

Run `bash tests/test-repository-contract.sh`.

Expected: FAIL at missing `kubernetes/dev-pod.yaml`.

- [ ] **Step 3: Create the pod manifest**

Create a multi-document YAML containing:

1. Namespace `kvcc-mbench`.
2. 200 GiB RWX VAST PVC `kvcc-workspace`.
3. ConfigMap `kvcc-pod-tools` with executable `github-credential-helper.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  get)
    printf 'protocol=https\nhost=github.com\nusername=%s\npassword=%s\n' \
      "$GITHUB_USER" "$GITHUB_TOKEN"
    ;;
  store|erase) ;;
  *) exit 1 ;;
esac
```

4. Init container `repo-installer` that downloads the Google repo tool to
   `/opt/kvcc-bin/repo` and marks it executable.
5. Main container with the approved Dynamo image, Secret `envFrom`, `HOME=/root`,
   `CUDARC_CUDA_VERSION=13000`, tools PATH, two B200 GPUs, AMD64 selection,
   workspace PVC, 64 GiB dshm, and read-only credential helper.
6. Main command that runs:

```bash
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_EMAIL"
git config --global credential.helper   '/opt/kvcc-pod-tools/github-credential-helper.sh'
trap : TERM INT
sleep infinity & wait
```

Update `manifests/develop.xml` to remove the unused SSH remote/comment and set the manifest project revision to `cuda-env-fixes`.

Update the disabled GPU workflow's future `repo init` command from its SSH URL
to the HTTPS URL and add `-b cuda-env-fixes`, keeping CI consistent with the
documented authentication model.

- [ ] **Step 4: Run and verify GREEN**

Run:

```bash
bash tests/test-repository-contract.sh
git diff --check
```

Expected: repository contract passes until the next not-yet-created artifact assertion; pod-specific assertions pass.

- [ ] **Step 5: Commit**

```bash
git add kubernetes/dev-pod.yaml manifests/develop.xml tests/test-repository-contract.sh
git commit -m "feat: add authenticated Dynamo development pod"
```

### Task 3: Make bootstrap self-contained and idempotent

**Files:**
- Create: `tests/test-bootstrap.sh`
- Modify: `tests/run.sh`
- Modify: `scripts/bootstrap.sh`

**Interfaces:**
- Consumes: root-capable Debian/Ubuntu Dynamo container, `repo` on PATH, workspace source directories.
- Produces: isolated `.venv` with Python 3.12, uv, Rust, NIXL, build tools, and pytest.

- [ ] **Step 1: Write the failing bootstrap fixture**

Create a temporary workspace with `manifests/scripts/bootstrap.sh`, fake
`dynamo`, `vllm`, and `kvcc` directories, and a `bin/` containing stubs.
The stubs append invocations to `CALL_LOG`. Test two cases:

```bash
# Root-like case: dpkg reports one missing package.
FAKE_UID=0 bash "$fixture/manifests/scripts/bootstrap.sh"
grep -F 'apt-get install -y --no-install-recommends' "$CALL_LOG"
grep -F 'uv venv --python 3.12' "$CALL_LOG"
grep -F "maturin[patchelf]" "$CALL_LOG"
grep -F 'nixl' "$CALL_LOG"

# Hooks are opt-in.
! grep -F 'pre-commit install' "$CALL_LOG"

# Explicit opt-in installs the hook.
KVCC_INSTALL_PRECOMMIT=1 bash "$fixture/manifests/scripts/bootstrap.sh"
grep -F 'pre-commit install' "$CALL_LOG"
```

Make `bootstrap.sh` consume `KVCC_EFFECTIVE_UID` only for tests, defaulting to
`id -u`. Stub `dpkg`, `apt-get`, `curl`, `uv`, `rustc`, `repo`,
`python`, and `pre-commit`.

Add `bash "$SCRIPT_DIR/test-bootstrap.sh"` to `tests/run.sh`.

- [ ] **Step 2: Run and verify RED**

Run `bash tests/test-bootstrap.sh`.

Expected: FAIL because current bootstrap exits on missing packages and never invokes apt.

- [ ] **Step 3: Implement minimal bootstrap behavior**

Refactor `bootstrap.sh` to:

```bash
KVCC_INSTALL_SYSTEM_DEPS="${KVCC_INSTALL_SYSTEM_DEPS:-1}"
KVCC_INSTALL_PRECOMMIT="${KVCC_INSTALL_PRECOMMIT:-0}"
EFFECTIVE_UID="${KVCC_EFFECTIVE_UID:-$(id -u)}"
```

If packages are missing and system installation is enabled:

- require UID 0;
- run `apt-get update`;
- install only the missing package array with `--no-install-recommends`.

Install uv into `/root/.local/bin` or the current home when absent. Install Rust
with rustup when absent. Require `repo` before proceeding. Create the venv with:

```bash
uv venv --python 3.12 "$VENV_DIR"
```

Target every package installation with `--python "$VENV_DIR/bin/python"`.
Install pre-commit hooks only when `KVCC_INSTALL_PRECOMMIT=1`.

- [ ] **Step 4: Run and verify GREEN**

Run:

```bash
bash tests/test-bootstrap.sh
bash tests/run.sh
bash -n scripts/bootstrap.sh
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/bootstrap.sh tests/test-bootstrap.sh tests/run.sh
git commit -m "feat: bootstrap Dynamo pod prerequisites"
```

### Task 4: Reconcile CUDA dependencies without a warm cache

**Files:**
- Create: `tests/test-build-all.sh`
- Modify: `tests/run.sh`
- Modify: `scripts/build-all.sh`

**Interfaces:**
- Consumes: bootstrapped venv and synchronized source trees.
- Produces: editable Dynamo, vLLM, and KVCC installs with the exact cu130 torch and matching FlashInfer native packages.

- [ ] **Step 1: Write the failing fresh-cache fixture**

Create a temporary workspace with the expected source directories and fake
`.venv/bin/uv`, `maturin`, and `python3`. Log invocations to `UV_LOG`.
The Python stub returns `0.6.15.post1` for metadata queries.

Run with an empty HOME:

```bash
HOME="$fixture/empty-home" CUDARC_CUDA_VERSION=13000   bash "$fixture/manifests/scripts/build-all.sh"
```

Assert:

```bash
grep -F -- '--extra-index-url https://download.pytorch.org/whl/cu130' "$UV_LOG"
grep -F -- 'torch==2.13.0+cu130' "$UV_LOG"
grep -F -- 'torchvision==0.28.0+cu130' "$UV_LOG"
grep -F -- 'flashinfer-cubin==0.6.15.post1' "$UV_LOG"
grep -F -- 'flashinfer-jit-cache==0.6.15.post1+cu130' "$UV_LOG"
grep -F -- 'https://flashinfer.ai/whl/cu130' "$UV_LOG"
! grep -F -- '.cache/uv/archive-v0'   "$fixture/manifests/scripts/build-all.sh"
```

Also compare line numbers in `UV_LOG` to prove Dynamo installation precedes
customized vLLM and KVCC follows vLLM.

Add the test to `tests/run.sh`.

- [ ] **Step 2: Run and verify RED**

Run `bash tests/test-build-all.sh`.

Expected: FAIL because the original script searches `$HOME/.cache/uv/archive-v0`
and never installs FlashInfer JIT cache.

- [ ] **Step 3: Implement minimal dependency reconciliation**

At the top of `build-all.sh`, define overridable constants:

```bash
KVCC_TORCH_VERSION="${KVCC_TORCH_VERSION:-2.13.0+cu130}"
KVCC_TORCHVISION_VERSION="${KVCC_TORCHVISION_VERSION:-0.28.0+cu130}"
KVCC_TORCH_INDEX_URL="${KVCC_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
KVCC_FLASHINFER_INDEX_URL="${KVCC_FLASHINFER_INDEX_URL:-https://flashinfer.ai/whl/}"
KVCC_FLASHINFER_CUDA_INDEX_URL="${KVCC_FLASHINFER_CUDA_INDEX_URL:-https://flashinfer.ai/whl/cu130}"
```

Delete all uv archive scanning. After vLLM and test requirements, run the exact
torch/torchvision install with `--index-strategy unsafe-best-match`. Query
FlashInfer using `"$VENV_DIR/bin/python"`, then install cubin and JIT cache
from their two indexes. Keep KVCC installation last.

Invoke `"$SCRIPT_DIR/verify-env.sh"` at the end; temporarily create a no-op
fixture verifier in the test until Task 5.

- [ ] **Step 4: Run and verify GREEN**

Run:

```bash
bash tests/test-build-all.sh
bash tests/run.sh
bash -n scripts/build-all.sh
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build-all.sh tests/test-build-all.sh tests/run.sh
git commit -m "fix: reconcile vLLM dependencies for CUDA 13"
```

### Task 5: Add strict runtime verification

**Files:**
- Create: `scripts/verify-env.sh`
- Create: `tests/test-verify-env.sh`
- Modify: `tests/run.sh`
- Test: `scripts/build-all.sh`

**Interfaces:**
- Consumes: `KVCC_WORKSPACE` and its `.venv/bin/python`.
- Produces: nonzero status for version, CUDA, FlashInfer, provenance, native import, NIXL, or dependency-metadata mismatch.

- [ ] **Step 1: Write the failing verifier tests**

Create a fixture Python executable that emits JSON from environment-controlled
values when invoked with the verifier's embedded Python program. Prefer testing
the real embedded assertions by setting `PYTHONPATH` to fixture modules and
creating matching `*.dist-info` metadata.

Required cases:

```bash
# Correct versions and workspace paths pass.
KVCC_WORKSPACE="$fixture/workspace"   bash scripts/verify-env.sh

# Wrong CUDA torch fails with the discovered value.
FAKE_TORCH_VERSION=2.13.0+cu132   expect_failure 'torch 2.13.0+cu130 required'

# Stale JIT cache fails.
FAKE_FLASHINFER_JIT=0.6.14+cu130   expect_failure 'FlashInfer JIT cache mismatch'

# vLLM outside workspace fails.
FAKE_VLLM_PATH=/usr/local/lib/python3.12/site-packages/vllm/__init__.py   expect_failure 'resolved outside workspace'
```

Stub `uv pip check --python ...` and assert it is invoked.

Add the test to `tests/run.sh`.

- [ ] **Step 2: Run and verify RED**

Run `bash tests/test-verify-env.sh`.

Expected: FAIL because `scripts/verify-env.sh` does not exist.

- [ ] **Step 3: Implement the verifier**

Create `scripts/verify-env.sh` with Bash preflight and an embedded Python
program that:

- requires Python 3.12;
- checks exact torch and torchvision versions;
- requires `torch.version.cuda == "13.0"`;
- checks FlashInfer Python/cubin equality and JIT `+cu130`;
- imports `dynamo.vllm`, `vllm`, `vllm._C_stable_libtorch`, `kvcc`, and
  `nixl`;
- checks Dynamo, vLLM, and KVCC module paths are under the workspace;
- prints a sorted JSON summary.

Then run:

```bash
uv pip check --python "$PYTHON"
```

Do not suppress or whitelist dependency errors in this clean isolated venv.

- [ ] **Step 4: Run and verify GREEN**

Run:

```bash
bash tests/test-verify-env.sh
bash tests/run.sh
bash -n scripts/verify-env.sh
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify-env.sh tests/test-verify-env.sh tests/run.sh
git commit -m "feat: verify CUDA runtime compatibility"
```

### Task 6: Complete deterministic unit and E2E orchestration

**Files:**
- Create: `tests/test-test-all.sh`
- Modify: `tests/run.sh`
- Modify: `scripts/test-all.sh`

**Interfaces:**
- Consumes: built workspace, `KVCC_TEST_GPUS`, `KVCC_RUN_E2E`.
- Produces: aggregate status across KVCC, vLLM/KVCC, focused Dynamo, and optional E2E suites.

- [ ] **Step 1: Write failing orchestration tests**

Create a fixture workspace with a fake Python executable that logs pytest
arguments and simulates `torch.cuda.device_count() == 2`. Run:

```bash
KVCC_RUN_E2E=1 KVCC_TEST_GPUS=0,1   bash "$fixture/manifests/scripts/test-all.sh"
```

Assert the log includes:

```bash
grep -F 'kvcc/tests/' "$PYTEST_LOG"
grep -F 'test_config.py' "$PYTEST_LOG"
grep -F 'test_harness.py' "$PYTEST_LOG"
grep -F 'test_kvcc_tier.py' "$PYTEST_LOG"
grep -F 'dynamo/components/src/dynamo/vllm/tests/test_router_hints.py' "$PYTEST_LOG"
grep -F 'validate_kvcc_e2e.py' "$PYTEST_LOG"
grep -F -- '--gpus 0,1' "$PYTEST_LOG"
```

Make the fake E2E invocation record
`VLLM_DEEP_GEMM_WARMUP` and assert it equals `skip`.

Add a failure case where requested GPU 2 is unavailable and assert E2E is not
started and the script returns nonzero. Add a `KVCC_RUN_E2E=0` case that runs
all CPU/focused Dynamo tests but omits E2E.

Add the test to `tests/run.sh`.

- [ ] **Step 2: Run and verify RED**

Run `bash tests/test-test-all.sh`.

Expected: FAIL because Dynamo tests and the warmup environment are absent and
there is no GPU preflight.

- [ ] **Step 3: Implement minimal orchestration**

Modify `test-all.sh` to:

- export `VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"`;
- add focused Dynamo router-hint pytest;
- validate comma-separated GPU indices with a small Python command before E2E;
- preserve aggregate `FAILED` behavior;
- install an exit trap that terminates only jobs returned by `jobs -pr`;
- print the exact supported scope in the success message.

Do not add broad `pkill` commands.

- [ ] **Step 4: Run and verify GREEN**

Run:

```bash
bash tests/test-test-all.sh
bash tests/run.sh
bash -n scripts/test-all.sh
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/test-all.sh tests/test-test-all.sh tests/run.sh
git commit -m "test: complete deterministic KVCC test orchestration"
```

### Task 7: Document the exact manual workflow

**Files:**
- Modify: `README.md`
- Test: `tests/test-repository-contract.sh`

**Interfaces:**
- Consumes: local shell variables and `tsh kubectl`.
- Produces: copy-paste workflow from Secret creation through test completion and safe cleanup.

- [ ] **Step 1: Expand the failing README contract**

Assert README includes these literals or equivalent patterns:

```bash
export GITHUB_USER=hxieustc
export GITHUB_EMAIL=
GITHUB_TOKEN
create secret generic kvcc-github-auth
--from-env-file
tsh kubectl apply -f kubernetes/dev-pod.yaml
repo init -u https://github.com/hxieustc/kvcc-manifests.git -b cuda-env-fixes
repo sync
bash manifests/scripts/bootstrap.sh
bash manifests/scripts/build-all.sh
bash manifests/scripts/test-all.sh
delete pod kvcc-dev
PVC
```

Assert it does not recommend a persistent `http.*.extraHeader`.

- [ ] **Step 2: Run and verify RED**

Run `bash tests/test-repository-contract.sh`.

Expected: FAIL on missing branch/pod/Secret workflow documentation.

- [ ] **Step 3: Rewrite README quick start**

Document:

1. Source `GITHUB_TOKEN` from the user shell without printing it.
2. Create a mode-0600 temporary env file containing the three GitHub values.
3. Create/update the Secret with `--from-env-file`, then remove the file.
4. Apply the pod manifest and wait for readiness with `tsh kubectl`.
5. Enter `kvcc-dev`, create `/opt/kvcc-workspace`, and run:

```bash
repo init   -u https://github.com/hxieustc/kvcc-manifests.git   -b cuda-env-fixes   -m manifests/develop.xml
repo sync -j8
bash manifests/scripts/bootstrap.sh
bash manifests/scripts/build-all.sh
bash manifests/scripts/test-all.sh
```

6. Explain environment overrides and supported test scope.
7. Document pod deletion that retains the PVC and separate explicit PVC deletion.
8. Remove the old global extraHeader recommendation and stale SSH statements.

- [ ] **Step 4: Run and verify GREEN**

Run:

```bash
bash tests/test-repository-contract.sh
bash tests/run.sh
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test-repository-contract.sh
git commit -m "docs: add manual Dynamo pod workflow"
```

### Task 8: Final repository and live-fixture verification

**Files:**
- Modify only if verification exposes a defect.

**Interfaces:**
- Consumes: completed branch.
- Produces: clean, reviewable `cuda-env-fixes` branch with evidence.

- [ ] **Step 1: Run all local tests fresh**

```bash
bash tests/run.sh
```

Expected: all fixture and contract tests pass.

- [ ] **Step 2: Run syntax and whitespace checks**

```bash
find scripts tests -type f -name '*.sh' -print0 |
  xargs -0 -n1 bash -n
git diff develop...HEAD --check
```

Expected: exit 0.

- [ ] **Step 3: Validate Kubernetes structure**

If `kubectl` is available:

```bash
kubectl apply --dry-run=client -f kubernetes/dev-pod.yaml >/dev/null
```

Otherwise parse all YAML documents with the available Python YAML library and
assert their `kind` sequence.

Expected: valid Namespace, PVC, ConfigMap, and Pod documents.

- [ ] **Step 4: Scan for credential leakage**

```bash
if rg -n 'ghp_|github_pat_|Authorization: Basic|GITHUB_TOKEN=.+'   --glob '!.git/**' .; then
  exit 1
fi
```

Expected: no token values or encoded credentials.

- [ ] **Step 5: Review branch scope**

```bash
git status --short
git log --oneline develop..HEAD
git diff --stat develop...HEAD
```

Expected: clean worktree; changes limited to design/plan docs, Kubernetes,
manifest, scripts, tests, workflow, and README.

- [ ] **Step 6: Commit any verification-only correction**

Only if Step 1-5 required a correction, stage the scoped implementation files:

```bash
git add .github README.md kubernetes manifests scripts tests
git commit -m "fix: address final workflow verification"
```
