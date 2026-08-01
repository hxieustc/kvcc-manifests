# CUDA Environment Fixes Design

## Goal

Provide a self-contained, manual Kubernetes development workflow in which a
user launches a pod based on NVIDIA's Dynamo vLLM runtime image, synchronizes
the repositories over authenticated HTTPS, and runs only these commands to
install, build, and test the customized stack:

```bash
bash manifests/scripts/bootstrap.sh
bash manifests/scripts/build-all.sh
bash manifests/scripts/test-all.sh
```

The customized Dynamo, vLLM, and KVCC source revisions are assumed correct.
This change owns only container preparation, dependency reconciliation,
runtime verification, test orchestration, and documentation.

## Supported Environment

- Container image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest`
- Architecture: AMD64
- GPUs: two NVIDIA B200 GPUs
- Python: 3.12 in an isolated workspace venv
- CUDA runtime selected by PyTorch: 13.0
- PyTorch: `2.13.0+cu130`
- torchvision: `0.28.0+cu130`
- FlashInfer Python and cubin: identical public versions
- FlashInfer JIT cache: the FlashInfer Python version with `+cu130`
- Repository access: HTTPS with `GITHUB_USER`, `GITHUB_EMAIL`, and
  `GITHUB_TOKEN` supplied through a Kubernetes Secret

The development manifest continues to track the active Dynamo, vLLM, and KVCC
branches. The manifest repository project points at `cuda-env-fixes` while
this work is reviewed so the synchronized `manifests/` checkout contains the
new scripts.

## Kubernetes Architecture

`kubernetes/dev-pod.yaml` defines the namespace, a 200 GiB ReadWriteMany VAST
PVC, a ConfigMap containing a Git credential helper, and the development pod.
The pod requests two B200 GPUs, mounts a 64 GiB memory-backed `/dev/shm`, and
sleeps after initialization so the user can execute the workflow manually.

An init container downloads the Google `repo` executable into a shared tools
volume before the main container starts. This is necessary because `repo sync`
precedes `bootstrap.sh`. The main container adds that volume to `PATH`.

The Git credential helper reads credentials from environment variables and
writes them only to Git's credential protocol output when invoked. The token
is sourced from `kvcc-github-auth`; it is not embedded in the manifest, a Git
remote, or persistent Git configuration. All projects use the HTTPS remote.

## Bootstrap Behavior

`bootstrap.sh` is idempotent. It:

1. Determines the workspace root from its own location.
2. Installs only missing apt development packages when running as root.
3. Reports an exact installation command and fails when packages are missing
   under a non-root user.
4. Installs uv and Rust when absent and makes their paths available within the
   script.
5. Verifies that `repo` is already available from the pod tools volume.
6. Creates an isolated Python 3.12 venv without system site-packages.
7. Installs build tools, NIXL, pytest, and optional pre-commit tooling.

Pre-commit hook installation is disabled by default in the pod and can be
enabled with `KVCC_INSTALL_PRECOMMIT=1`.

## Build and Dependency Reconciliation

`build-all.sh` retains the required order:

1. Build Dynamo's Rust/Python bindings and install customized Dynamo.
2. Install Dynamo's vLLM backend extras.
3. Install customized vLLM as an editable package using its precompiled native
   artifacts and install the CUDA test requirements.
4. Reconcile torch and torchvision to the exact CUDA 13.0 builds from the
   official PyTorch cu130 index.
5. Read the installed `flashinfer-python` version and install matching
   `flashinfer-cubin` and `flashinfer-jit-cache` packages from the FlashInfer
   indexes.
6. Install KVCC as an editable package.
7. Run `verify-env.sh`.

The final torch installation uses the authoritative package index and never
scrapes uv's internal cache. Reconciliation deliberately occurs after vLLM's
requirements because those requirements can temporarily select a different
torch build.

`verify-env.sh` fails unless package versions, CUDA selection, workspace import
provenance, native vLLM extension loading, NIXL, and dependency metadata are
consistent. It prints the discovered values before returning a failure.

## Test Orchestration

`test-all.sh` runs the following supported scope:

1. KVCC standalone unit tests.
2. vLLM/KVCC configuration, harness, and tiering unit tests.
3. Focused Dynamo/vLLM router integration tests used by KVCC.
4. The two-GPU KVCC E2E test when `KVCC_RUN_E2E=1`.

The script exports `VLLM_DEEP_GEMM_WARMUP=skip` for the E2E workers because
DeepGEMM autotuning is unrelated to KVCC correctness and can exceed the worker
startup timeout. It validates the requested GPU indices before starting E2E.
Tests continue after an individual suite fails so the final output reports all
failing sections, then the script returns nonzero.

The E2E harness owns process cleanup. `test-all.sh` also installs an exit trap
that terminates only child processes started by its own process group; it does
not use broad process-name kills.

“All tests passed” refers to the four explicitly supported suites above, not
every test in the Dynamo or vLLM monorepositories. An intentionally skipped
E2E parameter remains a skip unless its source test configuration enables it.

## Tests for the Manifest Repository

The repository gains shell tests under `tests/` that execute against temporary
fixtures and stub external tools. Tests cover:

- pod resources, image, Secret usage, tools volume, and absence of embedded
  credentials;
- HTTPS-only repository configuration;
- root and non-root bootstrap behavior;
- isolated Python 3.12 venv creation;
- build ordering and fresh-cache cu130 installation;
- FlashInfer cubin/JIT-cache alignment;
- environment-verifier version and provenance failures;
- test-suite selection, GPU preflight, and deterministic E2E environment;
- README commands and the exact manual workflow.

Each behavior change follows red-green development: its test is committed only
after it has been observed failing against the original implementation and
passing with the minimal production change.

## Documentation and Operations

`README.md` documents:

1. Exporting `GITHUB_USER`, `GITHUB_EMAIL`, and `GITHUB_TOKEN` in the launching
   shell.
2. Creating or updating `kvcc-github-auth` without putting the token on the
   command line.
3. Applying the pod with `tsh kubectl` and waiting for readiness.
4. Entering the pod and initializing `repo` with `-b cuda-env-fixes`.
5. Running `repo sync`, `bootstrap.sh`, `build-all.sh`, and `test-all.sh`.
6. Inspecting versions and rerunning selected tests.
7. Deleting the pod while retaining the PVC, plus explicit destructive PVC
   cleanup instructions.

The README distinguishes the moving development manifest from an exact source
lock and recommends recording a lock manifest for repeatable benchmark runs.

## Error Handling and Security

- Every script uses `set -euo pipefail`.
- Required tools, directories, environment variables, and GPU indices produce
  explicit errors.
- No GitHub token is written to repository files, pod YAML, logs, or Git remote
  URLs.
- Kubernetes cleanup defaults to retaining the workspace PVC.
- Dependency verification fails before tests when the shared native runtime is
  inconsistent.
- Source-code compatibility failures are out of scope and are not masked by
  manifest scripts.
