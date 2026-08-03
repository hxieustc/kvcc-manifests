# kvcc-manifests

Google Repo manifests and scripts for synchronizing, building, and testing
customized Dynamo, vLLM, and KVCC together. The `cuda-env-fixes` branch
supports two workflows:

1. Run Repo and the build/test scripts manually on a compatible Linux host or
   container, without Kubernetes.
2. Launch a Kubernetes Pod that runs the complete workflow automatically and
   remains available for interactive development after all tests pass.

Both workflows execute:

```text
repo sync → bootstrap.sh → build-all.sh → test-all.sh
```

The scripts assume the synchronized source revisions are correct. They
reconcile the shared CUDA/Python environment; they do not patch Dynamo, vLLM,
or KVCC source code.

The complete synchronization, bootstrap, build, unit-test, and two-GPU E2E
sequence has been validated through manual execution.

## What the workflow installs

- Python 3.12 in `/opt/kvcc-workspace/.venv`
- customized Dynamo and its Rust/Python bindings
- customized vLLM as an editable install using precompiled native artifacts
- customized KVCC as an editable install
- torch `2.13.0+cu130` and torchvision `0.28.0+cu130`
- matching FlashInfer Python, cubin, and `+cu130` JIT-cache packages

`build-all.sh` verifies package versions, CUDA selection, native imports, and
workspace import provenance before reporting success.

## Workflow 1: Manual execution without Kubernetes

Use this path on a compatible Linux host or an existing container without a
Pod.

### Prerequisites

- Linux on AMD64 with `apt` and `dpkg`
- root access when `bootstrap.sh` needs to install missing system packages
- NVIDIA drivers and at least two visible GPUs for the default E2E test
- `/models-shared` mounted when shared model data is required
- network access to GitHub and the configured Python package indexes
- `GITHUB_USER`, `GITHUB_EMAIL`, and `GITHUB_TOKEN` in the shell environment

Export and validate the GitHub values:

```bash
export GITHUB_USER=hxieustc
export GITHUB_EMAIL="harryx@nvidia.com"
# GITHUB_TOKEN should already be exported by your shell startup configuration.
test -n "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
```

Install Git, Git LFS, curl, and Google Repo:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ca-certificates curl git git-lfs

mkdir -p "$HOME/.local/bin"
curl -fsSL https://storage.googleapis.com/git-repo-downloads/repo \
  -o "$HOME/.local/bin/repo"
chmod 0755 "$HOME/.local/bin/repo"
export PATH="$HOME/.local/bin:$PATH"
git lfs install
```

Configure Git identity and GitHub-scoped HTTPS authorization:

```bash
git config --global user.name "$GITHUB_USER"
git config --global user.email "$GITHUB_EMAIL"
git config --global credential.https://github.com/.username "$GITHUB_USER"
git config --global credential.https://github.com/.email "$GITHUB_EMAIL"

github_basic_auth="$(
  printf '%s:%s' "$GITHUB_USER" "$GITHUB_TOKEN" | base64 | tr -d '\n'
)"
git config --global http.https://github.com/.extraHeader \
  "Authorization: Basic ${github_basic_auth}"
unset github_basic_auth
```

The authorization header is stored in the user’s global Git configuration.
Anyone who can read that file can recover the credential. Remove it when it is
no longer needed:

```bash
git config --global --unset-all http.https://github.com/.extraHeader
```

Create the workspace, synchronize all repositories, and run the complete
workflow:

```bash
mkdir -p /opt/kvcc-workspace
cd /opt/kvcc-workspace

repo init -u https://github.com/hxieustc/kvcc-manifests.git \
  -b cuda-env-fixes \
  -m manifests/develop.xml
repo sync -j8

bash manifests/scripts/bootstrap.sh
bash manifests/scripts/build-all.sh
bash manifests/scripts/test-all.sh
```

No venv activation is required between scripts; every script explicitly
targets `/opt/kvcc-workspace/.venv`.

## Workflow 2: Fully automated Kubernetes Pod

The Pod installs Git, Git LFS, and Repo, configures GitHub authentication,
synchronizes all repositories, bootstraps the environment, builds every
component, and runs all supported tests. It becomes Ready only after the
complete workflow succeeds, then remains running for interactive development.

If any command fails, `set -euo pipefail` terminates the container. The Pod
does not become Ready, and the failing stage remains visible in its logs.

### 1. Prepare the namespace and GitHub Secret

Export credentials in the shell where `tsh kubectl` runs:

```bash
export GITHUB_USER=hxieustc
export GITHUB_EMAIL="harryx@nvidia.com"
test -n "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
```

Create the namespace and pipe the three values into a Kubernetes Secret. This
avoids a temporary credential file and a token-bearing process argument:

```bash
tsh kubectl create namespace kvcc-mbench \
  --dry-run=client -o yaml | tsh kubectl apply -f -

printf 'GITHUB_USER=%s\nGITHUB_EMAIL=%s\nGITHUB_TOKEN=%s\n' \
  "$GITHUB_USER" "$GITHUB_EMAIL" "$GITHUB_TOKEN" | \
  tsh kubectl -n kvcc-mbench create secret generic kvcc-github-auth \
    --from-env-file=/dev/stdin \
    --dry-run=client -o yaml | \
  tsh kubectl apply -f -
```

The Pod writes a GitHub-scoped Basic authorization header to its ephemeral
`/root/.gitconfig`. The token is not stored in the Pod manifest, a Git remote
URL, or a local process argument. Anyone who can read the Secret or exec into
the Pod can access it.

### 2. Verify the cluster and persistent model cache

Confirm the Teleport context and the pre-existing shared model PVC:

```bash
tsh status
tsh kubectl config current-context
tsh kubectl -n kvcc-mbench get pvc shared-model-cache
```

`shared-model-cache` must be Bound. The Pod mounts it at `/models-shared`.
The manifest does not create or delete this PVC.

### 3. Recreate and launch the Pod

Kubernetes does not allow adding or removing containers, volumes, or mounts
from an existing Pod. Delete any previous `kvcc-dev` Pod before applying a
changed manifest:

```bash
tsh kubectl -n kvcc-mbench delete pod kvcc-dev \
  --ignore-not-found --wait=true
tsh kubectl apply -f kubernetes/dev-pod.yaml
```

The Pod requests:

```text
image:          nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
arch:           AMD64
GPU:            2 × NVIDIA B200
/dev/shm:       64 GiB memory-backed emptyDir
workspace:      /opt/kvcc-workspace, container-local and ephemeral
model storage:  /models-shared, persistent shared-model-cache PVC
```

### 4. Monitor the automated workflow

Follow installation, synchronization, build, and test output:

```bash
tsh kubectl -n kvcc-mbench logs -f pod/kvcc-dev
```

The container remains alive after success, so press Ctrl-C after this message:

```text
==> Automated workflow passed; pod is ready for development
```

In another terminal, wait for the final readiness marker:

```bash
tsh kubectl -n kvcc-mbench wait \
  --for=condition=Ready pod/kvcc-dev --timeout=8h
```

If the Pod fails or does not become Ready, inspect the failing stage:

```bash
tsh kubectl -n kvcc-mbench get pod kvcc-dev
tsh kubectl -n kvcc-mbench describe pod kvcc-dev
tsh kubectl -n kvcc-mbench logs pod/kvcc-dev
```

### 5. Enter the validated development environment

After the Pod is Ready:

```bash
tsh kubectl -n kvcc-mbench exec -it kvcc-dev -- bash
cd /opt/kvcc-workspace
```

The synchronized layout is:

```text
/opt/kvcc-workspace/
├── manifests/
├── dynamo/
├── vllm/
├── kvcc/
└── .venv/
```

The branch manifest tracks:

- Dynamo `oandreeva/router_hints`
- vLLM `kvcc_repo`
- KVCC `main`
- this manifest repository’s `cuda-env-fixes` branch

For benchmark reproducibility, record an exact lock after a known-green sync:

```bash
mkdir -p manifests/locked
repo manifest -r -o manifests/locked/cuda-env-fixes.lock.xml
```

## Test scope and useful overrides

`test-all.sh` runs:

1. KVCC standalone unit tests.
2. vLLM/KVCC configuration, harness, and tiering unit tests.
3. focused Dynamo/vLLM router integration tests.
4. the two-GPU KVCC E2E test.

DeepGEMM autotuning is skipped for E2E workers because it is unrelated to KVCC
correctness and can exceed worker startup time. The script verifies requested
GPU indices before starting E2E.

The following overrides apply to manual script invocations. To use them during
automated Pod startup, add matching `env` entries to the workspace container.

Run CPU and focused router tests without E2E:

```bash
KVCC_RUN_E2E=0 bash manifests/scripts/test-all.sh
```

Select different visible GPU indices:

```bash
KVCC_TEST_GPUS=2,3 bash manifests/scripts/test-all.sh
```

Enable vLLM pre-commit hooks:

```bash
KVCC_INSTALL_PRECOMMIT=1 bash manifests/scripts/bootstrap.sh
```

Override the runtime matrix only when intentionally testing other packages:

```bash
KVCC_TORCH_VERSION=2.13.0+cu130 \
KVCC_TORCHVISION_VERSION=0.28.0+cu130 \
KVCC_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  bash manifests/scripts/build-all.sh
```

## Day-to-day development

Inside a Ready Pod or a manually prepared workspace:

```bash
cd /opt/kvcc-workspace
repo sync -j8
bash manifests/scripts/bootstrap.sh
bash manifests/scripts/build-all.sh
bash manifests/scripts/test-all.sh
```

Python source packages use editable installs. Python-only changes generally
need only a worker restart and test rerun. Re-run `build-all.sh` after native
vLLM, Dynamo Rust, or dependency metadata changes.

## Manifest repository tests

Run local fixture and contract tests from a manifest repository checkout:

```bash
bash tests/run.sh
```

These tests do not download CUDA packages or require GPUs. They validate the
Pod contract, bootstrap behavior, clean-cache dependency resolution, runtime
verification failures, and test orchestration.

## Storage and cleanup

`/opt/kvcc-workspace` uses the Pod’s writable container layer and is
ephemeral. Deleting the Pod permanently removes synchronized sources, the
venv, and compiler caches. `/models-shared` is backed by the
`shared-model-cache` PVC and survives Pod deletion.

Delete the Pod:

```bash
tsh kubectl -n kvcc-mbench delete pod kvcc-dev
```

Delete the credential Secret when it is no longer needed:

```bash
tsh kubectl -n kvcc-mbench delete secret kvcc-github-auth
```

Do not delete `shared-model-cache` as part of normal cleanup.
