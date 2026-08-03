# kvcc-manifests

Google Repo manifests and scripts for building and testing customized Dynamo,
vLLM, and KVCC together. The `cuda-env-fixes` branch provides a manual
Kubernetes workflow based on NVIDIA's Dynamo vLLM runtime image.

The supported workflow is:

```text
launch pod → repo sync → bootstrap.sh → build-all.sh → test-all.sh
```

The scripts assume the synchronized Dynamo, vLLM, and KVCC source revisions are
correct. They reconcile the shared CUDA/Python environment; they do not patch
source repositories.

## What the workflow installs

- Python 3.12 in `/opt/kvcc-workspace/.venv`
- customized Dynamo and its Rust/Python bindings
- customized vLLM as an editable install using precompiled native artifacts
- customized KVCC as an editable install
- torch `2.13.0+cu130` and torchvision `0.28.0+cu130`
- matching FlashInfer Python, cubin, and `+cu130` JIT-cache packages

`build-all.sh` verifies package versions, CUDA selection, native imports, and
workspace import provenance before it reports success.

## 1. Prepare GitHub credentials

The private repositories are synchronized over HTTPS. Export these values in
the shell where you run `tsh kubectl`:

```bash
export GITHUB_USER=hxieustc
export GITHUB_EMAIL="harryx@nvidia.com"
# GITHUB_TOKEN should already be exported by your shell startup configuration.
test -n "${GITHUB_TOKEN:?GITHUB_TOKEN is required}"
```

Create the namespace and pipe the three values directly into a Kubernetes
Secret. This avoids both a temporary credential file and a token-bearing
process argument:

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

The pod reads the Secret into its environment at startup and writes a
GitHub-scoped Basic authorization header to its ephemeral `/root/.gitconfig`.
The token is not stored in the pod manifest, a Git remote URL, or a local
process argument. Anyone who can read the Secret or exec into the pod can
access the credential.

## 2. Launch the Dynamo development pod

Check that Teleport points at the intended cluster, then apply the manifest:

```bash
tsh status
tsh kubectl config current-context
tsh kubectl apply -f kubernetes/dev-pod.yaml
tsh kubectl -n kvcc-mbench wait \
  --for=condition=Ready pod/kvcc-dev --timeout=30m
```

The pod requests:

```text
image:      nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
arch:       AMD64
GPU:        2 × NVIDIA B200
/dev/shm:   64 GiB
workspace:  container-local ephemeral storage
```

Before the pod becomes Ready, the main container installs Git and Git LFS and
downloads Google's `repo` tool to `/usr/local/bin/repo`. `bootstrap.sh`
installs any remaining system build prerequisites, uv, and Rust as needed.

## 3. Synchronize the repositories

Open a shell in the pod:

```bash
tsh kubectl -n kvcc-mbench exec -it kvcc-dev -- bash
```

Inside the pod, initialize this branch and synchronize all projects:

```bash
mkdir -p /opt/kvcc-workspace
cd /opt/kvcc-workspace

repo init -u https://github.com/hxieustc/kvcc-manifests.git \
  -b cuda-env-fixes \
  -m manifests/develop.xml
repo sync -j8
```

The workspace layout is:

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
- this manifest repository's `cuda-env-fixes` branch

For benchmark reproducibility, record an exact lock after a known-green sync:

```bash
mkdir -p manifests/locked
repo manifest -r -o manifests/locked/cuda-env-fixes.lock.xml
```

## 4. Bootstrap, build, and test

Run the three scripts manually from `/opt/kvcc-workspace`:

```bash
bash manifests/scripts/bootstrap.sh
bash manifests/scripts/build-all.sh
bash manifests/scripts/test-all.sh
```

No venv activation is required between scripts; each script explicitly targets
`/opt/kvcc-workspace/.venv`.

`test-all.sh` runs:

1. KVCC standalone unit tests.
2. vLLM/KVCC configuration, harness, and tiering unit tests.
3. focused Dynamo/vLLM router integration tests.
4. the two-GPU KVCC E2E test.

DeepGEMM autotuning is skipped for the E2E workers because it is unrelated to
KVCC correctness and can exceed worker startup time. The script verifies the
requested GPU indices before starting E2E.

### Useful overrides

Run CPU and focused router tests without E2E:

```bash
KVCC_RUN_E2E=0 bash manifests/scripts/test-all.sh
```

Select different visible GPU indices:

```bash
KVCC_TEST_GPUS=2,3 bash manifests/scripts/test-all.sh
```

Enable vLLM pre-commit hook installation during bootstrap:

```bash
KVCC_INSTALL_PRECOMMIT=1 bash manifests/scripts/bootstrap.sh
```

Override package indexes or versions only when intentionally testing another
runtime matrix:

```bash
KVCC_TORCH_VERSION=2.13.0+cu130 \
KVCC_TORCHVISION_VERSION=0.28.0+cu130 \
KVCC_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  bash manifests/scripts/build-all.sh
```

## Day-to-day development

Refresh all branches and rebuild:

```bash
cd /opt/kvcc-workspace
repo sync -j8
bash manifests/scripts/bootstrap.sh
bash manifests/scripts/build-all.sh
bash manifests/scripts/test-all.sh
```

Python source packages are editable installs. Python-only changes generally
need only a worker restart and a test rerun. Re-run `build-all.sh` after native
vLLM, Dynamo Rust, or dependency metadata changes.

## Manifest repository tests

Run the shell fixture and contract tests from a local checkout:

```bash
bash tests/run.sh
```

These tests do not download CUDA packages or require GPUs. They validate the
pod contract, bootstrap behavior, clean-cache dependency resolution, runtime
verification failures, and test orchestration.

## Inspect and clean up

Inspect pod state:

```bash
tsh kubectl -n kvcc-mbench get pod
tsh kubectl -n kvcc-mbench describe pod kvcc-dev
```

The workspace is container-local and ephemeral. Deleting the pod permanently
removes synchronized sources, the venv, model caches, and compiler caches:

```bash
tsh kubectl -n kvcc-mbench delete pod kvcc-dev
```

Delete the credential Secret when it is no longer needed:

```bash
tsh kubectl -n kvcc-mbench delete secret kvcc-github-auth
```
