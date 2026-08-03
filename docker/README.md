# Customized Dynamo + vLLM + KVCC Runtime

This project turns the source revisions tracked by
[`hxieustc/kvcc-manifests@cuda-env-fixes`](https://github.com/hxieustc/kvcc-manifests/blob/cuda-env-fixes/README.md)
into either:

- one immutable, multi-stage container image; or
- a persistent Kubernetes development pod on an AMD64 NVIDIA B200 node.

Both paths start from NVIDIA's Dynamo vLLM runtime so they reuse its CUDA,
Torch, NIXL, UCX, and FlashInfer stack. The immutable-image path selectively
replaces only what is customized:

```text
build customized Dynamo → overlay vLLM's KVCC backend → install KVCC
```

The base image's vLLM distribution and native CUDA extensions remain in
place. Dynamo is rebuilt because its customized branch contains Rust/native
changes; vLLM is not rebuilt because its customized runtime changes are
limited to `vllm/v1/kv_offload/tiering/kvcc`.

## Credentials

Set the GitHub identity in the shell that launches the operation:

```bash
export GITHUB_USER=hxieustc
export GITHUB_EMAIL="harryx@nvidia.com"

# GITHUB_TOKEN is expected to be exported by ~/.zshrc.
test -n "${GITHUB_TOKEN:-}"
```

`GITHUB_TOKEN` must not be committed, copied into the Docker build context, or
written into the Kubernetes manifest. The image build supplies it through a
BuildKit secret. The Kubernetes launcher creates a namespaced Secret from a
temporary mode-0600 env file and deletes that file on exit.

## Source Revisions

Google's `repo` tool reads `manifests/develop.xml`, which currently selects:

| Workspace path | Repository | Revision |
|---|---|---|
| `manifests/` | `hxieustc/kvcc-manifests` | `cuda-env-fixes` |
| `dynamo/` | `ai-dynamo/dynamo` | `oandreeva/router_hints` |
| `vllm/` | `mkhazraee/vllm-priv` | `kvcc_repo` |
| `kvcc/` | `NVIDIA-dev/kvcc` | `main` |

The installer does not duplicate these revisions. Changing the manifest and
running `repo sync` changes the synchronized source state.

## Build an Immutable Image

Prerequisites:

- Docker with BuildKit/buildx;
- access to `nvcr.io`;
- access to the private GitHub repositories.

Run:

```bash
scripts/build-image.sh
```

Defaults:

```text
base image: nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest
platform:   linux/amd64
output:     kvcc-custom-runtime:dev
```

Override them when needed:

```bash
export DYNAMO_BASE_IMAGE="nvcr.io/nvidia/ai-dynamo/vllm-runtime-nightly:latest"
export KVCC_PLATFORM="linux/amd64"
export KVCC_IMAGE="my-registry.example/kvcc-custom-runtime:$(date +%Y%m%d)"
scripts/build-image.sh
```

The build synchronizes all three source repositories, builds Dynamo's
Rust/Python bindings, installs Dynamo and KVCC, and stages only the customized
vLLM KVCC backend. The final stage replaces that one backend directory in the
base vLLM package. It does not apply the legacy patch files, install vLLM from
source, install vLLM test dependencies, or run `build-all.sh`/`test-all.sh`.

Before installing KVCC, the builder records constraints for the base image's
Torch, torchvision, vLLM, Triton, and FlashInfer packages. Dependency
resolution must retain those versions, preventing the KVCC install from
silently replacing the CUDA runtime stack inherited from the base image.

The builder caches uv packages, Cargo dependencies, Cargo targets, and ccache
objects. The final stage starts again from the same Dynamo base and copies the
customized environment and synchronized sources.

Before production use, replace the floating base tag with the digest recorded
after a successful validation.

## Launch the Kubernetes Development Pod

The active Teleport context must point to the intended cluster:

```bash
tsh status
tsh kubectl config current-context
```

Apply the Secret, tool and patch ConfigMaps, 200 GiB VAST PVC, and two-GPU
development pod. The default `KVCC_WHOLE_TEST=1` mode runs `repo sync`, applies
the checked-in patches, bootstraps, builds every project, verifies the resolved
environment, and runs the unit and E2E tests:

```bash
scripts/apply-pod.sh
```

The launcher uses only `tsh kubectl`. It targets:

```text
namespace: kvcc-mbench
pod:       kvcc-dev
PVC:       kvcc-workspace
storage:   vast (ReadWriteMany)
GPU:       2 × NVIDIA B200
arch:      AMD64
```

Inspect state:

```bash
tsh kubectl -n kvcc-mbench get pod,pvc
tsh kubectl -n kvcc-mbench describe pod kvcc-dev
```

Open an interactive shell:

```bash
scripts/exec-pod.sh
```

The synchronized workspace and venv are at:

```text
/opt/kvcc-workspace
/opt/kvcc-workspace/.venv
```

## Incremental Development

Inside the pod, rerun the shared installer after upstream updates. This
selective installer rebuilds Dynamo, restages the vLLM KVCC backend, and
installs KVCC; it does not install vLLM or run the test suites:

```bash
bash /opt/kvcc-tools/sync-build.sh
```

To repeat the exact bootstrap, build-all, and test-all phases after a sync:

```bash
bash /opt/kvcc-tools/test-all-in-pod.sh
```

For only a Dynamo Rust binding change:

```bash
cd /opt/kvcc-workspace/dynamo/lib/bindings/python
maturin develop --uv
```

KVCC and Dynamo are editable installs. For vLLM development, edit the synced
source and rerun the overlay installer (or rebuild the image) before restarting
the worker; the base vLLM package itself is not editable.

## Verification and Tests

Static artifact validation:

```bash
bash tests/validate-artifacts.sh
```

Verify the live dependency environment:

```bash
tsh kubectl -n kvcc-mbench exec kvcc-dev -- \
  bash /opt/kvcc-tools/verify-environment.sh

tsh kubectl -n kvcc-mbench exec kvcc-dev -- \
  cat /opt/kvcc-workspace/build-provenance.txt
```

Run KVCC CPU tests:

```bash
tsh kubectl -n kvcc-mbench exec kvcc-dev -- bash -lc \
  'source /opt/kvcc-workspace/.venv/bin/activate &&
   python -m pytest /opt/kvcc-workspace/kvcc/tests -q'
```

Run the manifest repository's selected vLLM KVCC tests:

```bash
tsh kubectl -n kvcc-mbench exec kvcc-dev -- bash -lc \
  'source /opt/kvcc-workspace/.venv/bin/activate &&
   KVCC_RUN_E2E=0 bash /opt/kvcc-workspace/manifests/scripts/test-all.sh'
```

Run the full two-GPU E2E suite explicitly:

```bash
tsh kubectl -n kvcc-mbench exec kvcc-dev -- bash -lc \
  'source /opt/kvcc-workspace/.venv/bin/activate &&
   KVCC_TEST_GPUS=0,1 KVCC_RUN_E2E=1
   bash /opt/kvcc-workspace/manifests/scripts/test-all.sh'
```

## Compatibility Behavior

The immutable image does not hard-code Torch, vLLM, Triton, torchvision, or
FlashInfer versions. It records the versions installed in the selected Dynamo
base image, supplies them as resolver constraints, and verifies that KVCC
installation did not replace them. The runtime-stage provenance check requires:

- customized Dynamo and KVCC imports from the synchronized workspace;
- the vLLM Python package and native extension from the base image; and
- the customized KVCC backend under the installed vLLM tiering package.

The full-build Kubernetes development workflow remains independent. Its
`verify-environment.sh` checks the exact CUDA package versions produced by
`build-all.sh`, because that workflow intentionally installs customized vLLM
and its test dependency set instead of preserving the base vLLM package.

## Cleanup

Delete only the pod:

```bash
scripts/delete-pod.sh
```

The PVC is retained by this command so source checkouts, compiler caches, and
the venv survive pod recreation.

To remove all persistent development state later, do so explicitly:

```bash
tsh kubectl -n kvcc-mbench delete pvc kvcc-workspace
tsh kubectl -n kvcc-mbench delete secret kvcc-github-auth
tsh kubectl -n kvcc-mbench delete configmap kvcc-tools
tsh kubectl -n kvcc-mbench delete configmap kvcc-patches
```

Deleting the PVC destroys the synchronized workspace and build cache.
