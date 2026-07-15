# kvcc-manifests

Google Repo manifest repository for the KVCC KV-cache secondary-tier project.
Tracks the [Dynamo](https://github.com/ai-dynamo/dynamo) router and
[vLLM](https://github.com/mkhazraee/vllm-priv) KVCC integration branches.

## Quick start

```bash
# Install the repo tool if you don't have it
mkdir -p ~/.local/bin
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/.local/bin/repo
chmod +x ~/.local/bin/repo

# Initialise a workspace with the develop manifest
mkdir kvcc-workspace && cd kvcc-workspace
repo init -u git@github.com:hxieustc/kvcc-manifests.git -m manifests/develop.xml
repo sync

# Bootstrap dependencies and build
bash scripts/bootstrap.sh
bash scripts/build-all.sh
```

## Manifests

| File | Purpose |
|---|---|
| `default.xml` | Alias — includes `manifests/develop.xml` |
| `manifests/develop.xml` | Active development branches (dynamo + vllm) |

Add new manifests under `manifests/` (e.g. `manifests/release-YYYY.MM.xml`).
Pin exact SHAs in `locked/` for reproducible builds.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/bootstrap.sh` | Create venv and install all dependencies |
| `scripts/build-all.sh` | Build Dynamo (Rust) and vLLM (C++ extensions) |
| `scripts/test-all.sh` | Run unit tests and KVCC E2E GPU test |

Override GPU indices and E2E toggle via environment variables:

```bash
KVCC_TEST_GPUS=2,3 KVCC_RUN_E2E=1 bash scripts/test-all.sh
```

## Repository layout

```
kvcc-manifests/
├── default.xml              ← repo init default (includes manifests/develop.xml)
├── manifests/
│   └── develop.xml          ← dynamo + vllm development branches
├── locked/                  ← pinned-SHA snapshots (add as needed)
├── scripts/
│   ├── bootstrap.sh
│   ├── build-all.sh
│   └── test-all.sh
└── .github/
    └── workflows/
        └── integration.yml  ← CI (manifest validation + GPU E2E skeleton)
```
