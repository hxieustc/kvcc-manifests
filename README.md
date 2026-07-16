# kvcc-manifests

Google Repo manifest repository for the KVCC KV-cache secondary-tier project.
Tracks the [Dynamo](https://github.com/ai-dynamo/dynamo) router and
[vLLM](https://github.com/mkhazraee/vllm-priv) KVCC integration branches.

## Quick start — full dev cycle from scratch

### 1. Install system tools

```bash
# Google repo tool
mkdir -p ~/.local/bin
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/.local/bin/repo
chmod +x ~/.local/bin/repo
# Make sure ~/.local/bin is on your PATH
export PATH="$HOME/.local/bin:$PATH"

# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# System dev libraries
sudo apt-get install -y \
  build-essential libhwloc-dev libudev-dev pkg-config \
  libclang-dev protobuf-compiler python3-dev cmake
```

### 2. Create workspace and sync sources

```bash
mkdir kvcc-workspace && cd kvcc-workspace
repo init -u git@github.com:hxieustc/kvcc-manifests.git -m manifests/develop.xml
repo sync -j8
```

### 3. Bootstrap the build environment

```bash
bash scripts/bootstrap.sh
```

This checks prerequisites, creates `.venv` with Python 3.12, installs
`maturin`, `pandas`, `pre-commit`, and hooks pre-commit into the vLLM repo.

### 4. Build and install Dynamo and vLLM

If your CUDA version is **13.0 or newer**, set this before building — the
`cudarc` Rust crate requires it:

```bash
export CUDARC_CUDA_VERSION=13000
```

Then build:

```bash
bash scripts/build-all.sh
```

This installs Dynamo (Rust bindings via `maturin` + Python packages) first,
then vLLM (pre-compiled wheel, editable install). Order matters: Dynamo's
`[vllm]` extras would overwrite the editable vLLM install if built second.

### 5. Activate the venv and run tests

```bash
source .venv/bin/activate

# CPU-only unit tests (no GPU required)
python -m pytest vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e/test_config.py \
                 vllm/tests/v1/kv_offload/kvcc-tests/kvcc-e2e/test_harness.py -q

# Full GPU end-to-end test
bash scripts/test-all.sh
```

### 6. Day-to-day development

After the initial setup, the typical cycle is:

```bash
# Pull the latest changes across all repos
repo sync -j8

# Rebuild only the component you changed
# (Dynamo Rust change)
cd dynamo/lib/bindings/python && maturin develop --uv && cd -
# (vLLM Python-only change — editable install, no rebuild needed)

# Run tests
bash scripts/test-all.sh
```

---

## Repo commands for develop

### Sync

```bash
# Sync all projects (fetch + checkout tracked branches)
repo sync -j8

# Sync a single project
repo sync dynamo
repo sync vllm
```

### Branching

```bash
# Create a local topic branch across all projects
repo start my-feature --all

# Create a topic branch in one project only
repo start my-feature dynamo
```

### Status and diff

```bash
# Show working-tree status across all projects
repo status

# Diff all uncommitted changes
repo diff

# Diff staged changes only
repo diff --cached
```

### Committing and uploading for review

```bash
# Commit in a specific project as usual
cd dynamo && git add -p && git commit && cd -

# Upload a branch for Gerrit code review (if using Gerrit)
repo upload --cbr
```

### Switching manifests

```bash
# Re-initialise with a different manifest (e.g. a release snapshot)
repo init -m manifests/release-2026.07.xml
repo sync -j8
```

### Pinning the current state (lockfile)

```bash
# Write a snapshot of all current SHAs to locked/
repo manifest -r -o locked/develop-$(date +%Y.%m.%d).lock.xml
```

---

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
| `scripts/bootstrap.sh` | Prereq checks, venv creation, build-tool pip installs, pre-commit |
| `scripts/build-all.sh` | Build Dynamo (Rust + Python) then vLLM (editable) |
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
