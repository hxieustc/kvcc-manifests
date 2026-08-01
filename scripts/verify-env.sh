#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${KVCC_WORKSPACE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="$WORKSPACE_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: workspace Python is missing: $PYTHON" >&2
  exit 1
fi

echo "==> Checking Python and native-runtime compatibility"
"$PYTHON" - "$WORKSPACE_ROOT" <<'PY'
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

workspace = Path(sys.argv[1]).resolve()


def distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required distribution is missing: {name}") from error


def workspace_import(name: str) -> str:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{name} has no import path")
    path = Path(module_file).resolve()
    if not path.is_relative_to(workspace):
        raise RuntimeError(f"{name} resolved outside workspace: {path}")
    return str(path)


if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Python 3.12 required, found {platform.python_version()}")

import torch

if torch.__version__ != "2.13.0+cu130":
    raise RuntimeError(
        f"torch 2.13.0+cu130 required, found {torch.__version__}"
    )
if torch.version.cuda != "13.0":
    raise RuntimeError(
        f"torch CUDA 13.0 required, found {torch.version.cuda}"
    )

torchvision = distribution_version("torchvision")
if torchvision != "0.28.0+cu130":
    raise RuntimeError(
        f"torchvision 0.28.0+cu130 required, found {torchvision}"
    )

flashinfer_python = distribution_version("flashinfer-python")
flashinfer_cubin = distribution_version("flashinfer-cubin")
flashinfer_jit = distribution_version("flashinfer-jit-cache")
if flashinfer_cubin != flashinfer_python:
    raise RuntimeError(
        "FlashInfer cubin mismatch: "
        f"python={flashinfer_python}, cubin={flashinfer_cubin}"
    )
if flashinfer_jit != f"{flashinfer_python}+cu130":
    raise RuntimeError(
        "FlashInfer JIT cache mismatch: "
        f"python={flashinfer_python}, jit={flashinfer_jit}"
    )

records = {
    "cuda": torch.version.cuda,
    "dynamo_import": workspace_import("dynamo.vllm"),
    "flashinfer_cubin": flashinfer_cubin,
    "flashinfer_jit_cache": flashinfer_jit,
    "flashinfer_python": flashinfer_python,
    "kvcc_import": workspace_import("kvcc"),
    "nixl_import": str(Path(importlib.import_module("nixl").__file__).resolve()),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision,
    "vllm_import": workspace_import("vllm"),
    "vllm_native_import": workspace_import("vllm._C_stable_libtorch"),
}
print(json.dumps(records, indent=2, sort_keys=True))
PY

echo "==> Checking installed dependency metadata"
uv pip check --python "$PYTHON"

echo "==> Runtime verification passed"
