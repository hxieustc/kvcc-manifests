#!/usr/bin/env python3
"""Verify source provenance for the selectively customized runtime image."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/opt/kvcc-workspace"),
    )
    return parser.parse_args()


def module_path(name: str) -> Path:
    module = importlib.import_module(name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"{name} has no import path")
    return Path(module_file).resolve()


def is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> None:
    workspace = parse_args().workspace.resolve()
    dynamo_path = module_path("dynamo.vllm")
    kvcc_path = module_path("kvcc")
    vllm_path = module_path("vllm")
    native_path = module_path("vllm._C_stable_libtorch")

    for name, path in (("dynamo.vllm", dynamo_path), ("kvcc", kvcc_path)):
        if not is_below(path, workspace):
            raise RuntimeError(
                f"{name} did not resolve from synchronized source: {path}"
            )

    if is_below(vllm_path, workspace):
        raise RuntimeError(
            f"vLLM was unexpectedly installed from source: {vllm_path}"
        )
    if is_below(native_path, workspace):
        raise RuntimeError(
            "vLLM native code was unexpectedly installed from source: "
            f"{native_path}"
        )

    overlay = vllm_path.parent / "v1" / "kv_offload" / "tiering" / "kvcc"
    if not overlay.is_dir():
        raise RuntimeError(f"vLLM KVCC overlay is missing: {overlay}")

    print(f"Dynamo source: {dynamo_path}")
    print(f"base vLLM: {vllm_path}")
    print(f"base vLLM native module: {native_path}")
    print(f"vLLM KVCC overlay: {overlay}")
    print(f"KVCC source: {kvcc_path}")


if __name__ == "__main__":
    main()
