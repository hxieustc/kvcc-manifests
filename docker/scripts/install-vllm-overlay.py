#!/usr/bin/env python3
"""Replace the installed vLLM KVCC tiering backend with synchronized source."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/opt/kvcc-vllm-overlay/kvcc"),
        help="staged vLLM v1/kv_offload/tiering/kvcc directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise SystemExit(f"vLLM KVCC overlay source is missing: {source}")

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("the base image's vLLM package is not importable")

    vllm_package = Path(spec.origin).resolve().parent
    destination = vllm_package / "v1" / "kv_offload" / "tiering" / "kvcc"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"Installed vLLM KVCC overlay: {source} -> {destination}")


if __name__ == "__main__":
    main()
