#!/usr/bin/env python3
"""Verify that the ML runtime can use CUDA on Jetson."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


def _load_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to import {name}: {exc}") from exc


def _cuda_probe(torch: Any) -> tuple[bool, str | None]:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return False, "torch.cuda.is_available() is false"
    try:
        sample = torch.ones((64, 64), device="cuda")
        result = sample @ sample
        cuda.synchronize()
        float(result[0, 0].detach().cpu().item())
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit with an error when a real CUDA tensor operation fails.",
    )
    parser.add_argument(
        "--require-jetson",
        action="store_true",
        help="Exit with an error when this is not running on a Jetson rootfs.",
    )
    args = parser.parse_args()

    is_jetson = _is_jetson()
    if args.require_jetson and not is_jetson:
        print("ERROR: /etc/nv_tegra_release was not found; this is not a Jetson rootfs.")
        return 1

    torch = _load_module("torch")
    torchvision = _load_module("torchvision")
    ultralytics = _load_module("ultralytics")
    cuda_available = bool(torch.cuda.is_available())
    cuda_usable, cuda_error = _cuda_probe(torch)

    print(f"Jetson rootfs: {is_jetson}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"Ultralytics: {getattr(ultralytics, '__version__', 'unknown')}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA version: {getattr(torch.version, 'cuda', None)}")
    get_arch_list = getattr(torch.cuda, "get_arch_list", None)
    if callable(get_arch_list):
        print(f"CUDA arch list: {get_arch_list()}")
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            print(
                f"CUDA device {index}: {torch.cuda.get_device_name(index)} "
                f"(CC {capability[0]}.{capability[1]})"
            )
    print(f"CUDA usable: {cuda_usable}")
    if cuda_error is not None:
        print(f"CUDA probe error: {cuda_error}")
    if args.require_cuda and not cuda_usable:
        print(
            "ERROR: CUDA is not usable by PyTorch. On Jetson, install a "
            "PyTorch build that includes kernels for this GPU's compute "
            "capability instead of a CPU-only or incompatible CUDA wheel."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
