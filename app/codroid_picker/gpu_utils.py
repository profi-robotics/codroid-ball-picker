"""
GPU status utilities.

This module provides functions for detecting and reporting GPU availability
and status information for NVIDIA CUDA and Apple MPS backends.
"""

from __future__ import annotations

import logging

import dash_mantine_components as dmc

LOGGER = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_gpu_info() -> dict:
    """Get GPU status information including availability, clock speed, usage, and memory.

    Supports CUDA (NVIDIA), Metal Performance Shaders (Apple Silicon), and CPU fallback.
    """
    info = {
        "available": False,
        "name": "N/A",
        "driver_version": "N/A",
        "cuda_available": False,
        "cuda_version": "N/A",
        "mps_available": False,
        "gpu_count": 0,
        "devices": [],
    }

    # Check CUDA availability via PyTorch
    if TORCH_AVAILABLE:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_count"] = torch.cuda.device_count()
            info["available"] = True

        # Check for Metal Performance Shaders (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["available"] = True
            info["name"] = "Metal Performance Shaders"
            info["gpu_count"] = 1

    # Get detailed GPU info via NVML (NVIDIA only)
    if not NVML_AVAILABLE or not info["cuda_available"]:
        return info

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        info["available"] = device_count > 0

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / (1024 ** 2)
            mem_total_mb = mem_info.total / (1024 ** 2)
            mem_percent = (mem_info.used / mem_info.total) * \
                100 if mem_info.total > 0 else 0

            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0

            # Get clock speeds
            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                )
                clock_mem = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                clock_graphics = 0
                clock_mem = 0

            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = 0

            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(
                    handle) / 1000.0  # mW to W
            except Exception:
                power = 0

            device_info = {
                "index": i,
                "name": name,
                "memory_used_mb": round(mem_used_mb, 1),
                "memory_total_mb": round(mem_total_mb, 1),
                "memory_percent": round(mem_percent, 1),
                "gpu_utilization": gpu_util,
                "clock_graphics_mhz": clock_graphics,
                "clock_memory_mhz": clock_mem,
                "temperature_c": temp,
                "power_watts": round(power, 1),
            }
            info["devices"].append(device_info)

            if i == 0:
                info["name"] = name
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode("utf-8")
                    info["driver_version"] = driver_version
                except Exception:
                    pass

        pynvml.nvmlShutdown()
    except Exception as exc:
        # Only log at debug level for expected failures on non-NVIDIA platforms
        LOGGER.debug(
            "NVML GPU detection failed (expected on non-NVIDIA systems): %s", exc
        )

    return info


def format_gpu_status() -> list:
    """Format GPU status for display in the UI."""
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        return [
            dmc.Alert(
                "No GPU detected (NVIDIA CUDA or Apple MPS not available)",
                color="yellow",
                title="GPU Status",
            )
        ]

    elements = []

    # Overall status - handle both CUDA and MPS
    if gpu_info.get("mps_available"):
        # Apple Silicon MPS
        elements.append(
            dmc.Group(
                [
                    dmc.Badge("MPS: ✓ Available", color="green"),
                    dmc.Badge(f"{gpu_info['name']}", color="blue"),
                    dmc.Badge(
                        f"{gpu_info['gpu_count']} GPU(s)", color="brand"),
                ],
                gap="xs",
            )
        )
    else:
        # NVIDIA CUDA
        cuda_status = "✓ Available" if gpu_info["cuda_available"] else "✗ Not available"
        elements.append(
            dmc.Group(
                [
                    dmc.Badge(
                        f"CUDA: {cuda_status}",
                        color="green" if gpu_info["cuda_available"] else "red",
                    ),
                    dmc.Badge(
                        f"Driver: {gpu_info['driver_version']}", color="blue"),
                    dmc.Badge(
                        f"{gpu_info['gpu_count']} GPU(s)", color="brand"),
                ],
                gap="xs",
            )
        )

    # Per-device info
    for dev in gpu_info["devices"]:
        elements.append(
            dmc.Card(
                [
                    dmc.Text(
                        f"GPU {dev['index']}: {dev['name']}", fw=600, size="sm"),
                    dmc.Group(
                        [
                            dmc.Stack(
                                [
                                    dmc.Text("Memory", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['memory_used_mb']:.0f} / {dev['memory_total_mb']:.0f} MB ({dev['memory_percent']:.1f}%)",
                                        size="sm",
                                    ),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text(
                                        "GPU Usage", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['gpu_utilization']}%", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Clock", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['clock_graphics_mhz']} MHz", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Temp", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['temperature_c']}°C", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text("Power", size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['power_watts']:.1f} W", size="sm"),
                                ],
                                gap=2,
                            ),
                        ],
                        gap="md",
                    ),
                ],
                withBorder=True,
                padding="sm",
                radius="md",
            )
        )

    return elements
