"""
GPU status utilities.

This module provides functions for detecting and reporting GPU availability
and status information for NVIDIA CUDA, Apple MPS, and NVIDIA Tegra (Jetson) backends.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import dash_mantine_components as dmc
from app.codroid_picker.i18n import t

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


def _detect_jetson_gpu() -> Optional[dict]:
    """Detect NVIDIA Jetson/Tegra GPU via tegrastats or sysfs.

    Returns a dict with Jetson GPU info if detected, None otherwise.
    """
    tegra_release = Path("/etc/nv_tegra_release")
    if not tegra_release.exists():
        return None

    info = {
        "available": True,
        "jetson": True,
        "name": "NVIDIA Tegra GPU",
        "driver_version": "N/A",
        "cuda_available": True,  # Jetson has CUDA even if PyTorch is CPU-only
        "cuda_version": "N/A",
        "gpu_count": 1,
        "devices": [],
    }

    # Parse tegra release for version info
    try:
        release_text = tegra_release.read_text()
        # Example: "# R36 (release), REVISION: 4.7, ..."
        match = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", release_text)
        if match:
            info["driver_version"] = f"L4T R{match.group(1)}.{match.group(2)}"
    except Exception:
        pass

    # Try to find CUDA version from /usr/local/cuda
    cuda_version_file = Path("/usr/local/cuda/version.json")
    if cuda_version_file.exists():
        try:
            import json
            cuda_data = json.loads(cuda_version_file.read_text())
            info["cuda_version"] = cuda_data.get("cuda", {}).get("version", "N/A")
        except Exception:
            pass

    # Get GPU stats from tegrastats (one-shot)
    device_info = {
        "index": 0,
        "name": "Tegra GPU",
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "memory_percent": 0,
        "gpu_utilization": 0,
        "clock_graphics_mhz": 0,
        "clock_memory_mhz": 0,
        "temperature_c": 0,
        "power_watts": 0,
    }

    # Detect Jetson model from device tree first (fast, no subprocess)
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            model_name = model_path.read_text().strip().rstrip("\x00")
            device_info["name"] = model_name
            info["name"] = model_name
        except Exception:
            pass

    try:
        # Use timeout shell command to get one line from tegrastats
        # tegrastats runs indefinitely, so we use 'timeout' to kill it after one output
        result = subprocess.run(
            ["timeout", "0.5", "tegrastats", "--interval", "200"],
            capture_output=True,
            text=True,
            timeout=2.0,  # Safety timeout for subprocess.run itself
        )
        line = result.stdout.strip().split("\n")[0] if result.stdout else ""

        # Parse RAM: "RAM 1967/7620MB"
        ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if ram_match:
            device_info["memory_used_mb"] = int(ram_match.group(1))
            device_info["memory_total_mb"] = int(ram_match.group(2))
            if device_info["memory_total_mb"] > 0:
                device_info["memory_percent"] = round(
                    100 * device_info["memory_used_mb"] / device_info["memory_total_mb"], 1
                )

        # Parse GR3D_FREQ: "GR3D_FREQ 0%"
        gr3d_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if gr3d_match:
            device_info["gpu_utilization"] = int(gr3d_match.group(1))

        # Parse gpu temperature: "gpu@49.125C"
        gpu_temp_match = re.search(r"gpu@([\d.]+)C", line)
        if gpu_temp_match:
            device_info["temperature_c"] = int(float(gpu_temp_match.group(1)))

        # Parse power: "VDD_IN 5055mW/5055mW" or "VDD_GPU_SOC ..."
        power_match = re.search(r"VDD_IN\s+(\d+)mW", line)
        if power_match:
            device_info["power_watts"] = round(int(power_match.group(1)) / 1000, 1)

    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        # tegrastats not available, use sysfs fallback
        try:
            gpu_load_path = Path("/sys/devices/gpu.0/load")
            if gpu_load_path.exists():
                load = int(gpu_load_path.read_text().strip())
                device_info["gpu_utilization"] = load // 10  # load is 0-1000
        except Exception:
            pass
    except Exception as exc:
        LOGGER.debug("Jetson GPU stats failed: %s", exc)

    info["devices"].append(device_info)
    return info


def get_gpu_info() -> dict:
    """Get GPU status information including availability, clock speed, usage, and memory.

    Supports CUDA (NVIDIA), Metal Performance Shaders (Apple Silicon), Jetson/Tegra, and CPU fallback.
    """
    info = {
        "available": False,
        "name": "N/A",
        "driver_version": "N/A",
        "cuda_available": False,
        "cuda_version": "N/A",
        "mps_available": False,
        "jetson": False,
        "gpu_count": 0,
        "devices": [],
    }

    # Check for Jetson/Tegra first (may have CPU-only PyTorch but still has GPU)
    jetson_info = _detect_jetson_gpu()
    if jetson_info:
        return jetson_info

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
                "No GPU detected (NVIDIA CUDA, Jetson/Tegra, or Apple MPS not available)",
                color="yellow",
                title="GPU Status",
            )
        ]

    elements = []

    # Overall status - handle Jetson, CUDA, and MPS
    if gpu_info.get("jetson"):
        # NVIDIA Jetson/Tegra
        cuda_version = gpu_info.get("cuda_version", "N/A")
        elements.append(
            dmc.Group(
                [
                    dmc.Badge("Jetson: ✓ Detected", color="green"),
                    dmc.Badge(f"CUDA: {cuda_version}", color="teal"),
                    dmc.Badge(f"{gpu_info['driver_version']}", color="blue"),
                ],
                gap="xs",
            )
        )
    elif gpu_info.get("mps_available"):
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
        # For Jetson, show RAM as shared memory
        memory_label = t("RAM (shared)") if gpu_info.get("jetson") else t("Memory")
        elements.append(
            dmc.Card(
                [
                    dmc.Text(
                        t("GPU {index}: {name}", index=dev["index"], name=dev["name"]), fw=600, size="sm"),
                    dmc.Group(
                        [
                            dmc.Stack(
                                [
                                    dmc.Text(memory_label, size="xs", c="dimmed"),
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
                                        t("GPU Usage"), size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['gpu_utilization']}%", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text(t("Clock"), size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['clock_graphics_mhz']} MHz", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text(t("Temp"), size="xs", c="dimmed"),
                                    dmc.Text(
                                        f"{dev['temperature_c']}°C", size="sm"),
                                ],
                                gap=2,
                            ),
                            dmc.Stack(
                                [
                                    dmc.Text(t("Power"), size="xs", c="dimmed"),
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
