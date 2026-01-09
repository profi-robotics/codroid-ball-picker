"""Shared UI helpers used by multiple calibration UI modules."""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dash import html
import dash_mantine_components as dmc

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

from .constants import DEFAULT_MODEL_DIR

LOGGER = logging.getLogger(__name__)


def _format_device_list(devices: list[object]) -> html.Ul:
    if not devices:
        return html.Ul([html.Li("No OAK-D devices detected.")])
    items = []
    for idx, device in enumerate(devices, start=1):
        name = None
        mx_id_fn = getattr(device, "getMxId", None)
        device_id_fn = getattr(device, "getDeviceId", None)
        if callable(mx_id_fn):
            name = mx_id_fn()
        elif callable(device_id_fn):
            name = device_id_fn()
        else:
            name = str(device)
        state = getattr(device, "state", None)
        items.append(html.Li(f"{idx}. {name} ({state})"))
    return html.Ul(items)


def _placeholder_frame(message: str, size: Tuple[int, int] = (720, 1280)) -> np.ndarray:
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return frame


def _encode_frame(frame: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode camera frame.")
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _list_model_files(output_dir: Path = DEFAULT_MODEL_DIR) -> list[Path]:
    if not output_dir.exists():
        return []
    files = [path for path in output_dir.iterdir() if path.is_file()]
    return sorted(files, key=lambda path: path.name.lower())


def _model_options() -> list[dict]:
    options = []
    for path in _list_model_files():
        options.append({"value": str(path), "label": path.name})
    return options


def get_gpu_info() -> dict:
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

    if TORCH_AVAILABLE:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_count"] = torch.cuda.device_count()
            info["available"] = True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["available"] = True
            info["name"] = "Metal Performance Shaders"
            info["gpu_count"] = 1

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

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem_info.used / (1024 ** 2)
            mem_total_mb = mem_info.total / (1024 ** 2)
            mem_percent = (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = 0

            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS)
                clock_mem = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                clock_graphics = 0
                clock_mem = 0

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
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
        LOGGER.debug("NVML GPU detection failed (expected on non-NVIDIA systems): %s", exc)

    return info


def format_gpu_status() -> list:
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        return [
            dmc.Alert(
                "No NVIDIA GPU detected or NVML not available",
                color="yellow",
                title="GPU Status",
            )
        ]

    elements = []
    cuda_status = "✓ Available" if gpu_info["cuda_available"] else "✗ Not available"
    elements.append(
        dmc.Group(
            [
                dmc.Text("CUDA", size="sm", fw=600),
                dmc.Text(cuda_status, size="sm"),
            ],
            position="apart",
        )
    )

    if gpu_info["devices"]:
        for device in gpu_info["devices"]:
            elements.append(
                dmc.Stack(
                    [
                        dmc.Text(device["name"], fw=600, size="sm"),
                        dmc.Text(
                            f"Memory: {device['memory_used_mb']} / {device['memory_total_mb']} MB", size="xs", c="dimmed"
                        ),
                        dmc.Text(
                            f"GPU: {device['gpu_utilization']}% · Temp: {device['temperature_c']}°C", size="xs", c="dimmed"
                        ),
                        dmc.Text(
                            f"Clock: {device['clock_graphics_mhz']} MHz · Power: {device['power_watts']} W", size="xs", c="dimmed"
                        ),
                    ],
                    gap=2,
                )
            )

    return elements
