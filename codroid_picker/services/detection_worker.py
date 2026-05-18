"""YOLO worker process entrypoint.

The worker imports Ultralytics only after receiving an activation command.
"""

from __future__ import annotations

import time
import signal
from importlib import import_module
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Empty
from typing import Any, cast

import numpy as np


def _labels(model: object) -> list[str]:
    names = getattr(model, "names", None)
    if callable(names):
        names = names()
    if isinstance(names, dict):
        return [str(label) for _idx, label in sorted(names.items(), key=lambda item: int(item[0]))]
    if isinstance(names, (list, tuple)):
        return [str(label) for label in names]
    return []


def _detections_from_results(results: Any, model: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    names = getattr(model, "names", {}) or {}
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        for box, conf, cls_idx in zip(xyxy, confs, clss):
            cls_int = int(cls_idx)
            label = names.get(cls_int, str(cls_int)) if isinstance(names, dict) else str(cls_int)
            x1, y1, x2, y2 = [float(value) for value in box]
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    "label": str(label),
                    "class_id": cls_int,
                    "confidence": float(conf),
                }
            )
    return detections


def _select_inference_device() -> str:
    """Prefer CUDA only when the installed PyTorch build can run kernels."""
    is_jetson = Path("/etc/nv_tegra_release").exists()
    try:
        torch = import_module("torch")
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            if is_jetson:
                raise RuntimeError(
                    "CUDA is not available to PyTorch on this Jetson. "
                    "Install a Jetson-compatible CUDA PyTorch build."
                )
            return "cpu"
        if cuda is not None and bool(cuda.is_available()):
            try:
                sample = torch.ones((64, 64), device="cuda")
                result = sample @ sample
                cuda.synchronize()
                float(result[0, 0].detach().cpu().item())
                return "cuda:0"
            except Exception as exc:  # noqa: BLE001
                if is_jetson:
                    raise RuntimeError(
                        "CUDA is visible to PyTorch but cannot execute kernels on this Jetson GPU. "
                        "Install a Jetson-compatible PyTorch build for this device compute capability."
                    ) from exc
    except Exception:  # noqa: BLE001
        if is_jetson:
            raise
    return "cpu"


def worker_main(command_queue: Any, result_queue: Any) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    model: object | None = None
    model_path: str | None = None
    device = "cpu"
    while True:
        try:
            command = command_queue.get(timeout=0.2)
        except Empty:
            continue
        kind = command.get("kind")
        if kind == "stop":
            return
        if kind == "activate":
            model_path = command.get("path")
            if not model_path:
                model = None
                result_queue.put({"kind": "activated", "path": None, "labels": []})
                continue
            try:
                from ultralytics import YOLO  # type: ignore[import-not-found]

                device = _select_inference_device()
                model = YOLO(str(Path(model_path)))
                with_device = getattr(model, "to", None)
                if callable(with_device):
                    with_device(device)
                result_queue.put(
                    {
                        "kind": "activated",
                        "path": model_path,
                        "labels": _labels(model),
                        "device": device,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                model = None
                result_queue.put({"kind": "error", "error": str(exc), "path": model_path})
            continue
        if kind != "infer":
            continue
        if model is None:
            result_queue.put({"kind": "detections", "detections": [], "latency_ms": None, "path": model_path})
            continue
        shm = None
        try:
            shm = SharedMemory(name=command["shm_name"])
            shape = tuple(command["shape"])
            dtype = np.dtype(command.get("dtype", "uint8"))
            frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            started = time.perf_counter()
            results = cast(Any, model).predict(
                frame,
                conf=float(command.get("confidence", 0.4)),
                iou=float(command.get("iou", 0.45)),
                max_det=int(command.get("max_detections", 10)),
                classes=command.get("classes"),
                device=device,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            result_queue.put(
                {
                    "kind": "detections",
                    "detections": _detections_from_results(results, model),
                    "latency_ms": latency_ms,
                    "path": model_path,
                    "device": device,
                    "frame_id": command.get("frame_id"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            result_queue.put({"kind": "error", "error": str(exc), "path": model_path})
        finally:
            if shm is not None:
                shm.close()
