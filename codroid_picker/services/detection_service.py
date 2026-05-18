"""Detection service with optional isolated worker process."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import queue
import time
import asyncio
import contextlib
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable, Optional, cast

import cv2
import numpy as np

from codroid_picker.config import AppSettings
from codroid_picker.core.models import benchmark_rating
from codroid_picker.services.detection_worker import worker_main
from codroid_picker.state.store import StateStore
from codroid_picker.video.frame_buffer import RawFrameBuffer


class DetectionService:
    def __init__(self, settings: AppSettings, state: StateStore, frame_buffer: RawFrameBuffer) -> None:
        self.settings = settings
        self.state = state
        self.frame_buffer = frame_buffer
        self._ctx = mp.get_context("spawn")
        self._command_queue: Any = None
        self._result_queue: Any = None
        self._process: Any = None
        self._shm: Optional[SharedMemory] = None
        self._shm_shape: Optional[tuple[int, ...]] = None
        self._scheduler_task: asyncio.Task[None] | None = None
        self._enricher: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._skip_counter = 0

    def start(self) -> None:
        if not self.settings.detection_worker or self._process is not None:
            self.state.update_domain("detection", worker_status="disabled" if not self.settings.detection_worker else "running")
            return
        self._command_queue = self._ctx.Queue(maxsize=1)
        self._result_queue = self._ctx.Queue(maxsize=4)
        process = self._ctx.Process(target=worker_main, args=(self._command_queue, self._result_queue), daemon=True)
        self._process = process
        process.start()
        self.state.update_domain("detection", worker_status="running")

    def stop(self) -> None:
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
        if self._command_queue is not None:
            self._put_latest({"kind": "stop"})
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None
        self._command_queue = None
        self._result_queue = None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
        self._shm = None
        self._shm_shape = None
        self.state.update_domain("detection", worker_status="stopped")

    def set_enricher(self, enricher: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._enricher = enricher

    def start_scheduler(self) -> None:
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop_scheduler(self) -> None:
        if self._scheduler_task is None:
            return
        self._scheduler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._scheduler_task
        self._scheduler_task = None

    def params(self) -> dict[str, Any]:
        return self.state.domain("detection")

    def update_params(self, **updates: Any) -> dict[str, Any]:
        allowed = {"confidence", "iou", "max_detections", "classes", "frame_skip"}
        clean = {key: value for key, value in updates.items() if key in allowed}
        return self.state.update_domain("detection", **clean)

    def list_models(self) -> list[dict[str, str]]:
        model_dir = self.settings.model_dir
        if not model_dir.exists():
            return []
        return [
            {"path": str(path), "name": path.name}
            for path in sorted(model_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".engine"}
        ]

    def activate(self, path: Optional[Path]) -> dict[str, Any]:
        self.start()
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
        if path is not None and importlib.util.find_spec("ultralytics") is None:
            message = (
                "ML dependencies are not installed. Run `uv sync` before "
                "activating YOLO models."
            )
            self.state.update_domain(
                "detection",
                active_model=None,
                labels=[],
                ml_available=False,
                last_error=message,
            )
            return self.state.domain("detection")
        if self._command_queue is None:
            self.state.update_domain("detection", active_model=str(path) if path else None, labels=[], ml_available=None)
            return self.state.domain("detection")
        self._put_latest({"kind": "activate", "path": str(path) if path else None})
        result = self._wait_result(timeout_s=20.0)
        if result.get("kind") == "error":
            self.state.update_domain("detection", last_error=result.get("error"), active_model=None, labels=[], ml_available=False)
        else:
            self.state.update_domain(
                "detection",
                active_model=result.get("path"),
                labels=result.get("labels", []),
                ml_available=True if path else None,
                last_error=None,
                backend_device=result.get("device"),
            )
        return self.state.domain("detection")

    def run_once(self, *, force: bool = True) -> list[dict[str, Any]]:
        self.start()
        if not force and self._should_skip_frame():
            current = self.state.domain("detection").get("detections", [])
            return [dict(item) for item in current if isinstance(item, dict)]
        snapshot = self.frame_buffer.latest(copy=True)
        if snapshot.frame is None:
            self.state.update_domain("detection", detections=[], last_error="No camera frame available.")
            return []
        if self._command_queue is None:
            return []
        result = self._infer_frame(snapshot.frame, frame_id=snapshot.frame_id)
        if result.get("kind") == "error":
            self.state.update_domain("detection", last_error=result.get("error"))
            return []
        detections = [self._enrich_detection(dict(item)) for item in cast(list[dict[str, Any]], result.get("detections", []))]
        detections = self._filter_by_classes(detections)
        self._publish_detections(detections, result.get("latency_ms"))
        return detections

    def benchmark(self, *, runs: int = 3, warmup: int = 1) -> dict[str, Any]:
        self.start()
        active_model = self.state.domain("detection").get("active_model")
        if not active_model:
            result = {"status": "unavailable", "message": "No model loaded."}
            self.state.update_domain("detection", benchmark=result)
            return result
        snapshot = self.frame_buffer.latest(copy=True)
        if snapshot.frame is None:
            width, height = self.settings.camera_resolution
            frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
            frame_source = "synthetic"
        else:
            frame = snapshot.frame
            frame_source = "camera"
        timings: list[float] = []
        total = max(int(warmup), 0) + max(int(runs), 1)
        for index in range(total):
            result = self._infer_frame(frame, frame_id=snapshot.frame_id if snapshot.frame is not None else 0)
            if result.get("kind") == "error":
                payload: dict[str, Any] = {
                    "status": "error",
                    "message": str(result.get("error") or "Benchmark failed."),
                    "frame_source": frame_source,
                    "input_shape": list(frame.shape),
                }
                self.state.update_domain("detection", benchmark=payload, last_error=payload["message"])
                return payload
            latency = result.get("latency_ms")
            if index >= warmup and isinstance(latency, (int, float)):
                timings.append(float(latency))
        avg = sum(timings) / len(timings) if timings else 0.0
        benchmark_payload: dict[str, Any] = {
            "status": "ok",
            "message": f"Benchmark {avg:.1f} ms avg over {len(timings)} runs.",
            "avg_latency_ms": avg,
            "min_latency_ms": min(timings) if timings else 0.0,
            "max_latency_ms": max(timings) if timings else 0.0,
            "runs": len(timings),
            "frame_source": frame_source,
            "input_shape": list(frame.shape),
            "backend": "ultralytics-worker",
            "device": self.state.domain("detection").get("backend_device") or "unknown",
            "rating": benchmark_rating(avg),
        }
        self.state.update_domain("detection", benchmark=benchmark_payload, last_error=None)
        return benchmark_payload

    def _infer_frame(self, raw_frame: np.ndarray, *, frame_id: int) -> dict[str, Any]:
        if self._command_queue is None:
            return {"kind": "error", "error": "Detection worker is not available."}
        frame = self._prepare_frame(raw_frame)
        shm = self._ensure_shm(frame)
        shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        shm_array[:] = frame
        params = self.state.domain("detection")
        classes = self._parse_classes(params.get("classes"))
        self._put_latest(
            {
                "kind": "infer",
                "shm_name": shm.name,
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "frame_id": frame_id,
                "confidence": params.get("confidence", 0.4),
                "iou": params.get("iou", 0.45),
                "max_detections": params.get("max_detections", 10),
                "classes": classes,
            }
        )
        return self._wait_result(timeout_s=30.0)

    async def _scheduler_loop(self) -> None:
        interval_s = max(float(self.settings.camera_safety_check_interval_ms) / 1000.0, 0.1)
        while True:
            await asyncio.sleep(interval_s)
            state = self.state.domain("detection")
            if not state.get("active_model"):
                continue
            await asyncio.to_thread(self.run_once, force=False)

    def _should_skip_frame(self) -> bool:
        frame_skip = max(int(self.state.domain("detection").get("frame_skip") or self.settings.detection_frame_skip), 1)
        self._skip_counter = (self._skip_counter + 1) % frame_skip
        return self._skip_counter != 0

    def _enrich_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        if self._enricher is None:
            return detection
        try:
            return self._enricher(detection)
        except Exception:
            return detection

    @staticmethod
    def _parse_classes(value: Any) -> list[int] | None:
        if value is None:
            return None
        if isinstance(value, str):
            parsed = []
            for token in value.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    parsed.append(int(token))
                except ValueError:
                    return None
            return parsed or None
        if isinstance(value, list):
            try:
                return [int(item) for item in value]
            except (TypeError, ValueError):
                return None
        return None

    def _filter_by_classes(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        classes = self._parse_classes(self.state.domain("detection").get("classes"))
        if classes is None:
            return detections
        allowed = {str(item) for item in classes}
        return [item for item in detections if str(item.get("class_id", item.get("label", ""))) in allowed]

    def _publish_detections(self, detections: list[dict[str, Any]], latency_ms: Any) -> None:
        now = time.time()
        monitoring = self.state.domain("monitoring")
        history = list(monitoring.get("detection_history") or [])
        history.append({"ts": now, "count": len(detections), "latency_ms": latency_ms})
        self.state.update_domain(
            "detection",
            detections=detections,
            latency_ms=latency_ms,
            updated_at=now,
            last_error=None,
        )
        self.state.update_domain("monitoring", detection_history=history[-90:])

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        max_width = 640
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            frame = cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame)

    def _ensure_shm(self, frame: np.ndarray) -> SharedMemory:
        required_shape = tuple(frame.shape)
        required_size = int(frame.nbytes)
        if self._shm is not None and self._shm_shape == required_shape and self._shm.size >= required_size:
            return self._shm
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
        self._shm = SharedMemory(create=True, size=required_size)
        self._shm_shape = required_shape
        return self._shm

    def _put_latest(self, command: dict[str, Any]) -> None:
        if self._command_queue is None:
            return
        try:
            self._command_queue.put_nowait(command)
        except queue.Full:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                pass
            self._command_queue.put_nowait(command)

    def _wait_result(self, *, timeout_s: float) -> dict[str, Any]:
        if self._result_queue is None:
            return {}
        try:
            return cast(dict[str, Any], self._result_queue.get(timeout=timeout_s))
        except queue.Empty:
            return {"kind": "error", "error": "Detection worker timed out."}
