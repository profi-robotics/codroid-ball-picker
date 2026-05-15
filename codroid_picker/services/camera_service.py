"""Camera lifecycle service."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from codroid_picker.config import AppSettings
from codroid_picker.hardware.fakes import FakeCameraStream, FakeStreamConfig
from codroid_picker.state.store import StateStore
from codroid_picker.video.frame_buffer import RawFrameBuffer


class CameraService:
    def __init__(self, settings: AppSettings, state: StateStore, frame_buffer: RawFrameBuffer) -> None:
        self.settings = settings
        self.state = state
        self.frame_buffer = frame_buffer
        self._stream: Optional[Any] = None
        self._lock = threading.RLock()

    def start(
        self,
        *,
        source: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        host_camera_index: Optional[int] = None,
        host_camera_device: Optional[str] = None,
        enable_depth: Optional[bool] = None,
        lock_af: bool = False,
    ) -> str:
        with self._lock:
            self.stop()
            source = self._normalize_source(source or self.settings.camera_source)
            resolution = (
                int(width or self.settings.camera_resolution[0]),
                int(height or self.settings.camera_resolution[1]),
            )
            fps_value = float(fps or self.settings.camera_fps)
            stream = self._create_stream(
                source=source,
                resolution=resolution,
                fps=fps_value,
                host_camera_index=host_camera_index,
                host_camera_device=host_camera_device,
                enable_depth=enable_depth,
                lock_af=lock_af,
            )
            stream.start()
            self._stream = stream
            self._publish_state()
            return f"Camera stream started ({source}, {resolution[0]}x{resolution[1]} @ {fps_value:.1f} FPS)."

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
            self._stream = None
            self.state.update_domain("camera", running=False, status="off", capture_fps=0.0, frame_age_s=None)

    def scan(self) -> list[dict[str, Any]]:
        try:
            from vision.stream import list_host_camera_devices

            return list_host_camera_devices()
        except Exception:
            return []

    def focus(self, *, lock: bool, lens_position: Optional[int] = None) -> None:
        if self._stream is None:
            raise RuntimeError("Camera is not running.")
        self._stream.set_focus_lock(lock, lens_position)
        self.state.update_domain("camera", focus_locked=lock)

    def latest_frame(self) -> Any:
        return self.frame_buffer.latest(copy=True)

    def poll(self) -> None:
        self._publish_state()

    def shutdown(self) -> None:
        self.stop()

    def _create_stream(
        self,
        *,
        source: str,
        resolution: tuple[int, int],
        fps: float,
        host_camera_index: Optional[int],
        host_camera_device: Optional[str],
        enable_depth: Optional[bool],
        lock_af: bool,
    ) -> Any:
        source = self._normalize_source(source)
        configured_source = self._normalize_source(self.settings.camera_source)
        use_fake_camera = source == "fake" or (self.settings.use_fake_hardware and configured_source == "fake")
        if use_fake_camera:
            return FakeCameraStream(
                FakeStreamConfig(
                    resolution=resolution,
                    fps=fps,
                    enable_depth=bool(enable_depth),
                    lock_af=lock_af,
                ),
                frame_hook=self._on_frame,
            )

        from vision.stream import StreamConfig

        config = StreamConfig(
            resolution=resolution,
            fps=fps,
            enable_depth=bool(self.settings.camera_enable_depth if enable_depth is None else enable_depth),
            fast_depth=bool(self.settings.camera_fast_depth),
            lock_af=lock_af and source == "oak",
        )
        if source == "host":
            from codroid_picker.hardware.camera_host import create_host_stream

            camera_source = host_camera_device or self.settings.host_camera_device
            return create_host_stream(
                config,
                frame_hook=self._on_frame,
                camera_index=int(host_camera_index if host_camera_index is not None else self.settings.host_camera_index),
                camera_source=camera_source,
            )
        from codroid_picker.hardware.camera_depthai import create_depthai_stream

        return create_depthai_stream(config, frame_hook=self._on_frame)

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = (source or "oak").strip().lower()
        if normalized in {"host", "webcam", "opencv", "local", "generic", "usb", "v4l2", "azure-kinect", "azure_kinect"}:
            return "host"
        if normalized == "fake":
            return "fake"
        return "oak"

    def _on_frame(self, frame: Any, depth: Any = None) -> None:
        self.frame_buffer.put(frame, depth=depth)

    def _publish_state(self) -> None:
        stream = self._stream
        if stream is None:
            return
        status = stream.status() if hasattr(stream, "status") else "live"
        cfg = getattr(stream, "config", None)
        resolution = getattr(cfg, "resolution", self.settings.camera_resolution)
        self.state.update_domain(
            "camera",
            running=bool(stream.is_running()),
            status=status,
            device_id=stream.device_id(),
            capture_fps=float(stream.get_fps()),
            frame_age_s=self.frame_buffer.age_s(),
            resolution=resolution,
            depth_enabled=bool(getattr(cfg, "enable_depth", False)),
            last_error=stream.last_error(),
        )
