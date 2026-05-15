"""MJPEG video service."""

from __future__ import annotations

import time
from typing import Any, Iterable

from codroid_picker.config import AppSettings
from codroid_picker.state.store import StateStore
from codroid_picker.video.frame_buffer import RawFrameBuffer
from codroid_picker.video.mjpeg import BOUNDARY, encode_jpeg, mjpeg_stream, placeholder_frame
from codroid_picker.video.overlays import render_overlay


class VideoService:
    def __init__(self, settings: AppSettings, state: StateStore, frame_buffer: RawFrameBuffer) -> None:
        self.settings = settings
        self.state = state
        self.frame_buffer = frame_buffer
        self._clients = 0
        self._frames_encoded = 0
        self._last_encoded_at = 0.0
        self._last_publish_at = 0.0
        self._last_publish_frames = 0
        self._encoded_fps = 0.0

    @property
    def media_type(self) -> str:
        return f"multipart/x-mixed-replace; boundary={BOUNDARY}"

    def stream(self, *, fps: float | None = None, width: int | None = None, quality: int | None = None, overlay: str | None = None) -> Iterable[bytes]:
        self._clients += 1
        if overlay:
            self.state.update_domain("video", overlay_mode=overlay.replace("-", "_"))
        self._publish()
        try:
            for frame in mjpeg_stream(
                self.frame_buffer,
                fps=fps or self.settings.video_max_fps,
                quality=quality or self.settings.video_jpeg_quality,
                max_width=width or self.settings.video_max_width,
                frame_transform=lambda raw: self._apply_overlay(raw, overlay),
            ):
                self._frames_encoded += 1
                self._last_encoded_at = time.time()
                if self._frames_encoded % 15 == 0:
                    self._publish()
                yield frame
        finally:
            self._clients = max(self._clients - 1, 0)
            self._publish()

    def snapshot_jpeg(self, *, width: int | None = None, quality: int | None = None, overlay: str | None = None) -> bytes:
        snapshot = self.frame_buffer.latest(copy=True)
        frame = snapshot.frame
        if frame is None:
            max_width = width or self.settings.video_snapshot_width
            frame = placeholder_frame(size=(max_width, int(max_width * 9 / 16)))
        frame = self._apply_overlay(frame, overlay)
        jpeg = encode_jpeg(
            frame,
            quality=quality or self.settings.video_jpeg_quality,
            max_width=width or self.settings.video_snapshot_width,
        )
        self._frames_encoded += 1
        self._last_encoded_at = time.time()
        self._publish()
        return jpeg

    def _apply_overlay(self, frame: Any, overlay: str | None) -> Any:
        mode = overlay or str(self.state.domain("video").get("overlay_mode") or "auto")
        return render_overlay(frame, mode=mode, state=self.state.snapshot())

    def stats(self) -> dict[str, Any]:
        return {
            "transport": "mjpeg",
            "clients": self._clients,
            "frames_encoded": self._frames_encoded,
            "encoded_fps": self._encoded_fps,
            "stream_fps": self._encoded_fps,
            "last_encoded_at": self._last_encoded_at,
            "max_width": self.settings.video_max_width,
            "jpeg_quality": self.settings.video_jpeg_quality,
        }

    def _publish(self) -> None:
        now = time.time()
        if self._last_publish_at > 0.0 and now > self._last_publish_at:
            frame_delta = self._frames_encoded - self._last_publish_frames
            elapsed = now - self._last_publish_at
            client_count = max(self._clients, 1)
            if frame_delta >= 0:
                self._encoded_fps = frame_delta / elapsed / client_count
        self._last_publish_at = now
        self._last_publish_frames = self._frames_encoded
        self.state.update_domain(
            "video",
            clients=self._clients,
            stream_fps=self._encoded_fps,
            encoded_fps=self._encoded_fps,
            frames_encoded=self._frames_encoded,
            max_width=self.settings.video_max_width,
            jpeg_quality=self.settings.video_jpeg_quality,
        )
