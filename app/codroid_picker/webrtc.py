"""WebRTC video transport for low-latency camera preview."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import Future
from fractions import Fraction
from typing import Any, Optional

import cv2
import numpy as np

from app.codroid_picker.utils import placeholder_frame

try:
    import av
    from aiortc import (
        RTCConfiguration,
        RTCPeerConnection,
        RTCSessionDescription,
        VideoStreamTrack,
    )
except ImportError:  # pragma: no cover - exercised on hosts without aiortc.
    av = None
    RTCConfiguration = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    VideoStreamTrack = object  # type: ignore[assignment,misc]


LOGGER = logging.getLogger(__name__)
WEBRTC_AVAILABLE = av is not None and RTCPeerConnection is not None
VIDEO_TIME_BASE = Fraction(1, 90_000)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def should_try_webrtc() -> bool:
    """Return whether the browser should attempt WebRTC."""
    mode = os.getenv("CODROID_VIDEO_TRANSPORT", "webrtc").strip().lower()
    if mode in {"off", "disabled", "none"}:
        return False
    return WEBRTC_AVAILABLE


class LatestFrameVideoTrack(VideoStreamTrack):  # type: ignore[misc]
    """aiortc video track that emits only the latest raw camera frame."""

    kind = "video"

    def __init__(
        self,
        raw_stream: Any,
        service: "WebRTCVideoService",
        *,
        max_fps: float,
        max_width: int,
    ) -> None:
        super().__init__()
        self._raw_stream = raw_stream
        self._service = service
        self._max_fps = max(max_fps, 1.0)
        self._max_width = max(max_width, 1)
        self._frame_interval_s = 1.0 / self._max_fps
        self._last_emit_ts = 0.0
        self._last_frame_index = 0
        self._pts = 0
        self._placeholder: Optional[np.ndarray] = None
        self._last_output_shape: Optional[tuple[int, int]] = None

    async def recv(self) -> Any:
        """Return the next WebRTC frame, dropping old camera frames."""
        await self._throttle()
        frame, frame_index, frame_ts = self._latest_frame()
        now = time.time()
        stale = frame_ts <= 0.0 or (now - frame_ts) > self._service.stale_after_s
        if frame is None or stale:
            frame = self._placeholder_frame(frame)
        elif frame_index > self._last_frame_index + 1:
            self._service.record_dropped_frames(
                frame_index - self._last_frame_index - 1
            )

        frame = self._resize_for_transport(frame)
        height, width = frame.shape[:2]
        self._last_output_shape = (height, width)
        self._last_frame_index = max(frame_index, self._last_frame_index)
        self._pts += int(90_000 / self._max_fps)
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._pts
        video_frame.time_base = VIDEO_TIME_BASE
        self._service.record_frame_emitted(stale=stale, shape=(height, width))
        return video_frame

    def _resize_for_transport(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self._max_width:
            return frame
        scale = self._max_width / float(width)
        target_size = (self._max_width, max(int(height * scale), 1))
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    async def _throttle(self) -> None:
        now = time.time()
        wait_s = self._frame_interval_s - (now - self._last_emit_ts)
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        self._last_emit_ts = time.time()

    def _latest_frame(self) -> tuple[Optional[np.ndarray], int, float]:
        getter = getattr(self._raw_stream, "get_latest_frame_with_metadata", None)
        if getter is None:
            frame = self._raw_stream.get_latest_frame()
            return frame, 0, time.time() if frame is not None else 0.0
        return getter()

    def _placeholder_frame(self, frame: Optional[np.ndarray]) -> np.ndarray:
        if frame is not None:
            height, width = frame.shape[:2]
        elif self._placeholder is not None:
            return self._placeholder
        else:
            height, width = 480, 640
        self._placeholder = placeholder_frame(
            "Camera stream offline", size=(height, width)
        )
        return self._placeholder


class WebRTCVideoService:
    """Owns WebRTC signaling, peer lifecycle, and video transport stats."""

    def __init__(
        self,
        raw_stream: Any,
        *,
        max_fps: Optional[float] = None,
        max_peers: Optional[int] = None,
        stale_after_s: float = 3.0,
    ) -> None:
        self.raw_stream = raw_stream
        self.max_fps = max_fps or _env_float("CODROID_WEBRTC_MAX_FPS", 30.0)
        self.max_peers = max_peers or _env_int("CODROID_WEBRTC_MAX_PEERS", 2)
        self.max_width = _env_int("CODROID_WEBRTC_MAX_WIDTH", 640)
        self.stale_after_s = stale_after_s
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._peers: set[Any] = set()
        self._lock = threading.Lock()
        self._frames_emitted = 0
        self._frames_dropped = 0
        self._stale_frames = 0
        self._last_emit_ts = 0.0
        self._last_output_shape: Optional[tuple[int, int]] = None

    @property
    def available(self) -> bool:
        """Return whether WebRTC dependencies are importable."""
        return WEBRTC_AVAILABLE

    def create_answer_sync(
        self,
        offer: dict[str, Any],
        timeout_s: float = 10.0,
    ) -> dict[str, str]:
        """Create a WebRTC answer for a browser offer from a Flask thread."""
        self._validate_offer(offer)
        future = self._submit(self._create_answer(offer))
        return future.result(timeout=timeout_s)

    def stats(self) -> dict[str, Any]:
        """Return lightweight service and connection stats."""
        frame, frame_index, frame_ts = self._latest_frame()
        now = time.time()
        age_s = now - frame_ts if frame_ts > 0 else None
        with self._lock:
            states = [getattr(peer, "connectionState", "unknown") for peer in self._peers]
            return {
                "available": self.available,
                "transport": "webrtc",
                "try_webrtc": should_try_webrtc(),
                "peer_count": len(self._peers),
                "peer_limit": self.max_peers,
                "peer_states": states,
                "max_fps": self.max_fps,
                "max_width": self.max_width,
                "frames_emitted": self._frames_emitted,
                "frames_dropped": self._frames_dropped,
                "stale_frames": self._stale_frames,
                "last_emit_age_s": (
                    now - self._last_emit_ts if self._last_emit_ts > 0 else None
                ),
                "output_frame_shape": (
                    list(self._last_output_shape)
                    if self._last_output_shape is not None
                    else None
                ),
                "raw_frame_index": frame_index,
                "raw_frame_age_s": age_s,
                "video_frame_stale": age_s is None or age_s > self.stale_after_s,
                "raw_frame_shape": list(frame.shape) if frame is not None else None,
            }

    def record_frame_emitted(
        self,
        *,
        stale: bool,
        shape: tuple[int, int],
    ) -> None:
        """Record a frame emitted by any peer track."""
        with self._lock:
            self._frames_emitted += 1
            self._last_emit_ts = time.time()
            self._last_output_shape = shape
            if stale:
                self._stale_frames += 1

    def record_dropped_frames(self, count: int) -> None:
        """Record skipped raw camera frames."""
        if count <= 0:
            return
        with self._lock:
            self._frames_dropped += count

    def close_sync(self, timeout_s: float = 5.0) -> None:
        """Close all active peer connections and stop the event loop."""
        if self._loop is None:
            return
        try:
            self._submit(self._close_all()).result(timeout=timeout_s)
        finally:
            loop = self._loop
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout_s)
            self._loop = None
            self._thread = None

    def _validate_offer(self, offer: dict[str, Any]) -> None:
        if not self.available:
            raise RuntimeError("WebRTC dependencies are unavailable")
        if not isinstance(offer, dict):
            raise ValueError("Offer must be a JSON object")
        if offer.get("type") != "offer":
            raise ValueError("Offer type must be 'offer'")
        if not isinstance(offer.get("sdp"), str) or not offer["sdp"].strip():
            raise ValueError("Offer SDP is required")

    def _submit(self, coroutine: Any) -> Future:
        self._ensure_loop()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _ensure_loop(self) -> None:
        if self._loop and self._loop.is_running():
            return
        ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(self._loop, ready),
            name="WebRTCVideoService",
            daemon=True,
        )
        self._thread.start()
        ready.wait(timeout=3.0)

    def _run_loop(self, loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    async def _create_answer(self, offer: dict[str, Any]) -> dict[str, str]:
        await self._prune_closed_peers()
        with self._lock:
            peer_count = len(self._peers)
        if peer_count >= self.max_peers:
            raise RuntimeError("WebRTC peer limit reached")

        config = RTCConfiguration(iceServers=[])
        peer = RTCPeerConnection(configuration=config)
        peer.addTrack(
            LatestFrameVideoTrack(
                self.raw_stream,
                self,
                max_fps=self.max_fps,
                max_width=self.max_width,
            )
        )

        @peer.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            state = peer.connectionState
            LOGGER.info("WebRTC peer connection state: %s", state)
            if state in {"failed", "closed", "disconnected"}:
                await self._discard_peer(peer)

        with self._lock:
            self._peers.add(peer)
        try:
            await peer.setRemoteDescription(
                RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
            )
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            assert peer.localDescription is not None
            return {
                "sdp": peer.localDescription.sdp,
                "type": peer.localDescription.type,
            }
        except Exception:
            await self._discard_peer(peer)
            raise

    async def _discard_peer(self, peer: Any) -> None:
        with self._lock:
            self._peers.discard(peer)
        if getattr(peer, "connectionState", None) != "closed":
            await peer.close()

    async def _prune_closed_peers(self) -> None:
        with self._lock:
            peers = list(self._peers)
        for peer in peers:
            if getattr(peer, "connectionState", None) in {"closed", "failed"}:
                await self._discard_peer(peer)

    async def _close_all(self) -> None:
        with self._lock:
            peers = list(self._peers)
        for peer in peers:
            await self._discard_peer(peer)
        with self._lock:
            self._peers.clear()

    def _latest_frame(self) -> tuple[Optional[np.ndarray], int, float]:
        getter = getattr(self.raw_stream, "get_latest_frame_with_metadata", None)
        if getter is None:
            frame = self.raw_stream.get_latest_frame()
            return frame, 0, time.time() if frame is not None else 0.0
        return getter()


_SERVICE: Optional[WebRTCVideoService] = None
_SERVICE_LOCK = threading.Lock()


def get_webrtc_video_service(raw_stream: Any) -> WebRTCVideoService:
    """Return the singleton WebRTC video service."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None or _SERVICE.raw_stream is not raw_stream:
            _SERVICE = WebRTCVideoService(raw_stream)
        return _SERVICE


def close_webrtc_video_service() -> None:
    """Close the singleton WebRTC service if it exists."""
    global _SERVICE
    with _SERVICE_LOCK:
        service = _SERVICE
        _SERVICE = None
    if service is not None:
        service.close_sync()
