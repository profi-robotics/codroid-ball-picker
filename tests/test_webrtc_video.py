"""Tests for the WebRTC camera transport."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from app.codroid_picker.streaming import RawFrameStore
from app.codroid_picker import webrtc


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_latest_frame_track_emits_latest_frame_and_counts_drops() -> None:
    pytest.importorskip("aiortc")
    pytest.importorskip("av")

    raw_stream = RawFrameStore()
    service = webrtc.WebRTCVideoService(raw_stream, max_fps=60.0)
    track = webrtc.LatestFrameVideoTrack(
        raw_stream,
        service,
        max_fps=60.0,
        max_width=640,
    )

    raw_stream.add_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    raw_stream.add_frame(np.ones((4, 4, 3), dtype=np.uint8) * 64)
    raw_stream.add_frame(np.ones((4, 4, 3), dtype=np.uint8) * 127)

    frame = await track.recv()
    stats = service.stats()

    assert frame.width == 4
    assert frame.height == 4
    assert stats["frames_emitted"] == 1
    assert stats["frames_dropped"] == 2
    assert stats["video_frame_stale"] is False


@pytest.mark.asyncio
async def test_latest_frame_track_emits_placeholder_when_camera_is_off() -> None:
    pytest.importorskip("aiortc")
    pytest.importorskip("av")

    raw_stream = RawFrameStore()
    service = webrtc.WebRTCVideoService(raw_stream, max_fps=60.0)
    track = webrtc.LatestFrameVideoTrack(
        raw_stream,
        service,
        max_fps=60.0,
        max_width=640,
    )

    frame = await track.recv()
    stats = service.stats()

    assert frame.width == 640
    assert frame.height == 480
    assert stats["frames_emitted"] == 1
    assert stats["stale_frames"] == 1
    assert stats["video_frame_stale"] is True


def test_webrtc_service_rejects_malformed_offer() -> None:
    raw_stream = RawFrameStore()
    service = webrtc.WebRTCVideoService(raw_stream)

    with pytest.raises(ValueError, match="Offer type"):
        service.create_answer_sync({"type": "answer", "sdp": "v=0"})


def test_webrtc_stats_report_frame_age_and_shape() -> None:
    raw_stream = RawFrameStore()
    raw_stream.add_frame(np.zeros((8, 6, 3), dtype=np.uint8))
    service = webrtc.WebRTCVideoService(raw_stream, max_fps=15.0, max_peers=1)

    stats = service.stats()

    assert stats["max_fps"] == 15.0
    assert stats["max_width"] == 640
    assert stats["peer_limit"] == 1
    assert stats["raw_frame_shape"] == [8, 6, 3]
    assert stats["raw_frame_age_s"] is not None
    assert stats["video_frame_stale"] is False


def test_video_routes_return_400_for_bad_webrtc_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    from dash import Dash, html
    from flask import Flask

    from app.codroid_picker import video_routes

    class FakeSnapshotService:
        def start(self) -> None:
            return None

        def iter_sse(self, _last_seq: int):
            yield ""

    class FakeRegistry:
        raw_stream = RawFrameStore()

    monkeypatch.setattr(
        video_routes,
        "get_ui_snapshot_service",
        lambda: FakeSnapshotService(),
    )
    monkeypatch.setattr(video_routes, "registry", lambda: FakeRegistry())

    app = Dash(__name__, server=Flask(__name__))
    app.layout = html.Div("ok")
    video_routes.register_video_routes(app)

    response = app.server.test_client().post("/webrtc/offer", json={"type": "bad"})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_webrtc_service_creates_answer_for_valid_offer() -> None:
    pytest.importorskip("aiortc")

    from aiortc import RTCPeerConnection

    raw_stream = RawFrameStore()
    raw_stream.add_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    service = webrtc.WebRTCVideoService(raw_stream, max_fps=10.0, max_peers=1)

    async def run_case() -> dict[str, str]:
        peer = RTCPeerConnection()
        try:
            peer.addTransceiver("video", direction="recvonly")
            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)
            deadline = time.time() + 2.0
            while peer.iceGatheringState != "complete" and time.time() < deadline:
                await asyncio.sleep(0.01)
            assert peer.localDescription is not None
            offer_payload = {
                "type": peer.localDescription.type,
                "sdp": peer.localDescription.sdp,
            }
            return service.create_answer_sync(offer_payload)
        finally:
            service.close_sync()
            await peer.close()

    answer = asyncio.run(run_case())

    assert answer["type"] == "answer"
    assert "m=video" in answer["sdp"]
