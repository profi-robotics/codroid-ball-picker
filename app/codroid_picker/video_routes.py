"""Flask WebRTC signaling routes for the calibration UI."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from typing import TYPE_CHECKING

from flask import request, stream_with_context

from app.codroid_picker.app_instance import registry
from app.codroid_picker.ui_snapshots import get_ui_snapshot_service
from app.codroid_picker.webrtc import get_webrtc_video_service

if TYPE_CHECKING:
    from dash import Dash

LOGGER = logging.getLogger(__name__)
_CLIENT_DIAGNOSTICS_LOCK = threading.Lock()
_CLIENT_DIAGNOSTICS: dict = {}


def _record_client_diagnostics(payload: object) -> dict:
    """Store the latest browser-side WebRTC diagnostics."""
    global _CLIENT_DIAGNOSTICS
    if not isinstance(payload, dict):
        payload = {}
    diagnostic = {
        "received_at": time.time(),
        **{
            str(key): value
            for key, value in payload.items()
            if isinstance(key, str)
            and isinstance(value, (str, int, float, bool, type(None), list, dict))
        },
    }
    with _CLIENT_DIAGNOSTICS_LOCK:
        _CLIENT_DIAGNOSTICS = diagnostic
        return deepcopy(_CLIENT_DIAGNOSTICS)


def _client_diagnostics_snapshot() -> dict:
    """Return the latest browser-side WebRTC diagnostics."""
    with _CLIENT_DIAGNOSTICS_LOCK:
        return deepcopy(_CLIENT_DIAGNOSTICS)


def register_video_routes(app: "Dash") -> None:
    """Register Flask state and WebRTC routes with the Dash app.

    Provides:
    - /ui_events: Server-Sent Events stream with cached UI snapshots
    - /webrtc/offer: WebRTC offer/answer signaling for raw camera preview
    - /webrtc/stats: WebRTC transport and camera stats
    """
    service = get_ui_snapshot_service()
    service.start()

    if not getattr(app.server, "_codroid_ui_events_registered", False):
        app.server._codroid_ui_events_registered = True

        @app.server.route("/ui_events")
        def ui_events():
            """Stream cached UI snapshot updates to the browser."""
            raw_last_id = request.headers.get("Last-Event-ID", "0")
            try:
                last_seq = max(int(raw_last_id), 0)
            except (TypeError, ValueError):
                last_seq = 0
            return app.server.response_class(
                stream_with_context(service.iter_sse(last_seq)),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

    @app.server.route("/webrtc/offer", methods=["POST"])
    def webrtc_offer():
        """Accept a browser WebRTC offer and return a server answer."""
        service = get_webrtc_video_service(registry().raw_stream)
        payload = request.get_json(silent=True)
        try:
            answer = service.create_answer_sync(payload)
        except ValueError as exc:
            return {"error": str(exc)}, 400
        except RuntimeError as exc:
            status = 503 if not service.available else 429
            return {"error": str(exc)}, status
        except FutureTimeoutError:
            return {"error": "Timed out creating WebRTC answer"}, 504
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to create WebRTC answer: %s", exc, exc_info=True)
            return {"error": "Failed to create WebRTC answer"}, 500
        return answer

    @app.server.route("/webrtc/stats")
    def webrtc_stats():
        """Return current WebRTC transport state."""
        service = get_webrtc_video_service(registry().raw_stream)
        stream = registry().session.camera_stream
        return {
            **service.stats(),
            "capture_fps": stream.get_fps() if stream is not None else 0.0,
            "camera_status": stream.status() if stream is not None else "off",
            "camera_id": stream.device_id() if stream is not None else None,
            "client_diagnostics": _client_diagnostics_snapshot(),
        }

    @app.server.route("/webrtc/client_diagnostics", methods=["POST"])
    def webrtc_client_diagnostics():
        """Record browser-side WebRTC capability diagnostics."""
        return _record_client_diagnostics(request.get_json(silent=True))
