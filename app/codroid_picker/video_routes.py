"""Flask video streaming routes for the calibration UI.

This module provides HTTP endpoints for serving camera frames
as Motion JPEG streams or individual JPEG images.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import cv2
from flask import jsonify

from app.codroid_picker.app_instance import registry
from app.codroid_picker.streaming import FRAME_PROCESSOR
from app.codroid_picker.utils import placeholder_frame

if TYPE_CHECKING:
    from dash import Dash

LOGGER = logging.getLogger(__name__)


def register_video_routes(app: "Dash") -> None:
    """Register Flask video streaming routes with the Dash app.

    Provides:
    - /video_feed: Raw Motion JPEG stream for fullscreen playback
    - /overlay_state.json: Latest overlay primitives for browser rendering
    - /camera_frame.jpg: Latest streamed camera frame
    - /raw_camera_frame.jpg: Latest raw camera frame without overlays
    """
    @app.server.route("/video_feed")
    def video_feed():
        """Stream Motion JPEG video for fullscreen playback."""
        LOGGER.debug("/video_feed endpoint accessed")

        def generate():
            boundary = b'--frame\r\n'
            frame_count = 0
            last_frame_index = -1
            while True:
                frame_data, frame_index = registry().mjpeg_stream.get_frame_with_index()

                if not frame_data or frame_index == last_frame_index:
                    time.sleep(0.005)
                    continue

                last_frame_index = frame_index
                frame_count += 1
                if frame_count == 1:
                    LOGGER.debug(
                        "/video_feed serving first frame, size=%s bytes", len(frame_data))

                yield boundary
                yield b'Content-Type: image/jpeg\r\n'
                yield b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                yield frame_data
                yield b'\r\n'

        return app.server.response_class(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.server.route("/camera_frame.jpg")
    def camera_frame():
        """Get the latest streamed camera frame as a JPEG image."""
        frame_data = registry().mjpeg_stream.get_frame()
        if not frame_data:
            # Return placeholder as JPEG bytes
            placeholder = placeholder_frame(
                "Camera stream offline", size=(480, 640))
            success, buffer = cv2.imencode(".jpg", placeholder)
            if success:
                frame_data = buffer.tobytes()
            else:
                return app.server.response_class(
                    b"",
                    mimetype='image/jpeg'
                )
        return app.server.response_class(
            frame_data,
            mimetype='image/jpeg'
        )

    @app.server.route("/raw_camera_frame.jpg")
    def raw_camera_frame():
        """Get the latest raw camera frame (no overlays) as a JPEG image."""
        raw_frame = registry().raw_stream.get_latest_frame()
        if raw_frame is None:
            placeholder = placeholder_frame(
                "Camera stream offline", size=(480, 640))
            raw_frame = placeholder
        success, buffer = cv2.imencode(".jpg", raw_frame)
        if not success:
            return app.server.response_class(
                b"",
                mimetype='image/jpeg'
            )
        return app.server.response_class(
            buffer.tobytes(),
            mimetype='image/jpeg'
        )

    @app.server.route("/overlay_state.json")
    def overlay_state():
        """Return the latest browser-rendered overlay state."""
        return jsonify(FRAME_PROCESSOR.get_active_state_payload())
