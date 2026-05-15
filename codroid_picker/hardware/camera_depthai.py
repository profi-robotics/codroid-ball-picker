"""DepthAI camera adapter."""

from __future__ import annotations

from typing import Any


def create_depthai_stream(config: Any, frame_hook: Any = None) -> Any:
    """Create the existing DepthAI stream lazily so imports stay hardware-safe."""
    from vision.stream import DepthAIFrameStream

    return DepthAIFrameStream(config, frame_hook=frame_hook)

