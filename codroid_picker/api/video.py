"""Video endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response, StreamingResponse

from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/video")


@router.get("/stream")
def video_stream(
    fps: float | None = Query(default=None),
    width: int | None = Query(default=None),
    quality: int | None = Query(default=None),
    overlay: str | None = Query(default=None),
    ctx: AppContext = Depends(get_context),
) -> StreamingResponse:
    return StreamingResponse(
        ctx.video.stream(fps=fps, width=width, quality=quality, overlay=overlay),
        media_type=ctx.video.media_type,
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/snapshot.jpg")
def video_snapshot(
    width: int | None = Query(default=None),
    quality: int | None = Query(default=None),
    overlay: str | None = Query(default=None),
    ctx: AppContext = Depends(get_context),
) -> Response:
    return Response(
        ctx.video.snapshot_jpeg(width=width, quality=quality, overlay=overlay),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/stats")
def video_stats(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return ctx.video.stats()
