"""State and SSE endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse

from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter()


@router.get("/state")
def state_snapshot(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return ctx.state.snapshot()


@router.get("/events/status")
def status_events(
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ctx: AppContext = Depends(get_context),
) -> StreamingResponse:
    try:
        last_seq = max(int(last_event_id or "0"), 0)
    except ValueError:
        last_seq = 0
    return StreamingResponse(
        ctx.publisher.iter_sse(ctx.state.snapshot(), last_seq=last_seq),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
