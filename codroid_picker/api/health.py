"""Health endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter()


@router.get("/health")
def health(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    ctx.monitoring.sample_system()
    state = ctx.state.snapshot()
    return {
        "ok": True,
        "version": "0.1.0",
        "state_seq": state["seq"],
        "camera": state["camera"]["status"],
        "robot": state["robot"]["status"],
        "worker": state["detection"]["worker_status"],
    }
