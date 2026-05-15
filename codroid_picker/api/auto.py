"""Auto-pick API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import ActionResponse, AutoConfigRequest, action
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/auto")


@router.get("/config")
def auto_config(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return ctx.auto_pick.config()


@router.put("/config", response_model=ActionResponse)
def update_auto_config(payload: AutoConfigRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    updates = payload.model_dump(exclude_none=True)
    state = ctx.auto_pick.update_config(updates)
    return action("Auto configuration updated.", state=state)


@router.post("/start", response_model=ActionResponse)
async def start_auto(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    operation_id = await ctx.auto_pick.start()
    return action("Auto pick started.", state=ctx.state.snapshot(), operation_id=operation_id)


@router.post("/stop", response_model=ActionResponse)
async def stop_auto(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.auto_pick.stop()
    return action(message, state=ctx.state.snapshot())
