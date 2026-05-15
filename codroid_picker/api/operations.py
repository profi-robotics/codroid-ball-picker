"""Operation and manual pick/drop routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import ActionResponse, OperationPauseRequest, PickDropRequest, action
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter()


@router.post("/operation/pause", response_model=ActionResponse)
def pause_operation(payload: OperationPauseRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    ctx.state.pause_operation(payload.reason, "Operation paused.", owner="operator")
    return action("Operation paused.", state=ctx.state.snapshot())


@router.post("/operation/resume", response_model=ActionResponse)
def resume_operation(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    ctx.state.resume_operation("Operation resumed.", owner="operator")
    return action("Operation resumed.", state=ctx.state.snapshot())


@router.post("/operation/reset", response_model=ActionResponse)
def reset_operation(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    ctx.state.reset_operation("Program idle.")
    return action("Operation reset.", state=ctx.state.snapshot())


@router.post("/pick/closest", response_model=ActionResponse)
async def pick_closest(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.pick_place.pick_closest(drop_after=False)
    return action(message, state=ctx.state.snapshot())


@router.post("/pick/drop", response_model=ActionResponse)
async def pick_drop(payload: PickDropRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.pick_place.drop(drop_height_mm=payload.drop_height_mm)
    return action(message, state=ctx.state.snapshot())


@router.post("/pick/pick-and-drop", response_model=ActionResponse)
async def pick_and_drop(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.pick_place.pick_closest(drop_after=True)
    return action(message, state=ctx.state.snapshot())


@router.post("/drop-target/teach", response_model=ActionResponse)
async def teach_drop_target(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.pick_place.teach_drop_target()
    return action(message, state=ctx.state.snapshot())


@router.post("/drop-target/return", response_model=ActionResponse)
async def return_drop_target(payload: PickDropRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.pick_place.return_to_drop_target(height_offset_mm=payload.drop_height_mm)
    return action(message, state=ctx.state.snapshot())
