"""Gripper API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import (
    ActionResponse,
    GripperCloseRequest,
    GripperConnectRequest,
    GripperOpenRequest,
    GripperOrientationRequest,
    action,
)
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/gripper")


@router.post("/connect", response_model=ActionResponse)
async def connect_gripper(payload: GripperConnectRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.gripper.connect(ip=payload.ip, gripper_type=payload.type, t_index=payload.t_index)
    return action(message, state=ctx.state.snapshot())


@router.post("/disconnect", response_model=ActionResponse)
async def disconnect_gripper(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.gripper.disconnect()
    return action(message, state=ctx.state.snapshot())


@router.post("/open", response_model=ActionResponse)
async def open_gripper(payload: GripperOpenRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.gripper.open(**payload.model_dump(exclude_none=True))
    return action(message, state=ctx.state.snapshot())


@router.post("/close", response_model=ActionResponse)
async def close_gripper(payload: GripperCloseRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.gripper.close(**payload.model_dump(exclude_none=True))
    return action(message, state=ctx.state.snapshot())


@router.post("/orientation", response_model=ActionResponse)
async def gripper_orientation(payload: GripperOrientationRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.gripper.orientation(payload.orientation)
    return action(message, state=ctx.state.snapshot())

