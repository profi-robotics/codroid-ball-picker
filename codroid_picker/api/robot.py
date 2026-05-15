"""Robot API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import (
    ActionResponse,
    RobotConnectRequest,
    RobotErrorAckRequest,
    RobotJogRequest,
    RobotMoveRequest,
    RobotPowerRequest,
    RobotSpeedRequest,
    action,
)
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/robot")


@router.post("/connect", response_model=ActionResponse)
async def connect_robot(payload: RobotConnectRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.connect(payload.resolved_uri(ctx.settings.current.default_robot_uri))
    return action(message, state=ctx.state.snapshot())


@router.post("/acquire", response_model=ActionResponse)
async def acquire_robot(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.acquire()
    return action(message, state=ctx.state.snapshot())


@router.post("/release", response_model=ActionResponse)
async def release_robot(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.release()
    return action(message, state=ctx.state.snapshot())


@router.post("/power", response_model=ActionResponse)
async def robot_power(payload: RobotPowerRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.power(payload.on)
    return action(message, state=ctx.state.snapshot())


@router.post("/speed", response_model=ActionResponse)
async def robot_speed(payload: RobotSpeedRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.speed(payload.rate)
    return action(message, state=ctx.state.snapshot())


@router.post("/jog", response_model=ActionResponse)
async def robot_jog(payload: RobotJogRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.jog(axis=payload.axis, direction=payload.direction, hold_s=payload.hold_s, mode=payload.mode)
    return action(message, state=ctx.state.snapshot())


@router.post("/move", response_model=ActionResponse)
async def robot_move(payload: RobotMoveRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.move(payload.pose.model_dump(), coordinate_id=payload.coordinate_id, timeout_s=payload.timeout_s)
    return action(message, state=ctx.state.snapshot())


@router.post("/stop", response_model=ActionResponse)
async def robot_stop(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.stop()
    return action(message, state=ctx.state.snapshot())


@router.post("/errors/ack", response_model=ActionResponse)
async def robot_error_ack(payload: RobotErrorAckRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.robot.ack_error(payload.error_type)
    return action(message, state=ctx.state.snapshot())
