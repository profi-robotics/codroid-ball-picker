"""Calibration API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import (
    ActionResponse,
    CalibrationResetRequest,
    CameraCalibrationCaptureRequest,
    PathRequest,
    RobotCalibrationPointRequest,
    WorkspaceComputeRequest,
    action,
)
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/calibration")


@router.post("/reset", response_model=ActionResponse)
def reset_calibration(payload: CalibrationResetRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.reset(payload.scope)
    return action(message, state=ctx.state.snapshot())


@router.post("/camera/capture", response_model=ActionResponse)
def capture_camera_calibration(payload: CameraCalibrationCaptureRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.capture_camera(
        frame_count=payload.frame_count,
        delay_s=payload.delay_s,
        min_usable_frames=payload.min_usable_frames,
        output_dir=payload.output_path,
    )
    return action(message, state=ctx.state.snapshot())


@router.post("/camera/load", response_model=ActionResponse)
def load_camera_calibration(payload: PathRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.load_camera(payload.path)
    return action(message, state=ctx.state.snapshot())


@router.post("/workspace/compute", response_model=ActionResponse)
def compute_workspace(payload: WorkspaceComputeRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.compute_workspace(payload.marker_ids, payload.marker_size_mm)
    return action(message, state=ctx.state.snapshot())


@router.post("/workspace/load", response_model=ActionResponse)
def load_workspace(payload: PathRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.load_workspace(payload.path)
    return action(message, state=ctx.state.snapshot())


@router.post("/workspace/clear", response_model=ActionResponse)
def clear_workspace(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.clear_workspace()
    return action(message, state=ctx.state.snapshot())


@router.post("/plate/detect")
def detect_plate(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return {"ok": True, "plate": ctx.calibration.detect_plate()}


@router.post("/robot/start", response_model=ActionResponse)
async def start_robot_calibration(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.calibration.start_robot_calibration(ctx.robot)
    return action(message, state=ctx.state.snapshot())


@router.post("/robot/point", response_model=ActionResponse)
def capture_robot_calibration_point(payload: RobotCalibrationPointRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.capture_robot_point(payload.index)
    return action(message, state=ctx.state.snapshot())


@router.post("/robot/finish", response_model=ActionResponse)
async def finish_robot_calibration(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.calibration.finish_robot_calibration(ctx.robot)
    return action(message, state=ctx.state.snapshot())


@router.post("/robot/cancel", response_model=ActionResponse)
def cancel_robot_calibration(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.cancel_robot_calibration()
    return action(message, state=ctx.state.snapshot())


@router.post("/robot/load", response_model=ActionResponse)
def load_robot_calibration(payload: PathRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.calibration.load_robot(payload.path)
    return action(message, state=ctx.state.snapshot())


@router.post("/robot/verify", response_model=ActionResponse)
async def verify_robot_calibration(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = await ctx.calibration.verify_robot_calibration(ctx.robot)
    return action(message, state=ctx.state.snapshot())
