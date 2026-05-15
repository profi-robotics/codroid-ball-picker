"""Camera API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import ActionResponse, CameraFocusRequest, CameraStartRequest, OverlayRequest, action
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/camera")


@router.post("/start", response_model=ActionResponse)
def start_camera(payload: CameraStartRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    message = ctx.camera.start(**payload.model_dump())
    return action(message, state=ctx.state.snapshot())


@router.post("/stop", response_model=ActionResponse)
def stop_camera(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    ctx.camera.stop()
    return action("Camera stopped.", state=ctx.state.snapshot())


@router.get("/devices")
def camera_devices(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return {"devices": ctx.camera.scan()}


@router.post("/focus", response_model=ActionResponse)
def camera_focus(payload: CameraFocusRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    ctx.camera.focus(lock=payload.lock, lens_position=payload.lens_position)
    return action("Camera focus updated.", state=ctx.state.snapshot())


@router.put("/overlay", response_model=ActionResponse)
def camera_overlay(payload: OverlayRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    mode = ctx.overlay.set_mode(payload.mode)
    return action(f"Overlay set to {mode}.", state=ctx.state.snapshot())
