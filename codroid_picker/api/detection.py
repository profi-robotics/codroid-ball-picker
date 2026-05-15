"""Detection API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import ActionResponse, DetectionParamsRequest, DetectionRunRequest, action
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/detection")


@router.get("/params")
def detection_params(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return ctx.detection.params()


@router.put("/params", response_model=ActionResponse)
def update_detection_params(payload: DetectionParamsRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    updates = payload.model_dump(exclude_none=True)
    ctx.detection.update_params(**updates)
    return action("Detection parameters updated.", state=ctx.state.snapshot())


@router.post("/run")
def run_detection(payload: DetectionRunRequest, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return {"ok": True, "detections": ctx.detection.run_once(force=payload.force)}
