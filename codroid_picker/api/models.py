"""Model management API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile

from codroid_picker.api.schemas import ActionResponse, ModelActivateRequest, action
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/models")


@router.get("")
def list_models(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    detection_state = ctx.state.domain("detection")
    return {
        "models": ctx.detection.list_models(),
        "active": detection_state.get("active_model"),
        "labels": detection_state.get("labels", []),
        "benchmark": detection_state.get("benchmark", {"status": "not_run"}),
    }


@router.post("/upload", response_model=ActionResponse)
async def upload_model(file: UploadFile = File(...), ctx: AppContext = Depends(get_context)) -> ActionResponse:
    model_dir = ctx.settings.current.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "model.bin").name
    path = model_dir / safe_name
    path.write_bytes(await file.read())
    return action(f"Uploaded model {safe_name}.", state={"path": str(path)})


@router.post("/activate", response_model=ActionResponse)
def activate_model(payload: ModelActivateRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    state = ctx.detection.activate(payload.path)
    return action("Model activation updated.", state=state)


@router.post("/benchmark")
def benchmark_model(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return ctx.detection.benchmark()
