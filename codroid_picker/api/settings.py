"""Settings API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import ActionResponse, LocaleRequest, SettingsUpdateRequest, action
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter()


@router.get("/settings")
def get_settings(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return {"settings": ctx.settings.as_dict()}


@router.put("/settings", response_model=ActionResponse)
def put_settings(payload: SettingsUpdateRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    settings = ctx.settings.update(payload.updates)
    return action("Settings saved.", state={"settings": settings.model_dump(mode="json")})


@router.post("/settings/reset", response_model=ActionResponse)
def reset_settings(ctx: AppContext = Depends(get_context)) -> ActionResponse:
    settings = ctx.settings.reset()
    return action("Settings reset.", state={"settings": settings.model_dump(mode="json")})


@router.post("/locale", response_model=ActionResponse)
def set_locale(payload: LocaleRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    settings = ctx.settings.update({"ui_language": payload.language})
    return action("Locale updated.", state={"language": settings.ui_language})
