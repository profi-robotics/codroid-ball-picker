"""Server-rendered page routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from codroid_picker.dependencies import get_context
from codroid_picker.i18n import catalog, normalize_locale
from codroid_picker.services.context import AppContext

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "frontend" / "templates"))


@router.get("/", response_class=HTMLResponse)
def index(request: Request, lang: str | None = None, ctx: AppContext = Depends(get_context)) -> HTMLResponse:
    locale = normalize_locale(lang or ctx.settings.current.ui_language)
    state = ctx.state.snapshot()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "locale": locale,
            "initial_state_json": json.dumps(state, ensure_ascii=False),
            "catalog_json": json.dumps(catalog(locale), ensure_ascii=False),
        },
    )

