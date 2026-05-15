"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from codroid_picker.api import build_router
from codroid_picker.errors import CodroidPickerError
from codroid_picker.services.context import build_context


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.context = build_context()
    try:
        yield
    finally:
        await app.state.context.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="codroid-ball-picker", version="0.1.0", lifespan=lifespan)
    static_dir = Path(__file__).resolve().parent / "frontend" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    legacy_assets = Path("assets")
    if legacy_assets.exists():
        app.mount("/legacy-assets", StaticFiles(directory=str(legacy_assets)), name="legacy-assets")
    app.include_router(build_router())

    @app.exception_handler(CodroidPickerError)
    async def _codroid_error(_request: Request, exc: CodroidPickerError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"ok": False, "error": exc.__class__.__name__, "message": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def _not_found(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"ok": False, "error": "FileNotFoundError", "message": str(exc)})

    return app

