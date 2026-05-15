"""FastAPI dependency helpers."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from codroid_picker.services.context import AppContext


def get_context(request: Request) -> AppContext:
    return cast(AppContext, request.app.state.context)
