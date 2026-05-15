"""Small JSON-backed i18n helper."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SUPPORTED_LOCALES = ("en", "uk")
DEFAULT_LOCALE = "en"
_DIR = Path(__file__).resolve().parent
_LEGACY_DIR = _DIR.parents[1] / "app" / "codroid_picker" / "locales"


def normalize_locale(value: str | None) -> str:
    if not value:
        return DEFAULT_LOCALE
    cleaned = value.strip().lower().replace("_", "-")
    if cleaned.startswith("uk"):
        return "uk"
    return "en"


@lru_cache(maxsize=16)
def catalog(locale: str) -> dict[str, str]:
    normalized = normalize_locale(locale)
    merged: dict[str, str] = {}
    for path in (_LEGACY_DIR / f"{normalized}.json", _DIR / f"{normalized}.json"):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict):
            merged.update({str(key): str(value) for key, value in raw.items() if str(key).strip()})
    return merged


def t(key: str, locale: str = DEFAULT_LOCALE, **kwargs: Any) -> str:
    text = catalog(locale).get(key) or catalog(DEFAULT_LOCALE).get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text
