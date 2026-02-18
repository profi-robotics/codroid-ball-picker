from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import app.settings as app_settings

SUPPORTED_LOCALES = ("en", "uk")
DEFAULT_LOCALE = "en"
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"


def normalize_locale(value: str | None) -> str:
    """Normalize locale input to a supported locale."""
    if not value:
        return DEFAULT_LOCALE
    cleaned = value.strip().lower().replace("_", "-")
    if cleaned.startswith("uk"):
        return "uk"
    if cleaned.startswith("en"):
        return "en"
    return DEFAULT_LOCALE


def _catalog_mtime_ns(locale: str) -> int:
    path = _LOCALES_DIR / f"{normalize_locale(locale)}.json"
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


@lru_cache(maxsize=16)
def _load_catalog_cached(locale: str, mtime_ns: int) -> dict[str, str]:
    locale = normalize_locale(locale)
    path = _LOCALES_DIR / f"{locale}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _load_catalog(locale: str) -> dict[str, str]:
    return _load_catalog_cached(normalize_locale(locale), _catalog_mtime_ns(locale))


def current_locale() -> str:
    """Read current locale from persisted settings."""
    return normalize_locale(getattr(app_settings.SETTINGS, "ui_language", DEFAULT_LOCALE))


def t(key: str, locale: str | None = None, **kwargs: Any) -> str:
    """Translate a string key with English fallback and optional formatting."""
    selected = normalize_locale(locale or current_locale())
    text = _load_catalog(selected).get(key)
    if text is None:
        text = _load_catalog(DEFAULT_LOCALE).get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text
