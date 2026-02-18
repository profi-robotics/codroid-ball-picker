"""Localization and language-switch callbacks."""

from __future__ import annotations

import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dash import Dash, Input, Output, State, callback_context

from app.codroid_picker.i18n import normalize_locale


def _with_lang_nonce(href: str | None, locale: str) -> str:
    parsed = urlparse(href or "/")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["lang"] = locale
    query["l10n_ts"] = str(int(time.time() * 1000))
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            parsed.params,
            urlencode(query),
            parsed.fragment,
        )
    )


def register_localization_callbacks(app: Dash) -> None:
    """Register language selector synchronization and refresh callbacks."""

    @app.callback(
        Output("language-select-header", "value"),
        Output("language-select-setup", "value"),
        Output("ui-language-state", "data"),
        Output("app-location", "href"),
        Input("language-select-header", "value"),
        Input("language-select-setup", "value"),
        State("app-location", "href"),
        State("ui-language-state", "data"),
        prevent_initial_call=True,
    )
    def _switch_language(
        header_lang: str | None,
        setup_lang: str | None,
        href: str | None,
        lang_state: dict | None,
    ):
        import app.codroid_picker as codroid_picker_module

        trigger = callback_context.triggered_id
        current = normalize_locale((lang_state or {}).get("value"))
        header_norm = normalize_locale(header_lang)
        setup_norm = normalize_locale(setup_lang)

        if trigger == "language-select-header":
            selected = header_norm
        elif trigger == "language-select-setup":
            selected = setup_norm
        elif header_norm != current:
            selected = header_norm
        elif setup_norm != current:
            selected = setup_norm
        else:
            selected = current

        codroid_picker_module._commit_settings(ui_language=selected)
        codroid_picker_module.reload_defaults()

        return (
            selected,
            selected,
            {"value": selected},
            _with_lang_nonce(href, selected),
        )
