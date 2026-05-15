from __future__ import annotations

from pathlib import Path


def test_fastapi_package_does_not_import_legacy_app_package() -> None:
    root = Path("codroid_picker")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "app.codroid_picker" in text:
            offenders.append(str(path))

    assert offenders == []
