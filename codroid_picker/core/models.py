"""Model file helper functions for the FastAPI runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def list_model_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return [
        path
        for path in sorted(model_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".engine"}
    ]


def extract_model_labels(model: object) -> list[str]:
    names: Any = getattr(model, "names", None)
    if callable(names):
        names = names()
    if isinstance(names, dict):
        return [str(label).strip() for _idx, label in sorted(names.items(), key=lambda item: int(item[0])) if str(label).strip()]
    if isinstance(names, (list, tuple)):
        return [str(label).strip() for label in names if str(label).strip()]
    return []


def benchmark_rating(avg_latency_ms: float) -> str:
    if avg_latency_ms <= 150.0:
        return "usable"
    if avg_latency_ms <= 350.0:
        return "warning"
    return "poor"

