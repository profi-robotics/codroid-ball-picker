"""
YOLO model loading and management utilities.

This module provides functions for managing YOLO detection models,
including uploading, listing, downloading, and activating models.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Optional

from app.settings import _commit_settings
from app.codroid_picker.session import SESSION
import app.settings as app_settings

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = app_settings.DEFAULT_MODEL_DIR

# Default model to download if none available
DEFAULT_MODEL_NAME = "yolov8n.pt"


# ---------------------------------------------------------------------------
# Frame capture utilities
# ---------------------------------------------------------------------------


def _frame_index(path: Path) -> int:
    """Extract the frame index from a calibration color frame filename."""
    match = re.search(r"calibration_color_(\d+)\.png", path.name)
    if match:
        return int(match.group(1))
    return 0


def _list_capture_frames(output_dir: Path) -> list[Path]:
    """List all calibration color frames in a directory, sorted by index."""
    frames = list(output_dir.glob("calibration_color_*.png"))
    frames.sort(key=_frame_index)
    return frames


def _update_capture_state(output_dir: Path) -> list[Path]:
    """Update the session capture state with frames from a directory."""
    SESSION.capture.output_dir = output_dir
    frames = _list_capture_frames(output_dir)
    SESSION.capture.frames = frames
    return frames


# ---------------------------------------------------------------------------
# Model file utilities
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str, fallback: str = "model.bin") -> str:
    """Sanitize a filename for safe filesystem use."""
    base = Path(name).name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return safe or fallback


def _list_model_files(output_dir: Path = DEFAULT_MODEL_DIR) -> list[Path]:
    """List all model files in a directory."""
    if not output_dir.exists():
        return []
    files = [path for path in output_dir.iterdir() if path.is_file()]
    return sorted(files, key=lambda path: path.name.lower())


def _model_options() -> list[dict]:
    """Generate select options for available models."""
    options = []
    for path in _list_model_files():
        options.append({"value": str(path), "label": path.name})
    return options


def download_default_model(output_dir: Path = DEFAULT_MODEL_DIR) -> Optional[Path]:
    """Download the default YOLOv8n model if no models are available.

    Args:
        output_dir: Directory to save the model to.

    Returns:
        Path to the downloaded model, or None if download failed.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        LOGGER.warning(
            "ultralytics not installed, cannot download default model. "
            "Run `uv add ultralytics` to enable."
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / DEFAULT_MODEL_NAME

    if model_path.exists():
        LOGGER.info("Default model already exists: %s", model_path)
        return model_path

    try:
        LOGGER.info("Downloading default YOLO model: %s", DEFAULT_MODEL_NAME)
        # YOLO() auto-downloads the model if not cached
        model = YOLO(DEFAULT_MODEL_NAME)
        # Save to our models directory
        model.save(model_path)
        LOGGER.info("Downloaded default model to: %s", model_path)
        return model_path
    except Exception as exc:
        LOGGER.error("Failed to download default model: %s", exc)
        return None


def ensure_model_available(output_dir: Path = DEFAULT_MODEL_DIR) -> Optional[Path]:
    """Ensure at least one model is available, downloading if necessary.

    This function checks if any models exist in the output directory.
    If none are found, it downloads the default YOLOv8n model.

    Args:
        output_dir: Directory to check/save models to.

    Returns:
        Path to an available model, or None if no models available.
    """
    existing = _list_model_files(output_dir)
    if existing:
        return existing[0]

    LOGGER.info("No models found, downloading default model...")
    return download_default_model(output_dir)


def save_uploaded_model(contents: str, filename: str, output_dir: Path) -> Path:
    """Save an uploaded model file from base64-encoded contents."""
    if "," not in contents:
        raise ValueError("Uploaded model payload is not base64 encoded.")
    _, encoded = contents.split(",", 1)
    data = base64.b64decode(encoded)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(filename or "model.bin")
    path = output_dir / safe_name
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Model loading and activation
# ---------------------------------------------------------------------------


def _load_ultralytics_model(model_path: Path) -> object:
    """Load a YOLO model using ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run `uv add ultralytics` to enable YOLO inference."
        ) from exc
    return YOLO(str(model_path))


def activate_model(model_path: Optional[Path]) -> str:
    """Activate a YOLO model for detection."""
    if model_path is None:
        SESSION.active_model_path = None
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        _commit_settings(last_model=None, active_model=None)
        return "No model selected."
    if not model_path.exists():
        SESSION.active_model_path = None
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        raise FileNotFoundError(f"Model not found: {model_path}")
    SESSION.active_model_path = model_path
    try:
        SESSION.yolo_model = _load_ultralytics_model(model_path)
        SESSION.model_loaded = True
        SESSION.last_detection_error = None
        _commit_settings(last_model=str(model_path),
                         active_model=str(model_path))
        LOGGER.info(
            f"Model loaded and detection auto-started: {model_path.name}")
        return f"Model loaded: {model_path.name}. Detection auto-started."
    except Exception as exc:
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        SESSION.last_detection_error = str(exc)
        return f"Model ready but not loaded: {exc}"
