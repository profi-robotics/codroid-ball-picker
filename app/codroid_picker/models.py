"""
YOLO model loading and management utilities.

This module provides functions for managing YOLO detection models,
including uploading, listing, downloading, and activating models.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.settings import _commit_settings
from app.codroid_picker.session import SESSION, normalize_auto_label_profiles
import app.settings as app_settings

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = app_settings.DEFAULT_MODEL_DIR

# Default model to download if none available
DEFAULT_MODEL_NAME = "yolov8n.pt"
MODEL_BENCHMARK_RUNS = 3
MODEL_BENCHMARK_WARMUP_RUNS = 1


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


def extract_model_labels(model: object) -> list[str]:
    """Extract class labels from an Ultralytics-compatible model."""
    names = getattr(model, "names", None)
    if callable(names):
        names = names()
    if isinstance(names, dict):
        items = sorted(names.items(), key=lambda item: int(item[0]))
        return [str(label).strip() for _idx, label in items if str(label).strip()]
    if isinstance(names, (list, tuple)):
        return [str(label).strip() for label in names if str(label).strip()]
    return []


def model_label_options() -> list[dict]:
    """Return selectable label options for the active model."""
    return [{"value": label, "label": label} for label in SESSION.model_labels]


def resolve_selected_pick_labels(
    available_labels: list[str],
    current_labels: Optional[list[str]] = None,
) -> list[str]:
    """Keep selected labels compatible with the active model labels."""
    available = [str(label).strip() for label in available_labels if str(label).strip()]
    if not available:
        return []
    current = current_labels
    if current is None:
        current = SESSION.auto_config.get("target_labels") or app_settings.SETTINGS.auto_target_labels
    selected = [str(label).strip() for label in current if str(label).strip()]
    selected = [label for label in selected if label in available]
    if selected:
        return selected
    legacy = str(
        SESSION.auto_config.get("target_label") or app_settings.SETTINGS.auto_target_label
    ).strip()
    if legacy in available:
        return [legacy]
    return [available[0]]


def _benchmark_rating(avg_latency_ms: float) -> str:
    """Classify measured model latency for embedded usage."""
    if avg_latency_ms <= 150.0:
        return "usable"
    if avg_latency_ms <= 350.0:
        return "warning"
    return "poor"


def _benchmark_frame() -> tuple[np.ndarray, str]:
    """Return a benchmark frame and source description."""
    try:
        import app.codroid_picker as codroid_picker_module

        frame, _depth = codroid_picker_module.get_latest_frame()
        if frame is not None:
            return frame, "camera"
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Unable to read current frame for benchmark: %s", exc)

    width, height = app_settings.SETTINGS.camera_resolution
    frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    return frame, "synthetic"


def run_model_benchmark(model: Optional[object] = None) -> dict[str, Any]:
    """Measure inference latency for the active model on this hardware."""
    yolo_model = model or SESSION.yolo_model
    if yolo_model is None:
        return {
            "status": "unavailable",
            "message": "No model loaded.",
        }

    frame, frame_source = _benchmark_frame()
    timings_ms: list[float] = []
    try:
        for idx in range(MODEL_BENCHMARK_WARMUP_RUNS + MODEL_BENCHMARK_RUNS):
            started = time.perf_counter()
            yolo_model.predict(frame, max_det=1, verbose=False)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if idx >= MODEL_BENCHMARK_WARMUP_RUNS:
                timings_ms.append(elapsed_ms)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "error",
            "message": f"Benchmark failed: {exc}",
            "frame_source": frame_source,
            "input_shape": list(frame.shape),
        }
        SESSION.model_benchmark = result
        return result

    avg_latency_ms = sum(timings_ms) / len(timings_ms)
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    try:
        from app.codroid_picker.gpu_utils import get_gpu_info

        gpu_info = get_gpu_info()
        backend = gpu_info.get("name") or "CPU"
    except Exception:  # noqa: BLE001
        backend = "unknown"

    result = {
        "status": "ok",
        "rating": _benchmark_rating(avg_latency_ms),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "fps": round(fps, 2),
        "backend": backend,
        "frame_source": frame_source,
        "input_shape": list(frame.shape),
        "runs": MODEL_BENCHMARK_RUNS,
    }
    SESSION.model_benchmark = result
    return result


def format_model_benchmark(result: Optional[dict] = None) -> str:
    """Format a benchmark result for compact UI display."""
    payload = result or SESSION.model_benchmark
    if not payload:
        return "Model benchmark not run yet."
    if payload.get("status") != "ok":
        return str(payload.get("message") or "Model benchmark unavailable.")
    return (
        f"Benchmark: {payload['rating']} · {payload['avg_latency_ms']} ms "
        f"({payload['fps']} FPS) · {payload['backend']} · {payload['frame_source']} frame"
    )


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
        SESSION.model_labels = []
        SESSION.model_benchmark = {}
        _commit_settings(last_model=None, active_model=None)
        return "No model selected."
    if not model_path.exists():
        SESSION.active_model_path = None
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        SESSION.model_labels = []
        SESSION.model_benchmark = {}
        raise FileNotFoundError(f"Model not found: {model_path}")
    SESSION.active_model_path = model_path
    try:
        SESSION.yolo_model = _load_ultralytics_model(model_path)
        SESSION.model_loaded = True
        SESSION.last_detection_error = None
        SESSION.model_labels = extract_model_labels(SESSION.yolo_model)
        selected_labels = resolve_selected_pick_labels(SESSION.model_labels)
        SESSION.auto_config["target_labels"] = selected_labels
        if selected_labels:
            SESSION.auto_config["target_label"] = selected_labels[0]
        SESSION.auto_config["label_profiles"] = normalize_auto_label_profiles(
            selected_labels,
            {
                **app_settings.SETTINGS.auto_label_profiles,
                **SESSION.auto_config.get("label_profiles", {}),
            },
        )
        _commit_settings(last_model=str(model_path),
                         active_model=str(model_path))
        LOGGER.info(
            f"Model loaded and detection auto-started: {model_path.name}")
        label_count = len(SESSION.model_labels)
        return (
            f"Model loaded: {model_path.name}. "
            f"{label_count} label(s) available. Detection auto-started."
        )
    except Exception as exc:
        SESSION.yolo_model = None
        SESSION.model_loaded = False
        SESSION.model_labels = []
        SESSION.model_benchmark = {}
        SESSION.last_detection_error = str(exc)
        return f"Model ready but not loaded: {exc}"
