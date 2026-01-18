"""Module initialization and wiring for the codroid_picker application.

This module initializes the central AppRegistry with all dependencies.
It should be called once during application startup after all modules
have been imported.
"""

from __future__ import annotations

from app.codroid_picker import gripper_utils as gripper_utils_module
from app.codroid_picker.app_instance import registry
from app.codroid_picker.components import set_builder_dependencies


def initialize_modules(
    *,
    # Core state objects
    session,
    settings,
    gripper_defaults: dict,
    robot_coordinate_id: int,
    # Path constants
    drop_target_path,
    bundle_path,
    workspace_path,
    robot_calibration_path,
    model_dir,
    default_robot_uri: str,
    # Stream references
    mjpeg_stream,
    raw_stream,
    # Logger
    logger=None,
    # Component builder dependencies
    gpu_status_fn,
    model_options_fn,
) -> None:
    """Initialize all modules with their required dependencies.

    This function initializes the central AppRegistry which provides
    access to all shared state. Individual modules access dependencies
    via registry() instead of attach_* functions.
    """
    # Initialize the central registry
    registry().initialize(
        session=session,
        settings=settings,
        gripper_defaults=gripper_defaults,
        robot_coordinate_id=robot_coordinate_id,
        default_robot_uri=default_robot_uri,
        drop_target_path=drop_target_path,
        bundle_path=bundle_path,
        workspace_path=workspace_path,
        robot_calibration_path=robot_calibration_path,
        model_dir=model_dir,
        mjpeg_stream=mjpeg_stream,
        raw_stream=raw_stream,
        logger=logger,
    )

    # Refresh gripper constants from settings
    gripper_utils_module.refresh_gripper_constants()

    # Initialize component builder dependencies (still needed)
    set_builder_dependencies(
        gpu_status_fn=gpu_status_fn,
        model_options_fn=model_options_fn,
    )


def restore_persisted_state(restore_fn) -> None:
    """Restore any persisted application state.

    Args:
        restore_fn: Function to call to restore persisted state.
    """
    restore_fn()


def auto_load_model(settings, activate_model_fn, logger) -> None:
    """Auto-load model on startup if one is configured.

    Args:
        settings: Settings object with last_model path.
        activate_model_fn: Function to activate a model.
        logger: Logger instance for status messages.
    """
    from pathlib import Path

    # Check for last_model since that's what gets persisted
    model_setting = settings.active_model or settings.last_model
    if not model_setting:
        return

    try:
        model_path = Path(model_setting)
        if model_path.exists():
            activate_model_fn(model_path)
            logger.info(f"Auto-loaded model on startup: {model_path.name}")
        else:
            logger.warning(f"Configured model not found: {model_path}")
    except Exception as exc:
        logger.error(f"Failed to auto-load model: {exc}")
