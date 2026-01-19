"""Programs package for the calibration UI.

This package contains robot program implementations that can be
started, paused, resumed, and stopped through a unified interface.
"""

from app.codroid_picker.programs.base import (
    RobotProgram,
    ProgramStatus,
    ProgramResult,
    OperationPaused,
    OperationAbort,
)
from app.codroid_picker.programs.registry import (
    ProgramRegistry,
    get_registry,
    register_program,
    get_active_program,
    is_program_running,
    pause_program,
    resume_program,
    stop_program,
)
from app.codroid_picker.programs.verification import VerificationProgram
from app.codroid_picker.programs.manual_pick_place import ManualPickPlaceProgram
from app.codroid_picker.programs.auto_pick_place import AutoPickPlaceProgram
from app.codroid_picker.programs.plate_test import PlateTestProgram, PlateTestConfig
from app.codroid_picker.programs.robot_calibration import (
    RobotCalibrationProgram,
    RobotCalibrationConfig,
    RobotCalibrationResult,
)


def register_all_programs() -> None:
    """Register all built-in robot programs with the global registry."""
    registry = get_registry()
    registry.register(VerificationProgram)
    registry.register(ManualPickPlaceProgram)
    registry.register(AutoPickPlaceProgram)
    registry.register(PlateTestProgram)


__all__ = [
    # Base classes
    "RobotProgram",
    "ProgramStatus",
    "ProgramResult",
    "OperationPaused",
    "OperationAbort",
    # Registry
    "ProgramRegistry",
    "get_registry",
    "register_program",
    "get_active_program",
    "is_program_running",
    "pause_program",
    "resume_program",
    "stop_program",
    "register_all_programs",
    # Programs
    "VerificationProgram",
    "ManualPickPlaceProgram",
    "AutoPickPlaceProgram",
    "PlateTestProgram",
    "PlateTestConfig",
    "RobotCalibrationProgram",
    "RobotCalibrationConfig",
    "RobotCalibrationResult",
]
