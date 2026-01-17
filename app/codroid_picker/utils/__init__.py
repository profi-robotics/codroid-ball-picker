"""Utility functions and helpers for the codroid_picker application.

This package provides shared utilities:
- safe_types: Type-safe conversion functions with defaults
- placeholders: Placeholder frame generation for camera displays
- notifications: Factory functions for Mantine notifications
"""

from .safe_types import safe_float, safe_int, safe_path
from .placeholders import placeholder_frame
from .notifications import (
    success_notification,
    error_notification,
    warning_notification,
    info_notification,
)

__all__ = [
    # safe_types
    "safe_float",
    "safe_int",
    "safe_path",
    # placeholders
    "placeholder_frame",
    # notifications
    "success_notification",
    "error_notification",
    "warning_notification",
    "info_notification",
]
