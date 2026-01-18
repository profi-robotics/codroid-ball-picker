"""Utility functions and helpers for the codroid_picker application.

This package provides shared utilities:
- safe_types: Type-safe conversion functions with defaults
- placeholders: Placeholder frame generation for camera displays
"""

from .safe_types import safe_float, safe_int, safe_path
from .placeholders import placeholder_frame

__all__ = [
    # safe_types
    "safe_float",
    "safe_int",
    "safe_path",
    # placeholders
    "placeholder_frame",
]
