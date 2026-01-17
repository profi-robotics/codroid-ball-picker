"""Type-safe conversion utilities with default values.

These functions safely convert values to specific types, returning
a default value if conversion fails or the input is None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def safe_int(value: Optional[Union[int, float, str]], default: int) -> int:
    """Safely convert a value to int with a default fallback.

    Args:
        value: Value to convert. Can be int, float, str, or None.
        default: Default value to return if conversion fails.

    Returns:
        Integer value, or default if conversion fails or value is None.

    Examples:
        >>> safe_int(3.7, 0)
        3
        >>> safe_int(None, 10)
        10
        >>> safe_int("invalid", 5)
        5
    """
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_float(value: Optional[Union[int, float, str]], default: float) -> float:
    """Safely convert a value to float with a default fallback.

    Args:
        value: Value to convert. Can be int, float, str, or None.
        default: Default value to return if conversion fails.

    Returns:
        Float value, or default if conversion fails or value is None.

    Examples:
        >>> safe_float("3.14", 0.0)
        3.14
        >>> safe_float(None, 1.5)
        1.5
        >>> safe_float("invalid", 0.0)
        0.0
    """
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def safe_path(value: Optional[Union[str, Path]], default: Path) -> Path:
    """Safely convert a value to Path with a default fallback.

    Args:
        value: Value to convert. Can be str, Path, or None.
        default: Default Path to return if conversion fails.

    Returns:
        Path object, or default if value is None or conversion fails.

    Examples:
        >>> safe_path("/some/path", Path("."))
        PosixPath('/some/path')
        >>> safe_path(None, Path("/default"))
        PosixPath('/default')
    """
    try:
        return Path(value) if value is not None else default
    except (ValueError, TypeError):
        return default
