"""Application-level exceptions for the FastAPI runtime."""

from __future__ import annotations


class CodroidPickerError(RuntimeError):
    """Base class for expected operator-facing errors."""


class HardwareUnavailableError(CodroidPickerError):
    """Raised when a requested hardware adapter is not connected or available."""


class OperationConflictError(CodroidPickerError):
    """Raised when an operation cannot start because another one is active."""


class ValidationError(CodroidPickerError):
    """Raised when a request is well-formed but invalid for the current state."""

