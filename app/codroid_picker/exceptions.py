"""Shared exceptions for the codroid_picker application.

This module provides common exception classes used across robot operations
and calibration procedures.
"""

from __future__ import annotations


class OperationPaused(Exception):
    """Raised when an operation is paused and should wait.

    Operations that support pause/resume should catch this exception
    and call wait_for_operation_resume() to block until resumed or stopped.
    """

    pass


class OperationAbort(Exception):
    """Raised when an operation should stop entirely.

    This exception signals that the operation has been cancelled by the user
    or due to an error condition, and should clean up and exit gracefully.
    """

    pass


class TrackingLost(Exception):
    """Raised when target tracking is lost during a follow move.

    This exception signals that the target can no longer be detected and
    the robot should retreat to a safe position (e.g., drop approach).
    """

    pass

