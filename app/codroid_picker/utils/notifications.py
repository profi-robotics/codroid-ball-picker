"""Factory functions for Mantine notifications.

Provides standardized notification creation with consistent styling
for success, error, warning, and info messages.
"""

from __future__ import annotations

from typing import Optional, Union

import dash_mantine_components as dmc


def success_notification(
    title: str,
    message: str = "",
    auto_close: Union[int, bool] = 5000,
    icon: Optional[str] = None,
) -> dmc.Notification:
    """Create a success notification (green).

    Args:
        title: Notification title.
        message: Optional notification message body.
        auto_close: Milliseconds before auto-close, or False to disable.
        icon: Optional icon name.

    Returns:
        Configured dmc.Notification component.
    """
    return dmc.Notification(
        action="show",
        title=title,
        message=message,
        color="green",
        autoClose=auto_close,
        icon=icon,
    )


def error_notification(
    title: str,
    message: str = "",
    auto_close: Union[int, bool] = 7000,
    icon: Optional[str] = None,
) -> dmc.Notification:
    """Create an error notification (red).

    Args:
        title: Notification title.
        message: Optional notification message body.
        auto_close: Milliseconds before auto-close, or False to disable.
        icon: Optional icon name.

    Returns:
        Configured dmc.Notification component.
    """
    return dmc.Notification(
        action="show",
        title=title,
        message=message,
        color="red",
        autoClose=auto_close,
        icon=icon,
    )


def warning_notification(
    title: str,
    message: str = "",
    auto_close: Union[int, bool] = 4500,
    icon: Optional[str] = None,
) -> dmc.Notification:
    """Create a warning notification (yellow/orange).

    Args:
        title: Notification title.
        message: Optional notification message body.
        auto_close: Milliseconds before auto-close, or False to disable.
        icon: Optional icon name.

    Returns:
        Configured dmc.Notification component.
    """
    return dmc.Notification(
        action="show",
        title=title,
        message=message,
        color="yellow",
        autoClose=auto_close,
        icon=icon,
    )


def info_notification(
    title: str,
    message: str = "",
    auto_close: Union[int, bool] = 5000,
    icon: Optional[str] = None,
) -> dmc.Notification:
    """Create an info notification (blue).

    Args:
        title: Notification title.
        message: Optional notification message body.
        auto_close: Milliseconds before auto-close, or False to disable.
        icon: Optional icon name.

    Returns:
        Configured dmc.Notification component.
    """
    return dmc.Notification(
        action="show",
        title=title,
        message=message,
        color="blue",
        autoClose=auto_close,
        icon=icon,
    )
