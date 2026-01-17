"""Robot API and device utilities.

Provides helper functions for probing robots, running actions via the API,
and formatting device lists for the UI.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from dash import html

from app.codroid_picker.app_instance import registry

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

LOGGER = logging.getLogger(__name__)

async def _probe_robot(uri: str) -> dict:
    """Probe a robot at the given URI and return status info."""
    robot_session = registry().get_or_create_robot_session()
    default_uri = registry().default_robot_uri
    return await robot_session.probe(uri or default_uri)


async def _with_robot_api(
    uri: str,
    action: Callable[["CodroidAPI"], Awaitable[None]],
) -> None:
    """Run an async action with a robot API connection."""
    robot_session = registry().get_or_create_robot_session()
    default_uri = registry().default_robot_uri
    await robot_session.run_with_robot(uri or default_uri, action)


async def _ensure_robot_coord5_for_uri(uri: str | None) -> None:
    """Switch the robot to CoordinateId 5 via the provided URI."""
    from app.codroid_picker.robot_motion import ensure_coordinate5_active

    async def _activate(api: "CodroidAPI") -> None:
        await ensure_coordinate5_active(api)

    default_uri = registry().default_robot_uri
    await _with_robot_api(uri or default_uri, _activate)
