"""Robot program registry.

This module provides a central registry for managing robot programs.
It allows programs to be registered, started, and controlled through
a unified interface that the UI callbacks can use.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Dict, Optional, Type

if TYPE_CHECKING:
    from codroid_api import CodroidAPI

from app.codroid_picker.programs.base import (
    RobotProgram,
    ProgramStatus,
    ProgramResult,
)

LOGGER = logging.getLogger(__name__)


class ProgramRegistry:
    """Registry for robot programs.

    This singleton manages all registered robot programs and provides
    a unified interface for starting, pausing, resuming, and stopping
    programs. Only one program can be active at a time.

    Usage:
        # Register programs (typically at startup)
        registry.register(VerificationProgram)
        registry.register(PlateTestProgram)

        # Start a program
        result = await registry.start("verification", robot_api)

        # Control the active program
        registry.pause()
        registry.resume()
        registry.stop()
    """

    _instance: Optional["ProgramRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ProgramRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for clean hot-reload.

        This clears the singleton so the next call creates a fresh instance.
        """
        global _registry
        with cls._lock:
            cls._instance = None
            _registry = None

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._programs: Dict[str, Type[RobotProgram]] = {}
        self._active_program: Optional[RobotProgram] = None
        self._active_task: Optional[asyncio.Task] = None

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, program_class: Type[RobotProgram]) -> None:
        """Register a program class.

        Args:
            program_class: The RobotProgram subclass to register.
        """
        # Create a temporary instance to get the name
        instance = program_class()
        name = instance.name
        self._programs[name] = program_class
        LOGGER.info(f"Registered program: {name} ({instance.label})")

    def unregister(self, name: str) -> None:
        """Unregister a program by name."""
        if name in self._programs:
            del self._programs[name]
            LOGGER.info(f"Unregistered program: {name}")

    def list_programs(self) -> Dict[str, str]:
        """List all registered programs.

        Returns:
            Dict mapping program names to their labels.
        """
        result = {}
        for name, cls in self._programs.items():
            instance = cls()
            result[name] = instance.label
        return result

    # -------------------------------------------------------------------------
    # Program Access
    # -------------------------------------------------------------------------

    def get_program(self, name: str) -> Optional[RobotProgram]:
        """Get a new instance of a registered program.

        Args:
            name: Program name.

        Returns:
            New program instance, or None if not registered.
        """
        cls = self._programs.get(name)
        if cls is None:
            return None
        return cls()

    @property
    def active_program(self) -> Optional[RobotProgram]:
        """Get the currently active program instance."""
        return self._active_program

    @property
    def is_program_running(self) -> bool:
        """Check if any program is currently running."""
        return (
            self._active_program is not None
            and self._active_program.is_running
        )

    @property
    def active_program_name(self) -> Optional[str]:
        """Get the name of the active program, or None."""
        if self._active_program is None:
            return None
        return self._active_program.name

    @property
    def active_program_status(self) -> Optional[ProgramStatus]:
        """Get the status of the active program, or None."""
        if self._active_program is None:
            return None
        return self._active_program.status

    # -------------------------------------------------------------------------
    # Program Control
    # -------------------------------------------------------------------------

    async def start(
        self,
        name: str,
        robot_api: "CodroidAPI",
        *,
        initial_message: Optional[str] = None,
    ) -> ProgramResult:
        """Start a registered program.

        Args:
            name: Program name to start.
            robot_api: Connected robot API instance.
            initial_message: Optional initial status message.

        Returns:
            ProgramResult with execution result.

        Raises:
            ValueError: If program not registered.
            RuntimeError: If another program is already running.
        """
        if self.is_program_running:
            raise RuntimeError(
                f"Cannot start {name}: another program is already running "
                f"({self.active_program_name})"
            )

        cls = self._programs.get(name)
        if cls is None:
            raise ValueError(f"Program not registered: {name}")

        program = cls()
        self._active_program = program

        try:
            result = await program.start(
                robot_api,
                initial_message=initial_message,
            )
            return result
        finally:
            # Clear active program if it's done
            if not program.is_running:
                self._active_program = None

    async def start_instance(
        self,
        program: RobotProgram,
        robot_api: "CodroidAPI",
        *,
        initial_message: Optional[str] = None,
    ) -> ProgramResult:
        """Start a pre-created program instance.

        This is useful when the program needs to be configured with
        specific parameters before starting.

        Args:
            program: Pre-created program instance.
            robot_api: Connected robot API instance.
            initial_message: Optional initial status message.

        Returns:
            ProgramResult with execution result.

        Raises:
            RuntimeError: If another program is already running.
        """
        if self.is_program_running:
            raise RuntimeError(
                f"Cannot start {program.name}: another program is already running "
                f"({self.active_program_name})"
            )

        self._active_program = program

        try:
            result = await program.start(
                robot_api,
                initial_message=initial_message,
            )
            return result
        finally:
            # Clear active program if it's done
            if not program.is_running:
                self._active_program = None

    def start_background(
        self,
        name: str,
        robot_api: "CodroidAPI",
        *,
        initial_message: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> None:
        """Start a program as a background task.

        This is useful for starting programs from synchronous callbacks.
        The program runs in the background and the callback is called
        when it completes.

        Args:
            name: Program name to start.
            robot_api: Connected robot API instance.
            initial_message: Optional initial status message.
            callback: Optional callback(result) when program completes.
        """
        async def _run():
            try:
                result = await self.start(
                    name,
                    robot_api,
                    initial_message=initial_message,
                )
                if callback is not None:
                    callback(result)
            except Exception as exc:
                LOGGER.exception(f"Background program {name} failed")
                if callback is not None:
                    callback(ProgramResult(
                        status=ProgramStatus.ERROR,
                        message=str(exc),
                    ))

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            self._active_task = loop.create_task(_run())
        except RuntimeError:
            # No running loop, create one
            self._active_task = asyncio.create_task(_run())

    def pause(self, reason: str = "user_request") -> bool:
        """Pause the active program.

        Args:
            reason: Reason for pausing.

        Returns:
            True if a program was paused, False if no active program.
        """
        if self._active_program is None:
            return False
        self._active_program.request_pause(reason)
        LOGGER.info(f"Paused program: {self._active_program.name}")
        return True

    def resume(self) -> bool:
        """Resume the active program.

        Returns:
            True if a program was resumed, False if no active program.
        """
        if self._active_program is None:
            return False
        self._active_program.request_resume()
        LOGGER.info(f"Resumed program: {self._active_program.name}")
        return True

    def stop(self) -> bool:
        """Stop the active program.

        Returns:
            True if a program was stopped, False if no active program.
        """
        if self._active_program is None:
            return False
        self._active_program.request_stop()
        LOGGER.info(f"Stopped program: {self._active_program.name}")
        return True

    def clear(self) -> None:
        """Clear the active program reference.

        This should be called after a program completes to allow
        starting a new program.
        """
        self._active_program = None
        self._active_task = None


# Module-level singleton
_registry: Optional[ProgramRegistry] = None


def get_registry() -> ProgramRegistry:
    """Get the program registry singleton."""
    global _registry
    if _registry is None:
        _registry = ProgramRegistry()
    return _registry


def register_program(program_class: Type[RobotProgram]) -> None:
    """Register a program class with the global registry."""
    get_registry().register(program_class)


def get_active_program() -> Optional[RobotProgram]:
    """Get the currently active program."""
    return get_registry().active_program


def is_program_running() -> bool:
    """Check if any program is currently running."""
    return get_registry().is_program_running


def pause_program(reason: str = "user_request") -> bool:
    """Pause the active program."""
    return get_registry().pause(reason)


def resume_program() -> bool:
    """Resume the active program."""
    return get_registry().resume()


def stop_program() -> bool:
    """Stop the active program."""
    return get_registry().stop()
