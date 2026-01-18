"""Decorator-based callback registry for Dash applications.

This module provides a centralized way to register Dash callbacks using decorators,
reducing boilerplate and duplicate registration patterns across robot_callbacks
and ui_callbacks packages.

Usage:
    from app.codroid_picker.callback_registry import callback, get_registry

    @callback(
        outputs=[Output("my-output", "children")],
        inputs=[Input("my-input", "n_clicks")],
        states=[State("my-state", "data")],
    )
    async def my_callback(n_clicks: int, state_data: dict) -> str:
        return f"Clicked {n_clicks} times"

    # In app setup:
    get_registry().register_all(app)
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

from dash import Dash, Input, Output, State

LOGGER = logging.getLogger(__name__)


@dataclass
class CallbackSpec:
    """Specification for a single Dash callback."""

    func: Callable
    outputs: List[Output]
    inputs: List[Input]
    states: List[State] = field(default_factory=list)
    prevent_initial_call: bool = False
    background: bool = False
    module: str = ""

    def register(self, app: Dash) -> None:
        """Register this callback with the Dash app."""
        kwargs = {"prevent_initial_call": self.prevent_initial_call}
        if self.background:
            kwargs["background"] = True

        app.callback(
            *self.outputs,
            *self.inputs,
            *self.states,
            **kwargs,
        )(self.func)


class CallbackRegistry:
    """Central registry for all Dash callbacks.

    Provides decorator-based registration and bulk registration to app.
    """

    def __init__(self) -> None:
        self._callbacks: List[CallbackSpec] = []
        self._registered_apps: set = set()

    def add(self, spec: CallbackSpec) -> None:
        """Add a callback specification to the registry."""
        self._callbacks.append(spec)

    def register_all(self, app: Dash) -> None:
        """Register all callbacks with the Dash app.

        Guards against duplicate registration for the same app instance.
        """
        app_id = id(app)
        if app_id in self._registered_apps:
            LOGGER.debug("Callbacks already registered for app %s", app_id)
            return

        self._registered_apps.add(app_id)
        for spec in self._callbacks:
            try:
                spec.register(app)
                LOGGER.debug(
                    "Registered callback %s.%s",
                    spec.module,
                    spec.func.__name__,
                )
            except Exception as exc:
                LOGGER.error(
                    "Failed to register callback %s.%s: %s",
                    spec.module,
                    spec.func.__name__,
                    exc,
                )
                raise

        LOGGER.info("Registered %d callbacks with app", len(self._callbacks))

    def clear(self) -> None:
        """Clear all registered callbacks (for testing)."""
        self._callbacks.clear()
        self._registered_apps.clear()

    @property
    def callback_count(self) -> int:
        """Return the number of registered callbacks."""
        return len(self._callbacks)


# Global registry instance
_REGISTRY: Optional[CallbackRegistry] = None


def get_registry() -> CallbackRegistry:
    """Get the global callback registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = CallbackRegistry()
    return _REGISTRY


def _normalize_to_list(
    items: Union[Any, Sequence[Any], None],
    item_type: type,
) -> List[Any]:
    """Normalize a single item or sequence to a list."""
    if items is None:
        return []
    if isinstance(items, item_type):
        return [items]
    if isinstance(items, (list, tuple)):
        return list(items)
    return [items]


def callback(
    outputs: Union[Output, List[Output]],
    inputs: Union[Input, List[Input]],
    states: Union[State, List[State], None] = None,
    prevent_initial_call: bool = False,
    background: bool = False,
) -> Callable:
    """Decorator to register a Dash callback with the global registry.

    Args:
        outputs: Single Output or list of Outputs
        inputs: Single Input or list of Inputs
        states: Optional single State or list of States
        prevent_initial_call: If True, callback won't fire on initial load
        background: If True, callback runs in background worker

    Returns:
        Decorator function that registers the callback

    Example:
        @callback(
            outputs=Output("result", "children"),
            inputs=Input("button", "n_clicks"),
            states=State("input", "value"),
        )
        def handle_click(n_clicks, value):
            return f"Clicked {n_clicks}: {value}"
    """

    def decorator(func: Callable) -> Callable:
        spec = CallbackSpec(
            func=func,
            outputs=_normalize_to_list(outputs, Output),
            inputs=_normalize_to_list(inputs, Input),
            states=_normalize_to_list(states, State),
            prevent_initial_call=prevent_initial_call,
            background=background,
            module=func.__module__,
        )
        get_registry().add(spec)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def callback_with_duplicate_outputs(
    outputs: Union[Output, List[Output]],
    inputs: Union[Input, List[Input]],
    states: Union[State, List[State], None] = None,
    prevent_initial_call: bool = True,
) -> Callable:
    """Decorator for callbacks that use allow_duplicate=True on outputs.

    This is a convenience wrapper that sets prevent_initial_call=True by default
    (required for duplicate outputs) and handles the allow_duplicate flag.

    Args:
        outputs: Outputs with allow_duplicate=True set
        inputs: Callback inputs
        states: Optional callback states
        prevent_initial_call: Must be True for duplicate outputs (default)

    Returns:
        Decorator function
    """
    return callback(
        outputs=outputs,
        inputs=inputs,
        states=states,
        prevent_initial_call=prevent_initial_call,
    )
