"""UI callbacks package - organized by functional area.

This package contains UI-related Dash callbacks split into logical modules:
- footer: Footer visibility toggle and styling
- stepper: Calibration stepper navigation and menu state
- model: Model upload/load and GPU status
- notifications: Connection status notifications
- settings: Settings save/reset operations
- monitoring: Monitoring summaries and operation state
- prerequisites: System prerequisites state tracking
"""

from dash import Dash

from .footer import register_footer_callbacks
from .model import register_model_callbacks
from .localization import register_localization_callbacks
from .monitoring import register_monitoring_callbacks
from .notifications import register_notification_callbacks
from .prerequisites import register_prerequisites_callbacks
from .settings import register_settings_callbacks
from .stepper import register_stepper_callbacks


def register_ui_callbacks(app: Dash) -> None:
    """Register all UI-related callbacks with the app."""
    # Guard against duplicate registration
    if hasattr(app, '_ui_callbacks_registered'):
        return
    app._ui_callbacks_registered = True

    register_footer_callbacks(app)
    register_stepper_callbacks(app)
    register_model_callbacks(app)
    register_localization_callbacks(app)
    register_notification_callbacks(app)
    register_settings_callbacks(app)
    register_monitoring_callbacks(app)
    register_prerequisites_callbacks(app)


__all__ = [
    "register_ui_callbacks",
    "register_footer_callbacks",
    "register_stepper_callbacks",
    "register_model_callbacks",
    "register_localization_callbacks",
    "register_notification_callbacks",
    "register_settings_callbacks",
    "register_monitoring_callbacks",
    "register_prerequisites_callbacks",
]
