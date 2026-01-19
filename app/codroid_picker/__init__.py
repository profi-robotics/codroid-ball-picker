# Codroid Picker package

# Import all functions from the main module to make them available
# when importing app.codroid_picker

from .session import CALIBRATION_PROGRESS, SESSION, _refresh_session_auto_config_defaults
import sys


def __getattr__(name: str):
    """Lazy attribute access - get attributes from codroid_picker_main module.

    This is called when an attribute is not found in the module's namespace.
    It defers to the main module, which handles the circular import issue
    during startup.
    """
    if "codroid_picker_main" in sys.modules:
        main_module = sys.modules["codroid_picker_main"]
        if hasattr(main_module, name):
            return getattr(main_module, name)
    raise AttributeError(f"module 'app.codroid_picker' has no attribute '{name}'")
