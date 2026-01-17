# Codroid Picker package

# Import all functions from the main module to make them available
# when importing app.codroid_picker

from .session import CALIBRATION_PROGRESS, SESSION, _refresh_session_auto_config_defaults
import importlib.util
import sys
import os

# Load the main.py module file directly to avoid circular import
_module_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location(
    "codroid_picker_main", _module_path)
_codroid_picker_module = importlib.util.module_from_spec(_spec)

# Add to sys.modules before executing to fix the current_module issue
sys.modules["codroid_picker_main"] = _codroid_picker_module
_spec.loader.exec_module(_codroid_picker_module)

# Expose all the public functions and classes from the main module
# This allows callbacks to import app.codroid_picker and access the functions
for _name in dir(_codroid_picker_module):
    if not _name.startswith('_') or _name in getattr(_codroid_picker_module, 'globals_to_expose', []):
        globals()[_name] = getattr(_codroid_picker_module, _name)

# Also expose session-related imports that were in the merged ui package
