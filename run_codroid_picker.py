"""Run the Dash calibration UI."""

from __future__ import annotations

import argparse
import logging

import importlib.util
import sys

# Load main.py directly to avoid package/module naming conflict
spec = importlib.util.spec_from_file_location(
    "codroid_picker_main", "app/main.py")
codroid_picker = importlib.util.module_from_spec(spec)
# Add to sys.modules to fix @dataclass decorator issues
sys.modules["codroid_picker_main"] = codroid_picker
spec.loader.exec_module(codroid_picker)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Codroid calibration UI.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind the Dash server.")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port to bind the Dash server.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Dash debug mode.")
    args = parser.parse_args()

    # Configure logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress asyncio debug spam even in debug mode
    logging.getLogger('asyncio').setLevel(logging.INFO)
    logging.getLogger('websockets').setLevel(logging.INFO)
    logging.getLogger('websockets.client').setLevel(logging.INFO)

    app = codroid_picker.create_app()
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
        dev_tools_hot_reload=False,
    )


if __name__ == "__main__":
    main()
