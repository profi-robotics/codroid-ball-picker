"""Run the Dash calibration UI."""

from __future__ import annotations

import argparse

from app.calibration_ui import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Codroid calibration UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the Dash server.")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the Dash server.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
