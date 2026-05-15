"""Command-line entrypoint for the FastAPI runtime."""

from __future__ import annotations

import argparse
import logging

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run codroid-ball-picker FastAPI UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    uvicorn.run(
        "codroid_picker.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()
