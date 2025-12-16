"""Utility to verify an OAK-D camera is enumerated and reachable."""

from __future__ import annotations

import logging


def _log_missing_dependency() -> None:
    logging.error(
        "depthai is not installed. Install dependencies with `uv pip install -r requirements.txt` before running the camera check."
    )


def list_oak_devices():
    """Return a list of available OAK-D devices (empty if none are found)."""

    try:
        import depthai as dai
    except ImportError:
        _log_missing_dependency()
        return []

    devices = dai.Device.getAllAvailableDevices()
    return devices


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    devices = list_oak_devices()
    if not devices:
        logging.warning("No OAK-D devices detected. Check USB connection and permissions.")
        return

    logging.info("Detected %d OAK-D device(s):", len(devices))
    for idx, device in enumerate(devices, start=1):
        logging.info("%d) %s (%s)", idx, device.getMxId(), device.state)


if __name__ == "__main__":
    main()
