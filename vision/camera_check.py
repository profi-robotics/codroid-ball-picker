"""Utility to verify an OAK-D camera is enumerated and reachable."""

from __future__ import annotations

import logging


def _log_missing_dependency() -> None:
    logging.error(
        "depthai is not installed. Install dependencies with `uv install` (reads pyproject.toml) before running the camera check."
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
        mxid_fn = getattr(device, "getMxId", None)
        device_id_fn = getattr(device, "getDeviceId", None)
        name = None
        if callable(mxid_fn):
            name = mxid_fn()
        elif callable(device_id_fn):
            name = device_id_fn()
        else:
            name = str(device)
        state = getattr(device, "state", None)
        logging.info("%d) %s (%s)", idx, name, state)


if __name__ == "__main__":
    main()
