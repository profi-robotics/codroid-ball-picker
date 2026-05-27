# Non-Jetson Deployment

Use these notes for x86 or other non-Jetson hardware, including development
workstations and robot-cell computers that should run the Dash/WebRTC app and
YOLO locally.

## Install

```bash
uv sync
```

On non-Jetson hosts, the base dependency set installs `ultralytics` plus CPU
PyTorch and torchvision from the PyTorch CPU wheel index on Python 3.11+.
Jetson-only PyTorch wheels are selected only for Linux ARM Python 3.10 hosts.

## Run

```bash
python run_codroid_picker_supervised.py --debug
```

Configure robot, camera, and gripper endpoints through `CODROID_` environment
variables or `data/app_settings.json` as described in the README.
