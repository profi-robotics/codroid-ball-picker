# Non-Jetson Deployment

Use this profile for x86 or other non-Jetson hardware, including development
workstations and robot-cell computers that should run the Dash/WebRTC app and
YOLO locally.

## Install

```bash
uv sync --extra workstation
```

The `workstation` extra installs `ultralytics` plus CPU PyTorch and torchvision
from the PyTorch CPU wheel index. It intentionally avoids Jetson-only PyTorch
wheels and avoids pulling CUDA runtime packages from PyPI.

For a camera/robot bring-up without local YOLO inference, use the base install:

```bash
uv sync
```

## Run

```bash
python run_codroid_picker.py --debug
```

Configure robot, camera, and gripper endpoints through `CODROID_` environment
variables or `data/app_settings.json` as described in the README.
