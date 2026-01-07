# Jetson Orin Nano Docker Deployment

These steps build and run the calibration UI on a Jetson Orin Nano (JetPack 5.x / L4T R35.x) with GPU and USB access for the OAK-D and gripper.

## Prerequisites
- JetPack 5.x (L4T R35.x) with NVIDIA Container Runtime (`sudo apt install nvidia-container-runtime`).
- Docker + docker-compose plugin.
- Repo cloned to the Jetson.

## Build
```bash
docker compose build
```

This uses the L4T PyTorch base image and `uv` to install Python deps from `pyproject.toml`/`uv.lock`.

## Run
```bash
docker compose up
```

The UI will be reachable on `http://<jetson-ip>:8050`.

### Device access
- USB/OAK-D: passed via `/dev/bus/usb`.
- V4L2 camera (optional): `/dev/video0`.
- GPU: enabled via `runtime: nvidia` and `NVIDIA_*` env vars in `docker-compose.yml`.

### Persist data
`./data` is mounted into the container to keep calibration bundles, workspace definitions, and settings.

## Environment overrides
Add env vars to `docker-compose.yml` as needed (e.g., `ONROBOT_CB_IP`, `ROBOT_URI`, `MODEL_DIR`).

## Notes
- For additional cameras, add more `/dev/video*` entries.
- If depthai requires udev rules on your Jetson host, ensure they’re installed outside the container as well.
