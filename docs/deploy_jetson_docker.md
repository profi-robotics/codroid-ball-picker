# Jetson Docker Deployment

These steps run the FastAPI operator UI on a Jetson with USB access for the OAK-D, host networking for the robot/gripper LAN, MJPEG video, and optional GPU support for YOLO/TensorRT model execution.

## Prerequisites
- JetPack with NVIDIA Container Runtime (`sudo apt install nvidia-container-runtime`) when GPU/ML support is needed.
- Docker + docker-compose plugin.
- Repo cloned to the Jetson (`codroid-ball-picker`).

Use a base image that matches the deployed JetPack/L4T release. `Dockerfile.jetson` targets JetPack 6-class L4T R36 images; original Jetson Nano JetPack 4 deployments need a separately validated base image and Python 3.11 toolchain.

## Pull image
```bash
docker compose pull
```

This pulls `ghcr.io/profi-robotics/codroid-ball-picker:latest`.

## Run
```bash
docker compose up -d
```

The UI will be reachable on `http://<jetson-ip>:8050` and auto-restart after reboot.
When Tailscale is enabled, use `http://<tailscale-ip>:8050` as well.

The container starts:

```bash
uv run uvicorn codroid_picker.app:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8050 \
  --workers 1
```

Keep `--workers 1`; the camera stream, robot session, gripper client, state store, and detection worker are single in-process resources.

### Device access
- USB/OAK-D: passed via `/dev/bus/usb` bind mount plus `device_cgroup_rules: ["c 189:* rmw"]` so DepthAI can access the device after USB re-enumeration during boot.
- V4L2 camera (optional): `/dev/video0`.
- GPU: enabled via `runtime: nvidia` and `NVIDIA_*` env vars in `docker-compose.yml`.
- Network: `network_mode: host` is enabled so the container can directly reach robot/gripper subnets such as `192.168.101.0/24` while the UI is opened through Tailscale (`100.x.x.x`).

### Networking note
- `ports:` is intentionally not used with host networking.
- Ensure the Jetson host itself can reach the Compute Box (`192.168.101.95:41414`) because the container now shares the host network stack.

### Persist data
`./data` is mounted into the container to keep calibration bundles, workspace definitions, and settings.

## Automatic updates on Jetson
Install a systemd timer that checks for new image versions, pulls them, and recreates the container if needed:

```bash
./scripts/setup_jetson_auto_update.sh
```

By default, the timer checks every 5 minutes. Override interval:

```bash
AUTO_UPDATE_INTERVAL=10min ./scripts/setup_jetson_auto_update.sh
```

Run one update cycle manually:

```bash
./scripts/jetson_auto_update_check.sh
```

## Environment overrides
Add env vars to `docker-compose.yml` as needed:

- `CODROID_ROBOT_HOST`, `CODROID_ROBOT_PORT`, `CODROID_USER_PORT`
- `CODROID_GRIPPER_IP`, `CODROID_GRIPPER_TYPE`
- `CODROID_CAMERA_SOURCE`, `CODROID_HOST_CAMERA_INDEX`, `CODROID_HOST_CAMERA_DEVICE`
- `CODROID_VIDEO_MAX_FPS`, `CODROID_VIDEO_MAX_WIDTH`, `CODROID_VIDEO_JPEG_QUALITY`
- `CODROID_DETECTION_WORKER`
- `CODROID_LOG_LEVEL`

Use `CODROID_USE_FAKE_HARDWARE=1` for a container smoke test without robot, gripper, or camera hardware.

## Video and detection
The default browser stream is MJPEG at `GET /video/stream`, with `GET /video/snapshot.jpg` as the fallback path. Tune defaults with `CODROID_VIDEO_MAX_WIDTH`, `CODROID_VIDEO_MAX_FPS`, and `CODROID_VIDEO_JPEG_QUALITY`.

YOLO model loading is optional. Install/run the image with the `ml` extra when model activation is required. Jetson PyTorch wheels are platform-specific; use the NVIDIA wheel index that matches the device JetPack/L4T release, for example only after verifying compatibility:

```bash
uv pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v60
```

## Notes
- For additional cameras, add more `/dev/video*` entries.
- If depthai requires udev rules on your Jetson host, ensure they’re installed outside the container as well.
- If GHCR package visibility is private, log in first: `docker login ghcr.io`.
- The health check is `GET /health` on `127.0.0.1:8050`.
