# Jetson Orin Nano Docker Deployment

These steps run the calibration UI on a Jetson Orin Nano (JetPack 5.x/6.x) with GPU and USB access for the OAK-D and gripper.

## Prerequisites
- JetPack 5.x or 6.x with NVIDIA Container Runtime (`sudo apt install nvidia-container-runtime`).
- Docker + docker-compose plugin.
- Repo cloned to the Jetson (`codroid-ball-picker`).

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
Add env vars to `docker-compose.yml` as needed (e.g., `ONROBOT_CB_IP`, `ROBOT_URI`, `MODEL_DIR`).

## Notes
- For additional cameras, add more `/dev/video*` entries.
- If depthai requires udev rules on your Jetson host, ensure they’re installed outside the container as well.
- If GHCR package visibility is private, log in first: `docker login ghcr.io`.
