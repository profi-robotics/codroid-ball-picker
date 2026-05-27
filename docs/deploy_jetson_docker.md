# Jetson Orin Nano Docker Deployment

These steps run the calibration UI on a Jetson Orin Nano with JetPack 6.x,
GPU access, USB access for the OAK-D, and host networking for the robot and
gripper.

This image targets JetPack 6 / L4T R36.x. The original Jetson Nano generation
uses the older JetPack 4 stack and is not compatible with this Docker image's
Python 3.10 + CUDA 12.6 PyTorch dependency set.

## Deployment Flow

1. A push to `master` runs `.github/workflows/release-and-publish.yml`.
2. CI checks out `codroid-ball-picker`, `codroid-api`, and `onrobot-api`.
3. CI builds `linux/arm64/v8` from `Dockerfile` and publishes it to GHCR as:
   - `ghcr.io/profi-robotics/codroid-ball-picker:<tag>`
   - `ghcr.io/profi-robotics/codroid-ball-picker:latest`
   - `ghcr.io/profi-robotics/codroid-ball-picker:sha-<commit>`
4. The `codroid-watchtower` compose service checks the `codroid-ui` image.
5. The updater pulls the compose service image and recreates `codroid-ui` only
   when the pulled image ID differs from the running container image.

Application code updates arrive through the container image. Changes to
`docker-compose.yml`, systemd scripts, or this deployment guide require a
manual `git pull` on the Jetson and rerunning `scripts/setup_jetson_auto_update.sh`.
The compose file also sets Docker restart policies so containers come back after
host reboot. If systemd management is preferred, the installer enables
`codroid-ball-picker-compose.service`, which starts the compose service on boot
before periodic update checks run.

## Prerequisites

- Jetson Orin Nano flashed with JetPack 6.x / L4T R36.x.
- Docker and the Docker Compose plugin.
- NVIDIA Container Runtime available to Docker.
- This repo cloned to the Jetson so `docker-compose.yml`, `scripts/`, and
  persistent `data/` are present.
- GHCR login if the package is private:
  ```bash
  docker login ghcr.io
  ```

The image is based on `nvcr.io/nvidia/l4t-jetpack:r36.4.0` and installs the
project dependencies during CI, including the Jetson AI Lab `jp6/cu126`
PyTorch, torchvision, and torchaudio wheels declared in `pyproject.toml`.

## First Run

From the repo root on the Jetson:

```bash
docker compose pull
docker compose up -d
```

The UI will be reachable on `http://<jetson-ip>:8050`. When Tailscale is
enabled, use `http://<tailscale-ip>:8050`.
The same command starts `codroid-watchtower`, which checks every 300 seconds by
default. Override the interval with `CODROID_WATCHTOWER_INTERVAL_SECONDS`.
The UI command runs through `run_codroid_picker_supervised.py`, which restarts
the Dash app subprocess after unexpected native exits such as DepthAI/XLink
aborts. A clean app exit still stops normally.

Check logs:

```bash
docker compose logs -f ui
```

Verify the preinstalled runtime stack inside the container:

```bash
docker compose exec ui python - <<'PY'
import cv2
import depthai
import torch
import torchvision

print("opencv", cv2.__version__)
print("depthai", depthai.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda available", torch.cuda.is_available())
PY
```

## Automatic Updates

The default updater is the `codroid-watchtower` compose service:

```bash
docker compose logs -f updater
```

It monitors only the `codroid-ui` container, pulls `latest`, recreates the UI
container when the image changes, and removes old dangling images.

## Optional Systemd Management

Install the boot service and updater:

```bash
./scripts/setup_jetson_auto_update.sh
```

By default, the timer checks every 5 minutes. Override the interval during
installation:

```bash
AUTO_UPDATE_INTERVAL=10min ./scripts/setup_jetson_auto_update.sh
```

Run one update cycle manually:

```bash
./scripts/jetson_auto_update_check.sh
```

Inspect the timer:

```bash
systemctl status codroid-ball-picker-compose.service
systemctl status codroid-ball-picker-auto-update.timer
systemctl list-timers --all | grep codroid-ball-picker-auto-update
```

The installer writes `/etc/codroid-ball-picker-auto-update.env` with the
`latest` image reference. Do not pin deployment hosts to branch or release
image tags; the compose stack and updater are intended to track `latest` only:

```bash
CODROID_IMAGE=ghcr.io/profi-robotics/codroid-ball-picker:latest
CODROID_COMPOSE_SERVICE=ui
CODROID_CONTAINER_NAME=codroid-ui
```

After editing the env file, run:

```bash
sudo systemctl restart codroid-ball-picker-auto-update.timer
./scripts/jetson_auto_update_check.sh
```

## Device Access

- USB/OAK-D: `/dev/bus/usb` is bind-mounted and `device_cgroup_rules` allows
  USB re-enumeration after DepthAI boots the device. The UI container runs with
  `privileged: true` because DepthAI repeatedly resets and re-enumerates the
  OAK-D while booting firmware. The compose service also mounts `/run/udev`
  read-only so USB device metadata remains visible inside the container.
- USB buffer memory: OAK-D high-bandwidth streams need more than the Linux
  default 16 MB `usbfs_memory_mb` value. Run
  `sudo scripts/configure_depthai_usb.sh` once on the Jetson, or install the
  systemd service with `scripts/setup_jetson_auto_update.sh`; both set the live
  host value and persist `options usbcore usbfs_memory_mb=1000` for reboot.
- USB link speed: this compose file sets
  `CODROID_DEPTHAI_MAX_USB_SPEED=high`, which forces USB 2.0 HighSpeed. Use
  this when a Jetson/OAK-D link detects the camera at USB 3 SuperSpeed but then
  logs repeated `X_LINK_ERROR` messages. Remove the variable or set it to
  `auto` only after the USB 3 cable, hub, and power path have been proven
  stable.
- GPU: compose uses `runtime: nvidia` with `NVIDIA_VISIBLE_DEVICES=all`.
- Network: `network_mode: host` lets the container reach robot/gripper subnets
  such as `192.168.101.0/24` while the UI is exposed through the Jetson host.
- Persistent data: `./data` is mounted to `/app/data` for calibration bundles,
  workspace definitions, settings, and uploaded models.

Because host networking is enabled, `ports:` is intentionally not used. Ensure
the Jetson host itself can reach the robot controller and OnRobot Compute Box.

## Environment Overrides

The compose service always uses
`ghcr.io/profi-robotics/codroid-ball-picker:latest` for deployments.

Application settings still come from the normal `CODROID_` environment prefix
and persisted `data/app_settings.json`. Add deployment-specific variables to
`docker-compose.yml` only when they must be fixed for the device.

## Troubleshooting

- If `docker compose pull` fails for GHCR, run `docker login ghcr.io`.
- If the OAK-D is not visible, verify the Jetson host can see it with `lsusb`
  and check that no other process owns the device.
- If the OAK-D is detected but the camera stream repeatedly logs
  `X_LINK_ERROR`, check `cat /sys/module/usbcore/parameters/usbfs_memory_mb`.
  If it is below `1000`, run `sudo scripts/configure_depthai_usb.sh` and
  recreate the UI container.
- If the process exits after a native DepthAI line like
  `dispatcherResponseServe:... no request for this response: XLINK_READ_REL_RESP`,
  treat it as a native OAK-D/XLink link failure rather than a Dash callback
  exception. The app supervisor should restart the Dash subprocess
  automatically; Docker or systemd remains a second-level fallback. If the
  failure repeats, keep `CODROID_DEPTHAI_MAX_USB_SPEED=high`, reduce camera
  FPS/depth load, and check OAK-D cable, power, and temperature.
- If CUDA is unavailable in the verification command, confirm the host is
  JetPack 6 / L4T R36.x and the NVIDIA runtime is configured for Docker.
- If disk usage grows after many releases, the updater prunes dangling images
  after successful recreates, but `docker system df` is useful for inspection.
