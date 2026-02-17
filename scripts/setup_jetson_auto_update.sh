#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SERVICE_NAME="codroid-ball-picker-auto-update.service"
TIMER_NAME="codroid-ball-picker-auto-update.timer"
INTERVAL="${AUTO_UPDATE_INTERVAL:-5min}"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl is required for auto-update setup."
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required for auto-update setup."
  exit 1
fi

cat <<EOF | sudo tee "/etc/systemd/system/${SERVICE_NAME}" >/dev/null
[Unit]
Description=Codroid Ball Picker container auto-update check
After=docker.service network-online.target
Wants=docker.service network-online.target

[Service]
Type=oneshot
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/scripts/jetson_auto_update_check.sh
EOF

cat <<EOF | sudo tee "/etc/systemd/system/${TIMER_NAME}" >/dev/null
[Unit]
Description=Run Codroid Ball Picker auto-update check periodically

[Timer]
OnBootSec=2min
OnUnitActiveSec=${INTERVAL}
Persistent=true
Unit=${SERVICE_NAME}

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${TIMER_NAME}"

echo "Auto-update timer installed: ${TIMER_NAME}"
echo "Check status with: systemctl status ${TIMER_NAME}"
echo "List schedules with: systemctl list-timers --all | grep codroid-ball-picker-auto-update"
echo "Run manual update check with: ${REPO_DIR}/scripts/jetson_auto_update_check.sh"
