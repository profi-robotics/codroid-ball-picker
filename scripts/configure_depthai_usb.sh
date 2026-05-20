#!/usr/bin/env bash
set -euo pipefail

USBFS_MEMORY_MB="${CODROID_USBFS_MEMORY_MB:-1000}"
USBFS_PARAM="/sys/module/usbcore/parameters/usbfs_memory_mb"
MODPROBE_CONF="/etc/modprobe.d/codroid-depthai-usb.conf"

if ! [[ "${USBFS_MEMORY_MB}" =~ ^[0-9]+$ ]]; then
  echo "CODROID_USBFS_MEMORY_MB must be an integer, got: ${USBFS_MEMORY_MB}" >&2
  exit 2
fi

write_file() {
  local path="$1"
  local content="$2"

  if [ "$(id -u)" -eq 0 ]; then
    printf '%s\n' "${content}" > "${path}"
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    printf '%s\n' "${content}" | sudo tee "${path}" >/dev/null
    return
  fi

  echo "Root privileges are required to write ${path}." >&2
  exit 1
}

if [ -f "${USBFS_PARAM}" ]; then
  current_value=$(cat "${USBFS_PARAM}")
  if [ "${current_value}" != "${USBFS_MEMORY_MB}" ]; then
    write_file "${USBFS_PARAM}" "${USBFS_MEMORY_MB}"
    echo "Set usbfs_memory_mb=${USBFS_MEMORY_MB} (was ${current_value})."
  else
    echo "usbfs_memory_mb already set to ${USBFS_MEMORY_MB}."
  fi
else
  echo "Warning: ${USBFS_PARAM} not found; cannot set runtime usbfs memory." >&2
fi

write_file "${MODPROBE_CONF}" "options usbcore usbfs_memory_mb=${USBFS_MEMORY_MB}"
echo "Persisted DepthAI USB memory setting in ${MODPROBE_CONF}."
