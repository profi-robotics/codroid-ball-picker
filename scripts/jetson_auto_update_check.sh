#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
LOCK_FILE="/tmp/codroid-ball-picker-auto-update.lock"
SERVICE_NAME="${CODROID_COMPOSE_SERVICE:-ui}"
CONTAINER_NAME="${CODROID_CONTAINER_NAME:-codroid-ui}"
IMAGE_REF="ghcr.io/profi-robotics/codroid-ball-picker:latest"
DOCKER_CONFIG="${DOCKER_CONFIG:-/home/university/.docker}"
export DOCKER_CONFIG

log() {
  printf '[auto-update] %s\n' "$*"
}

if command -v flock >/dev/null 2>&1; then
  if { exec 9>"${LOCK_FILE}"; } 2>/dev/null; then
    if ! flock -n 9; then
      log "Another updater instance is running. Exiting."
      exit 0
    fi
  else
    log "Lock file ${LOCK_FILE} is not writable. Continuing without a lock."
  fi
fi

cd "${REPO_DIR}"

if ! docker info >/dev/null 2>&1; then
  log "Docker daemon is unavailable."
  exit 1
fi

if ! docker compose config --services | grep -Fxq "${SERVICE_NAME}"; then
  log "Compose service '${SERVICE_NAME}' was not found in ${REPO_DIR}/docker-compose.yml."
  exit 1
fi

old_container_image=""
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  old_container_image=$(docker inspect -f '{{.Image}}' "${CONTAINER_NAME}")
fi

log "Pulling latest image for ${SERVICE_NAME} (${IMAGE_REF})..."
if ! docker compose pull "${SERVICE_NAME}"; then
  log "Image pull failed. If the GHCR package is private, run: docker login ghcr.io"
  exit 1
fi

new_tagged_image_id=$(docker image inspect "${IMAGE_REF}" --format '{{.Id}}' 2>/dev/null || true)

if [ -z "${old_container_image}" ]; then
  log "Container not found. Creating service container."
  docker compose up -d "${SERVICE_NAME}"
  docker image prune -f >/dev/null 2>&1 || true
  exit 0
fi

if [ -n "${new_tagged_image_id}" ] && [ "${old_container_image}" != "${new_tagged_image_id}" ]; then
  log "New image detected. Recreating container."
  docker compose up -d --force-recreate "${SERVICE_NAME}"
  log "Container updated to ${new_tagged_image_id}."
  docker image prune -f >/dev/null 2>&1 || true
  exit 0
fi

current_status=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || true)
if [ "${current_status}" != "running" ]; then
  log "Container is not running (${current_status:-unknown}). Starting it."
  docker compose up -d "${SERVICE_NAME}"
  exit 0
fi

log "No new image found. Container already up to date."
