"""API router assembly."""

from __future__ import annotations

from fastapi import APIRouter

from codroid_picker.api import auto, calibration, camera, detection, gripper, health, models, operations, pages, plate_test, robot, settings, state, video


def build_router() -> APIRouter:
    router = APIRouter()
    router.include_router(pages.router)
    router.include_router(health.router)
    router.include_router(state.router)
    router.include_router(video.router)
    router.include_router(camera.router)
    router.include_router(robot.router)
    router.include_router(gripper.router)
    router.include_router(detection.router)
    router.include_router(models.router)
    router.include_router(auto.router)
    router.include_router(calibration.router)
    router.include_router(plate_test.router)
    router.include_router(settings.router)
    router.include_router(operations.router)
    return router
