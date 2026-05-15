"""Plate pick-and-place test routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from codroid_picker.api.schemas import ActionResponse, PlateTestRunRequest, action
from codroid_picker.dependencies import get_context
from codroid_picker.services.context import AppContext

router = APIRouter(prefix="/plate-test")


@router.post("/run", response_model=ActionResponse)
async def run_plate_test(payload: PlateTestRunRequest, ctx: AppContext = Depends(get_context)) -> ActionResponse:
    summary = await ctx.plate_test.run(
        approach_mm=payload.approach_mm,
        pick_mm=payload.pick_mm,
        drop_mm=payload.drop_mm,
        grip_timeout_s=payload.grip_timeout_s,
        use_base_coordinates=payload.use_base_coordinates,
        dry_run=payload.dry_run,
    )
    return action("Plate test complete.", state={"plate_test": summary})
