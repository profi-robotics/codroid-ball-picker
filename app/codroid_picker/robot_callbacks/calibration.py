"""Robot calibration workflow callbacks.

This module handles the complete robot calibration process:
- Calibration status and state synchronization
- Calibration modal UI updates
- Calibration workflow execution (3-point calibration)
- Accuracy assessment and verification
- Verification move sequence
"""

from typing import Any, Optional, Tuple

import dash_mantine_components as dmc
import numpy as np
from dash import callback_context, Dash, Input, no_update, Output, State


def register_calibration_callbacks(app: Dash) -> None:
    """Register robot calibration workflow callbacks."""

    # === Status and State Sync ===

    @app.callback(
        Output("robot-calibration-status", "children"),
        Input("robot-status-interval", "n_intervals"),
    )
    def _update_robot_calibration_guidance(_intervals: int) -> str:
        from app.codroid_picker import state

        return state.get_calibration_guidance_snapshot().get("message", "")

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-status-interval", "n_intervals"),
        State("robot-calibration-state", "data"),
        prevent_initial_call=True,
    )
    def _sync_robot_calibration_state(_intervals: int, current: Optional[dict]) -> dict:
        from app.codroid_picker.state import build_robot_calibration_state_payload

        desired = build_robot_calibration_state_payload()
        if current == desired:
            return no_update
        return desired

    # === Modal UI Updates ===

    @app.callback(
        Output("robot-calibration-modal", "opened"),
        Input("robot-calibration-state", "data"),
        Input("robot-calibration-modal-close", "n_clicks"),
    )
    def _toggle_robot_calibration_modal(
        calibration_state: Optional[dict],
        _close_clicks: int,
    ) -> bool:
        from app.codroid_picker import state

        status = (calibration_state or {}).get("status", "idle")
        trigger = callback_context.triggered_id
        guidance = state.get_calibration_guidance_snapshot()
        should_show_modal = not guidance.get("modal_suppressed", False)
        if status in {"running", "accuracy_ready", "awaiting_verification", "verifying"} and should_show_modal:
            return True
        if status in {"complete", "error", "canceled"}:
            return trigger != "robot-calibration-modal-close"
        if trigger == "robot-calibration-modal-close":
            return False
        return False

    @app.callback(
        Output("robot-calibration-modal-stepper", "active"),
        Output("robot-calibration-modal-alert", "children"),
        Output("robot-calibration-modal-alert", "color"),
        Output("robot-calibration-verify", "disabled"),
        Output("robot-calibration-modal-stepper", "children"),
        Input("robot-calibration-state", "data"),
    )
    def _update_robot_calibration_modal(
        calibration_state: Optional[dict],
    ) -> Tuple[int, str, str, bool, list[dmc.StepperStep]]:
        state = calibration_state or {}
        status = state.get("status", "idle")
        step = int(state.get("step", 0) or 0)
        message = state.get("message", "")
        active = max(0, min(4, step - 1)) if step else 0
        color_map = {
            "running": "blue",
            "awaiting_verification": "yellow",
            "accuracy_ready": "orange",
            "verifying": "blue",
            "complete": "green",
            "error": "red",
            "canceled": "red",
        }
        color = color_map.get(status, "gray")
        verify_enabled = (
            status == "awaiting_verification"
            and state.get("verification_ready", False)
        )
        verification_completed = status == "complete"

        def _step_state(is_done: bool) -> Optional[str]:
            return "completed" if is_done else None

        stepper_children = [
            dmc.StepperStep(
                label="Point 1",
                description="Indentation 1 (origin)",
                state=_step_state(step > 1),
            ),
            dmc.StepperStep(
                label="Point 2",
                description="Indentation 2 (+X)",
                state=_step_state(step > 2),
            ),
            dmc.StepperStep(
                label="Point 3",
                description="Indentation 3 (+X/+Y)",
                state=_step_state(step > 3),
            ),
            dmc.StepperStep(
                label="Accuracy",
                description="Calibration accuracy check",
                state=_step_state(step > 4 or bool(
                    state.get("accuracy_ready"))),
            ),
            dmc.StepperStep(
                label="Verification",
                description="Safe motion check",
                state=_step_state(verification_completed),
            ),
        ]
        return (
            active,
            message or "Waiting for calibration to start.",
            color,
            not verify_enabled,
            stepper_children,
        )

    @app.callback(
        Output("robot-calibration-accuracy-panel", "style"),
        Output("robot-calibration-accuracy-text", "children"),
        Output("robot-calibration-accuracy-text", "c"),
        Output("robot-calibration-retry", "disabled"),
        Output("robot-calibration-continue", "disabled"),
        Output("robot-calibration-continue", "children"),
        Input("robot-calibration-state", "data"),
    )
    def _sync_robot_calibration_accuracy_panel(
        calibration_state: Optional[dict],
    ) -> Tuple[dict[str, Any], str, str, bool, bool, str]:
        import app.codroid_picker as codroid_picker_module

        state = calibration_state or {}
        accuracy_ready = bool(state.get("accuracy_ready"))
        accuracy = state.get("accuracy") or {}
        if not accuracy_ready or not accuracy:
            return (
                {"display": "none"},
                "",
                "dimmed",
                True,
                True,
                "Proceed to verification",
            )
        max_err = float(accuracy.get("max_xy_error", 0.0))
        mean_err = float(accuracy.get("mean_xy_error", 0.0))
        passed = bool(state.get("accuracy_passed"))
        text = (
            f"Calibration accuracy: max XY error {max_err:.2f} mm · "
            f"mean XY error {mean_err:.2f} mm."
        )
        if passed:
            text += f" Within {codroid_picker_module.SETTINGS.verification_tolerance_mm:.2f} mm threshold."
            continue_label = "Proceed to verification"
            text_color = "green"
        else:
            text += (
                f" Above {codroid_picker_module.SETTINGS.verification_tolerance_mm:.2f} mm threshold; "
                "consider repeating calibration or proceed with caution."
            )
            continue_label = "Proceed with warning"
            text_color = "red"
        return (
            {"display": "flex", "flexDirection": "column", "gap": "0.6rem"},
            text,
            text_color,
            False,
            False,
            continue_label,
        )

    # === Accuracy Panel Actions ===

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-calibration-continue", "n_clicks"),
        State("robot-calibration-state", "data"),
        prevent_initial_call=True,
    )
    def _advance_to_verification_after_accuracy(
        _clicks: int,
        calibration_state: Optional[dict],
    ) -> dict:
        import app.codroid_picker as codroid_picker_module

        guidance = calibration_state or {}
        if not guidance.get("accuracy_ready"):
            return no_update
        codroid_picker_module._set_robot_guidance(
            "Verification ready. Run the verification move when ready.",
            status="awaiting_verification",
            step=5,
            verification_ready=True,
            modal_suppressed=False,
        )
        from app.codroid_picker.state import build_robot_calibration_state_payload
        return build_robot_calibration_state_payload()

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-calibration-retry", "n_clicks"),
        prevent_initial_call=True,
    )
    def _reset_calibration_after_accuracy(_clicks: int) -> dict:
        import app.codroid_picker as codroid_picker_module

        codroid_picker_module._clear_robot_accuracy_guidance()
        codroid_picker_module._set_robot_guidance(
            "Calibration reset. Close this dialog and restart from the main panel.",
            status="idle",
            step=0,
            verification_ready=False,
            modal_suppressed=True,
        )
        from app.codroid_picker.state import build_robot_calibration_state_payload
        return build_robot_calibration_state_payload()

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("use-existing-robot-calibration", "n_clicks"),
        State("settings-robot-calib-path", "value"),
        prevent_initial_call=True,
    )
    def _load_saved_robot_calibration(
        _clicks: int,
        robot_calib_path_value: Optional[str],
    ) -> dict:
        """Load persisted robot calibration data instead of running the guided flow."""
        import app.codroid_picker as codroid_picker_module

        path = codroid_picker_module._safe_path(
            robot_calib_path_value, codroid_picker_module.DEFAULT_ROBOT_CALIBRATION_PATH
        )
        if not path.exists():
            message = f"Saved robot calibration not found: {path}"
            codroid_picker_module._set_robot_guidance(
                message, status="error", verification_ready=False
            )
            return {"status": "error", "message": message}
        try:
            codroid_picker_module._load_robot_calibration(path)
            guidance = (
                "Saved robot calibration loaded. Place a calibration plate on the workspace "
                "and run the verification move to confirm."
            )
            codroid_picker_module._set_robot_guidance(
                guidance,
                status="awaiting_verification",
                verification_ready=True,
                step=5,
                modal_suppressed=True,
            )
            return {
                "status": "awaiting_verification",
                "message": guidance,
                "step": 5,
                "verification_ready": True,
            }
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to load saved robot calibration: {exc}"
            codroid_picker_module._set_robot_guidance(
                message,
                status="error",
                verification_ready=False,
                modal_suppressed=True,
            )
            return {"status": "error", "message": message}

    # === Calibration Execution ===

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("start-robot-calibration", "n_clicks"),
        State("plate-detection-state", "data"),
        State("robot-connection-state", "data"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _start_robot_calibration(
        _clicks: int,
        plate_state: Optional[dict],
        robot_state: Optional[dict],
        robot_uri: Optional[str],
    ) -> dict:
        import app.codroid_picker as codroid_picker_module

        codroid_picker_module.LOGGER.info(
            "=== Robot calibration button clicked ===")
        codroid_picker_module.LOGGER.info(f"Plate state: {plate_state}")
        codroid_picker_module.LOGGER.info(f"Robot URI: {robot_uri}")

        if (robot_state or {}).get("status") != "online":
            codroid_picker_module._set_robot_guidance(
                "Robot not connected. Click 'Test robot connection' to establish connection.",
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": "Robot not connected."}

        plate_ok, plate_message = codroid_picker_module._plate_is_stable(
            require_baseline=False)
        if not plate_ok:
            codroid_picker_module.LOGGER.warning(
                "Plate not ready: %s", plate_message)
            codroid_picker_module._set_robot_guidance(
                plate_message,
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": plate_message}

        posture_ok, posture_message = codroid_picker_module._robot_posture_ready()
        if not posture_ok:
            codroid_picker_module._set_robot_guidance(
                posture_message,
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": posture_message}

        codroid_picker_module._prepare_gripper_for_calibration()

        codroid_picker_module.LOGGER.info(
            "Plate state OK, starting calibration procedure...")
        try:
            layout_path = codroid_picker_module.DEFAULT_PLATE_PATH
            codroid_picker_module.LOGGER.info(
                f"Loading plate layout from: {layout_path}")
            plate_layout = codroid_picker_module.plate.load_plate_layout(
                layout_path)
            codroid_picker_module.LOGGER.info(
                "Plate layout loaded successfully")

            center = (plate_state or {}).get("center_px")
            if center is None:
                raise RuntimeError(
                    "Plate center not available; keep markers visible.")
            codroid_picker_module.SESSION.robot_calibration_baseline_px = np.asarray(
                center, dtype=np.float64)
            codroid_picker_module._ensure_plate_stable()
            codroid_picker_module.LOGGER.info("Plate stability baseline set")

            codroid_picker_module._set_robot_guidance(
                "Robot calibration started. Step 1/3: Place the gripper fingers in "
                "indentation 1, keep the flange vertical, and press flange button 1.",
                step=1,
                status="running",
                verification_ready=False,
            )
            codroid_picker_module.SESSION.robot_calibration_points = []
            codroid_picker_module.LOGGER.info(
                f"Starting robot plate calibration with URI: {robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI}")

            # Use RobotCalibrationProgram via registry
            from app.codroid_picker.programs import (
                get_registry,
                RobotCalibrationProgram,
                RobotCalibrationConfig,
            )
            from app.codroid_picker.programs.base import ProgramStatus

            registry = get_registry()
            robot_api = await codroid_picker_module._get_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI
            )
            program = RobotCalibrationProgram(
                plate_layout=plate_layout,
                config=RobotCalibrationConfig(),
            )
            result = await registry.start_instance(program, robot_api)

            if result.status != ProgramStatus.COMPLETED:
                raise RuntimeError(result.message)

            robot_points = program.result.robot_points
            codroid_picker_module.LOGGER.info(
                f"Robot plate calibration completed. Captured {len(robot_points)} points.")

            codroid_picker_module.LOGGER.info(
                "Applying robot calibration results...")
            _, accuracy = codroid_picker_module._apply_robot_calibration_results(
                robot_points,
                plate_layout,
            )
            try:
                plate_result = codroid_picker_module.compute_plate_calibration_from_latest(
                    layout_path)
                codroid_picker_module.SESSION.plate_reference_homography = plate_result.homography.copy()
                codroid_picker_module.LOGGER.info(
                    "Captured plate reference homography for plate test.")
            except Exception as exc:  # noqa: BLE001
                codroid_picker_module.LOGGER.warning(
                    "Failed to capture plate reference homography: %s", exc
                )
            codroid_picker_module._update_robot_calibration_meta(robot_points)
            codroid_picker_module.SESSION.robot_calibration_baseline_px = None
            codroid_picker_module.LOGGER.info(
                "Calibration results applied successfully")
            mean_err = accuracy.get("mean_xy_error", 0.0)
            max_err = accuracy.get("max_xy_error", 0.0)
            passes_threshold = max_err <= codroid_picker_module.SETTINGS.verification_tolerance_mm
            message = (
                f"Calibration accuracy: max XY error {max_err:.2f} mm · "
                f"mean XY error {mean_err:.2f} mm."
            )
            if passes_threshold:
                message = (
                    f"{message} Within {codroid_picker_module.SETTINGS.verification_tolerance_mm:.2f} mm "
                    "threshold. Proceed to verification."
                )
            else:
                message = (
                    f"{message} Above {codroid_picker_module.SETTINGS.verification_tolerance_mm:.2f} mm "
                    "threshold; consider repeating calibration or proceed with caution."
                )
            codroid_picker_module._set_robot_guidance(
                message,
                status="accuracy_ready",
                step=4,
                verification_ready=False,
                modal_suppressed=False,
            )
            return {
                "status": "accuracy_ready",
                "message": message,
                "step": 4,
                "verification_ready": False,
                "accuracy": accuracy,
                "accuracy_ready": True,
                "accuracy_passed": passes_threshold,
            }
        except Exception as exc:
            codroid_picker_module.SESSION.robot_calibration_baseline_px = None
            codroid_picker_module._restore_calibration_gripper_width()
            codroid_picker_module.LOGGER.error(
                "Robot calibration failed: %s", exc, exc_info=True)
            codroid_picker_module._set_robot_guidance(
                f"Robot calibration failed: {exc}",
                status="error",
                verification_ready=False,
            )
            return {"status": "error", "message": str(exc)}

    # === Verification Move ===

    @app.callback(
        Output("robot-calibration-state", "data", allow_duplicate=True),
        Input("robot-calibration-verify", "n_clicks"),
        Input("run-verification-inline", "n_clicks"),
        State("robot-uri", "value"),
        prevent_initial_call=True,
    )
    async def _run_robot_calibration_verification(
        _verify_clicks: int,
        _inline_clicks: int,
        robot_uri: Optional[str],
    ) -> dict:
        """Run the verification move using the VerificationProgram via registry."""
        import app.codroid_picker as codroid_picker_module
        from app.codroid_picker.programs import get_registry
        from app.codroid_picker.programs.base import ProgramStatus

        registry = get_registry()

        # Check if already running
        if registry.is_program_running:
            message = "Another program is already running."
            codroid_picker_module._set_robot_guidance(
                message, status="error", step=5, verification_ready=False
            )
            return {"status": "error", "message": message, "step": 5}

        # Pre-check prerequisites to provide UI feedback
        if not codroid_picker_module.SESSION.robot_calibration_points or len(codroid_picker_module.SESSION.robot_calibration_points) < 3:
            message = "Verification requires three calibration points."
            codroid_picker_module._set_robot_guidance(
                message, status="error", step=5, verification_ready=False
            )
            return {"status": "error", "message": message, "step": 5}

        # Ensure min_z is set in session
        points = codroid_picker_module.SESSION.robot_calibration_points[:3]
        if codroid_picker_module.SESSION.robot_calibration_min_z is None:
            codroid_picker_module.SESSION.robot_calibration_min_z = min(
                float(pt.get("z", 0.0)) for pt in points
            )

        posture_ok, posture_message = codroid_picker_module._robot_posture_ready()
        if not posture_ok:
            codroid_picker_module._set_robot_guidance(
                posture_message, status="error", step=5, verification_ready=False
            )
            return {"status": "error", "message": posture_message, "step": 5}

        # Update UI to show verification starting
        codroid_picker_module._set_robot_guidance(
            "Verification move running. Keep clear of the robot.",
            status="verifying",
            step=5,
            verification_ready=False,
            modal_suppressed=False,
        )

        async def _run_verification(api) -> dict:
            """Execute verification program through the registry."""
            result = await registry.start(
                "verification",
                api,
                initial_message="Verification move starting...",
            )

            codroid_picker_module._restore_calibration_gripper_width()

            if result.status == ProgramStatus.COMPLETED:
                codroid_picker_module._set_robot_guidance(
                    "Calibration verified. Coordinate ID 5 is active.",
                    status="complete",
                    step=5,
                    verification_ready=False,
                )
                return {"status": "complete", "message": "Calibration verified.", "step": 5}
            elif result.status == ProgramStatus.STOPPED:
                message = f"Verification move stopped: {result.message}"
                codroid_picker_module._set_robot_guidance(
                    message, status="canceled", step=5, verification_ready=False
                )
                return {"status": "canceled", "message": message, "step": 5}
            else:
                message = f"Verification move failed: {result.message}"
                codroid_picker_module._set_robot_guidance(
                    message, status="error", step=5, verification_ready=False
                )
                return {"status": "error", "message": message, "step": 5}

        try:
            return await codroid_picker_module._with_robot_api(
                robot_uri or codroid_picker_module.DEFAULT_ROBOT_URI,
                _run_verification,
            )
        except codroid_picker_module.OperationAbort as exc:
            message = f"Verification move stopped: {exc}"
            codroid_picker_module._set_robot_guidance(
                message, status="canceled", step=5, verification_ready=False
            )
            codroid_picker_module._restore_calibration_gripper_width()
            return {"status": "canceled", "message": message, "step": 5}
        except Exception as exc:  # noqa: BLE001
            codroid_picker_module.LOGGER.exception("Verification move failed")
            message = f"Verification move failed: {exc}"
            codroid_picker_module._set_robot_guidance(
                message, status="error", step=5, verification_ready=False
            )
            codroid_picker_module._restore_calibration_gripper_width()
            return {"status": "error", "message": message, "step": 5}
