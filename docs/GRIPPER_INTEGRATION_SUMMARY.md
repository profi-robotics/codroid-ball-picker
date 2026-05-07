# OnRobot Gripper Integration Summary

This project no longer keeps a local `robot/` package.

Current ownership:

- `codroid-ball-picker` contains UI wiring and runtime orchestration in `app/`.
- `onrobot-api` contains the gripper client implementation (`onrobot/` package).
- `codroid-api` contains Codroid robot WebSocket/HTTP control.

When updating gripper protocol logic:

1. Implement API-level changes in `onrobot-api`.
2. Keep UI behaviour changes in `codroid-ball-picker/app/`.
3. Update `docs/gripper_integration.md` in this repository for operator-facing guidance.
