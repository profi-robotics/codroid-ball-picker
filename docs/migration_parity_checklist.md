# FastAPI Migration Parity Checklist

This checklist tracks the feature parity between the legacy Dash/WebRTC UI and
the FastAPI/MJPEG/SSE rewrite.

| Area | Status | FastAPI surface | Notes |
| --- | --- | --- | --- |
| Camera connection and MJPEG video | ported | `CameraService`, `/camera/*`, `/video/*` | OAK-D, host, fake camera, snapshots, and stream stats are available. |
| Calibration workflows | partial | `CalibrationService`, `/calibration/*` | Camera, workspace, plate detection, robot teaching, finish, load, cancel, and verify are available. |
| Robot and gripper control | partial | `RobotService`, `GripperService`, `/robot/*`, `/gripper/*` | TCP/rescue jogs, polling, coordinate sync, flange DI sync, stale posture state, and recovery method mapping are implemented; real-cell validation is still required. |
| Manual pick/drop | ported | `PickPlaceService`, `/pick/*`, `/drop-target/*` | Closest pick, drop, pick-and-drop, and drop target teaching are available. |
| Auto pick loop | partial | `AutoPickService`, `/auto/*` | Start/stop, target planning, stage-by-stage plan snapshots, grip verification, retry after failed pick attempts, and safety blocking exist. |
| Plate pick-and-place test | partial | `PlateTestService`, `/plate-test/run` | Service route, dry-run, per-target persistence, fake-hardware path, plate motion checks, stop/pause checks, and skipped/failed target accounting are available; real-hardware validation remains. |
| Camera safety pause | partial | `SafetyService`, state `safety` | Periodic detection-driven pause/resume is implemented with safety-owned pause semantics. |
| Monitoring and diagnostics | partial | `MonitoringService`, SSE state | Pick counters, detection history, ML-extra availability, model benchmark, video stats, robot polling age, safety owner, plate-test results, and system/GPU info are published. |
| MJPEG overlays | partial | `VideoService`, `video.overlays` | Detection boxes, workspace/plate hints, auto plan, drop target, robot state, and safety blocked state draw into snapshots/streams. |
| Robot errors and rescue | partial | `robot_errors`, `RobotService` | Error detection is state-driven; hardware-specific clear methods are best-effort. |
| Bilingual UI | partial | `codroid_picker/i18n/*` | Core FastAPI labels are translated; remaining legacy strings should be copied as UI surfaces stabilize. |
| Docker and deployment | partial | `Dockerfile`, `Dockerfile.jetson`, compose | FastAPI runtime is documented; final cleanup waits for legacy Dash dependency removal. |

Current baseline checkpoint: `/tmp/codroid-picker-fastapi-baseline.patch` and
`/tmp/codroid-picker-fastapi-baseline.status` capture the pre-slice dirty tree
without committing unrelated work.
