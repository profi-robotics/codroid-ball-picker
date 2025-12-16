# Codroid WebSocket Command Catalog (reverse-engineering notes)

Observations were collected by inspecting the browser-based UI WebSocket frames
via Chrome DevTools. Payloads are JSON and include a `cmd` property. All
messages shown here are minimal; the UI may emit additional metadata fields
(e.g., request IDs, timestamps) that can be tolerated by the controller.

## Session / Heartbeat
- **keepalive**: `{ "cmd": "keepalive" }`
  - Sent periodically by the UI; the controller echoes an acknowledgement. Clients
    should replicate this to avoid idle timeouts.

## Safety and Power
- **enable**: `{ "cmd": "enable" }`
  - Transitions the robot from idle to enabled state. Expect a status payload in
    response.
- **disable**: `{ "cmd": "disable" }`
  - Disables the drives. Use after motions to leave the robot in a safe state.

## Motion
- **movl** (Cartesian linear):
  ```json
  {
    "cmd": "movl",
    "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
    "speed": 100.0,
    "acc": 200.0,
    "blend": 0.0
  }
  ```
  - `pose` is expressed in the active user coordinate system (typically base).
  - `rx/ry/rz` follow the UI convention (XYZ Euler radians).
- **movj** (joint space):
  ```json
  {
    "cmd": "movj",
    "joints": {"j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0, "j5": 0.0, "j6": 0.0},
    "speed": 30.0,
    "acc": 60.0
  }
  ```

## State Query
- **get_state**: `{ "cmd": "get_state" }`
  - Returns current joint positions, Cartesian pose, and IO summaries (fields may
    vary depending on firmware).

## Notes
- If the controller responds with `"result": "fail"` include the `msg` field in
  logs to assist debugging.
- Coordinate system and TCP selection were not observed in the captured frames;
  they may require additional UI interactions to surface.
