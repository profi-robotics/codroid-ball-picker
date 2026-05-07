"""Adapter layer for supported OnRobot grippers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Optional

from onrobot import SG, TWOFG, VG, detect_gripper, get_gripper_profile, get_static_dimensions
from onrobot.errors import OnRobotConnectionError, OnRobotValidationError

LOGGER = logging.getLogger(__name__)

AUTO_GRIPPER_TYPE = "auto"
SUPPORTED_GRIPPER_KEYS = {"twofg7", "sg", "vgc10"}


@dataclass(frozen=True)
class GripperCapabilities:
    """UI and command capabilities for a connected gripper."""

    grip_kind: str
    width: bool = False
    force: bool = False
    speed: bool = False
    vacuum: bool = False
    gentle: bool = False
    orientation: bool = False


@dataclass(frozen=True)
class GripperDetection:
    """Normalized detection result used by the app session."""

    key: str
    display_name: str
    product_code: int
    dimensions: Any
    adapter: "BaseGripperAdapter"


def gripper_type_options() -> list[dict[str, str]]:
    """Return app gripper mode options including auto detection."""
    options = [{"value": AUTO_GRIPPER_TYPE, "label": "Auto detect"}]
    for key in ("twofg7", "sg", "vgc10"):
        profile = get_gripper_profile(key)
        options.append({"value": key, "label": profile.display_name})
    return options


def normalize_gripper_type(value: Optional[str]) -> str:
    """Normalize persisted gripper mode."""
    key = (value or AUTO_GRIPPER_TYPE).strip().lower()
    if key == "2fg7":
        key = "twofg7"
    if key not in SUPPORTED_GRIPPER_KEYS | {AUTO_GRIPPER_TYPE}:
        return AUTO_GRIPPER_TYPE
    return key


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _is_open_action(action_label: str) -> bool:
    label = (action_label or "").lower()
    return any(token in label for token in ("open", "release", "drop"))


def _clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        lower, upper = upper, lower
    return min(max(float(value), float(lower)), float(upper))


class BaseGripperAdapter:
    """Common adapter contract for app gripper operations."""

    key = ""
    capabilities = GripperCapabilities(grip_kind="unknown")

    def __init__(self, client: Any, *, dimensions: Any = None) -> None:
        self.client = client
        self.profile = get_gripper_profile(self.key)
        self.dimensions = dimensions or get_static_dimensions(self.key)

    @property
    def display_name(self) -> str:
        return self.profile.display_name

    @property
    def product_code(self) -> int:
        return int(getattr(self.dimensions, "product_code", 0) or 0)

    def is_connected(self, *, t_index: int) -> bool:
        return bool(self.client.isConnected(t_index=t_index))

    def start_status_stream(self, *, t_index: int = 0) -> None:
        try:
            self.client.start_status_stream()
        except TypeError:
            self.client.start_status_stream(t_index=t_index)
        except AttributeError:
            return

    def ensure_ready(self, *, t_index: int) -> None:
        """Run one-time setup needed after detection."""

    def get_status_snapshot(self, *, t_index: int) -> Optional[dict[str, Any]]:
        try:
            snapshot = self.client.get_status_snapshot(t_index=t_index)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to read gripper status snapshot: %s", exc)
            return None
        return snapshot if isinstance(snapshot, dict) else None

    def read_telemetry(self, *, t_index: int) -> dict[str, Any]:
        raise NotImplementedError

    def width_limits(self, *, t_index: int) -> tuple[float, float]:
        return (
            float(self.profile.close_width_default),
            float(self.profile.open_width_default),
        )

    def is_busy(self, *, t_index: int) -> bool:
        try:
            return bool(self.client.isBusy(t_index=t_index))
        except Exception:  # noqa: BLE001
            return False

    def is_holding(
        self,
        *,
        t_index: int,
        vacuum_threshold: Optional[float] = None,
    ) -> bool:
        try:
            return bool(self.client.isGripped(t_index=t_index))
        except Exception:  # noqa: BLE001
            return False

    def command(
        self,
        target_width: float,
        action_label: str,
        *,
        t_index: int,
        force: Optional[int] = None,
        speed: Optional[int] = None,
        gentle: Optional[bool] = None,
        vacuum_setpoint: Optional[int] = None,
        vacuum_threshold: Optional[float] = None,
        prevent_overclose: bool = False,
    ) -> str:
        raise NotImplementedError

    def to_state(self) -> dict[str, Any]:
        """Return serializable state for Dash stores."""
        return {
            "key": self.key,
            "display_name": self.display_name,
            "product_code": self.product_code,
            "capabilities": asdict(self.capabilities),
        }


class TwoFG7Adapter(BaseGripperAdapter):
    """2FG7 width/force adapter."""

    key = "twofg7"
    capabilities = GripperCapabilities(
        grip_kind="width_force",
        width=True,
        force=True,
        speed=True,
        orientation=True,
    )

    def read_telemetry(self, *, t_index: int) -> dict[str, Any]:
        snapshot = self.get_status_snapshot(t_index=t_index) or {}
        width = snapshot.get("external_width")
        force = snapshot.get("force")
        min_width = snapshot.get("min_external_width")
        max_width = snapshot.get("max_external_width")
        busy = snapshot.get("busy")
        gripped = snapshot.get("grip_detected")
        status = snapshot.get("status")
        outward = snapshot.get("finger_orientation_outward")
        orientation = None
        if isinstance(outward, bool):
            orientation = "outward" if outward else "inward"
        elif isinstance(outward, str):
            cleaned = outward.strip().lower()
            if cleaned in {"true", "1", "yes", "on", "outward"}:
                orientation = "outward"
            elif cleaned in {"false", "0", "no", "off", "inward"}:
                orientation = "inward"

        if not snapshot:
            width = self.client.get_ext_width(t_index=t_index)
            force = self.client.get_force(t_index=t_index)
            min_width = self.client.get_min_ext_width(t_index=t_index)
            max_width = self.client.get_max_ext_width(t_index=t_index)
            busy = self.client.isBusy(t_index=t_index)
            gripped = self.client.isGripped(t_index=t_index)
            status = self.client.getStatus(t_index=t_index)
            orientation = self.client.get_finger_orientation_label(t_index=t_index)

        return {
            "width": width,
            "force": force,
            "min_width": min_width,
            "max_width": max_width,
            "busy": busy,
            "gripped": gripped,
            "status": status,
            "orientation": orientation,
        }

    def width_limits(self, *, t_index: int) -> tuple[float, float]:
        snapshot = self.get_status_snapshot(t_index=t_index) or {}
        min_width = snapshot.get("min_external_width")
        max_width = snapshot.get("max_external_width")
        if not isinstance(min_width, (int, float)):
            min_width = self.client.get_min_ext_width(t_index=t_index)
        if not isinstance(max_width, (int, float)):
            max_width = self.client.get_max_ext_width(t_index=t_index)
        return (
            _safe_float(min_width, self.profile.close_width_default),
            _safe_float(max_width, self.profile.open_width_default),
        )

    def command(
        self,
        target_width: float,
        action_label: str,
        *,
        t_index: int,
        force: Optional[int] = None,
        speed: Optional[int] = None,
        gentle: Optional[bool] = None,
        vacuum_setpoint: Optional[int] = None,
        vacuum_threshold: Optional[float] = None,
        prevent_overclose: bool = False,
    ) -> str:
        min_width, max_width = self.width_limits(t_index=t_index)
        safe_width = float(target_width)
        if prevent_overclose:
            current_width = self.client.get_ext_width(t_index=t_index)
            if isinstance(current_width, (int, float)) and safe_width < current_width:
                safe_width = float(current_width)
        if _is_open_action(action_label) and safe_width <= min_width and max_width > min_width:
            safe_width = max_width
        safe_width = _clamp(safe_width, min_width, max_width)
        applied_force = _clamp(
            _safe_int(force, int(self.profile.force_default)),
            self.profile.force_min,
            self.profile.force_max,
        )
        applied_speed = _clamp(
            _safe_int(speed, self.profile.speed_default),
            self.profile.speed_min,
            self.profile.speed_max,
        )
        self.client.grip(
            t_index=t_index,
            t_width=safe_width,
            n_force=int(applied_force),
            p_speed=int(applied_speed),
            f_wait=True,
        )
        return (
            f"Gripper {action_label} to {safe_width:.1f} mm @ "
            f"{applied_force:.0f} N / {applied_speed:.0f}% speed."
        )


class SoftGripperAdapter(BaseGripperAdapter):
    """Soft Gripper width adapter."""

    key = "sg"
    capabilities = GripperCapabilities(
        grip_kind="soft_width",
        width=True,
        gentle=True,
    )

    def ensure_ready(self, *, t_index: int) -> None:
        try:
            if not bool(self.client.is_initialized(t_index=t_index)):
                self.client.initialize(t_index=t_index, wait=True)
        except AttributeError:
            return

    def read_telemetry(self, *, t_index: int) -> dict[str, Any]:
        snapshot = self.get_status_snapshot(t_index=t_index) or {}
        width = snapshot.get("width") or snapshot.get("current_width")
        min_width = snapshot.get("min_open")
        max_width = snapshot.get("max_open")
        if not isinstance(width, (int, float)):
            width = self.client.get_width(t_index=t_index)
        if not isinstance(min_width, (int, float)):
            min_width = self.client.get_min_open(t_index=t_index)
        if not isinstance(max_width, (int, float)):
            max_width = self.client.get_max_open(t_index=t_index)
        return {
            "width": width,
            "depth": self.client.get_depth(t_index=t_index),
            "min_width": min_width,
            "max_width": max_width,
            "busy": self.client.isBusy(t_index=t_index),
            "gripped": self.client.isGripped(t_index=t_index),
            "status": self.client.get_status(t_index=t_index),
            "initialized": self.client.is_initialized(t_index=t_index),
        }

    def width_limits(self, *, t_index: int) -> tuple[float, float]:
        return (
            _safe_float(self.client.get_min_open(t_index=t_index), self.profile.close_width_default),
            _safe_float(self.client.get_max_open(t_index=t_index), self.profile.open_width_default),
        )

    def command(
        self,
        target_width: float,
        action_label: str,
        *,
        t_index: int,
        force: Optional[int] = None,
        speed: Optional[int] = None,
        gentle: Optional[bool] = None,
        vacuum_setpoint: Optional[int] = None,
        vacuum_threshold: Optional[float] = None,
        prevent_overclose: bool = False,
    ) -> str:
        min_width, max_width = self.width_limits(t_index=t_index)
        safe_width = _clamp(float(target_width), min_width, max_width)
        if _is_open_action(action_label):
            self.client.move_to_width(t_index=t_index, width=safe_width, gentle=True, wait=True)
            return f"Soft Gripper {action_label} to {safe_width:.1f} mm."
        use_gentle = self.profile.gentle_default if gentle is None else bool(gentle)
        self.client.grip(t_index=t_index, width=safe_width, gentle=use_gentle, wait=True)
        suffix = " gentle" if use_gentle else ""
        return f"Soft Gripper{suffix} {action_label} to {safe_width:.1f} mm."


class VGC10Adapter(BaseGripperAdapter):
    """VGC10 vacuum adapter."""

    key = "vgc10"
    capabilities = GripperCapabilities(grip_kind="vacuum", vacuum=True)

    def read_telemetry(self, *, t_index: int) -> dict[str, Any]:
        vacuum = self.client.get_vacuum(t_index=t_index)
        vac_a = vacuum.get("a_vacuum", 0.0) if isinstance(vacuum, dict) else 0.0
        vac_b = vacuum.get("b_vacuum", 0.0) if isinstance(vacuum, dict) else 0.0
        threshold = self.default_vacuum_threshold()
        return {
            "vacuum_a": vac_a,
            "vacuum_b": vac_b,
            "busy": False,
            "gripped": float(vac_a) >= threshold and float(vac_b) >= threshold,
            "status": None,
            "vacuum_threshold": threshold,
        }

    def width_limits(self, *, t_index: int) -> tuple[float, float]:
        return float(self.profile.vacuum_min), float(self.profile.vacuum_max)

    def default_vacuum_threshold(self) -> float:
        return float(self.profile.vacuum_default) * float(
            self.profile.vacuum_hold_threshold_ratio
        )

    def is_holding(
        self,
        *,
        t_index: int,
        vacuum_threshold: Optional[float] = None,
    ) -> bool:
        threshold = (
            float(vacuum_threshold)
            if vacuum_threshold is not None
            else self.default_vacuum_threshold()
        )
        vacuum = self.client.get_vacuum(t_index=t_index)
        if not isinstance(vacuum, dict):
            return False
        return (
            _safe_float(vacuum.get("a_vacuum"), 0.0) >= threshold
            and _safe_float(vacuum.get("b_vacuum"), 0.0) >= threshold
        )

    def command(
        self,
        target_width: float,
        action_label: str,
        *,
        t_index: int,
        force: Optional[int] = None,
        speed: Optional[int] = None,
        gentle: Optional[bool] = None,
        vacuum_setpoint: Optional[int] = None,
        vacuum_threshold: Optional[float] = None,
        prevent_overclose: bool = False,
    ) -> str:
        if _is_open_action(action_label):
            self.client.release_vacuum(
                t_index=t_index,
                channel_a=True,
                channel_b=True,
                wait=True,
            )
            return "VGC10 vacuum released on both channels."
        setpoint = _safe_int(
            vacuum_setpoint if vacuum_setpoint is not None else target_width,
            self.profile.vacuum_default,
        )
        setpoint = int(_clamp(setpoint, self.profile.vacuum_min, self.profile.vacuum_max))
        self.client.grip_vacuum(
            t_index=t_index,
            vacuum_a=setpoint,
            vacuum_b=setpoint,
            wait=True,
        )
        return f"VGC10 vacuum {action_label} at {setpoint} kPa on both channels."


def adapter_for_key(key: str, client: Any, *, dimensions: Any = None) -> BaseGripperAdapter:
    """Build the adapter for a supported gripper key."""
    normalized = normalize_gripper_type(key)
    if normalized == "twofg7":
        return TwoFG7Adapter(client, dimensions=dimensions)
    if normalized == "sg":
        return SoftGripperAdapter(client, dimensions=dimensions)
    if normalized == "vgc10":
        return VGC10Adapter(client, dimensions=dimensions)
    raise OnRobotValidationError(f"Unsupported gripper type: {key}")


def detect_or_create_gripper(
    device: Any,
    *,
    requested_key: str,
    t_index: int,
) -> GripperDetection:
    """Detect or manually construct a supported gripper adapter."""
    normalized = normalize_gripper_type(requested_key)
    if normalized == AUTO_GRIPPER_TYPE:
        detected = detect_gripper(device, t_index=t_index, create_client=True)
        if detected.key not in SUPPORTED_GRIPPER_KEYS:
            raise OnRobotValidationError(
                f"Detected {detected.display_name}, but only 2FG7, Soft Gripper, and VGC10 are supported."
            )
        client = detected.client
        dimensions = detected.dimensions
        key = detected.key
    elif normalized == "twofg7":
        key = normalized
        client = TWOFG(device)
        dimensions = client.get_dimensions(t_index=t_index)
    elif normalized == "sg":
        key = normalized
        client = SG(device)
        dimensions = client.get_dimensions(t_index=t_index)
    elif normalized == "vgc10":
        key = normalized
        client = VG(device)
        if not client.is_vgc10(t_index=t_index):
            raise OnRobotConnectionError("No VGC10 device connected")
        dimensions = client.get_dimensions(t_index=t_index)
    else:
        raise OnRobotValidationError(f"Unsupported gripper type: {requested_key}")

    adapter = adapter_for_key(key, client, dimensions=dimensions)
    if not adapter.is_connected(t_index=t_index):
        raise OnRobotConnectionError(f"No {adapter.display_name} connected")
    adapter.ensure_ready(t_index=t_index)
    return GripperDetection(
        key=adapter.key,
        display_name=adapter.display_name,
        product_code=adapter.product_code,
        dimensions=adapter.dimensions,
        adapter=adapter,
    )

