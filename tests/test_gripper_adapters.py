from __future__ import annotations

import pytest

from app.codroid_picker.gripper_adapters import (
    SoftGripperAdapter,
    TwoFG7Adapter,
    VGC10Adapter,
    gripper_type_options,
    normalize_gripper_type,
)


class _TwoFG7Client:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.width = 20.0

    def isConnected(self, t_index=0):  # noqa: ANN001, ANN201, N802
        return True

    def get_status_snapshot(self, t_index=0):  # noqa: ANN001, ANN201
        return None

    def get_min_ext_width(self, t_index=0):  # noqa: ANN001, ANN201
        return 5.0

    def get_max_ext_width(self, t_index=0):  # noqa: ANN001, ANN201
        return 25.0

    def get_ext_width(self, t_index=0):  # noqa: ANN001, ANN201
        return self.width

    def grip(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(("grip", kwargs))


class _SoftClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.initialized = False

    def isConnected(self, t_index=0):  # noqa: ANN001, ANN201, N802
        return True

    def is_initialized(self, t_index=0):  # noqa: ANN001, ANN201
        return self.initialized

    def initialize(self, t_index=0, wait=True):  # noqa: ANN001, ANN201
        self.calls.append(("initialize", t_index, wait))
        self.initialized = True

    def get_min_open(self, t_index=0):  # noqa: ANN001, ANN201
        return 11.0

    def get_max_open(self, t_index=0):  # noqa: ANN001, ANN201
        return 75.0

    def grip(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(("grip", kwargs))

    def move_to_width(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(("move_to_width", kwargs))


class _VGC10Client:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.vacuum = {"a_vacuum": 0.0, "b_vacuum": 0.0}

    def isConnected(self, t_index=0):  # noqa: ANN001, ANN201, N802
        return True

    def grip_vacuum(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(("grip_vacuum", kwargs))
        self.vacuum["a_vacuum"] = float(kwargs["vacuum_a"])
        self.vacuum["b_vacuum"] = float(kwargs["vacuum_b"])

    def release_vacuum(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(("release_vacuum", kwargs))
        self.vacuum = {"a_vacuum": 0.0, "b_vacuum": 0.0}

    def get_vacuum(self, t_index=0):  # noqa: ANN001, ANN201
        return dict(self.vacuum)


@pytest.mark.unit
def test_gripper_type_options_include_auto_and_supported_devices() -> None:
    assert [option["value"] for option in gripper_type_options()] == [
        "auto",
        "twofg7",
        "sg",
        "vgc10",
    ]
    assert normalize_gripper_type("2fg7") == "twofg7"
    assert normalize_gripper_type("unknown") == "auto"


@pytest.mark.unit
def test_twofg7_adapter_clamps_width_and_sends_force_speed() -> None:
    client = _TwoFG7Client()
    adapter = TwoFG7Adapter(client)

    message = adapter.command(
        2.0,
        "closed",
        t_index=0,
        force=500,
        speed=2,
    )

    assert "5.0 mm" in message
    assert client.calls == [
        (
            "grip",
            {
                "t_index": 0,
                "t_width": 5.0,
                "n_force": 140,
                "p_speed": 10,
                "f_wait": True,
            },
        )
    ]


@pytest.mark.unit
def test_soft_adapter_initializes_and_routes_grip_and_open() -> None:
    client = _SoftClient()
    adapter = SoftGripperAdapter(client)

    adapter.ensure_ready(t_index=0)
    adapter.command(56.0, "closed", t_index=0, gentle=True)
    adapter.command(75.0, "opened", t_index=0)

    assert client.calls[0] == ("initialize", 0, True)
    assert client.calls[1] == (
        "grip",
        {"t_index": 0, "width": 56.0, "gentle": True, "wait": True},
    )
    assert client.calls[2] == (
        "move_to_width",
        {"t_index": 0, "width": 75.0, "gentle": True, "wait": True},
    )


@pytest.mark.unit
def test_vgc10_adapter_uses_one_setpoint_for_both_channels_and_releases() -> None:
    client = _VGC10Client()
    adapter = VGC10Adapter(client)

    adapter.command(0.0, "closed", t_index=0, vacuum_setpoint=40)
    assert adapter.is_holding(t_index=0, vacuum_threshold=32.0)
    adapter.command(0.0, "opened", t_index=0)

    assert client.calls[0] == (
        "grip_vacuum",
        {"t_index": 0, "vacuum_a": 40, "vacuum_b": 40, "wait": True},
    )
    assert client.calls[1] == (
        "release_vacuum",
        {"t_index": 0, "channel_a": True, "channel_b": True, "wait": True},
    )
    assert not adapter.is_holding(t_index=0, vacuum_threshold=32.0)
