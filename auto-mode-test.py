import asyncio
import contextlib
import math
import time
from dataclasses import dataclass

from codroid_api import (
    AutoSocketConfig,
    CodroidAPI,
    CodroidConfig,
    DEFAULT_AUTO_SOCKET_PROJECT_ID,
)

HOST = '192.168.101.100'
PORT = 19093
USERNAME = 'admin'
PASSWORD = '123456'

MOVE_DELTA_Z_MM = 35.0
CYCLES = 4
REACH_TOLERANCE_MM = 3.0
MOVE_TIMEOUT_S = 10.0
AUTO_MODE_SETTLE_S = 0.5

POWER_KEYS = (
    'PowerOn',
    'Power',
    'power',
    'servo',
    'Servo',
    'servoOn',
    'ServoOn',
)
TRUE_VALUES = {'1', 'true', 'on', 'enabled', 'poweron', 'servoon'}
FALSE_VALUES = {'0', 'false', 'off', 'disabled', 'poweroff', 'servooff'}


@dataclass(frozen=True)
class RobotStateSnapshot:
    power_on: bool | None
    mode: str | None
    coordinate_id: int | None

    def summary(self):
        return {
            'power_on': self.power_on,
            'mode': self.mode or 'unknown',
            'coordinate_id': self.coordinate_id,
        }


def project_matches_managed(item):
    content = item.get('content') or {}
    return content.get('id') == DEFAULT_AUTO_SOCKET_PROJECT_ID


def dist_xyz(pose, target):
    dx = float(pose['x']) - float(target['x'])
    dy = float(pose['y']) - float(target['y'])
    dz = float(pose['z']) - float(target['z'])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
    if isinstance(value, dict) and 'value' in value:
        return coerce_bool(value.get('value'))
    return None


def find_power_state(payload):
    if isinstance(payload, dict):
        for key in POWER_KEYS:
            if key in payload:
                value = coerce_bool(payload.get(key))
                if value is not None:
                    return value

        for key, value in payload.items():
            if isinstance(key, str):
                lowered = key.lower()
                if 'power' in lowered or 'servo' in lowered:
                    power_on = coerce_bool(value)
                    if power_on is not None:
                        return power_on
            nested = find_power_state(value)
            if nested is not None:
                return nested

    elif isinstance(payload, list):
        for item in payload:
            nested = find_power_state(item)
            if nested is not None:
                return nested

    return None


def infer_power_on(status):
    power_on = find_power_state(status or {})
    if power_on is not None:
        return power_on

    state = (status or {}).get('state')
    if isinstance(state, (int, float)):
        return state >= 2

    return None


def mode_key(key):
    lowered = str(key).lower()
    return 'mode' in lowered and 'model' not in lowered


def normalize_mode_value(value):
    if isinstance(value, dict) and 'value' in value:
        return normalize_mode_value(value.get('value'))
    if isinstance(value, str):
        token = ''.join(char for char in value.lower() if char.isalnum())
        if 'auto' in token:
            return 'auto'
        if 'manual' in token or 'teach' in token or 'jog' in token:
            return 'manual'
    if isinstance(value, (int, float)):
        command_value = int(value)
        if command_value == 5:
            return 'auto'
        if command_value == 3:
            return 'manual'
    return None


def find_robot_mode(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if mode_key(key):
                mode = normalize_mode_value(value)
                if mode is not None:
                    return mode

        for value in payload.values():
            nested = find_robot_mode(value)
            if nested is not None:
                return nested

    elif isinstance(payload, list):
        for item in payload:
            nested = find_robot_mode(item)
            if nested is not None:
                return nested

    return None


def infer_coordinate_id(status):
    coord_id = (status or {}).get('CoordinateId')
    if coord_id is None:
        return None
    try:
        return int(coord_id)
    except (TypeError, ValueError):
        return None


def snapshot_robot_state(status):
    return RobotStateSnapshot(
        power_on=infer_power_on(status),
        mode=find_robot_mode(status or {}),
        coordinate_id=infer_coordinate_id(status),
    )


async def managed_count(api):
    listing = await api.list_projects()
    open_projects = (listing.get('data') or {}).get('open') or []
    managed = [item for item in open_projects if project_matches_managed(item)]

    socketish = []
    for item in open_projects:
        content = item.get('content') or {}
        label = content.get('label') or ''
        if label.startswith('CodroidAutoSocket') or label.startswith('CodroidSocket'):
            socketish.append((content.get('id'), label))

    return managed, socketish


def make_target_from_pose(pose, z):
    return CodroidAPI.build_target_cpos(
        float(pose['x']),
        float(pose['y']),
        float(z),
        float(pose['a']),
        float(pose['b']),
        float(pose['c']),
    )


async def restore_step(label, awaitable):
    try:
        await awaitable
    except Exception as exc:  # noqa: BLE001
        print('restore_warning', label, repr(exc), flush=True)


async def restore_mode(api, mode):
    if mode == 'auto':
        await restore_step('set_auto_mode', api.set_auto_mode())
        return
    if mode != 'manual':
        print('restore_mode_unknown_defaulting_to_manual', mode, flush=True)
    await restore_step('set_manual_mode', api.set_manual_mode())


async def restore_coordinate(api, coordinate_id):
    if coordinate_id is None:
        print('restore_coordinate_unknown_skipping', flush=True)
        return
    await restore_step(
        'set_current_coordinate_id',
        api.set_current_coordinate_id(coordinate_id),
    )


async def restore_robot_state(api, snapshot):
    print('restore_state_start', snapshot.summary(), flush=True)
    await restore_step('clear_target_position', api.clear_target_position())
    await restore_step('stop_command', api.stop_command())
    await restore_step('stop_auto_socket_project', api.stop_auto_socket_project())
    await asyncio.sleep(0.2)

    if snapshot.power_on is True:
        await restore_step('power_on', api.power_on())
        await asyncio.sleep(AUTO_MODE_SETTLE_S)
        await restore_coordinate(api, snapshot.coordinate_id)
        await restore_mode(api, snapshot.mode)
    elif snapshot.power_on is False:
        await restore_coordinate(api, snapshot.coordinate_id)
        await restore_mode(api, snapshot.mode)
        await asyncio.sleep(0.2)
        await restore_step('power_off', api.power_off())
    else:
        await restore_coordinate(api, snapshot.coordinate_id)
        await restore_mode(api, snapshot.mode)

    print('restore_state_done', snapshot.summary(), flush=True)


async def main():
    api = CodroidAPI(CodroidConfig(
        host=HOST,
        port=9000,
        origin=f'http://{HOST}:9098',
        token=f'user:{USERNAME}',
        username=USERNAME,
        user_password=PASSWORD,
    ))

    latest_pose = None
    latest_status = None
    project_messages = []
    nonempty_warnings = []
    nonempty_errors = []
    initial_state = None

    async def listener():
        nonlocal latest_pose, latest_status

        async for msg in api.listen():
            action = msg.get('action')

            if action == 'RobotPosture':
                data = (msg.get('data') or {}).get('data') or {}
                pose = data.get('end') or {}
                if pose:
                    latest_pose = pose

            elif action == 'RobotStatus':
                data = (msg.get('data') or {}).get('data') or {}
                latest_status = data.get('data') or data

            elif action == 'RobotWarning':
                data = msg.get('data') or {}
                if data.get('code') or data.get('data'):
                    nonempty_warnings.append(msg)

            elif action == 'RobotError':
                data = msg.get('data') or {}
                if data.get('code') or data.get('data'):
                    nonempty_errors.append(msg)

            elif msg.get('type') == 'projexecute':
                project_messages.append(msg)

    async def move_and_wait(target, label):
        await api.move_to_cartesian_target_linear(target)

        deadline = time.monotonic() + MOVE_TIMEOUT_S
        while time.monotonic() < deadline:
            if latest_pose and dist_xyz(latest_pose, target) < REACH_TOLERANCE_MM:
                print(label, latest_pose, flush=True)
                return
            await asyncio.sleep(0.2)

        error = None if not latest_pose else round(dist_xyz(latest_pose, target), 3)
        raise RuntimeError(
            f'Robot did not reach {label} within tolerance. error_mm={error}'
        )

    async def return_to_start_pose(target):
        if target is None:
            return
        if latest_pose and dist_xyz(latest_pose, target) < REACH_TOLERANCE_MM:
            print('restore_pose_already_at_start', latest_pose, flush=True)
            return

        print(
            'restore_pose_start',
            {k: target[k] for k in ('x', 'y', 'z', 'a', 'b', 'c')},
            flush=True,
        )
        try:
            await move_and_wait(target, 'restore_start_pose')
        except Exception as exc:  # noqa: BLE001
            print('restore_pose_failed', repr(exc), flush=True)

    await api.connect()
    task = asyncio.create_task(listener())

    try:
        await api.robot_login()
        await api.read_config()
        await api.read_system_data()
        await api.read_global_data()
        await asyncio.sleep(2.0)

        print('initial_status', latest_status, flush=True)
        print('initial_pose', latest_pose, flush=True)

        initial_state = snapshot_robot_state(latest_status)
        print('initial_robot_state', initial_state.summary(), flush=True)

        await api.set_current_coordinate_id(0)
        await asyncio.sleep(2.0)
        print('test_coordinate_pose', latest_pose, flush=True)

        if not latest_pose:
            raise RuntimeError('No RobotPosture received; refusing motion test.')

        local_ip = api._local_source_ip_for(HOST)
        config = AutoSocketConfig(
            socket_role='server',
            socket_bind_host='0.0.0.0',
            socket_port=PORT,
            project_socket_host=local_ip,
            project_socket_port=PORT,
            install_project=False,
            start_project=False,
            stop_project_on_exit=True,
            connect_on_enter=True,
            accept_timeout_s=25.0,
            read_timeout_ms=10000,
            move_speed='V100',
            move_acc='ACC100',
        )

        with contextlib.suppress(Exception):
            await api.stop_project()
            await asyncio.sleep(0.3)

        install1 = await api.install_auto_socket_project(config)
        install2 = await api.install_auto_socket_project(config)
        managed, socketish = await managed_count(api)

        print('install_ids', install1.project_id, install2.project_id, flush=True)
        print('managed_count_after_double_install', len(managed), flush=True)
        print('socketish_after_double_install', socketish, flush=True)

        if len(managed) != 1:
            raise RuntimeError(
                f'Expected exactly one managed project, found {len(managed)}'
            )

        initial_pose = dict(latest_pose)
        base_target = make_target_from_pose(initial_pose, float(initial_pose['z']))
        up_target = make_target_from_pose(
            initial_pose,
            float(initial_pose['z']) + MOVE_DELTA_Z_MM,
        )

        print(
            'base_target',
            {k: base_target[k] for k in ('x', 'y', 'z', 'a', 'b', 'c')},
            flush=True,
        )
        print(
            'up_target',
            {k: up_target[k] for k in ('x', 'y', 'z', 'a', 'b', 'c')},
            flush=True,
        )
        print('cycles', CYCLES, flush=True)

        print('activate_auto_mode_start', flush=True)
        await api.power_on()
        await asyncio.sleep(AUTO_MODE_SETTLE_S)
        await api.set_auto_mode()
        await asyncio.sleep(AUTO_MODE_SETTLE_S)
        print('activate_auto_mode_done', flush=True)

        run_config = AutoSocketConfig(
            socket_role='server',
            socket_bind_host='0.0.0.0',
            socket_port=PORT,
            project_socket_host=local_ip,
            project_socket_port=PORT,
            install_project=True,
            start_project=True,
            stop_project_on_exit=True,
            connect_on_enter=True,
            accept_timeout_s=25.0,
            read_timeout_ms=10000,
            move_speed='V100',
            move_acc='ACC100',
        )

        start = time.monotonic()

        async with api.auto_socket_mode(run_config) as backend:
            print(
                'connected_project',
                backend.project_info.project_id,
                backend.project_info.task_id,
                backend.project_info.run_label_id,
                flush=True,
            )

            try:
                for cycle in range(1, CYCLES + 1):
                    print('cycle_start', cycle, flush=True)
                    await move_and_wait(up_target, f'cycle_{cycle}_up_pose')
                    await move_and_wait(base_target, f'cycle_{cycle}_base_pose')
                    print('cycle_done', cycle, flush=True)
            finally:
                await return_to_start_pose(base_target)

        await asyncio.sleep(0.5)

        managed2, socketish2 = await managed_count(api)
        print('managed_count_after_run', len(managed2), flush=True)
        print('socketish_after_run', socketish2, flush=True)
        print('project_messages_seen', len(project_messages), flush=True)
        print(
            'nonempty_warnings',
            len(nonempty_warnings),
            'nonempty_errors',
            len(nonempty_errors),
            flush=True,
        )
        print('elapsed_s', round(time.monotonic() - start, 2), flush=True)

        if len(managed2) != 1:
            raise RuntimeError(
                f'Expected exactly one managed project after run, found {len(managed2)}'
            )

    finally:
        if initial_state is not None:
            await restore_robot_state(api, initial_state)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await api.close()


if __name__ == '__main__':
    asyncio.run(main())
