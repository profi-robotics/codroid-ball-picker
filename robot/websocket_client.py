"""
Async WebSocket client for the Estun Codroid controller.

The protocol is reverse engineered from browser DevTools traffic. Payloads are
JSON objects with a `cmd` key denoting intent. The client focuses on a minimal
subset required for the bootstrap demo: enabling the robot and issuing MovL and
MovJ motions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import websockets

LOGGER = logging.getLogger(__name__)


@dataclass
class MotionParameters:
    speed: float = 50.0
    acceleration: float = 100.0
    blend: float = 0.0


class CodroidWebSocketClient:
    def __init__(self, uri: str, keepalive_interval: float = 2.0) -> None:
        self.uri = uri
        self.keepalive_interval = keepalive_interval
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        LOGGER.info("Connecting to Codroid controller at %s", self.uri)
        self._ws = await websockets.connect(self.uri)
        self._keepalive_task = asyncio.create_task(self._heartbeat())

    async def close(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _heartbeat(self) -> None:
        assert self._ws is not None
        while True:
            await asyncio.sleep(self.keepalive_interval)
            await self._ws.send(json.dumps({"cmd": "keepalive"}))

    async def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("WebSocket is not connected. Call connect() first.")
        message = json.dumps(payload)
        LOGGER.debug("Sending WS payload: %s", message)
        await self._ws.send(message)
        response_raw = await self._ws.recv()
        try:
            response = json.loads(response_raw)
        except json.JSONDecodeError:
            LOGGER.warning("Received non-JSON response: %s", response_raw)
            response = {"raw": response_raw}
        LOGGER.debug("Received WS response: %s", response)
        return response

    async def enable(self) -> Dict[str, Any]:
        return await self.send({"cmd": "enable"})

    async def disable(self) -> Dict[str, Any]:
        return await self.send({"cmd": "disable"})

    async def movl(self, pose: Dict[str, float], params: MotionParameters) -> Dict[str, Any]:
        payload = {
            "cmd": "movl",
            "pose": pose,
            "speed": params.speed,
            "acc": params.acceleration,
            "blend": params.blend,
        }
        return await self.send(payload)

    async def movj(self, joints: Dict[str, float], params: MotionParameters) -> Dict[str, Any]:
        payload = {
            "cmd": "movj",
            "joints": joints,
            "speed": params.speed,
            "acc": params.acceleration,
        }
        return await self.send(payload)

    async def get_state(self) -> Dict[str, Any]:
        return await self.send({"cmd": "get_state"})


async def demo_movements(uri: str, center_pose: Dict[str, float], drop_pose: Dict[str, float]) -> None:
    client = CodroidWebSocketClient(uri)
    await client.connect()
    await client.enable()
    params = MotionParameters(speed=25.0, acceleration=50.0, blend=0.0)
    await client.movl(center_pose, params)
    await client.movl(drop_pose, params)
    await client.disable()
    await client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example invocation (update URI/poses for your setup):
    # asyncio.run(demo_movements("ws://192.168.0.10:9090/codroid/ws", center_pose, drop_pose))
