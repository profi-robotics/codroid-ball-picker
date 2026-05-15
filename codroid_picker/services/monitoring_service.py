"""Lightweight monitoring service."""

from __future__ import annotations

import time
from importlib import import_module
from typing import Any

from codroid_picker.state.store import StateStore


class MonitoringService:
    def __init__(self, state: StateStore) -> None:
        self.state = state

    def record_pick(self, *, success: bool, cycle_s: float | None = None) -> None:
        current = self.state.domain("monitoring")
        attempts = int(current.get("pick_attempts", 0)) + 1
        successes = int(current.get("pick_successes", 0)) + (1 if success else 0)
        failures = int(current.get("pick_failures", 0)) + (0 if success else 1)
        cycle_times = list(current.get("cycle_times", []))
        if cycle_s is not None:
            cycle_times.append(float(cycle_s))
            cycle_times = cycle_times[-50:]
        self.state.update_domain(
            "monitoring",
            pick_attempts=attempts,
            pick_successes=successes,
            pick_failures=failures,
            last_cycle_s=cycle_s,
            cycle_times=cycle_times,
            system={"updated_at": time.time()},
        )

    def sample_system(self) -> dict[str, Any]:
        system = {
            "updated_at": time.time(),
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
        }
        try:
            psutil = import_module("psutil")
            system["cpu_percent"] = float(psutil.cpu_percent(interval=None))
            system["memory_percent"] = float(psutil.virtual_memory().percent)
        except Exception:
            pass
        gpu = self._gpu_snapshot()
        self.state.update_domain("monitoring", system=system, gpu=gpu)
        return {"system": system, "gpu": gpu}

    @staticmethod
    def _gpu_snapshot() -> dict[str, Any]:
        try:
            pynvml = import_module("pynvml")

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            devices = []
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name_raw = pynvml.nvmlDeviceGetName(handle)
                name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                devices.append(
                    {
                        "index": index,
                        "name": name,
                        "memory_used_mb": round(memory.used / 1024 / 1024, 1),
                        "memory_total_mb": round(memory.total / 1024 / 1024, 1),
                    }
                )
            return {"available": count > 0, "devices": devices}
        except Exception:
            return {"available": False, "devices": []}
