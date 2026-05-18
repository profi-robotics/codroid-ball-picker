from __future__ import annotations

import sys

import pytest

from codroid_picker.services import detection_worker


class _Cuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def synchronize(self) -> None:
        return None


class _Tensor:
    def __matmul__(self, other: object) -> "_Tensor":
        return self

    def __getitem__(self, key: object) -> "_Tensor":
        return self

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def item(self) -> float:
        return 1.0


class _Torch:
    def __init__(self, available: bool, usable: bool = True) -> None:
        self.cuda = _Cuda(available)
        self.usable = usable

    def ones(self, shape: tuple[int, int], device: str) -> _Tensor:
        if not self.usable:
            raise RuntimeError("no kernel image is available for execution on the device")
        return _Tensor()


def test_select_inference_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _Torch(True))
    monkeypatch.setattr(detection_worker.Path, "exists", lambda _self: False)

    assert detection_worker._select_inference_device() == "cuda:0"


def test_select_inference_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _Torch(False))
    monkeypatch.setattr(detection_worker.Path, "exists", lambda _self: False)

    assert detection_worker._select_inference_device() == "cpu"


def test_select_inference_device_rejects_unusable_cuda_on_jetson(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _Torch(True, usable=False))
    monkeypatch.setattr(detection_worker.Path, "exists", lambda _self: True)

    with pytest.raises(RuntimeError, match="cannot execute kernels"):
        detection_worker._select_inference_device()
