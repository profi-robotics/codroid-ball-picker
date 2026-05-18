"""Tests for streaming frame admission."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.codroid_picker.streaming import MotionJPEGStream


class MotionJPEGStreamTests(unittest.TestCase):
    def test_add_frame_rate_limit_uses_admission_time(self) -> None:
        stream = MotionJPEGStream(min_push_interval_s=0.1)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with patch(
            "app.codroid_picker.streaming.time.monotonic",
            side_effect=[10.0, 10.05, 10.11],
        ):
            stream.add_frame(frame)
            stream.add_frame(frame)
            stream.add_frame(frame)

        self.assertEqual(stream.frame_count, 2)
        self.assertEqual(stream.get_frame_with_index()[1], 2)


if __name__ == "__main__":
    unittest.main()
