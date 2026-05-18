"""Tests for safe transit Z computation."""

from __future__ import annotations

import unittest

from app.codroid_picker.pick_place_helpers import compute_safe_transit_z


class TransitHeightTests(unittest.TestCase):
    def test_transit_uses_drop_and_approach_with_clearance(self) -> None:
        transit_z = compute_safe_transit_z(
            approach_mm=-30.0,
            drop_mm=0.0,
            drop_robot_xyz=(100.0, 200.0, -10.0),
            clearance_mm=10.0,
        )
        # min(-30, -10, -40) - 10 = -50
        self.assertAlmostEqual(transit_z, -50.0)

    def test_transit_without_drop_target_uses_approach(self) -> None:
        transit_z = compute_safe_transit_z(
            approach_mm=-25.0,
            drop_mm=0.0,
            drop_robot_xyz=None,
            clearance_mm=5.0,
        )
        self.assertAlmostEqual(transit_z, -30.0)


if __name__ == "__main__":
    unittest.main()
