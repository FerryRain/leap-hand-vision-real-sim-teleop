from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from franka_bridge.waypoints import Waypoint, WaypointSet, distance_m

ROOT = Path(__file__).resolve().parents[1]


def point(index: int, position: tuple[float, float, float]) -> Waypoint:
    return Waypoint(
        name=f"P{index}",
        position_m=position,
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        robot_timestamp_s=float(index),
    )


class WaypointTests(unittest.TestCase):
    def test_four_points_round_trip_atomically(self) -> None:
        original = WaypointSet.create(
            [
                point(1, (0.40, 0.00, 0.30)),
                point(2, (0.42, 0.00, 0.30)),
                point(3, (0.42, 0.02, 0.30)),
                point(4, (0.40, 0.02, 0.30)),
            ]
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "points.json"
            original.save(path, overwrite=False)
            loaded = WaypointSet.load(path)
            self.assertEqual(loaded.points, original.points)
            self.assertFalse((path.parent / ".points.json.tmp").exists())

    def test_requires_exactly_four_points(self) -> None:
        with self.assertRaises(ValueError):
            WaypointSet.create([point(1, (0.4, 0.0, 0.3))])

    def test_workspace_rejects_outside_point(self) -> None:
        waypoints = WaypointSet.create(
            [point(index, (0.4, 0.0, 0.8)) for index in range(1, 5)]
        )
        with self.assertRaises(ValueError):
            waypoints.validate_workspace(
                {
                    "workspace_min_m": [0.2, -0.4, 0.1],
                    "workspace_max_m": [0.7, 0.4, 0.7],
                }
            )

    def test_distance_is_in_metres(self) -> None:
        self.assertAlmostEqual(distance_m((0.0, 0.0, 0.0), (0.003, 0.004, 0.0)), 0.005)


if __name__ == "__main__":
    unittest.main()
