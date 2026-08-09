from __future__ import annotations

import time
import unittest

import numpy as np
from franka_bridge.mock_controller import MockFrankaController
from franka_bridge.runtime import FrankaRuntime

from tests.helpers import test_config


class PositionOnlyMockFrankaController(MockFrankaController):
    def get_state_snapshot(self) -> dict[str, object]:
        snapshot = super().get_state_snapshot()
        end_effector = snapshot["end_effector"]
        assert isinstance(end_effector, dict)
        end_effector.pop("rotation_matrix")
        return snapshot


class NumpyRotationMockFrankaController(MockFrankaController):
    def get_state_snapshot(self) -> dict[str, object]:
        snapshot = super().get_state_snapshot()
        end_effector = snapshot["end_effector"]
        assert isinstance(end_effector, dict)
        end_effector["rotation_matrix"] = np.asarray(
            end_effector["rotation_matrix"],
            dtype=float,
        )
        return snapshot


class RuntimeSafetyTests(unittest.TestCase):
    def test_stale_velocity_is_stopped_by_robot_side_watchdog(self) -> None:
        controller = MockFrankaController()
        runtime = FrankaRuntime(controller, test_config())
        try:
            self.assertTrue(runtime.acquire("owner"))
            runtime.submit_velocity(
                "owner", 1, "global", (0.01, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            time.sleep(0.05)
            self.assertGreater(controller.velocity_command_count, 0)
            time.sleep(0.12)
            self.assertFalse(runtime.bridge_status()["velocity_active"])
            self.assertGreaterEqual(controller.stop_continuous_count, 1)
            self.assertGreaterEqual(runtime.bridge_status()["watchdog_stop_count"], 1)
        finally:
            runtime.close()

    def test_only_one_client_can_own_control_and_sequences_cannot_replay(self) -> None:
        runtime = FrankaRuntime(MockFrankaController(), test_config())
        try:
            self.assertTrue(runtime.acquire("first"))
            self.assertFalse(runtime.acquire("second"))
            runtime.submit_velocity(
                "first", 4, "global", (0.0, 0.0, 0.01), (0.0, 0.0, 0.0)
            )
            with self.assertRaises(ValueError):
                runtime.submit_velocity(
                    "first", 4, "global", (0.0, 0.0, 0.01), (0.0, 0.0, 0.0)
                )
        finally:
            runtime.close()

    def test_release_immediately_stops_and_revokes_control(self) -> None:
        controller = MockFrankaController()
        runtime = FrankaRuntime(controller, test_config())
        try:
            runtime.acquire("owner")
            runtime.submit_velocity(
                "owner", 1, "global", (0.01, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
            time.sleep(0.03)
            runtime.release("owner")
            self.assertFalse(runtime.bridge_status()["control_lease_active"])
            self.assertFalse(runtime.bridge_status()["velocity_active"])
            self.assertGreaterEqual(controller.stop_continuous_count, 1)
        finally:
            runtime.close()

    def test_one_shot_motion_is_bounded_by_step_and_workspace(self) -> None:
        runtime = FrankaRuntime(MockFrankaController(), test_config())
        try:
            runtime.acquire("owner")
            runtime.move_relative("owner", (0.01, 0.0, 0.0), 0.1)
            with self.assertRaises(ValueError):
                runtime.move_relative("owner", (0.03, 0.0, 0.0), 0.1)
            with self.assertRaises(ValueError):
                runtime.move_global(
                    "owner", (2.0, 0.0, 0.0), absolute=True, dynamics_factor=0.1
                )
        finally:
            runtime.close()

    def test_absolute_global_motion_does_not_require_rotation_matrix(self) -> None:
        controller = PositionOnlyMockFrankaController()
        runtime = FrankaRuntime(controller, test_config())
        try:
            runtime.acquire("owner")
            target = (0.41, -0.01, 0.23)
            runtime.move_global(
                "owner",
                target,
                absolute=True,
                dynamics_factor=0.05,
            )
            self.assertEqual(tuple(controller.position), target)
        finally:
            runtime.close()

    def test_absolute_global_pose_commands_rotation(self) -> None:
        controller = MockFrankaController()
        runtime = FrankaRuntime(controller, test_config())
        rotation = (
            (0.9950041653, -0.0998334166, 0.0),
            (0.0998334166, 0.9950041653, 0.0),
            (0.0, 0.0, 1.0),
        )
        try:
            runtime.acquire("owner")
            runtime.move_global(
                "owner",
                (0.41, -0.01, 0.23),
                absolute=True,
                dynamics_factor=0.05,
                rotation_matrix=rotation,
            )
            self.assertEqual(
                controller.rotation_matrix, [list(row) for row in rotation]
            )
        finally:
            runtime.close()

    def test_absolute_pose_accepts_numpy_rotation_from_real_controller(self) -> None:
        controller = NumpyRotationMockFrankaController()
        runtime = FrankaRuntime(controller, test_config())
        try:
            runtime.acquire("owner")
            runtime.move_global(
                "owner",
                (0.41, -0.01, 0.23),
                absolute=True,
                dynamics_factor=0.05,
                rotation_matrix=np.eye(3),
            )
            self.assertEqual(controller.position, [0.41, -0.01, 0.23])
        finally:
            runtime.close()

    def test_orientation_motion_respects_enable_and_step_limit(self) -> None:
        rotation = (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        for config in (
            test_config(allow_orientation_motion=False),
            test_config(max_orientation_step_rad=0.2),
        ):
            runtime = FrankaRuntime(MockFrankaController(), config)
            try:
                runtime.acquire("owner")
                with self.assertRaises((PermissionError, ValueError)):
                    runtime.move_global(
                        "owner",
                        (0.41, -0.01, 0.23),
                        absolute=True,
                        dynamics_factor=0.05,
                        rotation_matrix=rotation,
                    )
            finally:
                runtime.close()

    def test_stop_command_revokes_lease(self) -> None:
        runtime = FrankaRuntime(MockFrankaController(), test_config())
        try:
            runtime.acquire("owner")
            runtime.stop_all()
            self.assertFalse(runtime.bridge_status()["control_lease_active"])
            with self.assertRaises(PermissionError):
                runtime.heartbeat("owner")
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
