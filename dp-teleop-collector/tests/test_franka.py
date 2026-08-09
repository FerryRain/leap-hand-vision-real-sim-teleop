"""Tests for FR3 state parsing and deadman-relative palm mapping."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import numpy as np
from dp_collector.config import FrankaTeleopSettings
from dp_collector.franka import (
    LatestFrankaState,
    PalmVelocityMapper,
    parse_franka_state,
)


def _mapping_settings(**overrides: object) -> FrankaTeleopSettings:
    values: dict[str, object] = {
        "mapping_confirmed": False,
        "camera_to_global_matrix": (
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
        "linear_gain_per_s": 1.0,
        "deadband_m": 0.01,
        "maximum_hand_offset_m": 0.20,
        "maximum_linear_speed_m_s": 0.05,
    }
    values.update(overrides)
    return FrankaTeleopSettings(**values)


def _state_message() -> dict[str, object]:
    return {
        "sequence": 41,
        "bridge": {
            "last_fault": None,
            "control_lease_active": True,
            "watchdog_stop_count": 3,
            "workspace_guard_stop_count": 2,
        },
        "robot": {
            "timestamp_s": 12.5,
            "current_errors": {},
            "joints": {
                "position": [0.1 * index for index in range(7)],
                "velocity": [-0.01 * index for index in range(7)],
            },
            "end_effector": {
                "position": [0.40, -0.02, 0.30],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "linear_velocity": [0.01, 0.02, 0.03],
                "angular_velocity": [-0.1, -0.2, -0.3],
                "collision": [False] * 6,
            },
        },
    }


def _latest(message: dict[str, object] | None = None) -> LatestFrankaState:
    return LatestFrankaState(
        message=_state_message() if message is None else message,
        received_monotonic_s=100.0,
    )


class PalmVelocityMapperTests(unittest.TestCase):
    def test_requires_deadman_and_recenters_after_release(self) -> None:
        mapper = PalmVelocityMapper(_mapping_settings())

        np.testing.assert_array_equal(
            mapper.update((0.0, 0.0, 0.5), deadman_down=False),
            np.zeros(3),
        )
        np.testing.assert_array_equal(
            mapper.update((0.0, 0.0, 0.5), deadman_down=True),
            np.zeros(3),
        )
        self.assertFalse(
            np.array_equal(
                mapper.update((0.03, 0.0, 0.5), deadman_down=True),
                np.zeros(3),
            )
        )

        mapper.update((0.03, 0.0, 0.5), deadman_down=False)
        np.testing.assert_array_equal(
            mapper.update((0.50, 0.0, 0.5), deadman_down=True),
            np.zeros(3),
        )

    def test_applies_deadband_axis_mapping_and_speed_limit(self) -> None:
        mapper = PalmVelocityMapper(_mapping_settings(maximum_linear_speed_m_s=0.05))
        mapper.update((0.0, 0.0, 0.5), deadman_down=True)

        below_deadband = mapper.update(
            (0.009, -0.009, 0.5),
            deadman_down=True,
        )
        np.testing.assert_array_equal(below_deadband, np.zeros(3))

        mapped = mapper.update((0.03, -0.02, 0.54), deadman_down=True)
        # Active camera displacement is [0.02, -0.01, 0.03] after deadband.
        np.testing.assert_allclose(mapped, [0.03, -0.02, 0.01], atol=1e-12)

        limited_mapper = PalmVelocityMapper(
            _mapping_settings(
                deadband_m=0.0,
                maximum_linear_speed_m_s=0.02,
            )
        )
        limited_mapper.update((0.0, 0.0, 0.5), deadman_down=True)
        limited = limited_mapper.update(
            (0.10, 0.10, 0.5),
            deadman_down=True,
        )
        self.assertAlmostEqual(float(np.linalg.norm(limited)), 0.02)

    def test_resets_when_hand_exceeds_maximum_offset(self) -> None:
        mapper = PalmVelocityMapper(_mapping_settings(maximum_hand_offset_m=0.05))
        mapper.update((0.0, 0.0, 0.5), deadman_down=True)

        np.testing.assert_array_equal(
            mapper.update((0.06, 0.0, 0.5), deadman_down=True),
            np.zeros(3),
        )
        # The out-of-range update reset the anchor, so the next in-range hand
        # pose establishes a new neutral point rather than commanding a jump.
        np.testing.assert_array_equal(
            mapper.update((0.02, 0.0, 0.5), deadman_down=True),
            np.zeros(3),
        )


class ParseFrankaStateTests(unittest.TestCase):
    def test_builds_expected_29_value_vector(self) -> None:
        parsed = parse_franka_state(
            _latest(),
            now_s=100.05,
            maximum_age_s=0.10,
        )

        self.assertIs(parsed.valid, True)
        self.assertEqual(parsed.invalid_reasons, ())
        self.assertEqual(parsed.sequence, 41)
        self.assertEqual(parsed.watchdog_stop_count, 3)
        self.assertEqual(parsed.workspace_guard_stop_count, 2)
        self.assertEqual(parsed.raw_bridge["control_lease_active"], True)
        self.assertEqual(parsed.robot_timestamp_s, 12.5)
        self.assertAlmostEqual(parsed.age_s, 0.05)
        self.assertEqual(parsed.low_dim.shape, (29,))
        self.assertEqual(parsed.low_dim.dtype, np.float32)
        np.testing.assert_allclose(
            parsed.low_dim[0:7],
            [0.1 * index for index in range(7)],
        )
        np.testing.assert_allclose(
            parsed.low_dim[7:14],
            [-0.01 * index for index in range(7)],
        )
        np.testing.assert_allclose(parsed.low_dim[14:17], [0.40, -0.02, 0.30])
        np.testing.assert_allclose(parsed.low_dim[17:23], [1, 0, 0, 0, 1, 0])
        np.testing.assert_allclose(parsed.low_dim[23:26], [0.01, 0.02, 0.03])
        np.testing.assert_allclose(parsed.low_dim[26:29], [-0.1, -0.2, -0.3])

    def test_reports_age_fault_error_and_collision(self) -> None:
        message = _state_message()
        message["bridge"] = {
            "last_fault": "network command failed",
            "velocity_workspace_blocked": True,
        }
        robot = message["robot"]
        self.assertIsInstance(robot, dict)
        assert isinstance(robot, dict)
        robot["current_errors"] = {"joint": {"cartesian_reflex": True}}
        end_effector = robot["end_effector"]
        self.assertIsInstance(end_effector, dict)
        assert isinstance(end_effector, dict)
        end_effector["collision"] = [False, False, True, False, False, False]

        parsed = parse_franka_state(
            _latest(message),
            now_s=100.25,
            maximum_age_s=0.10,
        )

        self.assertIs(parsed.valid, False)
        self.assertEqual(
            parsed.invalid_reasons,
            (
                "franka_state_stale",
                "franka_bridge_fault",
                "franka_workspace_guard",
                "franka_current_error",
                "franka_collision",
            ),
        )

    def test_rejects_invalid_shapes_and_rotation(self) -> None:
        invalid_cases = (
            (("joints", "position"), [0.0] * 6, "joint position"),
            (("joints", "velocity"), [0.0] * 8, "joint velocity"),
            (("end_effector", "position"), [0.0, 0.0], "EEF position"),
            (
                ("end_effector", "rotation_matrix"),
                [[1.0, 0.0], [0.0, 1.0]],
                "rotation_matrix",
            ),
            (
                ("end_effector", "rotation_matrix"),
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]],
                "orthonormal",
            ),
            (
                ("end_effector", "rotation_matrix"),
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "determinant",
            ),
            (
                ("end_effector", "linear_velocity"),
                [0.0] * 2,
                "linear velocity",
            ),
            (
                ("end_effector", "angular_velocity"),
                [0.0, 0.0, np.nan],
                "angular velocity",
            ),
        )
        for field_path, invalid_value, message_match in invalid_cases:
            with self.subTest(field_path=field_path, value=invalid_value):
                message = copy.deepcopy(_state_message())
                robot = message["robot"]
                assert isinstance(robot, dict)
                section = robot[field_path[0]]
                assert isinstance(section, dict)
                section[field_path[1]] = invalid_value
                with self.assertRaisesRegex(ValueError, message_match):
                    parse_franka_state(
                        _latest(message),
                        now_s=100.0,
                        maximum_age_s=0.10,
                    )

    def test_state_parse_does_not_call_numpy_linalg(self) -> None:
        with mock.patch(
            "numpy.linalg.det",
            side_effect=AssertionError("runtime must not call NumPy LAPACK"),
        ):
            parsed = parse_franka_state(
                _latest(),
                now_s=100.0,
                maximum_age_s=0.10,
            )

        self.assertTrue(parsed.valid)

    def test_required_control_lease_is_part_of_state_validity(self) -> None:
        message = _state_message()
        bridge = message["bridge"]
        assert isinstance(bridge, dict)
        bridge["control_lease_active"] = False

        observe = parse_franka_state(
            _latest(message),
            now_s=100.0,
            maximum_age_s=0.10,
        )
        teleop = parse_franka_state(
            _latest(message),
            now_s=100.0,
            maximum_age_s=0.10,
            require_control_lease=True,
        )

        self.assertTrue(observe.valid)
        self.assertFalse(teleop.valid)
        self.assertIn("franka_control_lease_inactive", teleop.invalid_reasons)


if __name__ == "__main__":
    unittest.main()
