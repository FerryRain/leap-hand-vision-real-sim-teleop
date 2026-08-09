"""Safety-boundary tests for collector configuration."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from dp_collector.config import CollectorSettings, FrankaTeleopSettings


def _collector_config() -> dict[str, object]:
    return {
        "sample_hz": 10.0,
        "image_width": 320,
        "image_height": 240,
        "jpeg_quality": 92,
        "minimum_episode_steps": 20,
        "maximum_episode_s": 90.0,
        "maximum_invalid_steps": 0,
        "maximum_camera_age_s": 0.10,
        "maximum_leap_state_age_s": 0.10,
        "maximum_franka_state_age_s": 0.15,
        "max_pending_frames": 4,
    }


def _franka_config() -> dict[str, object]:
    return {
        "mapping_confirmed": False,
        "camera_to_global_matrix": [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        "linear_gain_per_s": 0.6,
        "deadband_m": 0.01,
        "maximum_hand_offset_m": 0.2,
        "maximum_linear_speed_m_s": 0.02,
    }


class ConfigTests(unittest.TestCase):
    def test_valid_config_preserves_disabled_real_arm_mapping_gate(self) -> None:
        collector = CollectorSettings.from_dict(_collector_config())
        franka = FrankaTeleopSettings.from_dict(_franka_config())

        self.assertEqual(collector.sample_hz, 10.0)
        self.assertEqual(collector.maximum_invalid_steps, 0)
        self.assertIs(franka.mapping_confirmed, False)

    def test_collector_rejects_values_outside_safety_limits(self) -> None:
        unsafe_values = (
            ("sample_hz", 0.99),
            ("sample_hz", 30.01),
            ("image_width", 63),
            ("image_height", 63),
            ("jpeg_quality", 0),
            ("jpeg_quality", 101),
            ("minimum_episode_steps", 1),
            ("maximum_episode_s", 0.99),
            ("maximum_episode_s", 600.01),
            ("maximum_invalid_steps", -1),
            ("maximum_camera_age_s", 0.019),
            ("maximum_leap_state_age_s", 1.01),
            ("maximum_franka_state_age_s", float("nan")),
            ("max_pending_frames", 0),
            ("max_pending_frames", 17),
        )
        for field, unsafe_value in unsafe_values:
            with self.subTest(field=field, value=unsafe_value):
                raw = _collector_config()
                raw[field] = unsafe_value
                with self.assertRaises(ValueError):
                    CollectorSettings.from_dict(raw)

    def test_franka_mapping_rejects_values_outside_safety_limits(self) -> None:
        unsafe_values = (
            ("linear_gain_per_s", 0.0),
            ("linear_gain_per_s", 5.01),
            ("deadband_m", -0.001),
            ("deadband_m", 0.051),
            ("maximum_hand_offset_m", 0.019),
            ("maximum_hand_offset_m", 0.51),
            ("maximum_linear_speed_m_s", 0.0009),
            ("maximum_linear_speed_m_s", 0.101),
        )
        for field, unsafe_value in unsafe_values:
            with self.subTest(field=field, value=unsafe_value):
                raw = _franka_config()
                raw[field] = unsafe_value
                with self.assertRaises(ValueError):
                    FrankaTeleopSettings.from_dict(raw)

    def test_franka_mapping_rejects_invalid_camera_matrix(self) -> None:
        invalid_matrices = (
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]],
            [
                [1.0, 0.0, 0.0],
                [0.0, float("nan"), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                raw = copy.deepcopy(_franka_config())
                raw["camera_to_global_matrix"] = matrix
                with self.assertRaises(ValueError):
                    FrankaTeleopSettings.from_dict(raw)

    def test_franka_mapping_accepts_orthonormal_axis_reflection(self) -> None:
        raw = _franka_config()
        raw["camera_to_global_matrix"] = [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        settings = FrankaTeleopSettings.from_dict(raw)

        self.assertEqual(settings.camera_to_global_matrix[0], (-1.0, 0.0, 0.0))

    def test_franka_mapping_validation_does_not_call_numpy_linalg(self) -> None:
        with mock.patch(
            "numpy.linalg.det",
            side_effect=AssertionError("runtime must not call NumPy LAPACK"),
        ):
            FrankaTeleopSettings.from_dict(_franka_config())


if __name__ == "__main__":
    unittest.main()
