"""Runtime entry safety and image-shape tests."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
for path in (ROOT, REPOSITORY_ROOT, REPOSITORY_ROOT / "franka-lan-bridge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dp_collector.collector import (  # noqa: E402
    _build_step,
    _crop_preview_to_training_fov,
    _hand_inside_training_crop,
    _resize_rgbd,
    _scaled_intrinsics,
    _validate_args,
    _validate_franka_motion_server,
    build_parser,
)
from dp_collector.config import AppSettings, load_app_settings  # noqa: E402
from dp_collector.franka import ParsedFrankaState  # noqa: E402


class CollectorRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_app_settings(ROOT / "config.yml")

    def args(self, *extra: str):
        return build_parser().parse_args(list(extra))

    def test_defaults_cannot_move_real_hardware(self) -> None:
        args = self.args()
        self.assertEqual(args.leap_device, "mock")
        self.assertEqual(args.franka_mode, "off")
        self.assertFalse(args.enable_leap_torque)
        self.assertFalse(args.enable_franka_motion)
        _validate_args(args, self.settings)

    def test_real_leap_requires_port_and_explicit_torque(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disarmed"):
            _validate_args(self.args("--leap-device", "real"), self.settings)
        with self.assertRaisesRegex(RuntimeError, "leap-port"):
            _validate_args(
                self.args("--leap-device", "real", "--enable-leap-torque"),
                self.settings,
            )

    def test_franka_motion_has_independent_gates_and_preview_deadman(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "disarmed"):
            _validate_args(self.args("--franka-mode", "teleop"), self.settings)
        with self.assertRaisesRegex(ValueError, "requires --franka-mode"):
            _validate_args(self.args("--enable-franka-motion"), self.settings)
        with self.assertRaisesRegex(RuntimeError, "mapping_confirmed"):
            _validate_args(
                self.args(
                    "--franka-mode",
                    "teleop",
                    "--enable-franka-motion",
                ),
                self.settings,
            )

        confirmed = AppSettings(
            collector=self.settings.collector,
            franka_teleop=replace(
                self.settings.franka_teleop,
                mapping_confirmed=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "mouse deadman"):
            _validate_args(
                self.args(
                    "--franka-mode",
                    "teleop",
                    "--enable-franka-motion",
                    "--headless",
                ),
                confirmed,
            )

    def test_mock_auto_mode_cannot_enable_any_real_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "mock auto episodes require"):
            _validate_args(
                self.args(
                    "--headless",
                    "--mock-auto-episodes",
                    "grasp,release",
                    "--source",
                    "d455",
                ),
                self.settings,
            )

    def test_franka_teleop_requires_patched_workspace_guard_server(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "update and restart"):
            _validate_franka_motion_server(
                {"max_linear_speed": 0.05},
                self.settings,
            )

        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            _validate_franka_motion_server(
                {
                    "continuous_velocity_workspace_guard": True,
                    "max_linear_speed": 0.01,
                },
                self.settings,
            )

        _validate_franka_motion_server(
            {
                "continuous_velocity_workspace_guard": True,
                "max_linear_speed": 0.05,
            },
            self.settings,
        )

    def test_rgbd_resize_keeps_rgb_uint8_and_depth_uint16(self) -> None:
        color_bgr = np.zeros((12, 16, 3), dtype=np.uint8)
        color_bgr[..., 2] = 255
        depth = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)

        rgb, resized_depth = _resize_rgbd(
            color_bgr,
            depth,
            width=8,
            height=6,
        )

        self.assertEqual(rgb.shape, (6, 8, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.all(rgb[..., 0] == 255))
        self.assertEqual(resized_depth.shape, (6, 8))
        self.assertEqual(resized_depth.dtype, np.uint16)

    def test_preview_and_tracking_use_the_same_center_crop_as_training(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :160] = 50
        frame[:, 1120:] = 100
        cropped = _crop_preview_to_training_fov(
            frame,
            target_width=320,
            target_height=240,
        )
        self.assertEqual(cropped.shape, (720, 960, 3))
        self.assertTrue(np.all(cropped == 0))

        centered = np.full((21, 2), (0.5, 0.5), dtype=np.float64)
        self.assertTrue(
            _hand_inside_training_crop(
                centered,
                source_width=1280,
                source_height=720,
                target_width=320,
                target_height=240,
            )
        )
        centered[8, 0] = 0.05
        self.assertFalse(
            _hand_inside_training_crop(
                centered,
                source_width=1280,
                source_height=720,
                target_width=320,
                target_height=240,
            )
        )

    def test_full_franka_leap_step_has_45_state_and_22_action_values(self) -> None:
        args = self.args("--franka-mode", "teleop")
        franka = ParsedFrankaState(
            low_dim=np.arange(29, dtype=np.float32),
            raw_robot={},
            raw_bridge={
                "watchdog_stop_count": 4,
                "workspace_guard_stop_count": 2,
            },
            sequence=7,
            robot_timestamp_s=123.0,
            received_monotonic_s=9.99,
            age_s=0.01,
            watchdog_stop_count=4,
            workspace_guard_stop_count=2,
            valid=True,
            invalid_reasons=(),
        )
        source = SimpleNamespace(latest_palm_position_m=(0.0, 0.0, 0.5))
        rgbd = SimpleNamespace(
            capture_monotonic_s=9.99,
            camera_timestamp_ms=456.0,
            depth_scale_m=0.001,
        )
        leap = np.arange(16, dtype=np.float32) / 10.0
        twist = np.asarray((0.01, 0.0, -0.01, 0.0, 0.0, 0.0))

        state, action, ages, invalid_reasons, extra = _build_step(
            args=args,
            app_settings=self.settings,
            source=source,
            rgbd=rgbd,
            landmarks=np.zeros((21, 3), dtype=np.float64),
            handedness="right",
            confidence=0.99,
            tracking_valid=True,
            mapping_mode="tracking",
            leap_vision_target=leap,
            leap_applied=leap,
            leap_actual=leap,
            leap_read_s=9.995,
            leap_read_failed=False,
            leap_command_s=9.996,
            franka=franka,
            franka_twist=twist,
            franka_command_s=9.997,
            franka_ack_s=9.998,
            franka_watchdog_start_count=4,
            franka_workspace_guard_start_count=2,
            sample_s=10.0,
            run_start_s=0.0,
            deadman_requested=True,
        )

        self.assertEqual(state.shape, (45,))
        self.assertEqual(action.shape, (22,))
        np.testing.assert_allclose(action[:6], twist)
        np.testing.assert_allclose(action[6:], leap)
        self.assertEqual(set(ages), {"camera", "leap", "franka"})
        self.assertEqual(invalid_reasons, ())
        self.assertEqual(extra["franka"]["sequence"], 7)
        self.assertEqual(extra["franka"]["bridge"]["watchdog_stop_count"], 4)
        self.assertAlmostEqual(extra["franka_action_age_s"], 0.003)
        self.assertAlmostEqual(extra["franka_action_ack_latency_s"], 0.001)
        self.assertAlmostEqual(extra["leap_action_age_s"], 0.004)

    def test_bridge_counter_increase_invalidates_the_episode_step(self) -> None:
        args = self.args("--franka-mode", "teleop")
        franka = ParsedFrankaState(
            low_dim=np.zeros(29, dtype=np.float32),
            raw_robot={},
            raw_bridge={
                "watchdog_stop_count": 5,
                "workspace_guard_stop_count": 3,
            },
            sequence=8,
            robot_timestamp_s=124.0,
            received_monotonic_s=9.99,
            age_s=0.01,
            watchdog_stop_count=5,
            workspace_guard_stop_count=3,
            valid=True,
            invalid_reasons=(),
        )
        source = SimpleNamespace(latest_palm_position_m=(0.0, 0.0, 0.5))
        rgbd = SimpleNamespace(
            capture_monotonic_s=9.99,
            camera_timestamp_ms=456.0,
            depth_scale_m=0.001,
        )
        leap = np.zeros(16, dtype=np.float32)

        _state, _action, _ages, reasons, _extra = _build_step(
            args=args,
            app_settings=self.settings,
            source=source,
            rgbd=rgbd,
            landmarks=np.zeros((21, 3), dtype=np.float64),
            handedness="right",
            confidence=0.99,
            tracking_valid=True,
            mapping_mode="tracking",
            leap_vision_target=leap,
            leap_applied=leap,
            leap_actual=leap,
            leap_read_s=9.995,
            leap_read_failed=False,
            leap_command_s=9.996,
            franka=franka,
            franka_twist=np.zeros(6),
            franka_command_s=9.997,
            franka_ack_s=9.998,
            franka_watchdog_start_count=4,
            franka_workspace_guard_start_count=2,
            sample_s=10.0,
            run_start_s=0.0,
            deadman_requested=True,
        )

        self.assertIn("franka_velocity_watchdog_stopped", reasons)
        self.assertIn("franka_workspace_guard_triggered", reasons)

    def test_training_intrinsics_follow_image_resize(self) -> None:
        result = _scaled_intrinsics(
            {
                "width": 1280,
                "height": 720,
                "ppx": 640.0,
                "ppy": 360.0,
                "fx": 900.0,
                "fy": 900.0,
                "model": "brown_conrady",
                "coeffs": [0.0] * 5,
            },
            width=320,
            height=240,
        )

        assert result is not None
        self.assertEqual(result["width"], 320)
        self.assertEqual(result["height"], 240)
        self.assertAlmostEqual(result["ppx"], 160.0)
        self.assertAlmostEqual(result["ppy"], 120.0)
        self.assertAlmostEqual(result["fx"], 300.0)
        self.assertAlmostEqual(result["fy"], 300.0)
        self.assertEqual(result["source_crop_xywh"], [160.0, 0.0, 960.0, 720.0])
        self.assertFalse(result["horizontal_flip"])

        flipped = _scaled_intrinsics(
            {
                "width": 1280,
                "height": 720,
                "ppx": 640.0,
                "ppy": 360.0,
                "fx": 900.0,
                "fy": 900.0,
            },
            width=320,
            height=240,
            horizontal_flip=True,
        )
        assert flipped is not None
        self.assertAlmostEqual(flipped["ppx"], 479.0 / 3.0)
        self.assertTrue(flipped["horizontal_flip"])


if __name__ == "__main__":
    unittest.main()
