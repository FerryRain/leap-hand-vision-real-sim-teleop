from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
from src.d455 import (
    D455MediaPipeCamera,
    D455Settings,
    median_depth_m,
    palm_pixel_and_depth_m,
)
from teleop import _parse_args

ROOT = Path(__file__).resolve().parents[1]


class _FakeVideoProfile:
    def as_video_stream_profile(self) -> _FakeVideoProfile:
        return self

    def get_intrinsics(self) -> SimpleNamespace:
        return SimpleNamespace(
            width=3,
            height=2,
            ppx=1.25,
            ppy=0.75,
            fx=610.0,
            fy=611.0,
            model="brown_conrady",
            coeffs=[0.1, 0.2, 0.3, 0.4, 0.5],
        )


class _FakeFrame:
    def __init__(self, data: np.ndarray, *, timestamp_ms: float = 1234.5) -> None:
        self._data = data
        self._timestamp_ms = timestamp_ms
        self.profile = _FakeVideoProfile()

    def __bool__(self) -> bool:
        return True

    def get_data(self) -> np.ndarray:
        return self._data

    def get_timestamp(self) -> float:
        return self._timestamp_ms


class _FakeAlignedFrames:
    def __init__(self, color: _FakeFrame, depth: _FakeFrame) -> None:
        self._color = color
        self._depth = depth

    def get_color_frame(self) -> _FakeFrame:
        return self._color

    def get_depth_frame(self) -> _FakeFrame:
        return self._depth


class _FakePipeline:
    def wait_for_frames(self, _timeout_ms: int) -> object:
        return object()


class _FakeAlign:
    def __init__(self, frames: _FakeAlignedFrames) -> None:
        self._frames = frames

    def process(self, _frames: object) -> _FakeAlignedFrames:
        return self._frames


class _FakeHands:
    def process(self, _rgb: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(multi_hand_landmarks=None)


def _fake_camera(color: np.ndarray, depth: np.ndarray) -> D455MediaPipeCamera:
    camera = object.__new__(D455MediaPipeCamera)
    color_frame = _FakeFrame(color)
    depth_frame = _FakeFrame(depth)
    camera.pipeline = _FakePipeline()
    camera.align = _FakeAlign(_FakeAlignedFrames(color_frame, depth_frame))
    camera.settings = SimpleNamespace(
        frame_timeout_ms=100,
        color_width=3,
        color_height=2,
        depth_width=3,
        depth_height=2,
        fps=30,
    )
    camera.flip_horizontal = True
    camera.hands = _FakeHands()
    camera.depth_scale_m = 0.001
    camera.device_name = "Intel RealSense D455"
    camera.serial_number = "fake-serial"
    camera.firmware_version = "fake-firmware"
    camera.latest_palm_position_m = None
    camera.latest_palm_depth_m = None
    camera.depth_valid_frames = 0
    camera.depth_invalid_frames = 0
    camera._latest_color_bgr = None
    camera._latest_depth_units = None
    camera._latest_capture_monotonic_s = None
    camera._latest_camera_timestamp_ms = None
    camera._latest_color_intrinsics = None
    return camera


class D455TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load((ROOT / "config.yml").read_text(encoding="utf-8"))

    def test_config_uses_aligned_rgbd_streams(self) -> None:
        settings = D455Settings.from_config(self.config)

        self.assertEqual(settings.required_model, "D455")
        self.assertEqual(
            (settings.color_width, settings.color_height, settings.fps),
            (1280, 720, 30),
        )
        self.assertEqual(
            (settings.depth_width, settings.depth_height, settings.fps),
            (848, 480, 30),
        )
        self.assertEqual(settings.minimum_palm_depth_samples, 3)

    def test_serial_override_does_not_modify_config(self) -> None:
        settings = D455Settings.from_config(self.config, serial_override="123456789")

        self.assertEqual(settings.serial, "123456789")
        self.assertEqual(self.config["d455"]["serial"], "")

    def test_depth_patch_rejects_holes_and_out_of_range_values(self) -> None:
        depth = np.zeros((7, 7), dtype=np.uint16)
        depth[2:5, 2:5] = 500
        depth[3, 3] = 5000

        measured = median_depth_m(
            depth,
            depth_scale_m=0.001,
            x_px=3,
            y_px=3,
            radius_px=1,
            minimum_depth_m=0.15,
            maximum_depth_m=1.20,
        )

        self.assertAlmostEqual(measured or 0.0, 0.5)

    def test_palm_depth_uses_multiple_landmarks(self) -> None:
        landmarks = np.zeros((21, 3), dtype=np.float64)
        for index, point in zip(
            (0, 5, 9, 13, 17),
            ((0.45, 0.65), (0.40, 0.50), (0.50, 0.48), (0.60, 0.50), (0.65, 0.58)),
            strict=True,
        ):
            landmarks[index, :2] = point
        depth = np.full((100, 200), 600, dtype=np.uint16)
        depth[64:67, 89:92] = 0

        result = palm_pixel_and_depth_m(
            landmarks,
            depth,
            depth_scale_m=0.001,
            radius_px=1,
            minimum_samples=3,
            minimum_depth_m=0.15,
            maximum_depth_m=1.20,
        )

        self.assertIsNotNone(result)
        assert result is not None
        (x_px, y_px), depth_m = result
        self.assertTrue(0 <= x_px < depth.shape[1])
        self.assertTrue(0 <= y_px < depth.shape[0])
        self.assertAlmostEqual(depth_m, 0.6)

    def test_missing_depth_fails_closed(self) -> None:
        landmarks = np.full((21, 3), 0.5, dtype=np.float64)
        depth = np.zeros((20, 20), dtype=np.uint16)

        result = palm_pixel_and_depth_m(
            landmarks,
            depth,
            depth_scale_m=0.001,
            radius_px=1,
            minimum_samples=3,
            minimum_depth_m=0.15,
            maximum_depth_m=1.20,
        )

        self.assertIsNone(result)

    def test_cli_accepts_d455_source_and_serial(self) -> None:
        arguments = ["teleop", "--source", "d455", "--d455-serial", "987654321"]
        with patch.object(sys, "argv", arguments):
            args = _parse_args()

        self.assertEqual(args.source, "d455")
        self.assertEqual(args.d455_serial, "987654321")

    def test_missing_sdk_has_actionable_error(self) -> None:
        with (
            patch("src.d455.importlib.import_module", side_effect=ImportError),
            self.assertRaisesRegex(RuntimeError, "requirements-d455.txt"),
        ):
            D455MediaPipeCamera(self.config, None, disable_preview=True)

    def test_latest_rgbd_requires_a_captured_frame(self) -> None:
        camera = _fake_camera(
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint16),
        )

        with self.assertRaisesRegex(RuntimeError, "call read"):
            camera.latest_rgbd()

    def test_latest_rgbd_is_unannotated_flipped_and_safely_copied(self) -> None:
        color = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        depth = np.arange(6, dtype=np.uint16).reshape(2, 3)
        camera = _fake_camera(color, depth)

        with patch("src.d455.time.monotonic", return_value=42.25):
            landmarks, preview, handedness, confidence = camera.read()

        self.assertIsNone(landmarks)
        self.assertEqual(handedness, "unknown")
        self.assertEqual(confidence, 0.0)
        expected_color = np.ascontiguousarray(color[:, ::-1])
        expected_depth = np.ascontiguousarray(depth[:, ::-1])
        sample = camera.latest_rgbd()
        np.testing.assert_array_equal(sample.color_bgr, expected_color)
        np.testing.assert_array_equal(sample.depth_units, expected_depth)
        self.assertEqual(sample.color_bgr.dtype, np.uint8)
        self.assertEqual(sample.depth_units.dtype, np.uint16)
        self.assertTrue(sample.color_bgr.flags.c_contiguous)
        self.assertTrue(sample.depth_units.flags.c_contiguous)
        self.assertEqual(sample.capture_monotonic_s, 42.25)
        self.assertEqual(sample.camera_timestamp_ms, 1234.5)
        self.assertEqual(sample.depth_scale_m, 0.001)
        self.assertEqual(sample.color_intrinsics["width"], 3)
        self.assertEqual(sample.color_intrinsics["fx"], 610.0)

        # Neither preview annotation nor a caller mutating a returned sample can
        # corrupt the camera-owned observation used by a recorder.
        preview.fill(255)
        sample.color_bgr.fill(0)
        sample.depth_units.fill(0)
        sample.color_intrinsics["coeffs"][0] = 999.0
        second_sample = camera.latest_rgbd()
        np.testing.assert_array_equal(second_sample.color_bgr, expected_color)
        np.testing.assert_array_equal(second_sample.depth_units, expected_depth)
        self.assertEqual(second_sample.color_intrinsics["coeffs"][0], 0.1)

        diagnostics = camera.diagnostics()
        self.assertTrue(diagnostics["latest_rgbd_available"])
        self.assertEqual(diagnostics["latest_capture_monotonic_s"], 42.25)
        self.assertEqual(diagnostics["latest_camera_timestamp_ms"], 1234.5)
        self.assertTrue(diagnostics["flip_horizontal"])
        self.assertFalse(diagnostics["raw_rgbd_saved"])


if __name__ == "__main__":
    unittest.main()
