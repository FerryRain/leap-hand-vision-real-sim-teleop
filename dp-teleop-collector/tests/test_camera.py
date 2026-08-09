"""Hardware-free checks for the deterministic RGB-D vision source."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

# Camera tests exercise the collector's copy semantics, not MediaPipe.  Stub
# the one lightweight landmark helper so this pure-Python suite has no model
# or camera-runtime dependency.
runtime_stub = types.ModuleType("src.runtime")
runtime_stub.mock_landmarks = lambda elapsed_s: np.zeros((21, 3), dtype=np.float64)
with mock.patch.dict(sys.modules, {"src.runtime": runtime_stub}):
    from dp_collector import camera  # noqa: E402


class MockVisionSourceTests(unittest.TestCase):
    def test_returns_expected_rgbd_and_independent_copies(self) -> None:
        monotonic_values = iter((100.0, 100.5, 100.6))
        with mock.patch.object(
            camera.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            source = camera.MockVisionSource(width=96, height=64)
            landmarks, preview, handedness, confidence = source.read()
        first = source.latest_rgbd()

        self.assertEqual(landmarks.shape, (21, 3))
        self.assertEqual(preview.shape, (64, 96, 3))
        self.assertEqual(handedness, "right")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(first.color_bgr.shape, (64, 96, 3))
        self.assertEqual(first.color_bgr.dtype, np.uint8)
        self.assertEqual(first.depth_units.shape, (64, 96))
        self.assertEqual(first.depth_units.dtype, np.uint16)
        self.assertEqual(first.capture_monotonic_s, 100.6)
        self.assertEqual(first.camera_timestamp_ms, 500.0)
        self.assertEqual(first.depth_scale_m, 0.001)
        expected_depth_units = int(round(source.latest_palm_depth_m / 0.001))
        self.assertTrue(np.all(first.depth_units == expected_depth_units))
        self.assertEqual(first.color_intrinsics["width"], 96)
        self.assertEqual(first.color_intrinsics["height"], 64)

        preview[:] = 0
        first.color_bgr[:] = 0
        first.depth_units[:] = 0
        first.color_intrinsics["width"] = -1
        second = source.latest_rgbd()

        self.assertTrue(np.any(second.color_bgr != 0))
        self.assertTrue(np.all(second.depth_units == expected_depth_units))
        self.assertEqual(second.color_intrinsics["width"], 96)
        self.assertFalse(np.shares_memory(first.color_bgr, second.color_bgr))
        self.assertFalse(np.shares_memory(first.depth_units, second.depth_units))

    def test_requires_capture_before_rgbd_read(self) -> None:
        source = camera.MockVisionSource(width=96, height=64)

        try:
            with self.assertRaisesRegex(RuntimeError, "has not captured"):
                source.latest_rgbd()
        finally:
            source.close()

    def test_diagnostics_describe_aligned_raw_rgbd(self) -> None:
        source = camera.MockVisionSource(width=96, height=64)

        diagnostics = source.diagnostics()

        self.assertEqual(diagnostics["source"], "mock")
        self.assertEqual(diagnostics["serial_number"], "mock-d455")
        self.assertEqual(diagnostics["depth_scale_m"], 0.001)
        self.assertEqual(diagnostics["color_stream"], [96, 64, 30])
        self.assertEqual(diagnostics["depth_stream"], [96, 64, 30])
        self.assertIs(diagnostics["depth_aligned_to_color"], True)
        self.assertIs(diagnostics["raw_rgbd_saved"], True)
        self.assertIs(diagnostics["flip_horizontal"], False)
        self.assertEqual(diagnostics["color_intrinsics"]["width"], 96)
        self.assertEqual(diagnostics["color_intrinsics"]["height"], 64)


if __name__ == "__main__":
    unittest.main()
