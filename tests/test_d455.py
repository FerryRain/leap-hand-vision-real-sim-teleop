from __future__ import annotations

import sys
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
