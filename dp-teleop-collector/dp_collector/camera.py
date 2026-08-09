"""Vision sources used by the collector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from src.runtime import mock_landmarks


@dataclass(frozen=True)
class MockRGBDFrame:
    color_bgr: np.ndarray
    depth_units: np.ndarray
    capture_monotonic_s: float
    camera_timestamp_ms: float | None
    color_intrinsics: dict[str, Any]
    depth_scale_m: float


class MockVisionSource:
    """Deterministic RGB-D and hand landmarks for hardware-free validation."""

    def __init__(self, *, width: int = 640, height: int = 480) -> None:
        self.width = int(width)
        self.height = int(height)
        self.start_s = time.monotonic()
        self.frame_index = 0
        self.latest_palm_position_m: tuple[float, float, float] | None = None
        self.latest_palm_depth_m: float | None = None
        self.latest_hand_display_uv: np.ndarray | None = None
        self._latest: MockRGBDFrame | None = None

    def read(self) -> tuple[np.ndarray, np.ndarray, str, float]:
        elapsed_s = time.monotonic() - self.start_s
        landmarks = mock_landmarks(elapsed_s)
        palm = (
            0.015 * np.sin(0.8 * elapsed_s),
            -0.010 * np.sin(0.5 * elapsed_s),
            0.55 + 0.010 * np.sin(0.6 * elapsed_s),
        )
        self.latest_palm_position_m = tuple(float(item) for item in palm)
        self.latest_palm_depth_m = float(palm[2])
        self.latest_hand_display_uv = np.asarray(
            landmarks[:, :2], dtype=np.float64
        ).copy()

        x_gradient = np.linspace(0, 255, self.width, dtype=np.uint8)
        y_gradient = np.linspace(0, 255, self.height, dtype=np.uint8)[:, None]
        color = np.empty((self.height, self.width, 3), dtype=np.uint8)
        color[..., 0] = x_gradient[None, :]
        color[..., 1] = y_gradient
        color[..., 2] = np.uint8((self.frame_index * 3) % 256)
        depth = np.full(
            (self.height, self.width),
            int(round(palm[2] / 0.001)),
            dtype=np.uint16,
        )
        captured_s = time.monotonic()
        self._latest = MockRGBDFrame(
            color_bgr=color.copy(),
            depth_units=depth.copy(),
            capture_monotonic_s=captured_s,
            camera_timestamp_ms=1000.0 * elapsed_s,
            color_intrinsics={
                "width": self.width,
                "height": self.height,
                "ppx": 0.5 * self.width,
                "ppy": 0.5 * self.height,
                "fx": 600.0,
                "fy": 600.0,
                "model": "mock",
                "coeffs": [0.0] * 5,
            },
            depth_scale_m=0.001,
        )
        preview = color.copy()
        wrist_px = (
            int(np.clip(landmarks[0, 0], 0.0, 1.0) * (self.width - 1)),
            int(np.clip(landmarks[0, 1], 0.0, 1.0) * (self.height - 1)),
        )
        cv2.circle(preview, wrist_px, 8, (50, 230, 80), -1)
        self.frame_index += 1
        return landmarks, preview, "right", 1.0

    def latest_rgbd(self) -> MockRGBDFrame:
        if self._latest is None:
            raise RuntimeError("mock source has not captured a frame yet")
        sample = self._latest
        return MockRGBDFrame(
            color_bgr=sample.color_bgr.copy(),
            depth_units=sample.depth_units.copy(),
            capture_monotonic_s=sample.capture_monotonic_s,
            camera_timestamp_ms=sample.camera_timestamp_ms,
            color_intrinsics=dict(sample.color_intrinsics),
            depth_scale_m=sample.depth_scale_m,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "source": "mock",
            "serial_number": "mock-d455",
            "depth_scale_m": 0.001,
            "color_stream": [self.width, self.height, 30],
            "depth_stream": [self.width, self.height, 30],
            "depth_aligned_to_color": True,
            "flip_horizontal": False,
            "color_intrinsics": {
                "width": self.width,
                "height": self.height,
                "ppx": 0.5 * self.width,
                "ppy": 0.5 * self.height,
                "fx": 600.0,
                "fy": 600.0,
                "model": "mock",
                "coeffs": [0.0] * 5,
            },
            "raw_rgbd_saved": True,
        }

    def close(self) -> None:
        return None
