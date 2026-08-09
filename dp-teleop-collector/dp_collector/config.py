"""Validated settings for the demonstration collector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CollectorSettings:
    sample_hz: float
    image_width: int
    image_height: int
    jpeg_quality: int
    minimum_episode_steps: int
    maximum_episode_s: float
    maximum_invalid_steps: int
    maximum_camera_age_s: float
    maximum_leap_state_age_s: float
    maximum_franka_state_age_s: float
    max_pending_frames: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CollectorSettings":
        settings = cls(
            sample_hz=float(raw["sample_hz"]),
            image_width=int(raw["image_width"]),
            image_height=int(raw["image_height"]),
            jpeg_quality=int(raw["jpeg_quality"]),
            minimum_episode_steps=int(raw["minimum_episode_steps"]),
            maximum_episode_s=float(raw["maximum_episode_s"]),
            maximum_invalid_steps=int(raw["maximum_invalid_steps"]),
            maximum_camera_age_s=float(raw["maximum_camera_age_s"]),
            maximum_leap_state_age_s=float(raw["maximum_leap_state_age_s"]),
            maximum_franka_state_age_s=float(raw["maximum_franka_state_age_s"]),
            max_pending_frames=int(raw["max_pending_frames"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 1.0 <= self.sample_hz <= 30.0:
            raise ValueError("collector.sample_hz must be in [1, 30]")
        if self.image_width < 64 or self.image_height < 64:
            raise ValueError("collector image dimensions must be at least 64 pixels")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("collector.jpeg_quality must be in [1, 100]")
        if self.minimum_episode_steps < 2:
            raise ValueError("collector.minimum_episode_steps must be at least 2")
        if not 1.0 <= self.maximum_episode_s <= 600.0:
            raise ValueError("collector.maximum_episode_s must be in [1, 600]")
        if self.maximum_invalid_steps < 0:
            raise ValueError("collector.maximum_invalid_steps must be non-negative")
        for name, value in (
            ("maximum_camera_age_s", self.maximum_camera_age_s),
            ("maximum_leap_state_age_s", self.maximum_leap_state_age_s),
            ("maximum_franka_state_age_s", self.maximum_franka_state_age_s),
        ):
            if not 0.02 <= value <= 1.0:
                raise ValueError(f"collector.{name} must be in [0.02, 1.0]")
        if not 1 <= self.max_pending_frames <= 16:
            raise ValueError("collector.max_pending_frames must be in [1, 16]")


@dataclass(frozen=True)
class FrankaTeleopSettings:
    mapping_confirmed: bool
    camera_to_global_matrix: tuple[tuple[float, float, float], ...]
    linear_gain_per_s: float
    deadband_m: float
    maximum_hand_offset_m: float
    maximum_linear_speed_m_s: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FrankaTeleopSettings":
        matrix_raw = raw["camera_to_global_matrix"]
        if not isinstance(matrix_raw, (list, tuple)) or len(matrix_raw) != 3:
            raise ValueError("franka_teleop.camera_to_global_matrix must be 3x3")
        matrix = tuple(_number_tuple(row, 3) for row in matrix_raw)
        settings = cls(
            mapping_confirmed=bool(raw["mapping_confirmed"]),
            camera_to_global_matrix=matrix,
            linear_gain_per_s=float(raw["linear_gain_per_s"]),
            deadband_m=float(raw["deadband_m"]),
            maximum_hand_offset_m=float(raw["maximum_hand_offset_m"]),
            maximum_linear_speed_m_s=float(raw["maximum_linear_speed_m_s"]),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        matrix = np.asarray(self.camera_to_global_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("franka camera-to-global matrix must be finite 3x3")
        if not matrix3_is_orthonormal(matrix, atol=1.0e-4):
            raise ValueError("franka camera-to-global matrix must be orthonormal")
        if not math.isclose(
            abs(matrix3_determinant(matrix)),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-4,
        ):
            raise ValueError("franka camera-to-global matrix determinant must be +/-1")
        if not 0.0 < self.linear_gain_per_s <= 5.0:
            raise ValueError("franka linear_gain_per_s must be in (0, 5]")
        if not 0.0 <= self.deadband_m <= 0.05:
            raise ValueError("franka deadband_m must be in [0, 0.05]")
        if not 0.02 <= self.maximum_hand_offset_m <= 0.50:
            raise ValueError("franka maximum_hand_offset_m must be in [0.02, 0.50]")
        if not 0.001 <= self.maximum_linear_speed_m_s <= 0.10:
            raise ValueError("franka maximum_linear_speed_m_s must be in [0.001, 0.10]")


@dataclass(frozen=True)
class AppSettings:
    collector: CollectorSettings
    franka_teleop: FrankaTeleopSettings


def load_app_settings(path: Path) -> AppSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"collector config must be a mapping: {path}")
    collector = raw.get("collector")
    franka = raw.get("franka_teleop")
    if not isinstance(collector, dict) or not isinstance(franka, dict):
        raise ValueError("collector config needs collector and franka_teleop sections")
    return AppSettings(
        collector=CollectorSettings.from_dict(collector),
        franka_teleop=FrankaTeleopSettings.from_dict(franka),
    )


def _number_tuple(value: Any, expected: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(f"expected {expected} numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("configuration values must be finite")
    return result


def matrix3_determinant(value: Any) -> float:
    """Return a 3x3 determinant without invoking platform LAPACK libraries."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3")
    a, b, c = (float(item) for item in matrix[0])
    d, e, f = (float(item) for item in matrix[1])
    g, h, i = (float(item) for item in matrix[2])
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def matrix3_is_orthonormal(value: Any, *, atol: float) -> bool:
    """Check three orthonormal rows using scalar arithmetic only."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    for row_index in range(3):
        for other_index in range(3):
            dot = math.fsum(
                float(matrix[row_index, column]) * float(matrix[other_index, column])
                for column in range(3)
            )
            expected = 1.0 if row_index == other_index else 0.0
            if not math.isclose(dot, expected, rel_tol=0.0, abs_tol=atol):
                return False
    return True
