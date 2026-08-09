"""Camera, configuration, and mapper helpers for finger-only teleoperation."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import yaml

from .leap_hand_hardware import MAPPING_LOWER_RAD, MAPPING_UPPER_RAD
from .leap_hand_mapping import LandmarkEmaFilter, LeapLandmarkMapper, MappingSettings


class MediaPipeCamera:
    """Read one human hand and optionally show the annotated camera preview."""

    def __init__(
        self,
        config: dict[str, Any],
        camera_override: int | None,
        *,
        disable_preview: bool,
    ) -> None:
        camera_cfg = section(config, "camera")
        camera_index = (
            int(camera_cfg["index"])
            if camera_override is None
            else int(camera_override)
        )
        self.flip_horizontal = bool(camera_cfg["flip_horizontal"])
        self.preview = bool(camera_cfg["preview"]) and not disable_preview
        self.required_handedness = str(camera_cfg["required_handedness"]).lower()
        if self.required_handedness not in {"any", "left", "right"}:
            raise ValueError("camera.required_handedness must be any, left, or right")

        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open camera index {camera_index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg["width"]))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg["height"]))
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=float(camera_cfg["detection_confidence"]),
            min_tracking_confidence=float(camera_cfg["tracking_confidence"]),
        )
        self.drawer = mp.solutions.drawing_utils

    def read(self) -> tuple[np.ndarray | None, np.ndarray, str, float]:
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("camera returned no frame")
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None, frame, "unknown", 0.0

        hand_landmarks = result.multi_hand_landmarks[0]
        handedness = "unknown"
        confidence = 1.0
        if result.multi_handedness:
            classification = result.multi_handedness[0].classification[0]
            handedness = classification.label.lower()
            confidence = float(classification.score)
        if self.required_handedness not in {"any", handedness}:
            return None, frame, handedness, confidence
        self.drawer.draw_landmarks(
            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
        )
        points = np.asarray(
            [(point.x, point.y, point.z) for point in hand_landmarks.landmark],
            dtype=np.float64,
        )
        return points, frame, handedness, confidence

    def show(
        self,
        frame: np.ndarray,
        status: str,
        color: tuple[int, int, int],
        details: tuple[str, ...],
    ) -> int:
        if not self.preview:
            return -1
        cv2.putText(
            frame,
            status,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Q/E stop | SPACE pause/hold | L reload config.yml",
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        for line_index, line in enumerate(details):
            cv2.putText(
                frame,
                line,
                (20, 110 + line_index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow("LEAP Hand: camera + real + MuJoCo", frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        self.hands.close()
        self.capture.release()
        cv2.destroyAllWindows()


def new_mapper(
    config: dict[str, Any],
    *,
    seed_joints: np.ndarray | None = None,
) -> tuple[MappingSettings, LeapLandmarkMapper, LandmarkEmaFilter]:
    settings = mapping_settings(config)
    mapper = LeapLandmarkMapper(
        settings,
        joint_lower_rad=MAPPING_LOWER_RAD,
        joint_upper_rad=MAPPING_UPPER_RAD,
    )
    if seed_joints is not None:
        mapper.last_joints = np.clip(
            np.asarray(seed_joints, dtype=np.float64),
            MAPPING_LOWER_RAD,
            MAPPING_UPPER_RAD,
        )
    return settings, mapper, LandmarkEmaFilter(landmark_alpha(config))


def mapping_settings(config: dict[str, Any]) -> MappingSettings:
    control = section(config, "control")
    mapping = section(config, "mapping")
    settings = MappingSettings(
        smoothing_alpha=float(control["smoothing_alpha"]),
        maximum_joint_step_rad=float(control["maximum_joint_step_rad"]),
        # The shared mapper datatype also contains wrist fields. They are inert:
        # teleop.py never reads or sends the mapper's wrist result.
        maximum_wrist_step_m=1.0,
        joint_deadband_rad=float(control.get("joint_deadband_rad", 0.0)),
        wrist_deadband_m=(0.0, 0.0, 0.0),
        tracking_hold_s=float(control["tracking_hold_s"]),
        tracking_open_s=float(control["tracking_open_s"]),
        clutch_extension_ratio=0.0,
        wrist_neutral_m=(0.0, 0.0, 0.0),
        wrist_workspace_min_m=(-1.0, -1.0, -1.0),
        wrist_workspace_max_m=(1.0, 1.0, 1.0),
        wrist_maximum_offset_m=(1.0, 1.0, 1.0),
        wrist_signal_scale_m=(1.0, 1.0, 1.0),
        camera_to_robot_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        joint_open_rad=number_tuple(mapping["joint_open_rad"], 16),
        joint_closed_rad=number_tuple(mapping["joint_closed_rad"], 16),
        long_human_open_deg=number_tuple(mapping["long_human_open_deg"], 3),
        long_human_closed_deg=number_tuple(mapping["long_human_closed_deg"], 3),
        thumb_human_open_deg=number_tuple(mapping["thumb_human_open_deg"], 3),
        thumb_human_closed_deg=number_tuple(mapping["thumb_human_closed_deg"], 3),
        thumb_opposition_open_ratio=float(mapping["thumb_opposition_open_ratio"]),
        thumb_opposition_closed_ratio=float(mapping["thumb_opposition_closed_ratio"]),
        thumb_opposition_flexion_assist=number_tuple(
            mapping["thumb_opposition_flexion_assist"], 3
        ),
    )
    tuning = config.get("tuning")
    if tuning is not None:
        if not isinstance(tuning, dict):
            raise ValueError("config section 'tuning' must be a mapping")
        gains = np.asarray(number_tuple(tuning["finger_gain"], 4), dtype=np.float64)
        if np.any(gains <= 0.0) or np.any(gains > 2.0):
            raise ValueError("tuning.finger_gain must be in (0, 2]")
        opened = np.asarray(settings.joint_open_rad, dtype=np.float64)
        closed = np.asarray(settings.joint_closed_rad, dtype=np.float64)
        for finger_index, gain in enumerate(gains):
            joint_slice = slice(finger_index * 4, finger_index * 4 + 4)
            closed[joint_slice] = opened[joint_slice] + gain * (
                closed[joint_slice] - opened[joint_slice]
            )
        settings = replace(
            settings,
            joint_closed_rad=tuple(float(item) for item in closed),
        )
    settings.validate()
    return settings


def landmark_alpha(config: dict[str, Any]) -> float:
    tuning = config.get("tuning")
    if tuning is None:
        return 1.0
    if not isinstance(tuning, dict):
        raise ValueError("config section 'tuning' must be a mapping")
    return float(tuning["landmark_smoothing_alpha"])


def mock_landmarks(elapsed_s: float) -> np.ndarray:
    """Generate a deterministic open/close hand for hardware-free tests."""

    closure = 0.5 - 0.5 * math.cos(min(max(elapsed_s, 0.0), 2.0) * math.pi)
    wrist = np.asarray((0.50, 0.72, 0.0), dtype=np.float64)
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = wrist
    roots = {
        "thumb": (1, np.asarray((0.43, 0.65, -0.01))),
        "index": (5, np.asarray((0.44, 0.56, -0.01))),
        "middle": (9, np.asarray((0.50, 0.54, -0.01))),
        "ring": (13, np.asarray((0.56, 0.56, -0.01))),
        "pinky": (17, np.asarray((0.62, 0.60, -0.01))),
    }
    for finger, (root_index, root_point) in roots.items():
        points[root_index] = root_point
        base_vector = root_point - wrist
        base_heading = math.atan2(base_vector[1], base_vector[0])
        bend = math.radians((45.0 if finger == "thumb" else 65.0) * closure)
        current = root_point.copy()
        for segment_index, length in enumerate((0.055, 0.045, 0.035), start=1):
            heading = base_heading + bend * segment_index
            current = current + (
                length * math.cos(heading),
                length * math.sin(heading),
                0.0,
            )
            points[root_index + segment_index] = current
    return points


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing config section {name!r}")
    return value


def number_tuple(value: Any, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"expected {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("configuration values must be finite")
    return result
