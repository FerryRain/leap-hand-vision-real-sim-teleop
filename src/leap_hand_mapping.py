"""MediaPipe-to-LEAP mapping based on adjacent human bone angles.

The mapping follows the useful convention in Julianxng's LEAP teleoperation
project: three flexion angles per represented finger and neutral long-finger
abduction. The simulated LEAP thumb's fourth actuator is driven by an explicit
thumb-opposition signal so the hand can form an opposing grasp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

JOINT_NAMES = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)

FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
}


@dataclass(frozen=True)
class MappingSettings:
    smoothing_alpha: float
    maximum_joint_step_rad: float
    maximum_wrist_step_m: float
    joint_deadband_rad: float
    wrist_deadband_m: tuple[float, float, float]
    tracking_hold_s: float
    tracking_open_s: float
    clutch_extension_ratio: float
    wrist_neutral_m: tuple[float, float, float]
    wrist_workspace_min_m: tuple[float, float, float]
    wrist_workspace_max_m: tuple[float, float, float]
    wrist_maximum_offset_m: tuple[float, float, float]
    wrist_signal_scale_m: tuple[float, float, float]
    camera_to_robot_matrix: tuple[tuple[float, float, float], ...]
    joint_open_rad: tuple[float, ...]
    joint_closed_rad: tuple[float, ...]
    long_human_open_deg: tuple[float, float, float]
    long_human_closed_deg: tuple[float, float, float]
    thumb_human_open_deg: tuple[float, float, float]
    thumb_human_closed_deg: tuple[float, float, float]
    thumb_opposition_open_ratio: float
    thumb_opposition_closed_ratio: float
    thumb_opposition_flexion_assist: tuple[float, float, float]

    def validate(self) -> None:
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.maximum_joint_step_rad <= 0.0 or self.maximum_wrist_step_m <= 0.0:
            raise ValueError("command step limits must be positive")
        if self.joint_deadband_rad < 0.0:
            raise ValueError("joint_deadband_rad must be non-negative")
        _finite_vector(self.wrist_deadband_m, 3, "wrist_deadband_m")
        if np.any(np.asarray(self.wrist_deadband_m) < 0.0):
            raise ValueError("wrist_deadband_m must be non-negative")
        if self.tracking_hold_s < 0.0 or self.tracking_open_s <= 0.0:
            raise ValueError("tracking-loss timing is invalid")
        _finite_vector(self.wrist_neutral_m, 3, "wrist_neutral_m")
        _finite_vector(self.wrist_workspace_min_m, 3, "wrist_workspace_min_m")
        _finite_vector(self.wrist_workspace_max_m, 3, "wrist_workspace_max_m")
        _finite_vector(self.wrist_maximum_offset_m, 3, "wrist_maximum_offset_m")
        _finite_vector(self.wrist_signal_scale_m, 3, "wrist_signal_scale_m")
        _finite_vector(self.joint_open_rad, 16, "joint_open_rad")
        _finite_vector(self.joint_closed_rad, 16, "joint_closed_rad")
        matrix = np.asarray(self.camera_to_robot_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera_to_robot_matrix must be a finite 3x3 matrix")
        if abs(_determinant_3x3(matrix)) < 1.0e-6:
            raise ValueError("camera_to_robot_matrix must be invertible")
        if np.any(
            np.asarray(self.wrist_workspace_min_m)
            >= np.asarray(self.wrist_workspace_max_m)
        ):
            raise ValueError("wrist workspace minimum must be below maximum")
        if np.any(np.asarray(self.wrist_maximum_offset_m) <= 0.0):
            raise ValueError("wrist maximum offsets must be positive")
        for open_angles, closed_angles, label in (
            (
                self.long_human_open_deg,
                self.long_human_closed_deg,
                "long finger",
            ),
            (
                self.thumb_human_open_deg,
                self.thumb_human_closed_deg,
                "thumb",
            ),
        ):
            _finite_vector(open_angles, 3, f"{label} open angles")
            _finite_vector(closed_angles, 3, f"{label} closed angles")
            if np.any(np.asarray(open_angles) >= np.asarray(closed_angles)):
                raise ValueError(f"{label} open angles must be below closed angles")
        if self.thumb_opposition_open_ratio <= self.thumb_opposition_closed_ratio:
            raise ValueError("thumb opposition ratios are reversed")
        _finite_vector(
            self.thumb_opposition_flexion_assist,
            3,
            "thumb opposition flexion assist",
        )
        if np.any(np.asarray(self.thumb_opposition_flexion_assist) < 0.0) or np.any(
            np.asarray(self.thumb_opposition_flexion_assist) > 1.0
        ):
            raise ValueError("thumb opposition flexion assist must be in [0, 1]")


@dataclass(frozen=True)
class LeapCommand:
    wrist_position_m: tuple[float, float, float]
    joint_target_rad: tuple[float, ...]
    mode: str
    wrist_clutched: bool
    flexion_deg: tuple[float, ...] | None


class LandmarkEmaFilter:
    """Low-pass all MediaPipe landmarks before calculating joint angles."""

    def __init__(self, alpha: float) -> None:
        self.alpha = 0.0
        self.set_alpha(alpha)
        self._value: np.ndarray | None = None

    def set_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("tuning.landmark_smoothing_alpha must be in (0, 1]")
        self.alpha = alpha

    def reset(self) -> None:
        self._value = None

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        value = np.asarray(landmarks, dtype=np.float64)
        if value.shape != (21, 3) or not np.isfinite(value).all():
            raise ValueError("landmarks must be a finite 21x3 array")
        if self._value is None:
            self._value = value.copy()
        else:
            self._value += self.alpha * (value - self._value)
        return self._value.copy()


class LeapLandmarkMapper:
    def __init__(
        self,
        settings: MappingSettings,
        *,
        joint_lower_rad: np.ndarray,
        joint_upper_rad: np.ndarray,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.open_joints = np.asarray(settings.joint_open_rad, dtype=np.float64)
        self.closed_joints = np.asarray(settings.joint_closed_rad, dtype=np.float64)
        self.joint_lower = np.asarray(joint_lower_rad, dtype=np.float64)
        self.joint_upper = np.asarray(joint_upper_rad, dtype=np.float64)
        if self.joint_lower.shape != (16,) or self.joint_upper.shape != (16,):
            raise ValueError("LEAP joint limits must contain 16 values")
        if np.any(self.joint_lower > self.joint_upper):
            raise ValueError("LEAP joint limits are reversed")

        self.last_wrist = np.asarray(settings.wrist_neutral_m, dtype=np.float64)
        self.last_joints = np.clip(
            self.open_joints.copy(), self.joint_lower, self.joint_upper
        )
        self._human_anchor: np.ndarray | None = None
        self._robot_anchor = self.last_wrist.copy()
        self._was_clutched = False
        self._manual_clutch = False
        self._recenter_requested = False
        self._last_valid_time: float | None = None
        self._loss_start_joints: np.ndarray | None = None

    def recenter(self) -> None:
        """Re-anchor the next valid human pose without moving the robot wrist."""

        self._recenter_requested = True

    def toggle_manual_clutch(self) -> bool:
        self._manual_clutch = not self._manual_clutch
        if not self._manual_clutch:
            self._recenter_requested = True
        return self._manual_clutch

    def update(self, landmarks: np.ndarray, *, now_s: float) -> LeapCommand:
        points = validate_landmarks(landmarks)
        flexion = landmark_flexion_angles(points)
        desired_joints = self._map_fingers(points, flexion)
        self.last_joints = self._filter_joints(desired_joints)

        signal = wrist_motion_signal(points)
        automatic_clutch = (
            self.settings.clutch_extension_ratio > 0.0
            and extension_ratio(points) <= self.settings.clutch_extension_ratio
        )
        clutched = self._manual_clutch or automatic_clutch
        if clutched:
            self._was_clutched = True
        elif (
            self._was_clutched or self._human_anchor is None or self._recenter_requested
        ):
            self._human_anchor = signal
            self._robot_anchor = self.last_wrist.copy()
            self._was_clutched = False
            self._recenter_requested = False
        else:
            signal_delta = (signal - self._human_anchor) * np.asarray(
                self.settings.wrist_signal_scale_m, dtype=np.float64
            )
            desired_wrist = self._robot_anchor + (
                np.asarray(self.settings.camera_to_robot_matrix, dtype=np.float64)
                @ signal_delta
            )
            self.last_wrist = self._filter_wrist(desired_wrist)

        self._last_valid_time = float(now_s)
        self._loss_start_joints = None
        flat_flexion = tuple(
            float(value)
            for finger in ("index", "middle", "ring", "thumb")
            for value in flexion[finger]
        )
        return self._command(
            "wrist_clutch" if clutched else "tracking",
            clutched,
            flat_flexion,
        )

    def tracking_lost(self, *, now_s: float) -> LeapCommand:
        if self._loss_start_joints is None:
            self._loss_start_joints = self.last_joints.copy()
        lost_for = (
            float("inf")
            if self._last_valid_time is None
            else max(0.0, float(now_s) - self._last_valid_time)
        )
        if lost_for <= self.settings.tracking_hold_s:
            mode = "tracking_hold"
        else:
            opening_for = lost_for - self.settings.tracking_hold_s
            fraction = min(opening_for / self.settings.tracking_open_s, 1.0)
            desired = (
                1.0 - fraction
            ) * self._loss_start_joints + fraction * self.open_joints
            self.last_joints = self._rate_limit_joints(desired)
            mode = "tracking_open" if fraction < 1.0 else "tracking_safe_open"
        self._human_anchor = None
        self._was_clutched = False
        return self._command(mode, False, None)

    def _map_fingers(
        self,
        points: np.ndarray,
        flexion: dict[str, tuple[float, float, float]],
    ) -> np.ndarray:
        target = self.open_joints.copy()
        for finger_index, finger in enumerate(("index", "middle", "ring")):
            fractions = normalize_angles(
                flexion[finger],
                self.settings.long_human_open_deg,
                self.settings.long_human_closed_deg,
            )
            base = finger_index * 4
            target[base] = interpolate_joint(
                self.open_joints[base], self.closed_joints[base], fractions[0]
            )
            # The reference LEAP mapping holds abduction/rotation neutral.
            target[base + 1] = self.open_joints[base + 1]
            target[base + 2] = interpolate_joint(
                self.open_joints[base + 2],
                self.closed_joints[base + 2],
                fractions[1],
            )
            target[base + 3] = interpolate_joint(
                self.open_joints[base + 3],
                self.closed_joints[base + 3],
                fractions[2],
            )

        thumb = normalize_angles(
            flexion["thumb"],
            self.settings.thumb_human_open_deg,
            self.settings.thumb_human_closed_deg,
        )
        opposition_ratio = thumb_opposition_ratio(points)
        opposition = float(
            np.clip(
                (self.settings.thumb_opposition_open_ratio - opposition_ratio)
                / (
                    self.settings.thumb_opposition_open_ratio
                    - self.settings.thumb_opposition_closed_ratio
                ),
                0.0,
                1.0,
            )
        )
        assist = np.asarray(
            self.settings.thumb_opposition_flexion_assist,
            dtype=np.float64,
        )
        # A natural pinch can strongly oppose the thumb while its projected
        # bone angles stay fairly straight. Let opposition also curl CMC, MCP,
        # and IP so the simulated thumb can actually meet the long fingers.
        thumb = np.maximum(thumb, opposition * assist)
        for index, fraction in zip(
            (12, 13, 14, 15),
            (thumb[0], opposition, thumb[1], thumb[2]),
            strict=True,
        ):
            target[index] = interpolate_joint(
                self.open_joints[index], self.closed_joints[index], fraction
            )
        return np.clip(target, self.joint_lower, self.joint_upper)

    def _filter_joints(self, desired: np.ndarray) -> np.ndarray:
        error = desired - self.last_joints
        active = np.abs(error) >= self.settings.joint_deadband_rad
        smoothed = self.last_joints + self.settings.smoothing_alpha * np.where(
            active, error, 0.0
        )
        return self._rate_limit_joints(smoothed)

    def _rate_limit_joints(self, desired: np.ndarray) -> np.ndarray:
        delta = np.clip(
            desired - self.last_joints,
            -self.settings.maximum_joint_step_rad,
            self.settings.maximum_joint_step_rad,
        )
        return np.clip(self.last_joints + delta, self.joint_lower, self.joint_upper)

    def _filter_wrist(self, desired: np.ndarray) -> np.ndarray:
        neutral = np.asarray(self.settings.wrist_neutral_m, dtype=np.float64)
        maximum_offset = np.asarray(
            self.settings.wrist_maximum_offset_m, dtype=np.float64
        )
        lower = np.maximum(
            np.asarray(self.settings.wrist_workspace_min_m, dtype=np.float64),
            neutral - maximum_offset,
        )
        upper = np.minimum(
            np.asarray(self.settings.wrist_workspace_max_m, dtype=np.float64),
            neutral + maximum_offset,
        )
        bounded = np.clip(desired, lower, upper)
        error = bounded - self.last_wrist
        deadband = np.asarray(self.settings.wrist_deadband_m, dtype=np.float64)
        smoothed = self.last_wrist + self.settings.smoothing_alpha * np.where(
            np.abs(error) >= deadband,
            error,
            0.0,
        )
        delta = smoothed - self.last_wrist
        norm = float(np.linalg.norm(delta))
        if norm > self.settings.maximum_wrist_step_m:
            delta *= self.settings.maximum_wrist_step_m / norm
        return np.clip(self.last_wrist + delta, lower, upper)

    def _command(
        self,
        mode: str,
        clutched: bool,
        flexion: tuple[float, ...] | None,
    ) -> LeapCommand:
        return LeapCommand(
            wrist_position_m=tuple(float(value) for value in self.last_wrist),
            joint_target_rad=tuple(float(value) for value in self.last_joints),
            mode=mode,
            wrist_clutched=clutched,
            flexion_deg=flexion,
        )


def validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError("hand landmarks must have shape (21, 3)")
    if not np.isfinite(points).all():
        raise ValueError("hand landmarks must be finite")
    return points


def landmark_flexion_angles(
    landmarks: np.ndarray,
) -> dict[str, tuple[float, float, float]]:
    points = validate_landmarks(landmarks)
    result: dict[str, tuple[float, float, float]] = {}
    for finger, chain in FINGER_CHAINS.items():
        segments = [
            points[chain[index + 1]] - points[chain[index]] for index in range(4)
        ]
        result[finger] = tuple(
            angle_between_deg(segments[index], segments[index + 1])
            for index in range(3)
        )
    return result


def angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1.0e-9 or second_norm < 1.0e-9:
        raise ValueError("cannot calculate an angle from a zero bone")
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def extension_ratio(landmarks: np.ndarray) -> float:
    points = validate_landmarks(landmarks)
    palm_size = float(np.linalg.norm(points[9, :2] - points[0, :2]))
    if palm_size < 1.0e-6:
        raise ValueError("detected palm is too small")
    return float(np.linalg.norm(points[12, :2] - points[0, :2]) / palm_size)


def thumb_opposition_ratio(landmarks: np.ndarray) -> float:
    points = validate_landmarks(landmarks)
    palm_width = float(np.linalg.norm(points[17] - points[5]))
    if palm_width < 1.0e-6:
        raise ValueError("detected palm width is too small")
    return float(np.linalg.norm(points[4] - points[5]) / palm_width)


def wrist_motion_signal(landmarks: np.ndarray) -> np.ndarray:
    points = validate_landmarks(landmarks)
    palm_size = float(np.linalg.norm(points[9, :2] - points[0, :2]))
    if palm_size < 1.0e-6:
        raise ValueError("detected palm is too small")
    return np.asarray(
        (points[0, 0], points[0, 1], math.log(palm_size)), dtype=np.float64
    )


def normalize_angles(
    values: tuple[float, float, float],
    open_values: tuple[float, float, float],
    closed_values: tuple[float, float, float],
) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    open_array = np.asarray(open_values, dtype=np.float64)
    closed_array = np.asarray(closed_values, dtype=np.float64)
    return np.clip((values_array - open_array) / (closed_array - open_array), 0.0, 1.0)


def interpolate_joint(open_value: float, closed_value: float, fraction: float) -> float:
    return float(open_value + float(fraction) * (closed_value - open_value))


def _finite_vector(value: tuple[float, ...], expected: int, label: str) -> None:
    if len(value) != expected or not all(math.isfinite(item) for item in value):
        raise ValueError(f"{label} must contain {expected} finite values")


def _determinant_3x3(matrix: np.ndarray) -> float:
    # Avoid a LAPACK startup cost for this tiny validation-only calculation.
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return float(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
