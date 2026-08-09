"""FR3 state synchronization and camera-palm velocity mapping."""

from __future__ import annotations

import asyncio
import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import (
    FrankaTeleopSettings,
    matrix3_determinant,
    matrix3_is_orthonormal,
)
from .rotation import matrix_to_rotation_6d

FRANKA_STATE_NAMES = tuple(
    [f"fr3_joint_position_{index}_rad" for index in range(7)]
    + [f"fr3_joint_velocity_{index}_rad_s" for index in range(7)]
    + ["fr3_eef_x_m", "fr3_eef_y_m", "fr3_eef_z_m"]
    + [f"fr3_eef_rotation_6d_{index}" for index in range(6)]
    + ["fr3_eef_vx_m_s", "fr3_eef_vy_m_s", "fr3_eef_vz_m_s"]
    + ["fr3_eef_wx_rad_s", "fr3_eef_wy_rad_s", "fr3_eef_wz_rad_s"]
)


@dataclass(frozen=True)
class LatestFrankaState:
    message: dict[str, Any]
    received_monotonic_s: float


@dataclass(frozen=True)
class ParsedFrankaState:
    low_dim: np.ndarray
    raw_robot: dict[str, Any]
    raw_bridge: dict[str, Any]
    sequence: int
    robot_timestamp_s: float | None
    received_monotonic_s: float
    age_s: float
    watchdog_stop_count: int
    workspace_guard_stop_count: int
    valid: bool
    invalid_reasons: tuple[str, ...]


class FrankaStateCache:
    """Keep the newest network state and its local receive timestamp."""

    def __init__(self) -> None:
        self.latest: LatestFrankaState | None = None
        self.received_count = 0
        self.sequence_gap_count = 0
        self._last_sequence: int | None = None

    async def consume(self, client: Any) -> None:
        while True:
            message = await client.next_state()
            received = time.monotonic()
            sequence = int(message.get("sequence", -1))
            if self._last_sequence is not None and sequence > self._last_sequence + 1:
                self.sequence_gap_count += sequence - self._last_sequence - 1
            self._last_sequence = sequence
            self.latest = LatestFrankaState(
                message=copy.deepcopy(message),
                received_monotonic_s=received,
            )
            self.received_count += 1
            await asyncio.sleep(0)

    def parse_latest(
        self,
        *,
        now_s: float,
        maximum_age_s: float,
        require_control_lease: bool = False,
    ) -> ParsedFrankaState | None:
        if self.latest is None:
            return None
        return parse_franka_state(
            self.latest,
            now_s=now_s,
            maximum_age_s=maximum_age_s,
            require_control_lease=require_control_lease,
        )


class PalmVelocityMapper:
    """Map a deadman-relative D455 palm displacement to bounded global velocity."""

    def __init__(self, settings: FrankaTeleopSettings) -> None:
        self.settings = settings
        self._matrix = np.asarray(settings.camera_to_global_matrix, dtype=np.float64)
        self._anchor_m: np.ndarray | None = None
        self.deadman_was_down = False

    def update(
        self,
        palm_position_m: tuple[float, float, float] | None,
        *,
        deadman_down: bool,
    ) -> np.ndarray:
        if not deadman_down or palm_position_m is None:
            self.reset()
            return np.zeros(3, dtype=np.float64)
        palm = _vector(palm_position_m, 3, "D455 palm position")
        if self._anchor_m is None or not self.deadman_was_down:
            self._anchor_m = palm.copy()
            self.deadman_was_down = True
            return np.zeros(3, dtype=np.float64)
        self.deadman_was_down = True
        offset = palm - self._anchor_m
        if _vector_norm(offset) > self.settings.maximum_hand_offset_m:
            self.reset()
            return np.zeros(3, dtype=np.float64)
        active = np.sign(offset) * np.maximum(
            np.abs(offset) - self.settings.deadband_m,
            0.0,
        )
        velocity = self._matrix @ (self.settings.linear_gain_per_s * active)
        speed = _vector_norm(velocity)
        if speed > self.settings.maximum_linear_speed_m_s:
            velocity *= self.settings.maximum_linear_speed_m_s / speed
        return velocity

    def reset(self) -> None:
        self._anchor_m = None
        self.deadman_was_down = False


def parse_franka_state(
    latest: LatestFrankaState,
    *,
    now_s: float,
    maximum_age_s: float,
    require_control_lease: bool = False,
) -> ParsedFrankaState:
    message = latest.message
    robot = message.get("robot")
    if not isinstance(robot, dict):
        raise ValueError("FR3 state is missing robot")
    joints = robot.get("joints")
    end_effector = robot.get("end_effector")
    if not isinstance(joints, dict) or not isinstance(end_effector, dict):
        raise ValueError("FR3 state is missing joints or end_effector")
    q = _vector(joints.get("position"), 7, "FR3 joint position")
    dq = _vector(joints.get("velocity"), 7, "FR3 joint velocity")
    position = _vector(end_effector.get("position"), 3, "FR3 EEF position")
    rotation = np.asarray(end_effector.get("rotation_matrix"), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("FR3 EEF rotation_matrix must be finite 3x3")
    if not matrix3_is_orthonormal(rotation, atol=2.0e-3):
        raise ValueError("FR3 EEF rotation_matrix is not orthonormal")
    if not math.isclose(
        matrix3_determinant(rotation),
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-3,
    ):
        raise ValueError("FR3 EEF rotation_matrix determinant is not +1")
    linear = _vector(end_effector.get("linear_velocity"), 3, "FR3 linear velocity")
    angular = _vector(end_effector.get("angular_velocity"), 3, "FR3 angular velocity")
    low_dim = np.concatenate(
        (q, dq, position, matrix_to_rotation_6d(rotation), linear, angular)
    ).astype(np.float32)
    if low_dim.shape != (29,):
        raise AssertionError("internal FR3 low-dimensional state shape changed")

    age_s = max(0.0, float(now_s) - latest.received_monotonic_s)
    reasons: list[str] = []
    if age_s > maximum_age_s:
        reasons.append("franka_state_stale")
    bridge = message.get("bridge")
    raw_bridge = copy.deepcopy(bridge) if isinstance(bridge, dict) else {}
    watchdog_stop_count = _nonnegative_int(
        raw_bridge.get("watchdog_stop_count", 0),
        "FR3 bridge watchdog_stop_count",
    )
    workspace_guard_stop_count = _nonnegative_int(
        raw_bridge.get("workspace_guard_stop_count", 0),
        "FR3 bridge workspace_guard_stop_count",
    )
    if isinstance(bridge, dict) and bridge.get("last_fault"):
        reasons.append("franka_bridge_fault")
    if isinstance(bridge, dict) and bridge.get("velocity_workspace_blocked"):
        reasons.append("franka_workspace_guard")
    if (
        require_control_lease
        and isinstance(bridge, dict)
        and not bridge.get("control_lease_active", False)
    ):
        reasons.append("franka_control_lease_inactive")
    if _has_active_error(robot.get("current_errors")):
        reasons.append("franka_current_error")
    collision = end_effector.get("collision")
    if collision is not None and bool(np.any(np.asarray(collision))):
        reasons.append("franka_collision")

    timestamp = robot.get("timestamp_s")
    robot_timestamp_s = None if timestamp is None else float(timestamp)
    return ParsedFrankaState(
        low_dim=low_dim,
        raw_robot=copy.deepcopy(robot),
        raw_bridge=raw_bridge,
        sequence=int(message.get("sequence", -1)),
        robot_timestamp_s=robot_timestamp_s,
        received_monotonic_s=latest.received_monotonic_s,
        age_s=age_s,
        watchdog_stop_count=watchdog_stop_count,
        workspace_guard_stop_count=workspace_guard_stop_count,
        valid=not reasons,
        invalid_reasons=tuple(reasons),
    )


def _vector(value: Any, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return vector


def _vector_norm(value: np.ndarray) -> float:
    return math.sqrt(math.fsum(float(item) ** 2 for item in value.flat))


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if result < 0 or float(value) != float(result):
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _has_active_error(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_active_error(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_active_error(item) for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)
