"""Validated four-point calibration file shared by calibration and playback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

WAYPOINT_FILE_VERSION = 1
REQUIRED_POINT_COUNT = 4


@dataclass(frozen=True)
class Waypoint:
    name: str
    position_m: tuple[float, float, float]
    rotation_matrix: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    )
    robot_timestamp_s: float | None

    @classmethod
    def from_robot_state(cls, name: str, robot: dict[str, Any]) -> "Waypoint":
        end_effector = robot.get("end_effector")
        if not isinstance(end_effector, dict):
            raise ValueError("robot state is missing end_effector")
        position = _vector3(end_effector.get("position"), "end_effector.position")
        raw_rotation = end_effector.get("rotation_matrix")
        rotation = (
            None
            if raw_rotation is None
            else _matrix3(raw_rotation, "end_effector.rotation_matrix")
        )
        timestamp = robot.get("timestamp_s")
        robot_timestamp_s = None if timestamp is None else float(timestamp)
        if robot_timestamp_s is not None and not math.isfinite(robot_timestamp_s):
            raise ValueError("robot timestamp must be finite")
        return cls(
            name=name,
            position_m=position,
            rotation_matrix=rotation,
            robot_timestamp_s=robot_timestamp_s,
        )

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> "Waypoint":
        if not isinstance(raw, dict):
            raise ValueError(f"point {index + 1} must be an object")
        name = raw.get("name", f"P{index + 1}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"point {index + 1} has an invalid name")
        timestamp = raw.get("robot_timestamp_s")
        robot_timestamp_s = None if timestamp is None else float(timestamp)
        if robot_timestamp_s is not None and not math.isfinite(robot_timestamp_s):
            raise ValueError(f"point {index + 1} robot timestamp must be finite")
        raw_rotation = raw.get("rotation_matrix")
        return cls(
            name=name,
            position_m=_vector3(raw.get("position_m"), "position_m"),
            rotation_matrix=(
                None
                if raw_rotation is None
                else _matrix3(raw_rotation, "rotation_matrix")
            ),
            robot_timestamp_s=robot_timestamp_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position_m": list(self.position_m),
            "rotation_matrix": (
                None
                if self.rotation_matrix is None
                else [list(row) for row in self.rotation_matrix]
            ),
            "robot_timestamp_s": self.robot_timestamp_s,
        }


@dataclass(frozen=True)
class WaypointSet:
    created_at_utc: str
    frame: str
    points: tuple[Waypoint, Waypoint, Waypoint, Waypoint]

    @classmethod
    def create(cls, points: Sequence[Waypoint]) -> "WaypointSet":
        validated = _exactly_four(points)
        return cls(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            frame="franka_base",
            points=validated,
        )

    @classmethod
    def load(cls, path: Path) -> "WaypointSet":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("waypoint file must contain a JSON object")
        if raw.get("version") != WAYPOINT_FILE_VERSION:
            raise ValueError(
                f"unsupported waypoint file version: {raw.get('version')!r}"
            )
        if raw.get("frame") != "franka_base":
            raise ValueError("waypoint frame must be franka_base")
        raw_points = raw.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("waypoint file is missing points")
        points = tuple(
            Waypoint.from_dict(point, index) for index, point in enumerate(raw_points)
        )
        return cls(
            created_at_utc=str(raw.get("created_at_utc", "unknown")),
            frame="franka_base",
            points=_exactly_four(points),
        )

    def save(self, path: Path, *, overwrite: bool) -> None:
        destination = path.resolve()
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"waypoint file already exists: {destination}; use --overwrite"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        payload = {
            "version": WAYPOINT_FILE_VERSION,
            "created_at_utc": self.created_at_utc,
            "frame": self.frame,
            "orientation_behavior": "preserve_current_orientation_during_playback",
            "points": [point.to_dict() for point in self.points],
        }
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)

    def validate_workspace(self, safety: dict[str, Any]) -> None:
        minimum = _vector3(safety.get("workspace_min_m"), "workspace_min_m")
        maximum = _vector3(safety.get("workspace_max_m"), "workspace_max_m")
        for point in self.points:
            if any(
                value < lower or value > upper
                for value, lower, upper in zip(point.position_m, minimum, maximum)
            ):
                raise ValueError(
                    f"{point.name} {point.position_m} is outside server workspace "
                    f"{minimum} .. {maximum}"
                )


def distance_m(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 3 or len(second) != 3:
        raise ValueError("distance inputs must contain three values")
    return math.sqrt(
        math.fsum((float(a) - float(b)) ** 2 for a, b in zip(first, second))
    )


def rotation_distance_rad(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    first_matrix = _matrix3(first, "first rotation")
    second_matrix = _matrix3(second, "second rotation")
    trace = math.fsum(
        first_matrix[row][column] * second_matrix[row][column]
        for row in range(3)
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.acos(cosine)


def _exactly_four(
    points: Sequence[Waypoint],
) -> tuple[Waypoint, Waypoint, Waypoint, Waypoint]:
    if len(points) != REQUIRED_POINT_COUNT:
        raise ValueError(f"exactly {REQUIRED_POINT_COUNT} points are required")
    return tuple(points)  # type: ignore[return-value]


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result  # type: ignore[return-value]


def _matrix3(
    value: Any,
    name: str,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three rows")
    return tuple(
        _vector3(row, f"{name} row {index + 1}") for index, row in enumerate(value)
    )  # type: ignore[return-value]
