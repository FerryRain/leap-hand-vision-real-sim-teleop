"""Versioned, JSON-serializable schema for grasp/release demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "1.0"
TASKS = ("grasp", "release")
TASK_TO_INDEX = {task: index for index, task in enumerate(TASKS)}
ACTION_DIMS = {"hand_only": 16, "global_twist_leap": 22}


class Stage(IntEnum):
    """Task stage labels shared by grasp and release episodes."""

    UNKNOWN = 0
    APPROACH = 1
    CLOSE = 2
    LIFT = 3
    HOLD = 4
    LOWER = 5
    OPEN = 6
    RETREAT = 7


STAGE_NAMES = {int(stage): stage.name.lower() for stage in Stage}


def default_action_names(action_space: str) -> tuple[str, ...]:
    leap = tuple(f"leap_joint_{index}_target" for index in range(16))
    if action_space == "hand_only":
        return leap
    if action_space == "global_twist_leap":
        return ("ee_vx", "ee_vy", "ee_vz", "ee_wx", "ee_wy", "ee_wz") + leap
    raise ValueError(f"unsupported action_space: {action_space!r}")


@dataclass(frozen=True)
class EpisodeSpec:
    """Fields that must remain constant throughout one episode."""

    task: str
    action_space: str
    robot_state_dim: int
    image_shape: tuple[int, int]
    state_age_limits_s: Mapping[str, float]
    robot_state_names: tuple[str, ...] = ()
    action_names: tuple[str, ...] = ()
    timestamp_clock: str = "monotonic"
    camera_name: str = "camera_0"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = str(self.task).lower()
        action_space = str(self.action_space).lower()
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "image_shape", tuple(int(x) for x in self.image_shape))
        object.__setattr__(self, "robot_state_names", tuple(self.robot_state_names))
        names = tuple(self.action_names) or default_action_names(action_space)
        object.__setattr__(self, "action_names", names)
        object.__setattr__(self, "state_age_limits_s", dict(self.state_age_limits_s))
        object.__setattr__(self, "extra", dict(self.extra))

        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if action_space not in ACTION_DIMS:
            raise ValueError(
                f"action_space must be one of {tuple(ACTION_DIMS)}, "
                f"got {self.action_space!r}"
            )
        if int(self.robot_state_dim) <= 0:
            raise ValueError("robot_state_dim must be positive")
        if len(self.image_shape) != 2 or any(size <= 0 for size in self.image_shape):
            raise ValueError("image_shape must contain positive (height, width)")
        if not self.state_age_limits_s:
            raise ValueError(
                "state_age_limits_s must name at least one synchronized source"
            )
        for source, limit in self.state_age_limits_s.items():
            if not str(source):
                raise ValueError("state age source names cannot be empty")
            if not np.isfinite(limit) or float(limit) <= 0:
                raise ValueError(
                    f"state age limit for {source!r} must be finite and positive"
                )
        if (
            self.robot_state_names
            and len(self.robot_state_names) != self.robot_state_dim
        ):
            raise ValueError("robot_state_names length does not match robot_state_dim")
        if len(self.action_names) != self.action_dim:
            raise ValueError("action_names length does not match action_space")
        if not self.timestamp_clock:
            raise ValueError("timestamp_clock cannot be empty")
        if not self.camera_name:
            raise ValueError("camera_name cannot be empty")
        ensure_jsonable(self.extra, "extra")

    @property
    def action_dim(self) -> int:
        return ACTION_DIMS[self.action_space]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "robot_state_dim": self.robot_state_dim,
            "image_shape": list(self.image_shape),
            "state_age_limits_s": dict(self.state_age_limits_s),
            "robot_state_names": list(self.robot_state_names),
            "action_names": list(self.action_names),
            "timestamp_clock": self.timestamp_clock,
            "camera_name": self.camera_name,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeSpec":
        version = str(payload.get("schema_version", ""))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
            )
        spec = cls(
            task=str(payload["task"]),
            action_space=str(payload["action_space"]),
            robot_state_dim=int(payload["robot_state_dim"]),
            image_shape=tuple(payload["image_shape"]),
            state_age_limits_s=dict(payload["state_age_limits_s"]),
            robot_state_names=tuple(payload.get("robot_state_names", ())),
            action_names=tuple(payload.get("action_names", ())),
            timestamp_clock=str(payload.get("timestamp_clock", "monotonic")),
            camera_name=str(payload.get("camera_name", "camera_0")),
            extra=dict(payload.get("extra", {})),
        )
        recorded_dim = int(payload.get("action_dim", spec.action_dim))
        if recorded_dim != spec.action_dim:
            raise ValueError("recorded action_dim does not match action_space")
        return spec


@dataclass(frozen=True)
class StepRecord:
    """Numeric and label data synchronized with one RGB/depth pair."""

    timestamp_s: float
    robot_state: Sequence[float]
    action: Sequence[float]
    stage: int = int(Stage.UNKNOWN)
    state_ages_s: Mapping[str, float] = field(default_factory=dict)
    valid: bool = True
    invalid_reasons: Sequence[str] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


def normalize_step(
    spec: EpisodeSpec,
    step: StepRecord,
    *,
    index: int,
    rgb_path: str,
    depth_path: str,
    previous_timestamp_s: float | None,
) -> dict[str, Any]:
    """Validate a step and return its canonical JSON payload."""

    timestamp = float(step.timestamp_s)
    if not np.isfinite(timestamp):
        raise ValueError("timestamp_s must be finite")
    if previous_timestamp_s is not None and timestamp <= previous_timestamp_s:
        raise ValueError("timestamps must be strictly increasing within an episode")

    state = np.asarray(step.robot_state, dtype=np.float64).reshape(-1)
    action = np.asarray(step.action, dtype=np.float64).reshape(-1)
    if state.shape != (spec.robot_state_dim,):
        raise ValueError(
            f"robot_state must have shape ({spec.robot_state_dim},), got {state.shape}"
        )
    if action.shape != (spec.action_dim,):
        raise ValueError(
            f"action must have shape ({spec.action_dim},), got {action.shape}"
        )
    if not np.isfinite(state).all():
        raise ValueError("robot_state contains a non-finite value")
    if not np.isfinite(action).all():
        raise ValueError("action contains a non-finite value")

    stage = int(step.stage)
    if stage < 0 or stage > np.iinfo(np.int16).max:
        raise ValueError("stage must fit in a non-negative int16")

    ages = {str(source): float(age) for source, age in step.state_ages_s.items()}
    missing = sorted(set(spec.state_age_limits_s) - set(ages))
    if missing:
        raise ValueError(f"state_ages_s is missing required sources: {missing}")
    for source, age in ages.items():
        if not np.isfinite(age) or age < 0:
            raise ValueError(
                f"state age for {source!r} must be finite and non-negative"
            )

    stale_sources = sorted(
        source
        for source, limit in spec.state_age_limits_s.items()
        if ages[source] > float(limit)
    )
    explicit_reasons: list[str] = []
    for raw_reason in step.invalid_reasons:
        reason = str(raw_reason).strip()
        if not reason:
            raise ValueError("invalid_reasons cannot contain an empty value")
        if reason not in explicit_reasons:
            explicit_reasons.append(reason)
    invalid_reasons = explicit_reasons + [
        f"stale:{source}"
        for source in stale_sources
        if f"stale:{source}" not in explicit_reasons
    ]
    if not bool(step.valid) and not invalid_reasons:
        invalid_reasons.append("unspecified")
    valid = bool(step.valid) and not invalid_reasons
    extra = dict(step.extra)
    ensure_jsonable(extra, "step.extra")
    return {
        "index": int(index),
        "timestamp_s": timestamp,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "robot_state": state.astype(np.float32).tolist(),
        "action": action.astype(np.float32).tolist(),
        "task": spec.task,
        "stage": stage,
        "state_ages_s": ages,
        "valid": valid,
        "invalid_reasons": invalid_reasons,
        "extra": extra,
    }


def validate_image_arrays(
    spec: EpisodeSpec,
    rgb: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return contiguous RGB uint8 and depth uint16 arrays."""

    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    expected_rgb = spec.image_shape + (3,)
    if rgb_array.shape != expected_rgb or rgb_array.dtype != np.uint8:
        raise ValueError(
            f"rgb must be uint8 with shape {expected_rgb}, "
            f"got {rgb_array.dtype} {rgb_array.shape}"
        )
    if depth_array.shape != spec.image_shape or depth_array.dtype != np.uint16:
        raise ValueError(
            f"depth must be uint16 with shape {spec.image_shape}, "
            f"got {depth_array.dtype} {depth_array.shape}"
        )
    return np.ascontiguousarray(rgb_array), np.ascontiguousarray(depth_array)


def ensure_jsonable(value: Any, name: str) -> None:
    """Reject values that JSON would serialize ambiguously (NaN/Infinity)."""

    import json

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON-compatible data: {exc}") from exc
