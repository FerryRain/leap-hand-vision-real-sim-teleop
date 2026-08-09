"""Schema for image-free, LEAP-only grasp demonstrations.

The policy action is always the 16-D post-slew position goal that was actually
sent to the hand.  Measured motion is observation data, never relabelled as an
action.  This distinction matters for contact-rich grasps: a constant position
goal can coexist with slowly changing measured joints while the motor's local
controller remains current limited.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

PROPRIO_SCHEMA_VERSION = "1.0"
NUM_LEAP_JOINTS = 16
PROPRIO_STATE_DIM = 3 * NUM_LEAP_JOINTS
PROPRIO_ACTION_DIM = NUM_LEAP_JOINTS
TASK = "grasp"
DYNAMICS_FINGERPRINT_FIELDS = (
    "leap_device",
    "operating_mode",
    "goal_current_raw",
    "kp",
    "ki",
    "kd",
    "maximum_goal_step_rad",
    "motor_ids",
    "motor_model_numbers",
    "present_current_unit",
    "teleop_config_sha256",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _names(prefix: str, suffix: str) -> tuple[str, ...]:
    return tuple(f"leap_joint_{index}_{prefix}_{suffix}" for index in range(16))


ACTUAL_POSITION_NAMES = _names("actual", "rad")
FINITE_DIFFERENCE_VELOCITY_NAMES = _names("velocity_fd", "rad_s")
PRESENT_CURRENT_RAW_NAMES = _names("present_current", "raw_signed")
ROBOT_STATE_NAMES = (
    ACTUAL_POSITION_NAMES + FINITE_DIFFERENCE_VELOCITY_NAMES + PRESENT_CURRENT_RAW_NAMES
)
ACTION_NAMES = _names("post_slew_goal", "rad")
GOAL_POSITION_NAMES = ACTION_NAMES
POSITION_ERROR_NAMES = _names("goal_minus_actual", "rad")


@dataclass(frozen=True)
class ProprioEpisodeSpec:
    """Fields that remain fixed throughout one proprioceptive grasp episode."""

    sample_period_s: float
    sample_period_tolerance_s: float
    joint_names: tuple[str, ...]
    control_mode: str = "current_based_position"
    task: str = TASK
    timestamp_clock: str = "run_relative_monotonic"
    action_semantics: str = "post_slew_goal_position_sent_to_leap"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", tuple(str(x) for x in self.joint_names))
        object.__setattr__(self, "extra", dict(self.extra))
        period = float(self.sample_period_s)
        tolerance = float(self.sample_period_tolerance_s)
        if self.task != TASK:
            raise ValueError("proprio-only demonstrations support task='grasp' only")
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "sample_period_tolerance_s must be finite and non-negative"
            )
        if tolerance > period:
            raise ValueError("sample_period_tolerance_s cannot exceed sample_period_s")
        if len(self.joint_names) != NUM_LEAP_JOINTS:
            raise ValueError("joint_names must contain exactly 16 names")
        if len(set(self.joint_names)) != NUM_LEAP_JOINTS or any(
            not name for name in self.joint_names
        ):
            raise ValueError("joint_names must be non-empty and unique")
        if not self.control_mode:
            raise ValueError("control_mode cannot be empty")
        if not self.timestamp_clock:
            raise ValueError("timestamp_clock cannot be empty")
        if self.action_semantics != "post_slew_goal_position_sent_to_leap":
            raise ValueError(
                "action_semantics must remain the actual post-slew LEAP goal"
            )
        ensure_jsonable(self.extra, "extra")

    @property
    def robot_state_dim(self) -> int:
        return PROPRIO_STATE_DIM

    @property
    def action_dim(self) -> int:
        return PROPRIO_ACTION_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROPRIO_SCHEMA_VERSION,
            "task": self.task,
            "sample_period_s": float(self.sample_period_s),
            "sample_period_tolerance_s": float(self.sample_period_tolerance_s),
            "joint_names": list(self.joint_names),
            "control_mode": self.control_mode,
            "timestamp_clock": self.timestamp_clock,
            "action_semantics": self.action_semantics,
            "robot_state_dim": self.robot_state_dim,
            "robot_state_names": list(ROBOT_STATE_NAMES),
            "action_dim": self.action_dim,
            "action_names": list(ACTION_NAMES),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProprioEpisodeSpec":
        version = str(payload.get("schema_version", ""))
        if version != PROPRIO_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported proprio schema {version!r}; "
                f"expected {PROPRIO_SCHEMA_VERSION!r}"
            )
        spec = cls(
            sample_period_s=float(payload["sample_period_s"]),
            sample_period_tolerance_s=float(payload["sample_period_tolerance_s"]),
            joint_names=tuple(payload["joint_names"]),
            control_mode=str(payload.get("control_mode", "")),
            task=str(payload.get("task", "")),
            timestamp_clock=str(payload.get("timestamp_clock", "")),
            action_semantics=str(payload.get("action_semantics", "")),
            extra=dict(payload.get("extra", {})),
        )
        if (
            int(payload.get("robot_state_dim", spec.robot_state_dim))
            != spec.robot_state_dim
        ):
            raise ValueError("recorded robot_state_dim does not match proprio schema")
        if int(payload.get("action_dim", spec.action_dim)) != spec.action_dim:
            raise ValueError("recorded action_dim does not match proprio schema")
        if (
            tuple(payload.get("robot_state_names", ROBOT_STATE_NAMES))
            != ROBOT_STATE_NAMES
        ):
            raise ValueError("recorded robot_state_names do not match proprio schema")
        if tuple(payload.get("action_names", ACTION_NAMES)) != ACTION_NAMES:
            raise ValueError("recorded action_names do not match proprio schema")
        return spec


def dynamics_fingerprint(extra: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated, canonical hardware/control fingerprint.

    Raw episodes may be created with incomplete metadata so an interrupted
    experiment remains inspectable.  Training export is deliberately stricter:
    it calls this function and refuses legacy or incomplete episodes instead of
    guessing whether their contact dynamics are compatible.
    """

    if not isinstance(extra, Mapping):
        raise ValueError("proprio spec.extra must be a mapping")
    missing = [field for field in DYNAMICS_FINGERPRINT_FIELDS if field not in extra]
    if missing:
        raise ValueError(
            "missing required dynamics fingerprint fields: " + ", ".join(missing)
        )

    leap_device = str(extra["leap_device"]).strip().lower()
    if leap_device not in {"mock", "real"}:
        raise ValueError("dynamics fingerprint leap_device must be 'mock' or 'real'")
    operating_mode = _integer_metadata(
        extra["operating_mode"], "operating_mode", minimum=0, maximum=255
    )
    goal_current_raw = _integer_metadata(
        extra["goal_current_raw"],
        "goal_current_raw",
        minimum=0,
        maximum=np.iinfo(np.int16).max,
    )
    kp = _integer_metadata(extra["kp"], "kp", minimum=0)
    ki = _integer_metadata(extra["ki"], "ki", minimum=0)
    kd = _integer_metadata(extra["kd"], "kd", minimum=0)
    maximum_goal_step_rad = float(extra["maximum_goal_step_rad"])
    if not np.isfinite(maximum_goal_step_rad) or maximum_goal_step_rad <= 0.0:
        raise ValueError("maximum_goal_step_rad must be finite and positive")
    motor_ids = _motor_ids(extra["motor_ids"])
    motor_model_numbers = _motor_model_numbers(
        extra["motor_model_numbers"], leap_device=leap_device
    )
    present_current_unit = str(extra["present_current_unit"]).strip()
    if not present_current_unit:
        raise ValueError("present_current_unit cannot be empty")
    teleop_config_sha256 = str(extra["teleop_config_sha256"]).strip().lower()
    if not _SHA256_PATTERN.fullmatch(teleop_config_sha256):
        raise ValueError("teleop_config_sha256 must contain 64 hexadecimal digits")

    fingerprint = {
        "leap_device": leap_device,
        "operating_mode": operating_mode,
        "goal_current_raw": goal_current_raw,
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "maximum_goal_step_rad": maximum_goal_step_rad,
        "motor_ids": list(motor_ids),
        "motor_model_numbers": list(motor_model_numbers),
        "present_current_unit": present_current_unit,
        "teleop_config_sha256": teleop_config_sha256,
    }
    ensure_jsonable(fingerprint, "dynamics fingerprint")
    return fingerprint


@dataclass(frozen=True)
class ProprioStep:
    """One measured state paired with the position goal actually sent."""

    timestamp_s: float
    actual_position_rad: Sequence[float]
    present_current_raw: Sequence[int]
    goal_position_rad: Sequence[float]
    valid: bool = True
    invalid_reasons: Sequence[str] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


def normalize_proprio_step(
    spec: ProprioEpisodeSpec,
    step: ProprioStep,
    *,
    index: int,
    previous_timestamp_s: float,
    previous_actual_position_rad: Sequence[float],
) -> dict[str, Any]:
    """Validate one step and calculate its finite-difference state."""

    timestamp = float(step.timestamp_s)
    previous_timestamp = float(previous_timestamp_s)
    if not np.isfinite(timestamp) or not np.isfinite(previous_timestamp):
        raise ValueError("timestamps must be finite")
    sample_dt = timestamp - previous_timestamp
    if not np.isfinite(sample_dt) or sample_dt <= 0.0:
        raise ValueError("timestamps must be strictly increasing")

    actual = _float_vector(step.actual_position_rad, "actual_position_rad")
    previous_actual = _float_vector(
        previous_actual_position_rad, "previous_actual_position_rad"
    )
    goal = _float_vector(step.goal_position_rad, "goal_position_rad")
    current = _signed_current_vector(step.present_current_raw)
    velocity = (actual - previous_actual) / sample_dt
    position_error = goal - actual
    robot_state = np.concatenate((actual, velocity, current.astype(np.float64)), axis=0)

    reasons: list[str] = []
    for raw_reason in step.invalid_reasons:
        reason = str(raw_reason).strip()
        if not reason:
            raise ValueError("invalid_reasons cannot contain an empty value")
        if reason not in reasons:
            reasons.append(reason)
    if abs(sample_dt - float(spec.sample_period_s)) > float(
        spec.sample_period_tolerance_s
    ):
        reasons.append("sample_period_out_of_tolerance")
    if not bool(step.valid) and not reasons:
        reasons.append("unspecified")
    valid = bool(step.valid) and not reasons
    extra = dict(step.extra)
    ensure_jsonable(extra, "step.extra")

    return {
        "index": int(index),
        "task": TASK,
        "timestamp_s": timestamp,
        "sample_dt_s": sample_dt,
        "robot_state": robot_state.astype(np.float32).tolist(),
        "action": goal.astype(np.float32).tolist(),
        "actual_position_rad": actual.astype(np.float32).tolist(),
        "velocity_rad_s": velocity.astype(np.float32).tolist(),
        "present_current_raw": current.tolist(),
        "goal_position_rad": goal.astype(np.float32).tolist(),
        "position_error_rad": position_error.astype(np.float32).tolist(),
        "valid": valid,
        "invalid_reasons": reasons,
        "extra": extra,
    }


def _float_vector(values: Sequence[float], label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (NUM_LEAP_JOINTS,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain 16 finite values")
    return vector


def _signed_current_vector(values: Sequence[int]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (NUM_LEAP_JOINTS,):
        raise ValueError("present_current_raw must contain 16 values")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("present_current_raw must contain integers") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError("present_current_raw must contain finite integers")
    if np.any(numeric < np.iinfo(np.int16).min) or np.any(
        numeric > np.iinfo(np.int16).max
    ):
        raise ValueError("present_current_raw values must fit signed int16")
    return numeric.astype(np.int16)


def _integer_metadata(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not np.isfinite(numeric) or numeric != round(numeric):
        raise ValueError(f"{label} must be an integer")
    result = int(numeric)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _motor_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != NUM_LEAP_JOINTS:
        raise ValueError("motor_ids must contain exactly 16 values")
    result = tuple(
        _integer_metadata(item, "motor_ids entry", minimum=0, maximum=252)
        for item in value
    )
    if len(set(result)) != NUM_LEAP_JOINTS:
        raise ValueError("motor_ids must be unique")
    return result


def _motor_model_numbers(value: Any, *, leap_device: str) -> tuple[int | str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != NUM_LEAP_JOINTS:
        raise ValueError("motor_model_numbers must contain exactly 16 values")
    if leap_device == "real":
        return tuple(
            _integer_metadata(item, "motor_model_numbers entry", minimum=1)
            for item in value
        )
    normalized: list[int | str] = []
    for item in value:
        if isinstance(item, str) and item.strip().lower() == "mock":
            normalized.append("mock")
        elif not isinstance(item, (bool, np.bool_)):
            try:
                integer = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "mock motor_model_numbers entries must be 'mock' or -1"
                ) from exc
            if integer == -1 and float(item) == -1.0:
                normalized.append(-1)
            else:
                raise ValueError(
                    "mock motor_model_numbers entries must be 'mock' or -1"
                )
        else:
            raise ValueError("mock motor_model_numbers entries must be 'mock' or -1")
    if len(set(normalized)) != 1:
        raise ValueError("mock motor_model_numbers must use one consistent sentinel")
    return tuple(normalized)


def ensure_jsonable(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON-compatible data: {exc}") from exc
