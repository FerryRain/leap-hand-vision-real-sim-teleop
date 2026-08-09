"""Crash-tolerant storage for image-free LEAP grasp episodes."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .proprio_schema import (
    NUM_LEAP_JOINTS,
    PROPRIO_ACTION_DIM,
    PROPRIO_STATE_DIM,
    ProprioEpisodeSpec,
    ProprioStep,
    normalize_proprio_step,
)

_EPISODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FINAL_STATUSES = ("accepted", "rejected")


def make_proprio_episode_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"proprio_grasp_{timestamp}_{uuid.uuid4().hex[:8]}"


@dataclass
class ProprioValidationReport:
    path: Path
    status: str = "unknown"
    num_steps: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class ProprioEpisodeWriter:
    """Append JSONL steps under ``.partial`` and atomically finalize them.

    A measured position immediately before the first recorded policy step is
    required.  It provides a real finite-difference baseline, avoiding an
    invented zero-velocity first observation.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        spec: ProprioEpisodeSpec,
        *,
        initial_timestamp_s: float,
        initial_actual_position_rad: Sequence[float],
        episode_id: str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).absolute()
        self.spec = spec
        self.episode_id = episode_id or make_proprio_episode_id()
        if not _EPISODE_ID_PATTERN.fullmatch(self.episode_id):
            raise ValueError(
                "episode_id must contain only letters, digits, dot, underscore, or dash"
            )
        initial_timestamp = float(initial_timestamp_s)
        initial_actual = np.asarray(initial_actual_position_rad, dtype=np.float64)
        if not np.isfinite(initial_timestamp):
            raise ValueError("initial_timestamp_s must be finite")
        if (
            initial_actual.shape != (NUM_LEAP_JOINTS,)
            or not np.isfinite(initial_actual).all()
        ):
            raise ValueError(
                "initial_actual_position_rad must contain 16 finite values"
            )

        for name in (".partial", "accepted", "rejected"):
            (self.dataset_root / name).mkdir(parents=True, exist_ok=True)
        self.directory = self.dataset_root / ".partial" / self.episode_id
        self.directory.mkdir(parents=False, exist_ok=False)
        self._previous_timestamp_s = initial_timestamp
        self._previous_actual_position_rad = initial_actual.copy()
        self._step_count = 0
        self._active = True
        self._meta: dict[str, Any] = {
            "episode_id": self.episode_id,
            "dataset_kind": "leap_proprio_grasp",
            "status": "partial",
            "success": None,
            "rejection_reason": None,
            "created_at_utc": _utc_now(),
            "finalized_at_utc": None,
            "num_steps": 0,
            "spec": self.spec.to_dict(),
            "velocity_baseline": {
                "timestamp_s": initial_timestamp,
                "actual_position_rad": initial_actual.astype(np.float32).tolist(),
            },
        }
        _atomic_write_json(self.directory / "meta.json", self._meta)
        self._steps_file = (self.directory / "steps.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def active(self) -> bool:
        return self._active

    def validation_baseline(self) -> tuple[float, np.ndarray]:
        """Return the committed baseline used to validate the next step."""

        self._require_active()
        return (
            float(self._previous_timestamp_s),
            self._previous_actual_position_rad.copy(),
        )

    def append(
        self,
        *,
        timestamp_s: float,
        actual_position_rad: Sequence[float],
        present_current_raw: Sequence[int],
        goal_position_rad: Sequence[float],
        valid: bool = True,
        invalid_reasons: Iterable[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> int:
        return self.append_step(
            ProprioStep(
                timestamp_s=timestamp_s,
                actual_position_rad=actual_position_rad,
                present_current_raw=present_current_raw,
                goal_position_rad=goal_position_rad,
                valid=valid,
                invalid_reasons=tuple(invalid_reasons),
                extra=extra or {},
            )
        )

    def append_step(self, step: ProprioStep) -> int:
        self._require_active()
        payload = normalize_proprio_step(
            self.spec,
            step,
            index=self._step_count,
            previous_timestamp_s=self._previous_timestamp_s,
            previous_actual_position_rad=self._previous_actual_position_rad,
        )
        line = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._steps_file.write(line + "\n")
        self._steps_file.flush()
        os.fsync(self._steps_file.fileno())
        self._step_count += 1
        self._previous_timestamp_s = float(step.timestamp_s)
        self._previous_actual_position_rad = np.asarray(
            step.actual_position_rad, dtype=np.float64
        ).copy()
        return self._step_count - 1

    def accept(self, *, notes: str | None = None) -> Path:
        if self._step_count == 0:
            raise RuntimeError("cannot accept an empty episode")
        return self._finalize("accepted", success=True, notes=notes)

    def reject(self, reason: str, *, notes: str | None = None) -> Path:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("a non-empty rejection reason is required")
        return self._finalize("rejected", success=False, reason=reason, notes=notes)

    def close_partial(self, *, reason: str | None = None) -> Path:
        if not self._active:
            return self.directory
        self._close_steps_file()
        self._meta.update(
            num_steps=self._step_count,
            interruption_reason=str(reason) if reason else None,
        )
        _atomic_write_json(self.directory / "meta.json", self._meta)
        self._active = False
        return self.directory

    def _finalize(
        self,
        status: str,
        *,
        success: bool,
        reason: str | None = None,
        notes: str | None = None,
    ) -> Path:
        self._require_active()
        if status not in _FINAL_STATUSES:
            raise ValueError(f"invalid final status: {status}")
        self._close_steps_file()
        self._meta.update(
            status=status,
            success=bool(success),
            rejection_reason=reason,
            notes=notes,
            finalized_at_utc=_utc_now(),
            num_steps=self._step_count,
        )
        _atomic_write_json(self.directory / "meta.json", self._meta)
        destination = self.dataset_root / status / self.episode_id
        if destination.exists():
            raise FileExistsError(f"episode destination already exists: {destination}")
        os.replace(self.directory, destination)
        self.directory = destination
        self._active = False
        return destination

    def _close_steps_file(self) -> None:
        if not self._steps_file.closed:
            self._steps_file.flush()
            os.fsync(self._steps_file.fileno())
            self._steps_file.close()

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("episode writer is already closed or finalized")

    def __enter__(self) -> "ProprioEpisodeWriter":
        return self

    def __exit__(
        self, exc_type: Any, exc: BaseException | None, traceback: Any
    ) -> None:
        if self._active:
            self.close_partial(reason=repr(exc) if exc is not None else "not finalized")


def read_proprio_meta(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path) / "meta.json")


def read_proprio_steps(path: str | Path) -> list[dict[str, Any]]:
    steps_path = Path(path) / "steps.jsonl"
    rows: list[dict[str, Any]] = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"blank line in steps.jsonl at {line_number}")
            payload = _json_loads_strict(line)
            if not isinstance(payload, dict):
                raise ValueError(f"step {line_number} is not a JSON object")
            rows.append(payload)
    return rows


def validate_proprio_episode(path: str | Path) -> ProprioValidationReport:
    directory = Path(path).absolute()
    report = ProprioValidationReport(path=directory)
    try:
        meta = read_proprio_meta(directory)
        if meta.get("dataset_kind") != "leap_proprio_grasp":
            raise ValueError("meta dataset_kind is not leap_proprio_grasp")
        spec = ProprioEpisodeSpec.from_dict(meta["spec"])
        report.status = str(meta.get("status", "unknown"))
        if report.status not in ("partial", "accepted", "rejected"):
            raise ValueError("meta status is invalid")
        if (
            directory.parent.name in _FINAL_STATUSES
            and directory.parent.name != report.status
        ):
            raise ValueError("directory and meta statuses disagree")
        baseline = meta["velocity_baseline"]
        previous_timestamp = float(baseline["timestamp_s"])
        previous_actual = _vector(
            baseline["actual_position_rad"], NUM_LEAP_JOINTS, "velocity baseline"
        )
        steps = read_proprio_steps(directory)
        for index, payload in enumerate(steps):
            _validate_saved_step(
                spec,
                payload,
                index=index,
                previous_timestamp_s=previous_timestamp,
                previous_actual_position_rad=previous_actual,
            )
            previous_timestamp = float(payload["timestamp_s"])
            previous_actual = _vector(
                payload["actual_position_rad"],
                NUM_LEAP_JOINTS,
                "actual_position_rad",
            )
        report.num_steps = len(steps)
        if int(meta.get("num_steps", -1)) != len(steps):
            raise ValueError("meta num_steps does not match steps.jsonl")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report.errors.append(str(exc))
    return report


def _validate_saved_step(
    spec: ProprioEpisodeSpec,
    payload: Mapping[str, Any],
    *,
    index: int,
    previous_timestamp_s: float,
    previous_actual_position_rad: np.ndarray,
) -> None:
    if int(payload["index"]) != index:
        raise ValueError(f"step {index} has a non-contiguous index")
    if payload.get("task") != "grasp":
        raise ValueError(f"step {index} is not a grasp step")
    if "rgb_path" in payload or "depth_path" in payload:
        raise ValueError(f"step {index} unexpectedly references image data")
    timestamp = float(payload["timestamp_s"])
    dt = timestamp - float(previous_timestamp_s)
    recorded_dt = float(payload["sample_dt_s"])
    if not np.isfinite(timestamp) or dt <= 0.0 or not np.isclose(dt, recorded_dt):
        raise ValueError(f"step {index} has an invalid sample_dt_s")
    actual = _vector(payload["actual_position_rad"], 16, "actual_position_rad")
    velocity = _vector(payload["velocity_rad_s"], 16, "velocity_rad_s")
    current = _signed_current(payload["present_current_raw"])
    goal = _vector(payload["goal_position_rad"], 16, "goal_position_rad")
    error = _vector(payload["position_error_rad"], 16, "position_error_rad")
    action = _vector(payload["action"], PROPRIO_ACTION_DIM, "action")
    state = _vector(payload["robot_state"], PROPRIO_STATE_DIM, "robot_state")
    expected_velocity = (actual - previous_actual_position_rad) / dt
    expected_state = np.concatenate(
        (actual, expected_velocity, current.astype(np.float64)), axis=0
    )
    if not np.allclose(velocity, expected_velocity, rtol=1e-5, atol=1e-5):
        raise ValueError(f"step {index} finite-difference velocity is inconsistent")
    if not np.allclose(state, expected_state, rtol=1e-5, atol=1e-5):
        raise ValueError(f"step {index} robot_state is inconsistent")
    if not np.allclose(action, goal, rtol=0.0, atol=1e-6):
        raise ValueError(f"step {index} action is not its commanded goal")
    if not np.allclose(error, goal - actual, rtol=0.0, atol=1e-6):
        raise ValueError(f"step {index} position_error_rad is inconsistent")
    reasons = payload.get("invalid_reasons")
    if not isinstance(reasons, list) or any(not str(item).strip() for item in reasons):
        raise ValueError(f"step {index} invalid_reasons is malformed")
    period_bad = abs(dt - spec.sample_period_s) > spec.sample_period_tolerance_s
    if period_bad != ("sample_period_out_of_tolerance" in reasons):
        raise ValueError(f"step {index} sample-period validity is inconsistent")
    if bool(payload["valid"]) == bool(reasons):
        raise ValueError(f"step {index} valid flag disagrees with invalid_reasons")
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise ValueError(f"step {index} extra is not an object")


def _vector(values: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return vector


def _signed_current(values: Any) -> np.ndarray:
    vector = _vector(values, NUM_LEAP_JOINTS, "present_current_raw")
    if not np.equal(vector, np.rint(vector)).all():
        raise ValueError("present_current_raw must contain integers")
    if np.any(vector < np.iinfo(np.int16).min) or np.any(
        vector > np.iinfo(np.int16).max
    ):
        raise ValueError("present_current_raw must fit signed int16")
    return vector.astype(np.int16)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_loads_strict(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {constant!r}")
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = _json_loads_strict(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
