"""Crash-tolerant episode writer and on-disk dataset validation."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from .schema import EpisodeSpec, StepRecord, normalize_step, validate_image_arrays

_EPISODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FINAL_STATUSES = ("accepted", "rejected")


def make_episode_id() -> str:
    """Create a sortable, collision-resistant episode identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"episode_{timestamp}_{uuid.uuid4().hex[:8]}"


class EpisodeWriter:
    """Write a single episode to ``.partial`` and atomically finalize it.

    A step is committed to ``steps.jsonl`` only after both of its image files
    have been atomically written and fsynced.  Consequently a process crash may
    leave harmless orphan images, but never a committed row referring to an
    image that had not finished writing.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        spec: EpisodeSpec,
        *,
        episode_id: str | None = None,
        jpeg_quality: int = 95,
    ) -> None:
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.dataset_root = Path(dataset_root).absolute()
        self.spec = spec
        self.episode_id = episode_id or make_episode_id()
        if not _EPISODE_ID_PATTERN.fullmatch(self.episode_id):
            raise ValueError(
                "episode_id must contain only letters, digits, dot, underscore, or dash"
            )
        self.jpeg_quality = int(jpeg_quality)

        for name in (".partial", "accepted", "rejected"):
            (self.dataset_root / name).mkdir(parents=True, exist_ok=True)
        self.directory = self.dataset_root / ".partial" / self.episode_id
        self.directory.mkdir(parents=False, exist_ok=False)
        (self.directory / "rgb").mkdir()
        (self.directory / "depth").mkdir()

        now = _utc_now()
        self._meta: dict[str, Any] = {
            "episode_id": self.episode_id,
            "status": "partial",
            "success": None,
            "rejection_reason": None,
            "created_at_utc": now,
            "finalized_at_utc": None,
            "num_steps": 0,
            "spec": self.spec.to_dict(),
        }
        _atomic_write_json(self.directory / "meta.json", self._meta)
        self._steps_file = (self.directory / "steps.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )
        self._step_count = 0
        self._previous_timestamp_s: float | None = None
        self._active = True

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def active(self) -> bool:
        return self._active

    def append_frame(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp_s: float,
        robot_state: Iterable[float],
        action: Iterable[float],
        stage: int = 0,
        state_ages_s: Mapping[str, float],
        valid: bool = True,
        invalid_reasons: Iterable[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> int:
        """Validate and append one synchronized frame, returning its index."""

        return self.append(
            StepRecord(
                timestamp_s=timestamp_s,
                robot_state=tuple(robot_state),
                action=tuple(action),
                stage=stage,
                state_ages_s=state_ages_s,
                valid=valid,
                invalid_reasons=tuple(invalid_reasons),
                extra=extra or {},
            ),
            rgb=rgb,
            depth=depth,
        )

    def append(self, step: StepRecord, *, rgb: np.ndarray, depth: np.ndarray) -> int:
        """Append a :class:`StepRecord` and its aligned image pair."""

        self._require_active()
        rgb_array, depth_array = validate_image_arrays(self.spec, rgb, depth)
        index = self._step_count
        rgb_path = f"rgb/{index:06d}.jpg"
        depth_path = f"depth/{index:06d}.png"
        payload = normalize_step(
            self.spec,
            step,
            index=index,
            rgb_path=rgb_path,
            depth_path=depth_path,
            previous_timestamp_s=self._previous_timestamp_s,
        )

        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        rgb_ok, rgb_encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        depth_ok, depth_encoded = cv2.imencode(
            ".png", depth_array, [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )
        if not rgb_ok:
            raise OSError("OpenCV failed to encode RGB JPEG")
        if not depth_ok:
            raise OSError("OpenCV failed to encode uint16 depth PNG")

        _atomic_write_bytes(self.directory / rgb_path, rgb_encoded.tobytes())
        _atomic_write_bytes(self.directory / depth_path, depth_encoded.tobytes())
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
        return index

    def accept(self, *, notes: str | None = None) -> Path:
        """Mark the non-empty episode successful and move it to ``accepted``."""

        if self._step_count == 0:
            raise RuntimeError("cannot accept an empty episode")
        return self._finalize("accepted", success=True, notes=notes)

    def reject(self, reason: str, *, notes: str | None = None) -> Path:
        """Mark the episode unsuccessful and move it to ``rejected``."""

        reason = str(reason).strip()
        if not reason:
            raise ValueError("a non-empty rejection reason is required")
        return self._finalize("rejected", success=False, reason=reason, notes=notes)

    def close_partial(self, *, reason: str | None = None) -> Path:
        """Close files while deliberately leaving an interrupted episode partial."""

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
        # Same-volume directory rename is the episode commit point.  If the
        # process dies just before it, the complete episode remains recoverable
        # under .partial with its requested final status in meta.json.
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

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(
        self, exc_type: Any, exc: BaseException | None, traceback: Any
    ) -> None:
        if self._active:
            self.close_partial(reason=repr(exc) if exc is not None else "not finalized")


@dataclass
class ValidationReport:
    path: Path
    status: str
    num_steps: int = 0
    task: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "status": self.status,
            "task": self.task,
            "num_steps": self.num_steps,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass
class DatasetValidationReport:
    dataset_root: Path
    episodes: list[ValidationReport]
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(report.ok for report in self.episodes)

    @property
    def num_steps(self) -> int:
        return sum(report.num_steps for report in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "ok": self.ok,
            "num_episodes": len(self.episodes),
            "num_steps": self.num_steps,
            "errors": list(self.errors),
            "episodes": [report.to_dict() for report in self.episodes],
        }


def read_episode_meta(episode_dir: str | Path) -> dict[str, Any]:
    with (Path(episode_dir) / "meta.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("meta.json root must be an object")
    return payload


def read_episode_steps(episode_dir: str | Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    with (Path(episode_dir) / "steps.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"steps.jsonl contains a blank line at {line_number}")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"step {line_number} must be a JSON object")
            steps.append(payload)
    return steps


def iter_episode_dirs(
    dataset_root: str | Path,
    *,
    include_partial: bool = True,
    include_rejected: bool = True,
) -> list[Path]:
    root = Path(dataset_root)
    statuses = ["accepted"]
    if include_rejected:
        statuses.append("rejected")
    if include_partial:
        statuses.append(".partial")
    result: list[Path] = []
    for status in statuses:
        parent = root / status
        if parent.is_dir():
            result.extend(path for path in parent.iterdir() if path.is_dir())
    return sorted(result, key=lambda path: (path.name, path.parent.name))


def validate_episode(episode_dir: str | Path, *, deep: bool = True) -> ValidationReport:
    """Validate metadata, synchronization, vectors, state ages, and images."""

    directory = Path(episode_dir)
    location_status = (
        "partial" if directory.parent.name == ".partial" else directory.parent.name
    )
    report = ValidationReport(path=directory, status=location_status)
    try:
        meta = read_episode_meta(directory)
    except Exception as exc:
        report.errors.append(f"cannot read meta.json: {exc}")
        return report
    try:
        spec = EpisodeSpec.from_dict(meta["spec"])
        report.task = spec.task
    except Exception as exc:
        report.errors.append(f"invalid episode spec: {exc}")
        return report

    if meta.get("episode_id") != directory.name:
        report.errors.append("meta episode_id does not match directory name")
    recorded_status = meta.get("status")
    if location_status in _FINAL_STATUSES and recorded_status != location_status:
        report.errors.append(
            f"meta status {recorded_status!r} does not match "
            f"{location_status!r} directory"
        )
    elif location_status == "partial" and recorded_status != "partial":
        report.warnings.append(
            f"partial contains finalized metadata ({recorded_status!r}); "
            "it may be recoverable"
        )
    if location_status == "accepted" and meta.get("success") is not True:
        report.errors.append("accepted episode must have success=true")
    if location_status == "rejected" and meta.get("success") is not False:
        report.errors.append("rejected episode must have success=false")

    try:
        steps = read_episode_steps(directory)
    except Exception as exc:
        report.errors.append(f"cannot read steps.jsonl: {exc}")
        return report
    report.num_steps = len(steps)
    recorded_count = meta.get("num_steps")
    if recorded_count != len(steps):
        message = (
            f"meta num_steps={recorded_count!r}, but steps.jsonl has {len(steps)} rows"
        )
        if location_status == "partial":
            report.warnings.append(message)
        else:
            report.errors.append(message)
    if location_status == "accepted" and not steps:
        report.errors.append("accepted episode is empty")

    previous_timestamp: float | None = None
    seen_paths: set[str] = set()
    for expected_index, payload in enumerate(steps):
        prefix = f"step {expected_index}"
        try:
            if payload.get("index") != expected_index:
                raise ValueError(f"index is {payload.get('index')!r}")
            if payload.get("task") != spec.task:
                raise ValueError("task differs from episode spec")
            rgb_rel = str(payload["rgb_path"])
            depth_rel = str(payload["depth_path"])
            if rgb_rel in seen_paths or depth_rel in seen_paths:
                raise ValueError("image path is reused")
            seen_paths.update((rgb_rel, depth_rel))
            rgb_path = _safe_episode_path(directory, rgb_rel)
            depth_path = _safe_episode_path(directory, depth_rel)
            step = StepRecord(
                timestamp_s=float(payload["timestamp_s"]),
                robot_state=payload["robot_state"],
                action=payload["action"],
                stage=int(payload["stage"]),
                state_ages_s=payload["state_ages_s"],
                valid=bool(payload["valid"]),
                invalid_reasons=payload.get("invalid_reasons", ()),
                extra=payload.get("extra", {}),
            )
            canonical = normalize_step(
                spec,
                step,
                index=expected_index,
                rgb_path=rgb_rel,
                depth_path=depth_rel,
                previous_timestamp_s=previous_timestamp,
            )
            if bool(payload["valid"]) != canonical["valid"]:
                raise ValueError("valid=true despite a stale synchronized source")
            if payload.get("invalid_reasons", []) != canonical["invalid_reasons"]:
                raise ValueError("invalid_reasons are not canonical")
            previous_timestamp = float(payload["timestamp_s"])
            if not rgb_path.is_file() or not depth_path.is_file():
                raise ValueError("RGB or depth image is missing")
            if deep:
                _validate_encoded_images(spec, rgb_path, depth_path)
        except Exception as exc:
            report.errors.append(f"{prefix}: {exc}")
    return report


def validate_dataset(
    dataset_root: str | Path,
    *,
    deep: bool = True,
    include_partial: bool = True,
    include_rejected: bool = True,
) -> DatasetValidationReport:
    root = Path(dataset_root).absolute()
    top_errors: list[str] = []
    if not root.is_dir():
        top_errors.append("dataset root does not exist or is not a directory")
        return DatasetValidationReport(root, [], top_errors)
    episodes = [
        validate_episode(path, deep=deep)
        for path in iter_episode_dirs(
            root,
            include_partial=include_partial,
            include_rejected=include_rejected,
        )
    ]
    if not episodes:
        top_errors.append("dataset contains no episodes")
    return DatasetValidationReport(root, episodes, top_errors)


def _validate_encoded_images(
    spec: EpisodeSpec, rgb_path: Path, depth_path: Path
) -> None:
    rgb = cv2.imdecode(
        np.frombuffer(rgb_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if rgb is None or rgb.shape != spec.image_shape + (3,) or rgb.dtype != np.uint8:
        raise ValueError("RGB JPEG cannot be decoded to the declared uint8 shape")
    depth = cv2.imdecode(
        np.frombuffer(depth_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if depth is None or depth.shape != spec.image_shape or depth.dtype != np.uint16:
        raise ValueError("depth PNG cannot be decoded to the declared uint16 shape")


def _safe_episode_path(directory: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe image path: {relative_path!r}")
    candidate = directory.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(
            f"image path escapes episode directory: {relative_path!r}"
        ) from exc
    return candidate


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write_bytes(path, data)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
