"""Zarr v2 export for image-free, LEAP-only grasp demonstrations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .proprio_episode import (
    read_proprio_meta,
    read_proprio_steps,
    validate_proprio_episode,
)
from .proprio_schema import (
    ACTION_NAMES,
    ACTUAL_POSITION_NAMES,
    FINITE_DIFFERENCE_VELOCITY_NAMES,
    GOAL_POSITION_NAMES,
    POSITION_ERROR_NAMES,
    PRESENT_CURRENT_RAW_NAMES,
    PROPRIO_ACTION_DIM,
    PROPRIO_SCHEMA_VERSION,
    PROPRIO_STATE_DIM,
    ROBOT_STATE_NAMES,
    ProprioEpisodeSpec,
    dynamics_fingerprint,
)


class ProprioExportError(RuntimeError):
    """Raised when proprio episodes cannot form a coherent training store."""


@dataclass(frozen=True)
class ProprioEpisodeBundle:
    path: Path
    status: str
    meta: dict[str, Any]
    spec: ProprioEpisodeSpec
    steps: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProprioExportSummary:
    output_path: Path
    num_episodes: int
    num_steps: int
    robot_state_dim: int = PROPRIO_STATE_DIM
    action_dim: int = PROPRIO_ACTION_DIM

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "robot_state_dim": self.robot_state_dim,
            "action_dim": self.action_dim,
        }


def collect_proprio_bundles(
    dataset_root: str | Path,
    *,
    include_rejected: bool = False,
) -> list[ProprioEpisodeBundle]:
    root = Path(dataset_root).absolute()
    statuses = ["accepted"] + (["rejected"] if include_rejected else [])
    paths: list[Path] = []
    for status in statuses:
        parent = root / status
        if parent.is_dir():
            paths.extend(path for path in parent.iterdir() if path.is_dir())
    paths.sort(key=lambda path: (path.name, path.parent.name))

    bundles: list[ProprioEpisodeBundle] = []
    for path in paths:
        report = validate_proprio_episode(path)
        if not report.ok:
            raise ProprioExportError(
                f"invalid source episode {path}: {'; '.join(report.errors)}"
            )
        meta = read_proprio_meta(path)
        spec = ProprioEpisodeSpec.from_dict(meta["spec"])
        steps = tuple(read_proprio_steps(path))
        if not steps:
            continue
        bundles.append(
            ProprioEpisodeBundle(
                path=path,
                status=path.parent.name,
                meta=meta,
                spec=spec,
                steps=steps,
            )
        )
    if not bundles:
        raise ProprioExportError("no eligible proprio grasp episodes found")
    _validate_compatible_specs(bundles)
    return bundles


def export_proprio_to_zarr(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    include_rejected: bool = False,
    chunk_length: int = 256,
    overwrite: bool = False,
) -> ProprioExportSummary:
    """Export low-dimensional observations without creating image arrays."""

    if int(chunk_length) <= 0:
        raise ValueError("chunk_length must be positive")
    source_root = Path(dataset_root).absolute()
    output = Path(output_path).absolute()
    if output == source_root:
        raise ValueError("output_path cannot be the source dataset root")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use overwrite=True): {output}")

    bundles = collect_proprio_bundles(source_root, include_rejected=include_rejected)
    zarr, Blosc = _load_zarr_v2()
    spec = bundles[0].spec
    fingerprint = _bundle_dynamics_fingerprint(bundles[0])
    fingerprint_sha256 = _fingerprint_sha256(fingerprint)
    num_steps = sum(len(bundle.steps) for bundle in bundles)
    numeric_chunk = min(int(chunk_length), num_steps)
    temporary = output.with_name(f"{output.name}.partial-{uuid.uuid4().hex}")
    if temporary.exists():
        raise FileExistsError(f"temporary export path already exists: {temporary}")

    try:
        store = zarr.DirectoryStore(str(temporary))
        root = zarr.group(store=store, overwrite=True)
        data = root.create_group("data")
        meta_group = root.create_group("meta")
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        arrays = {
            "robot_state": data.create_dataset(
                "robot_state",
                shape=(num_steps, PROPRIO_STATE_DIM),
                chunks=(numeric_chunk, PROPRIO_STATE_DIM),
                dtype="<f4",
                compressor=compressor,
            ),
            "action": data.create_dataset(
                "action",
                shape=(num_steps, PROPRIO_ACTION_DIM),
                chunks=(numeric_chunk, PROPRIO_ACTION_DIM),
                dtype="<f4",
                compressor=compressor,
            ),
            "actual_position": data.create_dataset(
                "actual_position",
                shape=(num_steps, 16),
                chunks=(numeric_chunk, 16),
                dtype="<f4",
                compressor=compressor,
            ),
            "velocity": data.create_dataset(
                "velocity",
                shape=(num_steps, 16),
                chunks=(numeric_chunk, 16),
                dtype="<f4",
                compressor=compressor,
            ),
            "present_current_raw": data.create_dataset(
                "present_current_raw",
                shape=(num_steps, 16),
                chunks=(numeric_chunk, 16),
                dtype="<i2",
                compressor=compressor,
            ),
            "goal_position": data.create_dataset(
                "goal_position",
                shape=(num_steps, 16),
                chunks=(numeric_chunk, 16),
                dtype="<f4",
                compressor=compressor,
            ),
            "position_error": data.create_dataset(
                "position_error",
                shape=(num_steps, 16),
                chunks=(numeric_chunk, 16),
                dtype="<f4",
                compressor=compressor,
            ),
            "timestamp": data.create_dataset(
                "timestamp",
                shape=(num_steps,),
                chunks=(numeric_chunk,),
                dtype="<f8",
                compressor=compressor,
            ),
            "sample_dt": data.create_dataset(
                "sample_dt",
                shape=(num_steps,),
                chunks=(numeric_chunk,),
                dtype="<f8",
                compressor=compressor,
            ),
            "valid": data.create_dataset(
                "valid",
                shape=(num_steps,),
                chunks=(numeric_chunk,),
                dtype="bool",
                compressor=compressor,
            ),
        }

        episode_ends: list[int] = []
        offset = 0
        for bundle in bundles:
            for payload in bundle.steps:
                arrays["robot_state"][offset] = np.asarray(
                    payload["robot_state"], dtype=np.float32
                )
                arrays["action"][offset] = np.asarray(
                    payload["action"], dtype=np.float32
                )
                arrays["actual_position"][offset] = np.asarray(
                    payload["actual_position_rad"], dtype=np.float32
                )
                arrays["velocity"][offset] = np.asarray(
                    payload["velocity_rad_s"], dtype=np.float32
                )
                arrays["present_current_raw"][offset] = np.asarray(
                    payload["present_current_raw"], dtype=np.int16
                )
                arrays["goal_position"][offset] = np.asarray(
                    payload["goal_position_rad"], dtype=np.float32
                )
                arrays["position_error"][offset] = np.asarray(
                    payload["position_error_rad"], dtype=np.float32
                )
                arrays["timestamp"][offset] = float(payload["timestamp_s"])
                arrays["sample_dt"][offset] = float(payload["sample_dt_s"])
                arrays["valid"][offset] = bool(payload["valid"])
                offset += 1
            episode_ends.append(offset)
        meta_group.create_dataset(
            "episode_ends",
            data=np.asarray(episode_ends, dtype=np.int64),
            chunks=(min(numeric_chunk, len(episode_ends)),),
            compressor=compressor,
        )
        root.attrs.update(
            {
                "format": "diffusion_policy_zarr_v2",
                "dataset_kind": "leap_proprio_grasp",
                "source_schema_version": PROPRIO_SCHEMA_VERSION,
                "task": "grasp",
                "robot_state_names": list(ROBOT_STATE_NAMES),
                "action_names": list(ACTION_NAMES),
                "action_semantics": spec.action_semantics,
                "timestamp_clock": spec.timestamp_clock,
                "dynamics_fingerprint": fingerprint,
                "dynamics_fingerprint_sha256": fingerprint_sha256,
            }
        )
        if hasattr(store, "close"):
            store.close()
        manifest = build_proprio_manifest(
            bundles,
            episode_ends=episode_ends,
            num_steps=num_steps,
            include_rejected=include_rejected,
        )
        _write_manifest(temporary / "manifest.json", manifest)
        _commit_export(temporary, output, overwrite=overwrite)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return ProprioExportSummary(
        output_path=output,
        num_episodes=len(bundles),
        num_steps=num_steps,
    )


def build_proprio_manifest(
    bundles: list[ProprioEpisodeBundle],
    *,
    episode_ends: list[int],
    num_steps: int,
    include_rejected: bool,
) -> dict[str, Any]:
    if not bundles:
        raise ValueError("manifest requires at least one episode")
    if len(bundles) != len(episode_ends):
        raise ValueError("episode_ends length must match bundles")
    _validate_compatible_specs(bundles)
    spec = bundles[0].spec
    fingerprint = _bundle_dynamics_fingerprint(bundles[0])
    arrays = {
        "data/robot_state": {"shape": [num_steps, 48], "dtype": "float32"},
        "data/action": {"shape": [num_steps, 16], "dtype": "float32"},
        "data/actual_position": {"shape": [num_steps, 16], "dtype": "float32"},
        "data/velocity": {"shape": [num_steps, 16], "dtype": "float32"},
        "data/present_current_raw": {
            "shape": [num_steps, 16],
            "dtype": "int16",
        },
        "data/goal_position": {"shape": [num_steps, 16], "dtype": "float32"},
        "data/position_error": {"shape": [num_steps, 16], "dtype": "float32"},
        "data/timestamp": {"shape": [num_steps], "dtype": "float64"},
        "data/sample_dt": {"shape": [num_steps], "dtype": "float64"},
        "data/valid": {"shape": [num_steps], "dtype": "bool"},
        "meta/episode_ends": {"shape": [len(bundles)], "dtype": "int64"},
    }
    start = 0
    episodes: list[dict[str, Any]] = []
    for bundle, end in zip(bundles, episode_ends, strict=True):
        end = int(end)
        episodes.append(
            {
                "episode_id": str(bundle.meta["episode_id"]),
                "source": f"{bundle.status}/{bundle.path.name}",
                "status": bundle.status,
                "success": bool(bundle.meta["success"]),
                "start": start,
                "end": end,
                "num_steps": end - start,
                "metadata": dict(bundle.spec.extra),
            }
        )
        start = end
    if start != num_steps:
        raise ValueError("last episode end must equal num_steps")
    manifest = {
        "format": "diffusion_policy_zarr_v2",
        "format_version": 1,
        "dataset_kind": "leap_proprio_grasp",
        "source_schema_version": PROPRIO_SCHEMA_VERSION,
        "task": "grasp",
        "arrays": arrays,
        "observation": {
            "key": "robot_state",
            "dim": PROPRIO_STATE_DIM,
            "names": list(ROBOT_STATE_NAMES),
            "layout": [
                {
                    "field": "actual_position",
                    "slice": [0, 16],
                    "names": list(ACTUAL_POSITION_NAMES),
                },
                {
                    "field": "velocity",
                    "slice": [16, 32],
                    "names": list(FINITE_DIFFERENCE_VELOCITY_NAMES),
                },
                {
                    "field": "present_current_raw",
                    "slice": [32, 48],
                    "names": list(PRESENT_CURRENT_RAW_NAMES),
                },
            ],
        },
        "action": {
            "key": "action",
            "dim": PROPRIO_ACTION_DIM,
            "names": list(ACTION_NAMES),
            "semantics": spec.action_semantics,
            "warning": (
                "A constant close goal teaches a constant position command. "
                "Slow measured closure under changing resistance comes from the "
                "LEAP low-level controller and mechanics; it is not relabelled "
                "as a learned force-control action."
            ),
        },
        "diagnostics": {
            "goal_position_names": list(GOAL_POSITION_NAMES),
            "position_error_names": list(POSITION_ERROR_NAMES),
        },
        "control": {
            "mode": spec.control_mode,
            "sample_period_s": float(spec.sample_period_s),
            "sample_period_tolerance_s": float(spec.sample_period_tolerance_s),
        },
        "dynamics_fingerprint": {
            "sha256": _fingerprint_sha256(fingerprint),
            "fields": fingerprint,
        },
        "units": {
            "actual_position": "radian",
            "velocity": "radian_per_second_finite_difference",
            "present_current_raw": (
                f"{fingerprint['present_current_unit']}; not_force_calibrated"
            ),
            "goal_position": "radian",
            "position_error": "radian_goal_minus_actual",
            "timestamp": "second",
            "sample_dt": "second",
        },
        "timestamp_clock": spec.timestamp_clock,
        "image_observations": False,
        "franka_observations_or_actions": False,
        "filters": {"accepted_only": not bool(include_rejected)},
        "episodes": episodes,
    }
    json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return manifest


def _validate_compatible_specs(bundles: list[ProprioEpisodeBundle]) -> None:
    baseline = bundles[0].spec
    baseline_fingerprint = _bundle_dynamics_fingerprint(bundles[0])
    fields = (
        "sample_period_s",
        "sample_period_tolerance_s",
        "joint_names",
        "control_mode",
        "timestamp_clock",
        "action_semantics",
    )
    for bundle in bundles[1:]:
        mismatched = [
            field
            for field in fields
            if getattr(bundle.spec, field) != getattr(baseline, field)
        ]
        if mismatched:
            raise ProprioExportError(
                f"episode {bundle.path.name} is incompatible in {mismatched}; "
                "export distinct proprio schemas separately"
            )
        fingerprint = _bundle_dynamics_fingerprint(bundle)
        dynamics_mismatches = [
            field
            for field, baseline_value in baseline_fingerprint.items()
            if fingerprint[field] != baseline_value
        ]
        if dynamics_mismatches:
            qualified = [f"spec.extra.{field}" for field in dynamics_mismatches]
            raise ProprioExportError(
                f"episode {bundle.path.name} is incompatible in {qualified}; "
                "mock/real or differing LEAP dynamics must be exported separately"
            )


def _bundle_dynamics_fingerprint(
    bundle: ProprioEpisodeBundle,
) -> dict[str, Any]:
    try:
        return dynamics_fingerprint(bundle.spec.extra)
    except ValueError as exc:
        raise ProprioExportError(
            f"episode {bundle.path.name} has incomplete or invalid dynamics "
            f"metadata: {exc}; recollect the episode or repair metadata from "
            "verified hardware/config records before export"
        ) from exc


def _fingerprint_sha256(fingerprint: dict[str, Any]) -> str:
    canonical = json.dumps(
        fingerprint,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_zarr_v2() -> tuple[Any, Any]:
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ProprioExportError(
            "proprio Zarr export requires zarr<3 and numcodecs"
        ) from exc
    major = int(str(zarr.__version__).split(".", maxsplit=1)[0])
    if major >= 3:
        raise ProprioExportError(
            f"zarr {zarr.__version__} is installed, but this exporter requires zarr<3"
        )
    return zarr, Blosc


def _commit_export(temporary: Path, output: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        backup = output.with_name(f"{output.name}.backup-{uuid.uuid4().hex}")
        os.replace(output, backup)
    try:
        os.replace(temporary, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            manifest,
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
