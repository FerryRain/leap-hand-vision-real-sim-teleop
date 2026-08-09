"""Deterministic conversion from accepted episodes to Diffusion Policy Zarr v2."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .episode import read_episode_meta, read_episode_steps, validate_episode
from .schema import SCHEMA_VERSION, TASK_TO_INDEX, TASKS, EpisodeSpec


class DatasetExportError(RuntimeError):
    """Raised when source episodes cannot form one coherent training dataset."""


@dataclass(frozen=True)
class EpisodeBundle:
    path: Path
    status: str
    meta: dict[str, Any]
    spec: EpisodeSpec
    steps: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExportSummary:
    output_path: Path
    num_episodes: int
    num_steps: int
    task_counts: dict[str, int]
    action_space: str
    action_dim: int
    robot_state_dim: int
    image_shape: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "task_counts": dict(self.task_counts),
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "robot_state_dim": self.robot_state_dim,
            "image_shape": list(self.image_shape),
        }


def collect_episode_bundles(
    dataset_root: str | Path,
    *,
    include_rejected: bool = False,
    task: str | None = None,
    valid_only: bool = False,
    deep_validation: bool = True,
) -> list[EpisodeBundle]:
    """Load, validate, filter, and deterministically sort source episodes."""

    root = Path(dataset_root).absolute()
    if task is not None:
        task = str(task).lower()
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
    statuses = ["accepted"] + (["rejected"] if include_rejected else [])
    paths: list[Path] = []
    for status in statuses:
        parent = root / status
        if parent.is_dir():
            paths.extend(path for path in parent.iterdir() if path.is_dir())
    paths.sort(key=lambda path: (path.name, path.parent.name))

    bundles: list[EpisodeBundle] = []
    for path in paths:
        report = validate_episode(path, deep=deep_validation)
        if not report.ok:
            details = "; ".join(report.errors)
            raise DatasetExportError(f"invalid source episode {path}: {details}")
        meta = read_episode_meta(path)
        spec = EpisodeSpec.from_dict(meta["spec"])
        if task is not None and spec.task != task:
            continue
        steps = read_episode_steps(path)
        if valid_only:
            steps = [step for step in steps if bool(step["valid"])]
        if not steps:
            continue
        bundles.append(
            EpisodeBundle(
                path=path,
                status=path.parent.name,
                meta=meta,
                spec=spec,
                steps=tuple(steps),
            )
        )
    if not bundles:
        qualifier = f" for task={task!r}" if task else ""
        raise DatasetExportError(f"no eligible episodes found{qualifier}")
    _validate_compatible_specs(bundles)
    return bundles


def export_to_zarr(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    include_rejected: bool = False,
    task: str | None = None,
    valid_only: bool = False,
    deep_validation: bool = True,
    chunk_length: int = 64,
    overwrite: bool = False,
) -> ExportSummary:
    """Create a canonical Zarr v2 store consumable by Diffusion Policy.

    Accepted episodes are the default and rejected episodes are opt-in.  The
    store is first built in a sibling ``.partial-*`` directory and committed by
    directory rename, so an interruption cannot masquerade as a complete
    export.
    """

    if int(chunk_length) <= 0:
        raise ValueError("chunk_length must be positive")
    source_root = Path(dataset_root).absolute()
    output = Path(output_path).absolute()
    if output == source_root:
        raise ValueError("output_path cannot be the source dataset root")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use overwrite=True): {output}")

    bundles = collect_episode_bundles(
        source_root,
        include_rejected=include_rejected,
        task=task,
        valid_only=valid_only,
        deep_validation=deep_validation,
    )
    zarr, Blosc = _load_zarr_v2()
    first = bundles[0].spec
    height, width = first.image_shape
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

        camera = data.create_dataset(
            "camera_0",
            shape=(num_steps, height, width, 3),
            chunks=(1, height, width, 3),
            dtype="u1",
            compressor=compressor,
        )
        depth = data.create_dataset(
            "depth_0",
            shape=(num_steps, height, width),
            chunks=(1, height, width),
            dtype="<u2",
            compressor=compressor,
        )
        robot_state = data.create_dataset(
            "robot_state",
            shape=(num_steps, first.robot_state_dim),
            chunks=(numeric_chunk, first.robot_state_dim),
            dtype="<f4",
            compressor=compressor,
        )
        action = data.create_dataset(
            "action",
            shape=(num_steps, first.action_dim),
            chunks=(numeric_chunk, first.action_dim),
            dtype="<f4",
            compressor=compressor,
        )
        task_onehot = data.create_dataset(
            "task_onehot",
            shape=(num_steps, len(TASKS)),
            chunks=(numeric_chunk, len(TASKS)),
            dtype="<f4",
            compressor=compressor,
        )
        stage = data.create_dataset(
            "stage",
            shape=(num_steps,),
            chunks=(numeric_chunk,),
            dtype="<i2",
            compressor=compressor,
        )
        timestamp = data.create_dataset(
            "timestamp",
            shape=(num_steps,),
            chunks=(numeric_chunk,),
            dtype="<f8",
            compressor=compressor,
        )
        valid = data.create_dataset(
            "valid",
            shape=(num_steps,),
            chunks=(numeric_chunk,),
            dtype="bool",
            compressor=compressor,
        )

        episode_ends: list[int] = []
        episode_tasks: list[int] = []
        episode_success: list[bool] = []
        task_counts = {name: 0 for name in TASKS}
        offset = 0
        for bundle in bundles:
            task_index = TASK_TO_INDEX[bundle.spec.task]
            onehot = np.zeros(len(TASKS), dtype=np.float32)
            onehot[task_index] = 1.0
            for payload in bundle.steps:
                rgb, depth_image = _decode_images(
                    bundle.path, payload, first.image_shape
                )
                camera[offset] = rgb
                depth[offset] = depth_image
                robot_state[offset] = np.asarray(
                    payload["robot_state"], dtype=np.float32
                )
                action[offset] = np.asarray(payload["action"], dtype=np.float32)
                task_onehot[offset] = onehot
                stage[offset] = int(payload["stage"])
                timestamp[offset] = float(payload["timestamp_s"])
                valid[offset] = bool(payload["valid"])
                offset += 1
            episode_ends.append(offset)
            episode_tasks.append(task_index)
            episode_success.append(bool(bundle.meta["success"]))
            task_counts[bundle.spec.task] += 1

        meta_group.create_dataset(
            "episode_ends",
            data=np.asarray(episode_ends, dtype=np.int64),
            chunks=(min(numeric_chunk, len(episode_ends)),),
            compressor=compressor,
        )
        meta_group.create_dataset(
            "task",
            data=np.asarray(episode_tasks, dtype=np.int8),
            chunks=(min(numeric_chunk, len(episode_tasks)),),
            compressor=compressor,
        )
        meta_group.create_dataset(
            "success",
            data=np.asarray(episode_success, dtype=np.bool_),
            chunks=(min(numeric_chunk, len(episode_success)),),
            compressor=compressor,
        )

        root.attrs.update(
            {
                "format": "diffusion_policy_zarr_v2",
                "source_schema_version": SCHEMA_VERSION,
                "task_encoding": {name: index for index, name in enumerate(TASKS)},
                "episode_ids": [bundle.meta["episode_id"] for bundle in bundles],
                "episode_status": [bundle.status for bundle in bundles],
                "action_space": first.action_space,
                "action_names": list(first.action_names),
                "robot_state_names": list(first.robot_state_names),
                "camera_name": first.camera_name,
                "timestamp_clock": first.timestamp_clock,
                "valid_only": bool(valid_only),
            }
        )
        if hasattr(store, "close"):
            store.close()
        manifest = build_manifest(
            bundles,
            episode_ends=episode_ends,
            num_steps=num_steps,
            include_rejected=include_rejected,
            task_filter=task,
            valid_only=valid_only,
        )
        _write_manifest(temporary / "manifest.json", manifest)
        _commit_export(temporary, output, overwrite=overwrite)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return ExportSummary(
        output_path=output,
        num_episodes=len(bundles),
        num_steps=num_steps,
        task_counts={
            task_name: count for task_name, count in task_counts.items() if count
        },
        action_space=first.action_space,
        action_dim=first.action_dim,
        robot_state_dim=first.robot_state_dim,
        image_shape=first.image_shape,
    )


def _validate_compatible_specs(bundles: list[EpisodeBundle]) -> None:
    baseline = bundles[0].spec
    fields = ("action_space", "action_dim", "robot_state_dim", "image_shape")
    for bundle in bundles[1:]:
        mismatched = [
            field
            for field in fields
            if getattr(bundle.spec, field) != getattr(baseline, field)
        ]
        if mismatched:
            raise DatasetExportError(
                f"episode {bundle.path.name} is incompatible in {mismatched}; "
                "export different action/state schemas separately"
            )
        if bundle.spec.robot_state_names != baseline.robot_state_names:
            raise DatasetExportError(
                f"episode {bundle.path.name} has different robot_state_names"
            )
        if bundle.spec.action_names != baseline.action_names:
            raise DatasetExportError(
                f"episode {bundle.path.name} has different action_names"
            )


def build_manifest(
    bundles: list[EpisodeBundle],
    *,
    episode_ends: list[int],
    num_steps: int,
    include_rejected: bool,
    task_filter: str | None,
    valid_only: bool,
) -> dict[str, Any]:
    """Build the deterministic, human-readable Zarr companion manifest."""

    if not bundles:
        raise ValueError("manifest requires at least one episode")
    if len(episode_ends) != len(bundles):
        raise ValueError("episode_ends length must match bundles")
    spec = bundles[0].spec
    height, width = spec.image_shape
    arrays = {
        "data/camera_0": {"shape": [num_steps, height, width, 3], "dtype": "uint8"},
        "data/depth_0": {"shape": [num_steps, height, width], "dtype": "uint16"},
        "data/robot_state": {
            "shape": [num_steps, spec.robot_state_dim],
            "dtype": "float32",
        },
        "data/action": {"shape": [num_steps, spec.action_dim], "dtype": "float32"},
        "data/task_onehot": {"shape": [num_steps, len(TASKS)], "dtype": "float32"},
        "data/stage": {"shape": [num_steps], "dtype": "int16"},
        "data/timestamp": {"shape": [num_steps], "dtype": "float64"},
        "data/valid": {"shape": [num_steps], "dtype": "bool"},
        "meta/episode_ends": {"shape": [len(bundles)], "dtype": "int64"},
        "meta/task": {"shape": [len(bundles)], "dtype": "int8"},
        "meta/success": {"shape": [len(bundles)], "dtype": "bool"},
    }
    start = 0
    episodes: list[dict[str, Any]] = []
    for bundle, end in zip(bundles, episode_ends, strict=True):
        end = int(end)
        episodes.append(
            {
                "episode_id": str(bundle.meta["episode_id"]),
                "source": f"{bundle.status}/{bundle.path.name}",
                "task": bundle.spec.task,
                "task_index": TASK_TO_INDEX[bundle.spec.task],
                "success": bool(bundle.meta["success"]),
                "status": bundle.status,
                "start": start,
                "end": end,
                "num_steps": end - start,
                # Keep camera/depth calibration, server safety limits, config
                # hashes and Git provenance with the standalone Zarr artifact.
                "metadata": dict(bundle.spec.extra),
            }
        )
        start = end
    if start != num_steps:
        raise ValueError("last episode end must equal num_steps")
    manifest = {
        "format": "diffusion_policy_zarr_v2",
        "format_version": 1,
        "source_schema_version": SCHEMA_VERSION,
        "arrays": arrays,
        "action": {
            "space": spec.action_space,
            "dim": spec.action_dim,
            "names": list(spec.action_names),
        },
        "robot_state": {
            "dim": spec.robot_state_dim,
            "names": list(spec.robot_state_names),
        },
        "camera": {
            "name": spec.camera_name,
            "image_shape": [height, width],
            "rgb_color_order": "RGB",
            "depth_dtype": "uint16",
        },
        "timestamp_clock": spec.timestamp_clock,
        "task_encoding": {name: index for index, name in enumerate(TASKS)},
        "filters": {
            "accepted_only": not bool(include_rejected),
            "task": task_filter,
            "valid_only": bool(valid_only),
        },
        "episodes": episodes,
    }
    # This check is intentional: it prevents NaN/Infinity or an accidental
    # non-JSON object from entering the training artifact.
    json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return manifest


def _decode_images(
    episode_dir: Path,
    payload: dict[str, Any],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rgb_bytes = (episode_dir / Path(*str(payload["rgb_path"]).split("/"))).read_bytes()
    bgr = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    depth_bytes = (
        episode_dir / Path(*str(payload["depth_path"]).split("/"))
    ).read_bytes()
    depth = cv2.imdecode(
        np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if bgr is None or bgr.shape != image_shape + (3,) or bgr.dtype != np.uint8:
        raise DatasetExportError("validated RGB image changed or cannot be decoded")
    if depth is None or depth.shape != image_shape or depth.dtype != np.uint16:
        raise DatasetExportError("validated depth image changed or cannot be decoded")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth


def _load_zarr_v2() -> tuple[Any, Any]:
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise DatasetExportError(
            "Zarr export requires 'zarr<3' and 'numcodecs'; "
            "install the collector requirements"
        ) from exc
    major = int(str(zarr.__version__).split(".", maxsplit=1)[0])
    if major >= 3:
        raise DatasetExportError(
            f"zarr {zarr.__version__} is installed, but this deterministic "
            "exporter requires zarr<3"
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


def summary_as_json(summary: ExportSummary) -> str:
    return json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
