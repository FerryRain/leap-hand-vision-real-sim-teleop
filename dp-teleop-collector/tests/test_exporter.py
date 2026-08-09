from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from dp_collector.episode import EpisodeWriter
from dp_collector.exporter import (
    DatasetExportError,
    build_manifest,
    collect_episode_bundles,
    export_to_zarr,
)
from dp_collector.schema import EpisodeSpec


def _record(
    root: Path,
    episode_id: str,
    task: str,
    *,
    accepted: bool = True,
    state_dim: int = 4,
) -> None:
    spec = EpisodeSpec(
        task=task,
        action_space="hand_only",
        robot_state_dim=state_dim,
        image_shape=(6, 8),
        state_age_limits_s={"camera": 0.1, "leap": 0.1},
        robot_state_names=tuple(f"state_{i}" for i in range(state_dim)),
    )
    writer = EpisodeWriter(root, spec, episode_id=episode_id)
    for index in range(2):
        writer.append_frame(
            rgb=np.full((6, 8, 3), [20, 40 + index, 180], dtype=np.uint8),
            depth=np.full((6, 8), 600 + index, dtype=np.uint16),
            timestamp_s=5.0 + index * 0.1,
            robot_state=np.arange(state_dim, dtype=float),
            action=np.arange(16, dtype=float),
            state_ages_s={"camera": 0.01, "leap": 0.02},
            valid=index == 0,
            invalid_reasons=() if index == 0 else ("tracking_lost",),
        )
    if accepted:
        writer.accept()
    else:
        writer.reject("test rejection")


def test_collector_defaults_to_accepted_and_can_filter_task(tmp_path: Path) -> None:
    _record(tmp_path, "b_release", "release")
    _record(tmp_path, "a_grasp", "grasp")
    _record(tmp_path, "c_rejected", "grasp", accepted=False)
    bundles = collect_episode_bundles(tmp_path, deep_validation=True)
    assert [bundle.path.name for bundle in bundles] == ["a_grasp", "b_release"]
    assert [bundle.spec.task for bundle in bundles] == ["grasp", "release"]

    grasp = collect_episode_bundles(tmp_path, task="grasp", valid_only=True)
    assert len(grasp) == 1
    assert len(grasp[0].steps) == 1
    with_rejected = collect_episode_bundles(
        tmp_path, include_rejected=True, task="grasp"
    )
    assert [bundle.path.name for bundle in with_rejected] == ["a_grasp", "c_rejected"]


def test_incompatible_state_dimensions_cannot_be_mixed(tmp_path: Path) -> None:
    _record(tmp_path, "episode_a", "grasp", state_dim=4)
    _record(tmp_path, "episode_b", "release", state_dim=5)
    with pytest.raises(DatasetExportError, match="incompatible"):
        collect_episode_bundles(tmp_path)


def test_manifest_is_human_readable_and_uses_relative_sources(tmp_path: Path) -> None:
    _record(tmp_path, "episode_grasp", "grasp")
    bundles = collect_episode_bundles(tmp_path)
    manifest = build_manifest(
        bundles,
        episode_ends=[2],
        num_steps=2,
        include_rejected=False,
        task_filter=None,
        valid_only=False,
    )
    assert manifest["arrays"]["data/camera_0"] == {
        "shape": [2, 6, 8, 3],
        "dtype": "uint8",
    }
    assert manifest["action"]["names"][-1] == "leap_joint_15_target"
    assert manifest["episodes"] == [
        {
            "episode_id": "episode_grasp",
            "source": "accepted/episode_grasp",
            "task": "grasp",
            "task_index": 0,
            "success": True,
            "status": "accepted",
            "start": 0,
            "end": 2,
            "num_steps": 2,
            "metadata": {},
        }
    ]
    serialized = json.dumps(manifest, allow_nan=False, sort_keys=True)
    assert str(tmp_path) not in serialized


@pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None, reason="zarr<3 not installed"
)
def test_zarr_v2_export_contains_canonical_arrays(tmp_path: Path) -> None:
    import zarr

    _record(tmp_path, "episode_grasp", "grasp")
    _record(tmp_path, "episode_release", "release")
    output = tmp_path / "training.zarr"
    summary = export_to_zarr(tmp_path, output)
    assert summary.num_episodes == 2
    assert summary.num_steps == 4

    root = zarr.open_group(str(output), mode="r")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["arrays"]["data/action"]["shape"] == [4, 16]
    assert [episode["end"] for episode in manifest["episodes"]] == [2, 4]
    assert root["data/camera_0"].shape == (4, 6, 8, 3)
    assert root["data/depth_0"].dtype == np.dtype("uint16")
    assert root["data/robot_state"].shape == (4, 4)
    assert root["data/action"].shape == (4, 16)
    assert root["data/task_onehot"][:].tolist() == [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    assert root["data/valid"][:].tolist() == [True, False, True, False]
    assert root["meta/episode_ends"][:].tolist() == [2, 4]
    assert root["meta/task"][:].tolist() == [0, 1]
    assert root["meta/success"][:].tolist() == [True, True]
