from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from dp_collector.episode import EpisodeWriter, validate_dataset, validate_episode
from dp_collector.schema import EpisodeSpec, Stage, StepRecord, normalize_step


def make_spec(task: str = "grasp", action_space: str = "hand_only") -> EpisodeSpec:
    return EpisodeSpec(
        task=task,
        action_space=action_space,
        robot_state_dim=16,
        image_shape=(8, 10),
        state_age_limits_s={"camera": 0.10, "leap": 0.10},
        robot_state_names=tuple(f"leap_joint_{i}" for i in range(16)),
    )


def make_images(index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.full((8, 10, 3), (10 + index, 40, 200), dtype=np.uint8)
    depth = np.full((8, 10), 500 + index, dtype=np.uint16)
    return rgb, depth


def append_step(writer: EpisodeWriter, index: int = 0, **overrides: object) -> int:
    rgb, depth = make_images(index)
    values = {
        "rgb": rgb,
        "depth": depth,
        "timestamp_s": 100.0 + index * 0.05,
        "robot_state": np.arange(16, dtype=float),
        "action": np.arange(16, dtype=float) / 10,
        "stage": int(Stage.CLOSE),
        "state_ages_s": {"camera": 0.02, "leap": 0.03},
    }
    values.update(overrides)
    return writer.append_frame(**values)


def test_accept_is_atomic_and_images_keep_declared_types(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, make_spec(), episode_id="episode_test")
    assert append_step(writer, 0) == 0
    assert append_step(writer, 1) == 1
    destination = writer.accept(notes="clean grasp")

    assert destination == tmp_path / "accepted" / "episode_test"
    assert not (tmp_path / ".partial" / "episode_test").exists()
    meta = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "accepted"
    assert meta["success"] is True
    assert meta["num_steps"] == 2
    assert (destination / "rgb" / "000000.jpg").read_bytes().startswith(b"\xff\xd8")
    depth = cv2.imread(str(destination / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16
    np.testing.assert_array_equal(depth, np.full((8, 10), 500, dtype=np.uint16))
    assert validate_episode(destination).ok


def test_reject_moves_episode_and_records_reason(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, make_spec("release"), episode_id="bad_release")
    append_step(writer)
    destination = writer.reject("object did not release")
    meta = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
    assert destination.parent.name == "rejected"
    assert meta["success"] is False
    assert meta["rejection_reason"] == "object did not release"
    assert validate_episode(destination).ok


def test_context_exception_leaves_recoverable_partial(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="camera disconnected"):
        with EpisodeWriter(tmp_path, make_spec(), episode_id="interrupted") as writer:
            append_step(writer)
            raise RuntimeError("camera disconnected")
    partial = tmp_path / ".partial" / "interrupted"
    assert partial.is_dir()
    meta = json.loads((partial / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "partial"
    assert meta["num_steps"] == 1
    assert "camera disconnected" in meta["interruption_reason"]
    assert validate_episode(partial).ok


def test_stale_and_explicit_invalid_reasons_are_preserved(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, make_spec(), episode_id="invalid_frame")
    append_step(
        writer,
        state_ages_s={"camera": 0.25, "leap": 0.03},
        invalid_reasons=("tracking_lost", "tracking_lost"),
    )
    destination = writer.accept()
    step = json.loads((destination / "steps.jsonl").read_text(encoding="utf-8"))
    assert step["valid"] is False
    assert step["invalid_reasons"] == ["tracking_lost", "stale:camera"]
    assert validate_episode(destination).ok


def test_writer_rejects_bad_shape_before_committing_a_step(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, make_spec(), episode_id="bad_shape")
    with pytest.raises(ValueError, match="rgb must"):
        append_step(writer, rgb=np.zeros((7, 10, 3), dtype=np.uint8))
    assert writer.step_count == 0
    assert not list((writer.directory / "rgb").iterdir())
    writer.reject("invalid camera frame")


def test_normalize_step_rejects_nonfinite_and_nonmonotonic_state() -> None:
    spec = make_spec()
    step = StepRecord(
        timestamp_s=1.0,
        robot_state=np.zeros(16),
        action=np.zeros(16),
        state_ages_s={"camera": 0.01, "leap": 0.01},
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_step(
            spec,
            step,
            index=1,
            rgb_path="rgb/000001.jpg",
            depth_path="depth/000001.png",
            previous_timestamp_s=1.0,
        )
    bad = StepRecord(
        timestamp_s=2.0,
        robot_state=np.full(16, np.nan),
        action=np.zeros(16),
        state_ages_s={"camera": 0.01, "leap": 0.01},
    )
    with pytest.raises(ValueError, match="non-finite"):
        normalize_step(
            spec,
            bad,
            index=0,
            rgb_path="rgb/000000.jpg",
            depth_path="depth/000000.png",
            previous_timestamp_s=None,
        )


def test_dataset_validator_reports_corrupt_step(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, make_spec(), episode_id="corrupt")
    append_step(writer)
    destination = writer.accept()
    line = json.loads((destination / "steps.jsonl").read_text(encoding="utf-8"))
    line["rgb_path"] = "../outside.jpg"
    (destination / "steps.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")
    report = validate_dataset(tmp_path, include_partial=False)
    assert not report.ok
    assert any("unsafe image path" in error for error in report.episodes[0].errors)
