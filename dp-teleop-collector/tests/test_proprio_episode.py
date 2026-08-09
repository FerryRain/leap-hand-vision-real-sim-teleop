from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from dp_collector.proprio_episode import (
    ProprioEpisodeWriter,
    validate_proprio_episode,
)
from dp_collector.proprio_schema import ProprioEpisodeSpec, ProprioStep

JOINT_NAMES = tuple(f"joint_{index}" for index in range(16))


def make_spec(**overrides: object) -> ProprioEpisodeSpec:
    values: dict[str, object] = {
        "sample_period_s": 0.05,
        "sample_period_tolerance_s": 0.01,
        "joint_names": JOINT_NAMES,
        "extra": {"goal_current_raw": 350},
    }
    values.update(overrides)
    return ProprioEpisodeSpec(**values)


def make_writer(root: Path, episode_id: str = "proprio_test") -> ProprioEpisodeWriter:
    return ProprioEpisodeWriter(
        root,
        make_spec(),
        initial_timestamp_s=1.0,
        initial_actual_position_rad=np.zeros(16),
        episode_id=episode_id,
    )


def test_writer_computes_true_finite_difference_and_accepts_without_images(
    tmp_path: Path,
) -> None:
    writer = make_writer(tmp_path)
    actual = np.arange(16, dtype=np.float64) * 0.01
    current = np.arange(-8, 8, dtype=np.int16)
    goal = actual + 0.2
    assert (
        writer.append(
            timestamp_s=1.05,
            actual_position_rad=actual,
            present_current_raw=current,
            goal_position_rad=goal,
        )
        == 0
    )
    destination = writer.accept(notes="successful candy grasp")

    assert destination == tmp_path / "accepted" / "proprio_test"
    assert not (destination / "rgb").exists()
    assert not (destination / "depth").exists()
    payload = json.loads((destination / "steps.jsonl").read_text(encoding="utf-8"))
    np.testing.assert_allclose(payload["actual_position_rad"], actual)
    np.testing.assert_allclose(payload["velocity_rad_s"], actual / 0.05)
    np.testing.assert_array_equal(payload["present_current_raw"], current)
    np.testing.assert_allclose(payload["position_error_rad"], np.full(16, 0.2))
    np.testing.assert_allclose(payload["action"], goal)
    np.testing.assert_allclose(
        payload["robot_state"],
        np.concatenate((actual, actual / 0.05, current.astype(float))),
    )
    assert payload["valid"] is True
    assert validate_proprio_episode(destination).ok


def test_sample_jitter_is_recorded_and_marks_step_invalid(tmp_path: Path) -> None:
    writer = make_writer(tmp_path, "jitter")
    writer.append(
        timestamp_s=1.08,
        actual_position_rad=np.zeros(16),
        present_current_raw=np.zeros(16, dtype=np.int16),
        goal_position_rad=np.ones(16),
    )
    destination = writer.accept()
    payload = json.loads((destination / "steps.jsonl").read_text(encoding="utf-8"))
    assert payload["sample_dt_s"] == pytest.approx(0.08)
    assert payload["valid"] is False
    assert payload["invalid_reasons"] == ["sample_period_out_of_tolerance"]
    assert validate_proprio_episode(destination).ok


def test_measured_next_position_cannot_silently_replace_command_action(
    tmp_path: Path,
) -> None:
    writer = make_writer(tmp_path, "semantics")
    measured = np.full(16, 0.1)
    commanded = np.full(16, 1.4)
    writer.append_step(
        ProprioStep(
            timestamp_s=1.05,
            actual_position_rad=measured,
            present_current_raw=np.full(16, 200),
            goal_position_rad=commanded,
        )
    )
    destination = writer.accept()
    payload = json.loads((destination / "steps.jsonl").read_text(encoding="utf-8"))
    np.testing.assert_allclose(payload["action"], commanded)
    assert not np.allclose(payload["action"], measured)
    assert "post_slew_goal_position" in writer.spec.action_semantics


def test_current_must_be_signed_int16_raw_values(tmp_path: Path) -> None:
    writer = make_writer(tmp_path, "bad_current")
    common = {
        "timestamp_s": 1.05,
        "actual_position_rad": np.zeros(16),
        "goal_position_rad": np.ones(16),
    }
    with pytest.raises(ValueError, match="integers"):
        writer.append(present_current_raw=np.full(16, 1.5), **common)
    with pytest.raises(ValueError, match="signed int16"):
        writer.append(present_current_raw=np.full(16, 40_000), **common)
    assert writer.step_count == 0
    writer.reject("invalid current feedback")


def test_context_exception_leaves_recoverable_partial(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="serial disconnected"):
        with make_writer(tmp_path, "interrupted") as writer:
            writer.append(
                timestamp_s=1.05,
                actual_position_rad=np.zeros(16),
                present_current_raw=np.zeros(16),
                goal_position_rad=np.zeros(16),
            )
            raise RuntimeError("serial disconnected")
    partial = tmp_path / ".partial" / "interrupted"
    meta = json.loads((partial / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "partial"
    assert meta["num_steps"] == 1
    assert validate_proprio_episode(partial).ok


def test_spec_is_grasp_only_and_has_fixed_action_semantics() -> None:
    with pytest.raises(ValueError, match="grasp"):
        make_spec(task="release")
    with pytest.raises(ValueError, match="action_semantics"):
        make_spec(action_semantics="next_measured_position")
    with pytest.raises(ValueError, match="cannot exceed"):
        make_spec(sample_period_tolerance_s=0.06)
