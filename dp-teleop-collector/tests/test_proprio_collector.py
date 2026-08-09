from __future__ import annotations

import math

import numpy as np
from dp_collector.proprio_collector import ActiveEpisode, _record_resampled_samples
from src.leap_hand_hardware import LeapHandFeedback


class _CaptureWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def append(self, **fields: object) -> int:
        self.rows.append(fields)
        return len(self.rows) - 1


def _active(
    writer: _CaptureWriter,
    *,
    previous_time_s: float = 100.0,
    previous_goal: float = 0.0,
) -> ActiveEpisode:
    feedback = LeapHandFeedback(
        actual_position_rad=np.zeros(16),
        present_velocity_raw=np.zeros(16, dtype=np.int64),
        present_current_raw=np.zeros(16, dtype=np.int64),
        monotonic_s=previous_time_s,
    )
    return ActiveEpisode(
        writer=writer,  # type: ignore[arg-type]
        started_monotonic_s=100.0,
        next_sample_due_s=100.1,
        last_sample_timestamp_s=0.0,
        previous_feedback=feedback,
        previous_goal_position_rad=np.full(16, previous_goal),
    )


def test_due_sample_pairs_pre_action_feedback_with_sent_post_slew_goal() -> None:
    writer = _CaptureWriter()
    active = _active(writer, previous_time_s=100.05, previous_goal=0.6)
    feedback = LeapHandFeedback(
        actual_position_rad=np.full(16, 0.2),
        present_velocity_raw=np.arange(-8, 8),
        present_current_raw=np.arange(-16, 0),
        monotonic_s=100.1,
    )
    goal = np.full(16, 0.8)

    recorded = _record_resampled_samples(
        active,
        feedback=feedback,
        run_start_s=100.0,
        sample_period_s=0.1,
        sample_period_tolerance_s=0.001,
        maximum_feedback_bracket_s=0.075,
        tracking_ready=True,
        goal_position_rad=goal,
        vision_target_rad=np.full(16, 1.2),
    )

    assert recorded == 1
    row = writer.rows[0]
    np.testing.assert_allclose(row["actual_position_rad"], 0.2)
    # The 0.6 command was actually active at the fixed-grid time. The newer
    # 0.8 command is sent only after the current feedback snapshot.
    np.testing.assert_allclose(row["goal_position_rad"], 0.6)
    np.testing.assert_array_equal(row["present_current_raw"], np.arange(-16, 0))
    assert row["valid"] is True
    assert row["invalid_reasons"] == ()


def test_large_feedback_gap_is_resampled_but_marked_invalid() -> None:
    writer = _CaptureWriter()
    active = _active(writer)
    feedback = LeapHandFeedback(
        actual_position_rad=np.zeros(16),
        present_velocity_raw=np.zeros(16, dtype=np.int64),
        present_current_raw=np.zeros(16, dtype=np.int64),
        monotonic_s=100.25,
    )

    recorded = _record_resampled_samples(
        active,
        feedback=feedback,
        run_start_s=100.0,
        sample_period_s=0.1,
        sample_period_tolerance_s=0.001,
        maximum_feedback_bracket_s=0.075,
        tracking_ready=True,
        goal_position_rad=np.zeros(16),
        vision_target_rad=np.zeros(16),
    )

    assert recorded == 2
    assert all(row["valid"] is False for row in writer.rows)
    reasons = set(writer.rows[0]["invalid_reasons"])
    assert "feedback_bracket_too_large" in reasons
    assert active.invalid_steps == 2
    assert math.isclose(active.next_sample_due_s, 100.3, abs_tol=1e-12)


def test_not_due_does_not_duplicate_feedback() -> None:
    writer = _CaptureWriter()
    active = _active(writer)
    feedback = LeapHandFeedback(
        actual_position_rad=np.zeros(16),
        present_velocity_raw=np.zeros(16, dtype=np.int64),
        present_current_raw=np.zeros(16, dtype=np.int64),
        monotonic_s=100.05,
    )

    result = _record_resampled_samples(
        active,
        feedback=feedback,
        run_start_s=100.0,
        sample_period_s=0.1,
        sample_period_tolerance_s=0.001,
        maximum_feedback_bracket_s=0.075,
        tracking_ready=True,
        goal_position_rad=np.zeros(16),
        vision_target_rad=np.zeros(16),
    )

    assert result == 0
    assert writer.rows == []


def test_large_bracket_between_grid_points_stays_latched_until_next_sample() -> None:
    writer = _CaptureWriter()
    active = _active(writer, previous_time_s=100.101)
    active.next_sample_due_s = 100.2
    first = LeapHandFeedback(
        actual_position_rad=np.full(16, 0.1),
        present_velocity_raw=np.zeros(16, dtype=np.int64),
        present_current_raw=np.zeros(16, dtype=np.int64),
        monotonic_s=100.18,
    )

    assert (
        _record_resampled_samples(
            active,
            feedback=first,
            run_start_s=100.0,
            sample_period_s=0.1,
            sample_period_tolerance_s=0.001,
            maximum_feedback_bracket_s=0.075,
            tracking_ready=True,
            goal_position_rad=np.zeros(16),
            vision_target_rad=np.zeros(16),
        )
        == 0
    )
    assert "feedback_bracket_too_large" in active.sticky_invalid_reasons

    second = LeapHandFeedback(
        actual_position_rad=np.full(16, 0.2),
        present_velocity_raw=np.zeros(16, dtype=np.int64),
        present_current_raw=np.zeros(16, dtype=np.int64),
        monotonic_s=100.21,
    )
    assert (
        _record_resampled_samples(
            active,
            feedback=second,
            run_start_s=100.0,
            sample_period_s=0.1,
            sample_period_tolerance_s=0.001,
            maximum_feedback_bracket_s=0.075,
            tracking_ready=True,
            goal_position_rad=np.zeros(16),
            vision_target_rad=np.zeros(16),
        )
        == 1
    )
    assert writer.rows[0]["valid"] is False
    assert "feedback_bracket_too_large" in writer.rows[0]["invalid_reasons"]
    assert active.invalid_steps == 1
