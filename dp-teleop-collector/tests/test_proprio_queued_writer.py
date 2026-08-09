from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from dp_collector.proprio_episode import ProprioEpisodeWriter
from dp_collector.proprio_queued_writer import (
    ProprioQueueFullError,
    ProprioWorkerError,
    QueuedProprioEpisodeWriter,
)
from dp_collector.proprio_schema import ProprioEpisodeSpec


def make_backend(root: Path, episode_id: str) -> ProprioEpisodeWriter:
    spec = ProprioEpisodeSpec(
        sample_period_s=0.05,
        sample_period_tolerance_s=0.01,
        joint_names=tuple(f"joint_{index}" for index in range(16)),
    )
    return ProprioEpisodeWriter(
        root,
        spec,
        initial_timestamp_s=0.0,
        initial_actual_position_rad=np.zeros(16),
        episode_id=episode_id,
    )


def append(writer: QueuedProprioEpisodeWriter, index: int = 1) -> int:
    return writer.append(
        timestamp_s=0.05 * index,
        actual_position_rad=np.full(16, 0.01 * index),
        present_current_raw=np.arange(-8, 8),
        goal_position_rad=np.full(16, 1.2),
        extra={"index": index},
    )


def test_successful_queue_drains_before_atomic_accept(tmp_path: Path) -> None:
    queued = QueuedProprioEpisodeWriter(
        make_backend(tmp_path, "queued_ok"), max_pending_steps=4
    )
    assert append(queued, 1) == 0
    assert append(queued, 2) == 1
    destination = queued.accept()
    assert destination == tmp_path / "accepted" / "queued_ok"
    assert queued.committed_step_count == 2
    assert len((destination / "steps.jsonl").read_text().splitlines()) == 2


def test_append_snapshots_inputs_before_returning(tmp_path: Path) -> None:
    queued = QueuedProprioEpisodeWriter(
        make_backend(tmp_path, "snapshot"), max_pending_steps=2
    )
    actual = np.full(16, 0.1)
    current = np.arange(-8, 8)
    goal = np.full(16, 1.2)
    queued.append(
        timestamp_s=0.05,
        actual_position_rad=actual,
        present_current_raw=current,
        goal_position_rad=goal,
    )
    actual.fill(9.0)
    current.fill(999)
    goal.fill(-9.0)
    destination = queued.accept()
    row = json.loads((destination / "steps.jsonl").read_text(encoding="utf-8"))
    np.testing.assert_allclose(row["actual_position_rad"], np.full(16, 0.1))
    np.testing.assert_array_equal(row["present_current_raw"], np.arange(-8, 8))
    np.testing.assert_allclose(row["action"], np.full(16, 1.2))


def test_queue_overload_is_latched_and_accept_leaves_partial(tmp_path: Path) -> None:
    backend = make_backend(tmp_path, "queue_full")
    original_append = backend.append_step
    entered = threading.Event()
    release = threading.Event()

    def slow_append(step: object) -> int:
        entered.set()
        assert release.wait(timeout=2.0)
        return original_append(step)  # type: ignore[arg-type]

    backend.append_step = slow_append  # type: ignore[method-assign]
    queued = QueuedProprioEpisodeWriter(backend, max_pending_steps=1)
    started = time.monotonic()
    append(queued, 1)
    assert time.monotonic() - started < 0.2
    assert entered.wait(timeout=1.0)
    with pytest.raises(ProprioQueueFullError):
        append(queued, 2)
    release.set()
    with pytest.raises(ProprioQueueFullError):
        queued.accept()
    assert (tmp_path / ".partial" / "queue_full").is_dir()
    assert not (tmp_path / "accepted" / "queue_full").exists()


def test_worker_error_prevents_accept_and_retains_partial(tmp_path: Path) -> None:
    backend = make_backend(tmp_path, "worker_error")

    def failing_append(step: object) -> int:
        raise OSError("disk unavailable")

    backend.append_step = failing_append  # type: ignore[method-assign]
    queued = QueuedProprioEpisodeWriter(backend, max_pending_steps=2)
    append(queued, 1)
    with pytest.raises(ProprioWorkerError, match="disk unavailable"):
        queued.accept()
    assert (tmp_path / ".partial" / "worker_error").is_dir()
    assert not (tmp_path / "accepted" / "worker_error").exists()
