from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from dp_collector.queued_writer import (
    EpisodeQueueFullError,
    EpisodeWorkerError,
    QueuedEpisodeWriter,
)
from dp_collector.schema import EpisodeSpec, StepRecord


def make_spec() -> EpisodeSpec:
    return EpisodeSpec(
        task="grasp",
        action_space="hand_only",
        robot_state_dim=16,
        image_shape=(8, 10),
        state_age_limits_s={"camera": 0.10, "leap": 0.10},
        robot_state_names=tuple(f"leap_joint_{index}" for index in range(16)),
    )


def make_step(index: int = 0) -> tuple[StepRecord, np.ndarray, np.ndarray]:
    step = StepRecord(
        timestamp_s=100.0 + 0.05 * index,
        robot_state=np.arange(16, dtype=np.float32),
        action=np.arange(16, dtype=np.float32) / 10.0,
        state_ages_s={"camera": 0.01, "leap": 0.02},
    )
    rgb = np.full((8, 10, 3), index, dtype=np.uint8)
    depth = np.full((8, 10), 500 + index, dtype=np.uint16)
    return step, rgb, depth


class StubEpisodeWriter:
    def __init__(self, tmp_path: Path) -> None:
        self.spec = make_spec()
        self.episode_id = "stub_episode"
        self.directory = tmp_path / ".partial" / self.episode_id
        self.active = True
        self.step_count = 0
        self.append_started = threading.Event()
        self.allow_append = threading.Event()
        self.allow_append.set()
        self.append_error: BaseException | None = None
        self.operations: list[str] = []
        self.frames: list[tuple[StepRecord, np.ndarray, np.ndarray]] = []

    def append(
        self,
        step: StepRecord,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> int:
        self.append_started.set()
        if not self.allow_append.wait(timeout=5.0):
            raise TimeoutError("test did not release blocked writer")
        if self.append_error is not None:
            raise self.append_error
        index = self.step_count
        self.frames.append((step, rgb.copy(), depth.copy()))
        self.step_count += 1
        self.operations.append(f"append:{index}")
        return index

    def accept(self, *, notes: str | None = None) -> Path:
        self.operations.append("accept")
        self.active = False
        self.directory = self.directory.parents[1] / "accepted" / self.episode_id
        return self.directory

    def reject(self, reason: str, *, notes: str | None = None) -> Path:
        self.operations.append(f"reject:{reason}")
        self.active = False
        self.directory = self.directory.parents[1] / "rejected" / self.episode_id
        return self.directory

    def close_partial(self, *, reason: str | None = None) -> Path:
        self.operations.append(f"close_partial:{reason}")
        self.active = False
        return self.directory


def test_slow_disk_writer_never_blocks_frame_submission(tmp_path: Path) -> None:
    backend = StubEpisodeWriter(tmp_path)
    backend.allow_append.clear()
    writer = QueuedEpisodeWriter(backend, max_pending_frames=2)  # type: ignore[arg-type]
    step, rgb, depth = make_step()

    try:
        assert writer.append(step, rgb=rgb, depth=depth) == 0
        assert backend.append_started.wait(timeout=1.0)
        assert writer.pending_count == 1
        assert writer.step_count == 1
        assert writer.committed_step_count == 0
    finally:
        backend.allow_append.set()
    assert writer.close_partial(reason="test_complete") == backend.directory
    assert backend.operations == ["append:0", "close_partial:test_complete"]


def test_in_flight_frame_counts_toward_bound_and_overflow_latches(
    tmp_path: Path,
) -> None:
    backend = StubEpisodeWriter(tmp_path)
    backend.allow_append.clear()
    writer = QueuedEpisodeWriter(backend, max_pending_frames=1)  # type: ignore[arg-type]
    step, rgb, depth = make_step()

    writer.append(step, rgb=rgb, depth=depth)
    assert backend.append_started.wait(timeout=1.0)
    with pytest.raises(EpisodeQueueFullError, match="queue is full"):
        writer.append(step, rgb=rgb, depth=depth)
    with pytest.raises(EpisodeQueueFullError):
        writer.raise_if_failed()

    backend.allow_append.set()
    with pytest.raises(EpisodeQueueFullError):
        writer.close_partial(reason="overload")
    assert not writer.active
    assert backend.operations[0] == "append:0"
    assert backend.operations[1].startswith("close_partial:overload;")


def test_worker_error_is_propagated_and_failed_episode_cannot_be_accepted(
    tmp_path: Path,
) -> None:
    backend = StubEpisodeWriter(tmp_path)
    backend.append_error = OSError("disk offline")
    writer = QueuedEpisodeWriter(backend, max_pending_frames=2)  # type: ignore[arg-type]
    step, rgb, depth = make_step()

    writer.append(step, rgb=rgb, depth=depth)
    with pytest.raises(EpisodeWorkerError, match="disk offline") as caught:
        writer.accept()

    assert isinstance(caught.value.__cause__, OSError)
    assert "accept" not in backend.operations
    assert len(backend.operations) == 1
    assert backend.operations[0].startswith("close_partial:background persistence")
    assert not writer.active


def test_accept_waits_for_drain_before_finalizing(tmp_path: Path) -> None:
    backend = StubEpisodeWriter(tmp_path)
    backend.allow_append.clear()
    writer = QueuedEpisodeWriter(backend, max_pending_frames=2)  # type: ignore[arg-type]
    step, rgb, depth = make_step()
    writer.append(step, rgb=rgb, depth=depth)
    assert backend.append_started.wait(timeout=1.0)

    result: list[Path] = []
    errors: list[BaseException] = []

    def accept_in_background() -> None:
        try:
            result.append(writer.accept(notes="good"))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    finalizer = threading.Thread(target=accept_in_background)
    finalizer.start()
    assert finalizer.is_alive()
    assert backend.operations == []

    backend.allow_append.set()
    finalizer.join(timeout=2.0)
    assert not finalizer.is_alive()
    assert errors == []
    assert result == [backend.directory]
    assert backend.operations == ["append:0", "accept"]


@pytest.mark.parametrize("operation", ["reject", "close_partial"])
def test_reject_and_abort_drain_before_closing(
    tmp_path: Path,
    operation: str,
) -> None:
    backend = StubEpisodeWriter(tmp_path)
    writer = QueuedEpisodeWriter(backend, max_pending_frames=2)  # type: ignore[arg-type]
    step, rgb, depth = make_step()
    writer.append(step, rgb=rgb, depth=depth)

    if operation == "reject":
        writer.reject("operator_rejected")
        assert backend.operations == ["append:0", "reject:operator_rejected"]
    else:
        writer.close_partial(reason="operator_stop")
        assert backend.operations == ["append:0", "close_partial:operator_stop"]


def test_append_frame_snapshots_mutable_inputs(tmp_path: Path) -> None:
    backend = StubEpisodeWriter(tmp_path)
    backend.allow_append.clear()
    writer = QueuedEpisodeWriter(backend, max_pending_frames=2)  # type: ignore[arg-type]
    _, rgb, depth = make_step()
    state = np.arange(16, dtype=np.float32)
    action = np.arange(16, dtype=np.float32)

    writer.append_frame(
        rgb=rgb,
        depth=depth,
        timestamp_s=100.0,
        robot_state=state,
        action=action,
        state_ages_s={"camera": 0.01, "leap": 0.02},
    )
    assert backend.append_started.wait(timeout=1.0)
    rgb.fill(255)
    depth.fill(65535)
    state.fill(-1)
    action.fill(-1)
    backend.allow_append.set()
    writer.drain()
    writer.close_partial(reason="test_complete")
    saved_step, saved_rgb, saved_depth = backend.frames[0]
    np.testing.assert_array_equal(saved_rgb, np.zeros((8, 10, 3), dtype=np.uint8))
    np.testing.assert_array_equal(saved_depth, np.full((8, 10), 500, dtype=np.uint16))
    np.testing.assert_array_equal(saved_step.robot_state, np.arange(16))
    np.testing.assert_array_equal(saved_step.action, np.arange(16))
