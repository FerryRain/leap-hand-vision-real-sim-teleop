"""Bounded background persistence for teleoperation episode frames.

The control loop must never wait for JPEG/PNG encoding or filesystem syncs.
``QueuedEpisodeWriter`` therefore owns one worker thread and bounds the total
number of frames retained by the persistence pipeline, including the frame
currently being written.  Queue overload and worker failures are latched: the
producer sees them immediately (or on its next ``raise_if_failed`` call), and a
failed episode can never be accepted.
"""

from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .episode import EpisodeWriter
from .schema import StepRecord, validate_image_arrays


class EpisodePersistenceError(RuntimeError):
    """Base class for a latched background-persistence failure."""


class EpisodeQueueFullError(EpisodePersistenceError):
    """The bounded frame pipeline cannot accept another frame."""


class EpisodeWorkerError(EpisodePersistenceError):
    """The background writer failed while committing a frame."""


@dataclass(frozen=True)
class _AppendJob:
    index: int
    step: StepRecord
    rgb: np.ndarray
    depth: np.ndarray


_STOP = object()


class QueuedEpisodeWriter:
    """Persist frames off the control thread with a strict memory bound.

    ``max_pending_frames`` counts both queued frames and the frame currently
    being encoded/fsynced.  ``append`` and ``append_frame`` only validate and
    snapshot their inputs before a non-blocking enqueue; they never wait for the
    worker.  Call ``raise_if_failed`` once per control iteration so an
    asynchronous disk error enters the robot's fail-closed stop path promptly.

    Finalizers drain every submitted job.  If draining discovers a worker or
    overload failure, the underlying episode is safely closed as ``.partial``
    and the failure is propagated instead of accepting/rejecting incomplete
    data.
    """

    def __init__(
        self,
        writer: EpisodeWriter,
        *,
        max_pending_frames: int = 4,
    ) -> None:
        if int(max_pending_frames) != max_pending_frames or max_pending_frames < 1:
            raise ValueError("max_pending_frames must be a positive integer")
        if not writer.active:
            raise ValueError("cannot queue writes for an inactive episode writer")

        self._writer = writer
        self._max_pending_frames = int(max_pending_frames)
        self._queue: queue.Queue[_AppendJob | object] = queue.Queue(
            maxsize=self._max_pending_frames
        )
        self._condition = threading.Condition()
        self._outstanding = 0
        self._submitted_count = int(writer.step_count)
        self._failure: EpisodePersistenceError | None = None
        self._state = "open"
        self._worker_stopped = False
        self._worker = threading.Thread(
            target=self._worker_main,
            name=f"episode-writer-{writer.episode_id}",
            daemon=True,
        )
        self._worker.start()

    @property
    def episode_id(self) -> str:
        return self._writer.episode_id

    @property
    def directory(self) -> Path:
        return self._writer.directory

    @property
    def spec(self) -> Any:
        return self._writer.spec

    @property
    def step_count(self) -> int:
        """Number of frames successfully submitted to the bounded pipeline."""

        with self._condition:
            return self._submitted_count

    @property
    def committed_step_count(self) -> int:
        """Number of frames that have reached ``steps.jsonl`` on disk."""

        return int(self._writer.step_count)

    @property
    def pending_count(self) -> int:
        """Number of queued or currently-writing frames."""

        with self._condition:
            return self._outstanding

    @property
    def max_pending_frames(self) -> int:
        return self._max_pending_frames

    @property
    def active(self) -> bool:
        with self._condition:
            return self._state == "open" and self._writer.active

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
        """Snapshot and enqueue one synchronized frame without disk waiting."""

        step = StepRecord(
            timestamp_s=float(timestamp_s),
            robot_state=np.array(robot_state, copy=True),
            action=np.array(action, copy=True),
            stage=int(stage),
            state_ages_s=dict(state_ages_s),
            valid=bool(valid),
            invalid_reasons=tuple(invalid_reasons),
            extra=copy.deepcopy(dict(extra or {})),
        )
        return self.append(step, rgb=rgb, depth=depth)

    def append(self, step: StepRecord, *, rgb: np.ndarray, depth: np.ndarray) -> int:
        """Snapshot and enqueue a :class:`StepRecord` without blocking."""

        rgb_array, depth_array = validate_image_arrays(self._writer.spec, rgb, depth)
        snapshot = StepRecord(
            timestamp_s=float(step.timestamp_s),
            robot_state=np.array(step.robot_state, copy=True),
            action=np.array(step.action, copy=True),
            stage=int(step.stage),
            state_ages_s=dict(step.state_ages_s),
            valid=bool(step.valid),
            invalid_reasons=tuple(step.invalid_reasons),
            extra=copy.deepcopy(dict(step.extra)),
        )

        with self._condition:
            self._require_open_locked()
            self._raise_if_failed_locked()
            if self._outstanding >= self._max_pending_frames:
                failure = EpisodeQueueFullError(
                    "episode persistence queue is full "
                    f"({self._outstanding}/{self._max_pending_frames} frames); "
                    "collection must stop to avoid dropping a control sample"
                )
                self._latch_failure_locked(failure)
                raise failure

            index = self._submitted_count
            job = _AppendJob(
                index=index,
                step=snapshot,
                rgb=np.array(rgb_array, copy=True, order="C"),
                depth=np.array(depth_array, copy=True, order="C"),
            )
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:  # Defensive: the count check is authoritative.
                failure = EpisodeQueueFullError(
                    "episode persistence queue became full during enqueue"
                )
                self._latch_failure_locked(failure)
                raise failure from exc
            self._outstanding += 1
            self._submitted_count += 1
            return index

    def raise_if_failed(self) -> None:
        """Propagate a latched overload or worker error on the caller thread."""

        with self._condition:
            self._raise_if_failed_locked()

    # A descriptive alias for callers that prefer a health-check name.
    check_health = raise_if_failed

    def drain(self) -> None:
        """Wait for all submitted writes, then propagate any worker failure."""

        with self._condition:
            while self._outstanding:
                self._condition.wait()
            self._raise_if_failed_locked()

    def accept(self, *, notes: str | None = None) -> Path:
        """Drain and atomically accept, or leave a failed episode partial."""

        return self._finalize("accept", notes=notes)

    def reject(self, reason: str, *, notes: str | None = None) -> Path:
        """Drain and reject, or leave a failed episode safely partial."""

        clean_reason = str(reason).strip()
        if not clean_reason:
            raise ValueError("a non-empty rejection reason is required")
        return self._finalize("reject", reason=clean_reason, notes=notes)

    def close_partial(self, *, reason: str | None = None) -> Path:
        """Drain and close an interrupted episode, propagating write failures."""

        with self._condition:
            if self._state == "closed":
                return self._writer.directory
            self._begin_finalization_locked()

        drain_error: EpisodePersistenceError | None = None
        try:
            self.drain()
        except EpisodePersistenceError as exc:
            drain_error = exc
        self._stop_worker()

        close_reason = reason
        if drain_error is not None:
            failure_text = f"background persistence failed: {drain_error}"
            close_reason = f"{reason}; {failure_text}" if reason else failure_text
        try:
            path = self._writer.close_partial(reason=close_reason)
        finally:
            self._mark_closed()
        if drain_error is not None:
            raise drain_error
        return path

    def _finalize(
        self,
        operation: str,
        *,
        reason: str | None = None,
        notes: str | None = None,
    ) -> Path:
        with self._condition:
            self._begin_finalization_locked()

        try:
            self.drain()
        except EpisodePersistenceError as exc:
            self._stop_worker()
            try:
                self._writer.close_partial(
                    reason=f"background persistence failed before {operation}: {exc}"
                )
            finally:
                self._mark_closed()
            raise

        self._stop_worker()
        try:
            if operation == "accept":
                path = self._writer.accept(notes=notes)
            elif operation == "reject":
                assert reason is not None
                path = self._writer.reject(reason, notes=notes)
            else:  # pragma: no cover - private invariant
                raise AssertionError(f"unknown finalization operation: {operation}")
        except BaseException as exc:
            try:
                if self._writer.active:
                    self._writer.close_partial(
                        reason=f"{operation} finalization failed: {exc!r}"
                    )
            finally:
                self._mark_closed()
            raise
        self._mark_closed()
        return path

    def _worker_main(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _AppendJob)
                with self._condition:
                    already_failed = self._failure is not None
                if already_failed:
                    continue
                try:
                    committed_index = self._writer.append(
                        item.step,
                        rgb=item.rgb,
                        depth=item.depth,
                    )
                    if committed_index != item.index:
                        raise RuntimeError(
                            "episode writer returned an unexpected frame index: "
                            f"expected {item.index}, got {committed_index}"
                        )
                except BaseException as exc:
                    failure = EpisodeWorkerError(
                        f"background episode write failed at frame {item.index}: {exc}"
                    )
                    failure.__cause__ = exc
                    with self._condition:
                        self._latch_failure_locked(failure)
            finally:
                if item is not _STOP:
                    with self._condition:
                        self._outstanding -= 1
                        self._condition.notify_all()
                self._queue.task_done()

    def _stop_worker(self) -> None:
        with self._condition:
            if self._worker_stopped:
                return
            if self._outstanding:
                raise RuntimeError("cannot stop episode worker before draining it")
            self._worker_stopped = True
        self._queue.put_nowait(_STOP)
        self._worker.join()

    def _require_open_locked(self) -> None:
        if self._state != "open" or not self._writer.active:
            raise RuntimeError("queued episode writer is already closing or closed")

    def _begin_finalization_locked(self) -> None:
        self._require_open_locked()
        self._state = "closing"

    def _mark_closed(self) -> None:
        with self._condition:
            self._state = "closed"
            self._condition.notify_all()

    def _latch_failure_locked(self, failure: EpisodePersistenceError) -> None:
        if self._failure is None:
            self._failure = failure
        self._condition.notify_all()

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise self._failure

    def __enter__(self) -> "QueuedEpisodeWriter":
        return self

    def __exit__(
        self, exc_type: Any, exc: BaseException | None, traceback: Any
    ) -> None:
        if self.active:
            try:
                self.close_partial(
                    reason=repr(exc) if exc is not None else "not finalized"
                )
            except EpisodePersistenceError:
                if exc is None:
                    raise
