"""Bounded background persistence for LEAP proprioceptive episodes."""

from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .proprio_episode import ProprioEpisodeWriter
from .proprio_schema import ProprioStep, normalize_proprio_step


class ProprioPersistenceError(RuntimeError):
    """Base class for a latched proprio persistence failure."""


class ProprioQueueFullError(ProprioPersistenceError):
    """The bounded proprio pipeline cannot accept another sample."""


class ProprioWorkerError(ProprioPersistenceError):
    """The background proprio writer failed to commit a sample."""


@dataclass(frozen=True)
class _AppendJob:
    index: int
    step: ProprioStep


_STOP = object()


class QueuedProprioEpisodeWriter:
    """Persist low-dimensional samples without blocking the control loop.

    ``max_pending_steps`` includes the sample currently being fsynced.  Queue
    overload and worker errors are latched.  Any finalizer encountering a
    latched failure closes the backend under ``.partial`` and refuses to accept
    the episode.
    """

    def __init__(
        self,
        writer: ProprioEpisodeWriter,
        *,
        max_pending_steps: int = 16,
    ) -> None:
        if int(max_pending_steps) != max_pending_steps or max_pending_steps < 1:
            raise ValueError("max_pending_steps must be a positive integer")
        if not writer.active:
            raise ValueError("cannot queue writes for an inactive episode writer")
        baseline_timestamp, baseline_actual = writer.validation_baseline()
        self._writer = writer
        self._max_pending_steps = int(max_pending_steps)
        self._queue: queue.Queue[_AppendJob | object] = queue.Queue(
            maxsize=self._max_pending_steps
        )
        self._condition = threading.Condition()
        self._outstanding = 0
        self._submitted_count = int(writer.step_count)
        self._submitted_previous_timestamp_s = baseline_timestamp
        self._submitted_previous_actual_position_rad = baseline_actual
        self._failure: ProprioPersistenceError | None = None
        self._state = "open"
        self._worker_stopped = False
        self._worker = threading.Thread(
            target=self._worker_main,
            name=f"proprio-writer-{writer.episode_id}",
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
        with self._condition:
            return self._submitted_count

    @property
    def committed_step_count(self) -> int:
        return int(self._writer.step_count)

    @property
    def pending_count(self) -> int:
        with self._condition:
            return self._outstanding

    @property
    def max_pending_steps(self) -> int:
        return self._max_pending_steps

    @property
    def active(self) -> bool:
        with self._condition:
            return self._state == "open" and self._writer.active

    def append(
        self,
        *,
        timestamp_s: float,
        actual_position_rad: Sequence[float],
        present_current_raw: Sequence[int],
        goal_position_rad: Sequence[float],
        valid: bool = True,
        invalid_reasons: Iterable[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> int:
        step = ProprioStep(
            timestamp_s=float(timestamp_s),
            actual_position_rad=np.array(actual_position_rad, copy=True),
            present_current_raw=np.array(present_current_raw, copy=True),
            goal_position_rad=np.array(goal_position_rad, copy=True),
            valid=bool(valid),
            invalid_reasons=tuple(invalid_reasons),
            extra=copy.deepcopy(dict(extra or {})),
        )
        return self.append_step(step)

    def append_step(self, step: ProprioStep) -> int:
        snapshot = ProprioStep(
            timestamp_s=float(step.timestamp_s),
            actual_position_rad=np.array(step.actual_position_rad, copy=True),
            present_current_raw=np.array(step.present_current_raw, copy=True),
            goal_position_rad=np.array(step.goal_position_rad, copy=True),
            valid=bool(step.valid),
            invalid_reasons=tuple(step.invalid_reasons),
            extra=copy.deepcopy(dict(step.extra)),
        )
        with self._condition:
            self._require_open_locked()
            self._raise_if_failed_locked()
            normalize_proprio_step(
                self._writer.spec,
                snapshot,
                index=self._submitted_count,
                previous_timestamp_s=self._submitted_previous_timestamp_s,
                previous_actual_position_rad=(
                    self._submitted_previous_actual_position_rad
                ),
            )
            if self._outstanding >= self._max_pending_steps:
                failure = ProprioQueueFullError(
                    "proprio persistence queue is full "
                    f"({self._outstanding}/{self._max_pending_steps} steps); "
                    "collection must stop without dropping a sample"
                )
                self._latch_failure_locked(failure)
                raise failure
            index = self._submitted_count
            try:
                self._queue.put_nowait(_AppendJob(index=index, step=snapshot))
            except queue.Full as exc:  # Defensive; outstanding is authoritative.
                failure = ProprioQueueFullError(
                    "proprio persistence queue became full during enqueue"
                )
                self._latch_failure_locked(failure)
                raise failure from exc
            self._outstanding += 1
            self._submitted_count += 1
            self._submitted_previous_timestamp_s = float(snapshot.timestamp_s)
            self._submitted_previous_actual_position_rad = np.asarray(
                snapshot.actual_position_rad, dtype=np.float64
            ).copy()
            return index

    def raise_if_failed(self) -> None:
        with self._condition:
            self._raise_if_failed_locked()

    check_health = raise_if_failed

    def drain(self) -> None:
        with self._condition:
            while self._outstanding:
                self._condition.wait()
            self._raise_if_failed_locked()

    def accept(self, *, notes: str | None = None) -> Path:
        return self._finalize("accept", notes=notes)

    def reject(self, reason: str, *, notes: str | None = None) -> Path:
        clean_reason = str(reason).strip()
        if not clean_reason:
            raise ValueError("a non-empty rejection reason is required")
        return self._finalize("reject", reason=clean_reason, notes=notes)

    def close_partial(self, *, reason: str | None = None) -> Path:
        with self._condition:
            if self._state == "closed":
                return self._writer.directory
            self._begin_finalization_locked()
        drain_error: ProprioPersistenceError | None = None
        try:
            self.drain()
        except ProprioPersistenceError as exc:
            drain_error = exc
        self._stop_worker()
        close_reason = reason
        if drain_error is not None:
            detail = f"background proprio persistence failed: {drain_error}"
            close_reason = f"{reason}; {detail}" if reason else detail
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
        except ProprioPersistenceError as exc:
            self._stop_worker()
            try:
                self._writer.close_partial(
                    reason=(
                        "background proprio persistence failed before "
                        f"{operation}: {exc}"
                    )
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
                    committed_index = self._writer.append_step(item.step)
                    if committed_index != item.index:
                        raise RuntimeError(
                            "proprio writer returned an unexpected step index: "
                            f"expected {item.index}, got {committed_index}"
                        )
                except BaseException as exc:
                    failure = ProprioWorkerError(
                        f"background proprio write failed at step {item.index}: {exc}"
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
                raise RuntimeError("cannot stop proprio worker before draining it")
            self._worker_stopped = True
        self._queue.put_nowait(_STOP)
        self._worker.join()

    def _require_open_locked(self) -> None:
        if self._state != "open" or not self._writer.active:
            raise RuntimeError("queued proprio writer is already closing or closed")

    def _begin_finalization_locked(self) -> None:
        self._require_open_locked()
        self._state = "closing"

    def _mark_closed(self) -> None:
        with self._condition:
            self._state = "closed"
            self._condition.notify_all()

    def _latch_failure_locked(self, failure: ProprioPersistenceError) -> None:
        if self._failure is None:
            self._failure = failure
        self._condition.notify_all()

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise self._failure

    def __enter__(self) -> "QueuedProprioEpisodeWriter":
        return self

    def __exit__(
        self, exc_type: Any, exc: BaseException | None, traceback: Any
    ) -> None:
        if self.active:
            try:
                self.close_partial(
                    reason=repr(exc) if exc is not None else "not finalized"
                )
            except ProprioPersistenceError:
                if exc is None:
                    raise
