"""Robot-side safety state machine independent from the network transport."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .config import ServerConfig
from .protocol import vector_norm


class ControllerLike(Protocol):
    robot: Any

    def get_state_snapshot(self) -> dict[str, object]: ...

    def start_continuous(
        self, frame: str = "local", *, command_timeout_ms: int | None = None
    ) -> None: ...

    def send_continuous(
        self,
        linear_velocity: Sequence[float],
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> object: ...

    def stop_continuous(self) -> None: ...

    def stop(self) -> None: ...

    def move_relative(
        self,
        displacement: Sequence[float],
        *,
        dynamics_factor: float = 1.0,
        asynchronous: bool = False,
    ) -> object: ...

    def move_global(
        self,
        displacement_or_position: Sequence[float],
        *,
        absolute: bool = False,
        dynamics_factor: float = 1.0,
        asynchronous: bool = False,
    ) -> object: ...

    def move_global_pose(
        self,
        position: Sequence[float],
        rotation_matrix: Sequence[Sequence[float]],
        *,
        dynamics_factor: float = 1.0,
        asynchronous: bool = False,
    ) -> object: ...

    def recover_from_errors(self) -> bool: ...


@dataclass(frozen=True)
class VelocityTarget:
    owner: str
    sequence: int
    frame: str
    linear: tuple[float, float, float]
    angular: tuple[float, float, float]
    expires_at: float


class FrankaRuntime:
    """Own the single control lease and enforce watchdog stops locally."""

    def __init__(self, controller: ControllerLike, config: ServerConfig) -> None:
        self.controller = controller
        self.config = config
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._lease_owner: str | None = None
        self._lease_deadline = 0.0
        self._last_sequence = -1
        self._velocity_target: VelocityTarget | None = None
        self._velocity_active = False
        self._velocity_frame: str | None = None
        self._one_shot_active = False
        self._one_shot_owner: str | None = None
        self._last_fault: str | None = None
        self._watchdog_stop_count = 0
        self._velocity_workspace_blocked = False
        self._workspace_guard_stop_count = 0
        self._last_workspace_guard_reason: str | None = None
        self._thread = threading.Thread(
            target=self._control_loop,
            name="franka-bridge-control",
            daemon=True,
        )
        self._thread.start()

    def acquire(self, owner: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._expire_lease_locked(now)
            if self._lease_owner not in {None, owner}:
                return False
            self._lease_owner = owner
            self._lease_deadline = now + self.config.lease_timeout_ms / 1000.0
            if self._last_sequence < 0:
                self._last_sequence = -1
            return True

    def heartbeat(self, owner: str) -> None:
        with self._lock:
            self._require_owner_locked(owner)
            self._lease_deadline = (
                time.monotonic() + self.config.lease_timeout_ms / 1000.0
            )

    def release(self, owner: str) -> None:
        with self._lock:
            if self._lease_owner != owner:
                return
            self._clear_control_locked(stop_robot=True)
            self._lease_owner = None
            self._lease_deadline = 0.0
            self._last_sequence = -1

    def submit_velocity(
        self,
        owner: str,
        sequence: int,
        frame: str,
        linear: Sequence[float],
        angular: Sequence[float],
    ) -> None:
        normalized_frame = frame.lower()
        if normalized_frame not in {"local", "global"}:
            raise ValueError("frame must be local or global")
        linear_vector = self._finite_vector3(linear, "linear")
        angular_vector = self._finite_vector3(angular, "angular")
        if vector_norm(linear_vector) > self.config.max_linear_speed:
            raise ValueError("linear velocity exceeds server safety limit")
        if vector_norm(angular_vector) > self.config.max_angular_speed:
            raise ValueError("angular velocity exceeds server safety limit")

        now = time.monotonic()
        with self._lock:
            self._require_owner_locked(owner)
            if sequence <= self._last_sequence:
                raise ValueError("velocity sequence must increase monotonically")
            self._last_sequence = sequence
            self._lease_deadline = now + self.config.lease_timeout_ms / 1000.0
            if vector_norm(linear_vector) < 1e-9 and vector_norm(angular_vector) < 1e-9:
                self._velocity_target = None
                self._stop_velocity_locked()
                self._velocity_workspace_blocked = False
                return
            if self._one_shot_active:
                self._stop_all_locked()
            self._velocity_target = VelocityTarget(
                owner=owner,
                sequence=sequence,
                frame=normalized_frame,
                linear=linear_vector,
                angular=angular_vector,
                expires_at=now + self.config.velocity_timeout_ms / 1000.0,
            )

    def move_relative(
        self,
        owner: str,
        displacement: Sequence[float],
        dynamics_factor: float,
    ) -> None:
        self._require_one_shot_enabled()
        vector = self._finite_vector3(displacement, "displacement")
        self._validate_dynamics_factor(dynamics_factor)
        self._validate_relative_displacement(vector)
        with self._lock:
            self._require_owner_locked(owner)
            position, rotation = self._current_pose_locked()
            base_delta = tuple(
                sum(rotation[row][column] * vector[column] for column in range(3))
                for row in range(3)
            )
            target = tuple(position[index] + base_delta[index] for index in range(3))
            self._validate_workspace(target)
            self._stop_velocity_locked()
            self.controller.move_relative(
                vector,
                dynamics_factor=dynamics_factor,
                asynchronous=True,
            )
            self._one_shot_active = True
            self._one_shot_owner = owner
            self._lease_deadline = (
                time.monotonic() + self.config.lease_timeout_ms / 1000.0
            )

    def move_global(
        self,
        owner: str,
        value: Sequence[float],
        *,
        absolute: bool,
        dynamics_factor: float,
        rotation_matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self._require_one_shot_enabled()
        vector = self._finite_vector3(value, "position")
        self._validate_dynamics_factor(dynamics_factor)
        with self._lock:
            self._require_owner_locked(owner)
            target_rotation = (
                None
                if rotation_matrix is None
                else self._valid_rotation_matrix(rotation_matrix)
            )
            if target_rotation is not None:
                if not absolute:
                    raise ValueError(
                        "rotation_matrix is only supported for absolute motion"
                    )
                if not self.config.allow_orientation_motion:
                    raise PermissionError(
                        "orientation motion is disabled by server config"
                    )
                _position, current_rotation = self._current_pose_locked()
                rotation_step = self._rotation_distance_rad(
                    current_rotation,
                    target_rotation,
                )
                if rotation_step > self.config.max_orientation_step_rad:
                    raise ValueError(
                        f"orientation step {rotation_step:.4f} rad exceeds server "
                        f"limit {self.config.max_orientation_step_rad:.4f} rad"
                    )
            if absolute:
                target = vector
            else:
                self._validate_relative_displacement(vector)
                current_position = self._current_position_locked()
                target = tuple(
                    current_position[index] + vector[index] for index in range(3)
                )
            self._validate_workspace(target)
            self._stop_velocity_locked()
            if target_rotation is None:
                self.controller.move_global(
                    vector,
                    absolute=absolute,
                    dynamics_factor=dynamics_factor,
                    asynchronous=True,
                )
            else:
                self.controller.move_global_pose(
                    vector,
                    target_rotation,
                    dynamics_factor=dynamics_factor,
                    asynchronous=True,
                )
            self._one_shot_active = True
            self._one_shot_owner = owner
            self._lease_deadline = (
                time.monotonic() + self.config.lease_timeout_ms / 1000.0
            )

    def stop_all(self) -> None:
        with self._lock:
            self._stop_all_locked()
            self._lease_owner = None
            self._lease_deadline = 0.0
            self._last_sequence = -1

    def recover_from_errors(self, owner: str) -> bool:
        if not self.config.allow_error_recovery:
            raise PermissionError("error recovery is disabled by server config")
        with self._lock:
            self._require_owner_locked(owner)
            self._clear_control_locked(stop_robot=True)
            return bool(self.controller.recover_from_errors())

    def state_snapshot(self) -> dict[str, object]:
        return self.controller.get_state_snapshot()

    def bridge_status(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            return {
                "control_lease_active": self._lease_owner is not None
                and now <= self._lease_deadline,
                "velocity_active": self._velocity_active,
                "velocity_command_fresh": self._velocity_target is not None
                and now <= self._velocity_target.expires_at,
                "one_shot_active": self._one_shot_active,
                "last_velocity_sequence": self._last_sequence,
                "watchdog_stop_count": self._watchdog_stop_count,
                "velocity_workspace_blocked": self._velocity_workspace_blocked,
                "workspace_guard_stop_count": self._workspace_guard_stop_count,
                "last_workspace_guard_reason": self._last_workspace_guard_reason,
                "last_fault": self._last_fault,
            }

    def close(self) -> None:
        self._shutdown.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            self._clear_control_locked(stop_robot=True)
            self._lease_owner = None

    def _control_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        while not self._shutdown.wait(period):
            try:
                self._control_tick()
            except Exception as error:  # hardware faults must fail closed
                with self._lock:
                    self._last_fault = f"{type(error).__name__}: {error}"
                    self._velocity_target = None
                    try:
                        self._stop_all_locked()
                    except Exception as stop_error:
                        self._last_fault += (
                            f"; stop failed: {type(stop_error).__name__}: {stop_error}"
                        )

    def _control_tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._expire_lease_locked(now)
            if self._one_shot_active and not bool(
                getattr(self.controller.robot, "is_in_control", False)
            ):
                self._one_shot_active = False
                self._one_shot_owner = None

            target = self._velocity_target
            if target is None or now > target.expires_at:
                if target is not None:
                    self._watchdog_stop_count += 1
                    self._velocity_target = None
                self._stop_velocity_locked()
                return

            workspace_reason = self._velocity_workspace_reason_locked(target)
            if workspace_reason is not None:
                if not self._velocity_workspace_blocked:
                    self._workspace_guard_stop_count += 1
                self._velocity_workspace_blocked = True
                self._last_workspace_guard_reason = workspace_reason
                self._velocity_target = None
                self._stop_velocity_locked()
                return
            self._velocity_workspace_blocked = False

            if not self._velocity_active or self._velocity_frame != target.frame:
                self._stop_velocity_locked()
                self.controller.start_continuous(
                    frame=target.frame,
                    command_timeout_ms=self.config.franky_command_timeout_ms,
                )
                self._velocity_active = True
                self._velocity_frame = target.frame
            self.controller.send_continuous(target.linear, target.angular)

    def _expire_lease_locked(self, now: float) -> None:
        if self._lease_owner is not None and now > self._lease_deadline:
            self._watchdog_stop_count += 1
            self._clear_control_locked(stop_robot=True)
            self._lease_owner = None
            self._lease_deadline = 0.0
            self._last_sequence = -1

    def _require_owner_locked(self, owner: str) -> None:
        self._expire_lease_locked(time.monotonic())
        if self._lease_owner != owner:
            raise PermissionError("connection does not own the control lease")

    def _clear_control_locked(self, *, stop_robot: bool) -> None:
        self._velocity_target = None
        self._velocity_workspace_blocked = False
        if stop_robot:
            self._stop_all_locked()
        else:
            self._stop_velocity_locked()

    def _stop_velocity_locked(self) -> None:
        if self._velocity_active:
            self.controller.stop_continuous()
        self._velocity_active = False
        self._velocity_frame = None

    def _stop_all_locked(self) -> None:
        self._velocity_target = None
        self._velocity_workspace_blocked = False
        if self._velocity_active:
            self.controller.stop_continuous()
        if self._one_shot_active or bool(
            getattr(self.controller.robot, "is_in_control", False)
        ):
            self.controller.stop()
        self._velocity_active = False
        self._velocity_frame = None
        self._one_shot_active = False
        self._one_shot_owner = None

    def _velocity_workspace_reason_locked(
        self,
        target: VelocityTarget,
    ) -> str | None:
        position, rotation = self._current_pose_locked()
        for axis, (value, lower, upper) in enumerate(
            zip(
                position,
                self.config.workspace_min_m,
                self.config.workspace_max_m,
            )
        ):
            if value < lower or value > upper:
                return (
                    "current end-effector position is outside the configured "
                    f"workspace on axis {axis}: {value:.6f} not in "
                    f"[{lower:.6f}, {upper:.6f}]"
                )

        if target.frame == "global":
            global_linear = target.linear
        else:
            global_linear = tuple(
                math.fsum(
                    rotation[row][column] * target.linear[column] for column in range(3)
                )
                for row in range(3)
            )

        prediction_horizon_s = max(
            self.config.velocity_timeout_ms / 1000.0,
            2.0 / self.config.control_hz,
        )
        predicted_position = tuple(
            position[index] + global_linear[index] * prediction_horizon_s
            for index in range(3)
        )
        for axis, (value, lower, upper) in enumerate(
            zip(
                predicted_position,
                self.config.workspace_min_m,
                self.config.workspace_max_m,
            )
        ):
            if value < lower or value > upper:
                return (
                    "predicted end-effector position would leave the configured "
                    f"workspace on axis {axis} within "
                    f"{prediction_horizon_s:.3f} s: {value:.6f} not in "
                    f"[{lower:.6f}, {upper:.6f}]"
                )
        return None

    def _require_one_shot_enabled(self) -> None:
        if not self.config.allow_one_shot_motion:
            raise PermissionError("one-shot Cartesian motion is disabled")

    @staticmethod
    def _finite_vector3(
        value: Sequence[float], name: str
    ) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError(f"{name} must contain three values")
        result = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in result):
            raise ValueError(f"{name} must contain finite values")
        return result  # type: ignore[return-value]

    def _validate_dynamics_factor(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("dynamics_factor must be positive")
        if value > self.config.max_motion_dynamics_factor:
            raise ValueError("dynamics_factor exceeds server safety limit")

    def _validate_relative_displacement(self, vector: Sequence[float]) -> None:
        if vector_norm(vector) > self.config.max_relative_displacement_m:
            raise ValueError("relative displacement exceeds server safety limit")

    def _validate_workspace(self, position: Sequence[float]) -> None:
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                position,
                self.config.workspace_min_m,
                self.config.workspace_max_m,
            )
        ):
            raise ValueError("target position is outside the configured workspace")

    def _current_pose_locked(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ]:
        snapshot = self.controller.get_state_snapshot()
        end_effector = snapshot.get("end_effector")
        if not isinstance(end_effector, dict):
            raise RuntimeError("state snapshot is missing end_effector")
        position = self._finite_vector3(end_effector.get("position", ()), "position")
        raw_rotation = end_effector.get("rotation_matrix")
        try:
            row_count = len(raw_rotation)  # type: ignore[arg-type]
        except TypeError as error:
            raise RuntimeError(
                "state snapshot is missing a 3x3 rotation_matrix"
            ) from error
        if row_count != 3:
            raise RuntimeError("state snapshot is missing a 3x3 rotation_matrix")
        rotation = tuple(
            self._finite_vector3(row, "rotation_matrix row")
            for row in raw_rotation  # type: ignore[union-attr]
        )
        return position, rotation  # type: ignore[return-value]

    def _current_position_locked(self) -> tuple[float, float, float]:
        snapshot = self.controller.get_state_snapshot()
        end_effector = snapshot.get("end_effector")
        if not isinstance(end_effector, dict):
            raise RuntimeError("state snapshot is missing end_effector")
        return self._finite_vector3(end_effector.get("position", ()), "position")

    @classmethod
    def _valid_rotation_matrix(
        cls,
        value: Sequence[Sequence[float]],
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        if len(value) != 3:
            raise ValueError("rotation_matrix must contain three rows")
        rotation = tuple(
            cls._finite_vector3(row, "rotation_matrix row") for row in value
        )
        for row in range(3):
            for other in range(3):
                dot = math.fsum(
                    rotation[row][column] * rotation[other][column]
                    for column in range(3)
                )
                expected = 1.0 if row == other else 0.0
                if abs(dot - expected) > 1e-3:
                    raise ValueError("rotation_matrix must be orthonormal")
        determinant = (
            rotation[0][0]
            * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1]
            * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2]
            * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if abs(determinant - 1.0) > 1e-3:
            raise ValueError("rotation_matrix determinant must be +1")
        return rotation  # type: ignore[return-value]

    @staticmethod
    def _rotation_distance_rad(
        first: Sequence[Sequence[float]],
        second: Sequence[Sequence[float]],
    ) -> float:
        trace = math.fsum(
            first[row][column] * second[row][column]
            for row in range(3)
            for column in range(3)
        )
        cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
        return math.acos(cosine)
