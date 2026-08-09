"""Hardware-free controller used for deployment checks and tests."""

from __future__ import annotations

import time
from typing import Sequence


class MockFrankaController:
    def __init__(self) -> None:
        self.robot = self
        self.position = [0.4, 0.0, 0.3]
        self.linear_velocity = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.rotation_matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        self.current_frame = "global"
        self._continuous = False
        self._motion_end = 0.0
        self._last_update = time.monotonic()
        self.stop_count = 0
        self.stop_continuous_count = 0
        self.velocity_command_count = 0

    @property
    def is_in_control(self) -> bool:
        return self._continuous or time.monotonic() < self._motion_end

    def start_continuous(
        self, frame: str = "local", *, command_timeout_ms: int | None = None
    ) -> None:
        del command_timeout_ms
        self.current_frame = frame
        self._continuous = True

    def send_continuous(
        self,
        linear_velocity: Sequence[float],
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> object:
        self._integrate()
        self.linear_velocity = [float(item) for item in linear_velocity]
        self.angular_velocity = [float(item) for item in angular_velocity]
        self.velocity_command_count += 1
        self._continuous = True
        return object()

    def stop_continuous(self) -> None:
        self._integrate()
        self.linear_velocity = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self._continuous = False
        self.stop_continuous_count += 1

    def stop(self) -> None:
        self.stop_continuous()
        self._motion_end = 0.0
        self.stop_count += 1

    def move_relative(
        self,
        displacement: Sequence[float],
        *,
        dynamics_factor: float = 1.0,
        asynchronous: bool = False,
    ) -> object:
        del dynamics_factor, asynchronous
        self.position = [a + float(b) for a, b in zip(self.position, displacement)]
        self._motion_end = time.monotonic() + 0.1
        return object()

    def move_global(
        self,
        displacement_or_position: Sequence[float],
        *,
        absolute: bool = False,
        dynamics_factor: float = 1.0,
        asynchronous: bool = False,
    ) -> object:
        del dynamics_factor, asynchronous
        if absolute:
            self.position = [float(item) for item in displacement_or_position]
        else:
            self.position = [
                a + float(b) for a, b in zip(self.position, displacement_or_position)
            ]
        self._motion_end = time.monotonic() + 0.1
        return object()

    def recover_from_errors(self) -> bool:
        return True

    def move_global_pose(
        self,
        position: Sequence[float],
        rotation_matrix: Sequence[Sequence[float]],
        *,
        dynamics_factor: float = 1.0,
        asynchronous: bool = False,
    ) -> object:
        del dynamics_factor, asynchronous
        self.position = [float(item) for item in position]
        self.rotation_matrix = [
            [float(item) for item in row] for row in rotation_matrix
        ]
        self._motion_end = time.monotonic() + 0.1
        return object()

    def get_state_snapshot(self) -> dict[str, object]:
        self._integrate()
        return {
            "timestamp_s": time.time(),
            "robot_mode": "mock",
            "control_command_success_rate": 1.0,
            "current_errors": {},
            "last_motion_errors": {},
            "joints": {
                "position": [0.0] * 7,
                "velocity": [0.0] * 7,
                "torque": [0.0] * 7,
                "external_torque": [0.0] * 7,
                "contact": [0.0] * 7,
                "collision": [0.0] * 7,
            },
            "end_effector": {
                "position": self.position.copy(),
                "rotation_matrix": [row.copy() for row in self.rotation_matrix],
                "linear_velocity": self.linear_velocity.copy(),
                "angular_velocity": self.angular_velocity.copy(),
                "external_wrench_global": [0.0] * 6,
                "contact": [0.0] * 6,
                "collision": [0.0] * 6,
            },
        }

    def _integrate(self) -> None:
        now = time.monotonic()
        delta = max(0.0, now - self._last_update)
        self.position = [
            position + velocity * delta
            for position, velocity in zip(self.position, self.linear_velocity)
        ]
        self._last_update = now
