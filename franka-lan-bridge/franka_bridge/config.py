"""Validated server configuration for the Franka LAN bridge."""

from __future__ import annotations

import ipaddress
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ServerConfig:
    bind_host: str
    port: int
    allowed_client_cidrs: tuple[str, ...]
    controller_root: str
    controller_class: str
    robot_host: str
    state_hz: float
    control_hz: float
    velocity_timeout_ms: int
    lease_timeout_ms: int
    franky_command_timeout_ms: int
    max_linear_speed: float
    max_angular_speed: float
    max_relative_displacement_m: float
    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    max_motion_dynamics_factor: float
    relative_dynamics_factor: float
    recover_errors_on_start: bool
    allow_error_recovery: bool
    allow_one_shot_motion: bool

    @classmethod
    def from_file(cls, path: Path) -> "ServerConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("server config must be a JSON object")
        config = cls(
            bind_host=str(raw.get("bind_host", "127.0.0.1")),
            port=int(raw.get("port", 8765)),
            allowed_client_cidrs=tuple(
                str(item) for item in raw.get("allowed_client_cidrs", [])
            ),
            controller_root=str(raw.get("controller_root", ".")),
            controller_class=str(
                raw.get(
                    "controller_class",
                    "utils.franka_controller:FrankaController",
                )
            ),
            robot_host=str(raw.get("robot_host", "10.19.131.201")),
            state_hz=float(raw.get("state_hz", 20.0)),
            control_hz=float(raw.get("control_hz", 30.0)),
            velocity_timeout_ms=int(raw.get("velocity_timeout_ms", 200)),
            lease_timeout_ms=int(raw.get("lease_timeout_ms", 1000)),
            franky_command_timeout_ms=int(raw.get("franky_command_timeout_ms", 150)),
            max_linear_speed=float(raw.get("max_linear_speed", 0.05)),
            max_angular_speed=float(raw.get("max_angular_speed", 0.25)),
            max_relative_displacement_m=float(
                raw.get("max_relative_displacement_m", 0.02)
            ),
            workspace_min_m=cls._vector3(
                raw.get("workspace_min_m", [0.20, -0.40, 0.10]),
                "workspace_min_m",
            ),
            workspace_max_m=cls._vector3(
                raw.get("workspace_max_m", [0.70, 0.40, 0.70]),
                "workspace_max_m",
            ),
            max_motion_dynamics_factor=float(
                raw.get("max_motion_dynamics_factor", 0.2)
            ),
            relative_dynamics_factor=float(raw.get("relative_dynamics_factor", 0.05)),
            recover_errors_on_start=bool(raw.get("recover_errors_on_start", False)),
            allow_error_recovery=bool(raw.get("allow_error_recovery", False)),
            allow_one_shot_motion=bool(raw.get("allow_one_shot_motion", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.bind_host:
            raise ValueError("bind_host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if not self.allowed_client_cidrs:
            raise ValueError("allowed_client_cidrs must contain at least one network")
        for value in self.allowed_client_cidrs:
            ipaddress.ip_network(value, strict=False)
        if ":" not in self.controller_class:
            raise ValueError("controller_class must use module:Class syntax")
        if not self.robot_host:
            raise ValueError("robot_host must not be empty")
        for name, value in (
            ("state_hz", self.state_hz),
            ("control_hz", self.control_hz),
            ("max_linear_speed", self.max_linear_speed),
            ("max_angular_speed", self.max_angular_speed),
            ("max_relative_displacement_m", self.max_relative_displacement_m),
            ("max_motion_dynamics_factor", self.max_motion_dynamics_factor),
            ("relative_dynamics_factor", self.relative_dynamics_factor),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.state_hz > 100.0 or self.control_hz > 100.0:
            raise ValueError("state_hz and control_hz must not exceed 100 Hz")
        if not 0.0 < self.relative_dynamics_factor <= 1.0:
            raise ValueError("relative_dynamics_factor must be in (0, 1]")
        if not 0.0 < self.max_motion_dynamics_factor <= 1.0:
            raise ValueError("max_motion_dynamics_factor must be in (0, 1]")
        if any(
            lower >= upper
            for lower, upper in zip(self.workspace_min_m, self.workspace_max_m)
        ):
            raise ValueError("workspace_min_m must be below workspace_max_m")
        if self.franky_command_timeout_ms < 50:
            raise ValueError("franky_command_timeout_ms must be at least 50")
        if self.velocity_timeout_ms < 2_000.0 / self.control_hz:
            raise ValueError("velocity_timeout_ms is too short for control_hz")
        if self.lease_timeout_ms < self.velocity_timeout_ms:
            raise ValueError("lease_timeout_ms must be >= velocity_timeout_ms")

    def client_is_allowed(self, address: str) -> bool:
        peer = ipaddress.ip_address(address)
        return any(
            peer in ipaddress.ip_network(network, strict=False)
            for network in self.allowed_client_cidrs
        )

    def public_safety_settings(self) -> dict[str, Any]:
        return {
            "state_hz": self.state_hz,
            "control_hz": self.control_hz,
            "velocity_timeout_ms": self.velocity_timeout_ms,
            "lease_timeout_ms": self.lease_timeout_ms,
            "franky_command_timeout_ms": self.franky_command_timeout_ms,
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "max_relative_displacement_m": self.max_relative_displacement_m,
            "workspace_min_m": self.workspace_min_m,
            "workspace_max_m": self.workspace_max_m,
            "max_motion_dynamics_factor": self.max_motion_dynamics_factor,
            "allow_error_recovery": self.allow_error_recovery,
            "allow_one_shot_motion": self.allow_one_shot_motion,
        }

    @staticmethod
    def _vector3(value: Any, name: str) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name} must contain three values")
        result = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in result):
            raise ValueError(f"{name} must contain finite values")
        return result  # type: ignore[return-value]
