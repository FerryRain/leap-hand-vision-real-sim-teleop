from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from franka_bridge.config import ServerConfig

ROOT = Path(__file__).resolve().parents[1]


def test_config(**changes: object) -> ServerConfig:
    config = ServerConfig.from_file(ROOT / "server_config.example.json")
    return replace(
        config,
        bind_host="127.0.0.1",
        allowed_client_cidrs=("127.0.0.1/32",),
        state_hz=50.0,
        control_hz=100.0,
        velocity_timeout_ms=80,
        lease_timeout_ms=240,
        allow_one_shot_motion=True,
        **changes,
    )
