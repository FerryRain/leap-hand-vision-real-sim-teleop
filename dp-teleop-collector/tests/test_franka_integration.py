from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
BRIDGE_ROOT = REPOSITORY_ROOT / "franka-lan-bridge"
for path in (ROOT, REPOSITORY_ROOT, BRIDGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dp_collector.collector import build_parser, run  # noqa: E402
from dp_collector.episode import read_episode_steps, validate_dataset  # noqa: E402
from dp_collector.exporter import export_to_zarr  # noqa: E402
from franka_bridge.config import ServerConfig  # noqa: E402
from franka_bridge.mock_controller import MockFrankaController  # noqa: E402
from franka_bridge.runtime import FrankaRuntime  # noqa: E402
from franka_bridge.server import FrankaBridgeServer  # noqa: E402

TOKEN = "collector-integration-token-32-chars"


class FakePreviewUI:
    def __init__(self, *, enabled: bool) -> None:
        assert enabled
        self.calls = 0
        self._deadman_down = False

    @property
    def deadman_down(self) -> bool:
        return self._deadman_down

    @deadman_down.setter
    def deadman_down(self, value: bool) -> None:
        self._deadman_down = bool(value)

    def window_open(self) -> bool:
        return True

    def show(self, _frame: np.ndarray, _lines: tuple[str, ...]) -> int:
        self.calls += 1
        if self.calls == 10:
            return ord("g")
        if self.calls == 11:
            # Starting an episode intentionally clears the deadman.  This
            # simulates the operator pressing it again on the next frame.
            self._deadman_down = True
        if self.calls == 80:
            return ord(" ")
        if self.calls == 83:
            return ord("q")
        return -1

    def close(self) -> None:
        self._deadman_down = False


class FakeTerminalKeys:
    def __enter__(self) -> FakeTerminalKeys:
        return self

    def poll(self) -> None:
        return None

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None


def test_mock_fr3_leap_episode_round_trips_to_22d_zarr(tmp_path: Path) -> None:
    asyncio.run(_run_mock_fr3_episode(tmp_path))

    dataset = tmp_path / "dataset"
    report = validate_dataset(
        dataset,
        include_partial=False,
        include_rejected=False,
    )
    assert report.ok
    assert len(report.episodes) == 1
    accepted = next((dataset / "accepted").iterdir())
    steps = read_episode_steps(accepted)
    assert len(steps) >= 20
    assert all(len(step["robot_state"]) == 45 for step in steps)
    assert all(len(step["action"]) == 22 for step in steps)
    assert any(np.linalg.norm(step["action"][:3]) > 0.0 for step in steps)
    assert all("bridge" in step["extra"]["franka"] for step in steps)

    summary = export_to_zarr(dataset, tmp_path / "training.zarr")
    assert summary.num_episodes == 1
    assert summary.action_dim == 22
    assert summary.robot_state_dim == 45


async def _run_mock_fr3_episode(tmp_path: Path) -> None:
    config_path = tmp_path / "collector.yml"
    config_text = (ROOT / "config.yml").read_text(encoding="utf-8")
    config_text = config_text.replace(
        "mapping_confirmed: false", "mapping_confirmed: true"
    )
    config_text = config_text.replace("deadband_m: 0.010", "deadband_m: 0.000")
    config_path.write_text(config_text, encoding="utf-8")

    controller = MockFrankaController()
    server_config = ServerConfig.from_file(BRIDGE_ROOT / "server_config.example.json")
    runtime = FrankaRuntime(controller, server_config)
    bridge = FrankaBridgeServer(runtime, server_config, TOKEN)
    server = await websockets.serve(
        bridge.handle_connection,
        "127.0.0.1",
        0,
        compression=None,
        max_size=64 * 1024,
    )
    port = server.sockets[0].getsockname()[1]
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--source",
            "mock",
            "--leap-device",
            "mock",
            "--franka-mode",
            "teleop",
            "--franka-uri",
            f"ws://127.0.0.1:{port}",
            "--enable-franka-motion",
            "--output",
            str(tmp_path / "dataset"),
            "--duration",
            "6",
        ]
    )

    try:
        with (
            mock.patch.dict(os.environ, {"FRANKA_BRIDGE_TOKEN": TOKEN}),
            mock.patch("dp_collector.collector.PreviewUI", FakePreviewUI),
            mock.patch("dp_collector.collector.TerminalKeys", FakeTerminalKeys),
        ):
            summary = await run(args)
        assert summary.accepted_episodes == 1
        assert summary.rejected_episodes == 0
        assert summary.stop_reason == "operator_emergency_stop"
    finally:
        server.close()
        await server.wait_closed()
        runtime.close()

    # Keep the source config deterministic and ensure the test did not mutate it.
    source = (ROOT / "config.yml").read_text(encoding="utf-8")
    assert "mapping_confirmed: false" in source
    json.dumps(summary.__dict__, allow_nan=False)
