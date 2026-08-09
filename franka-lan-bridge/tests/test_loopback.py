from __future__ import annotations

import unittest

import websockets
from franka_bridge.client import FrankaBridgeClient
from franka_bridge.mock_controller import MockFrankaController
from franka_bridge.next_point import OperatorStop, wait_for_motion
from franka_bridge.runtime import FrankaRuntime
from franka_bridge.server import FrankaBridgeServer
from franka_bridge.waypoints import Waypoint

from tests.helpers import test_config

TOKEN = "loopback-test-token-32-characters"


class NoKeys:
    def poll(self) -> str | None:
        return None


class StopKeys:
    def poll(self) -> str | None:
        return "e"


class LoopbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.config = test_config()
        self.controller = MockFrankaController()
        self.runtime = FrankaRuntime(self.controller, self.config)
        self.bridge = FrankaBridgeServer(self.runtime, self.config, TOKEN)
        self.server = await websockets.serve(
            self.bridge.handle_connection,
            "127.0.0.1",
            0,
            compression=None,
            max_size=64 * 1024,
        )
        socket = self.server.sockets[0]
        self.uri = f"ws://127.0.0.1:{socket.getsockname()[1]}"

    async def asyncTearDown(self) -> None:
        self.server.close()
        await self.server.wait_closed()
        self.runtime.close()

    async def test_state_and_velocity_round_trip(self) -> None:
        async with FrankaBridgeClient(self.uri, TOKEN) as client:
            await client.acquire_control()
            await client.send_velocity((0.01, 0.0, 0.0), frame="global", wait_ack=True)
            state = await client.next_state(timeout_s=1.0)
            self.assertEqual(state["type"], "state")
            self.assertIn("joints", state["robot"])
            await client.stop()
            self.assertFalse(self.runtime.bridge_status()["control_lease_active"])

    async def test_observer_can_issue_stop_without_control_lease(self) -> None:
        owner = FrankaBridgeClient(self.uri, TOKEN, client_name="owner")
        observer = FrankaBridgeClient(self.uri, TOKEN, client_name="observer")
        await owner.connect()
        await observer.connect()
        try:
            await owner.acquire_control()
            await owner.send_velocity((0.01, 0.0, 0.0), wait_ack=True)
            await observer.stop()
            self.assertFalse(self.runtime.bridge_status()["control_lease_active"])
        finally:
            await owner.close()
            await observer.close()

    async def test_absolute_point_motion_completes_with_position_check(self) -> None:
        target = Waypoint(
            name="P1",
            position_m=(0.41, 0.0, 0.3),
            rotation_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            robot_timestamp_s=None,
        )
        async with FrankaBridgeClient(self.uri, TOKEN) as client:
            await client.acquire_control()
            response = await client.move_global(
                target.position_m,
                absolute=True,
                dynamics_factor=0.1,
            )
            error = await wait_for_motion(
                client,
                target,
                accepted_at_s=float(response["server_monotonic_s"]),
                timeout_s=1.0,
                arrival_tolerance_m=0.001,
                keys=NoKeys(),
            )
            self.assertLessEqual(error, 0.001)
            await client.release_control()

    async def test_operator_key_stops_motion_and_revokes_control(self) -> None:
        target = Waypoint(
            name="P1",
            position_m=(0.41, 0.0, 0.3),
            rotation_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            robot_timestamp_s=None,
        )
        async with FrankaBridgeClient(self.uri, TOKEN) as owner:
            await owner.acquire_control()
            response = await owner.move_global(
                target.position_m,
                absolute=True,
                dynamics_factor=0.1,
            )
            with self.assertRaises(OperatorStop):
                await wait_for_motion(
                    owner,
                    target,
                    accepted_at_s=float(response["server_monotonic_s"]),
                    timeout_s=1.0,
                    arrival_tolerance_m=0.001,
                    keys=StopKeys(),
                )
            self.assertFalse(self.runtime.bridge_status()["control_lease_active"])


if __name__ == "__main__":
    unittest.main()
