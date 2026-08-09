"""WebSocket server that keeps all Franka control on the robot workstation."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import logging
import os
import secrets
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import websockets

from .config import ServerConfig
from .mock_controller import MockFrankaController
from .protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    ack,
    decode_message,
    encode_message,
    error_response,
    finite_float,
    matrix3,
    positive_int,
    vector3,
)
from .runtime import ControllerLike, FrankaRuntime

LOGGER = logging.getLogger("franka_bridge.server")
TOKEN_ENV = "FRANKA_BRIDGE_TOKEN"


class FrankaBridgeServer:
    def __init__(
        self,
        runtime: FrankaRuntime,
        config: ServerConfig,
        token: str,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self._token = token

    async def handle_connection(self, websocket: Any, _path: str | None = None) -> None:
        peer_ip = self._peer_ip(websocket)
        if not self.config.client_is_allowed(peer_ip):
            LOGGER.warning(
                "rejected connection from non-allowlisted address %s", peer_ip
            )
            await websocket.close(code=4403, reason="client IP not allowed")
            return

        connection_id = uuid.uuid4().hex
        client_name = "unknown"
        state_task: asyncio.Task[None] | None = None
        send_lock = asyncio.Lock()
        authenticated = False
        try:
            raw_hello = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            hello = decode_message(raw_hello)
            if hello.get("type") != "hello":
                raise ProtocolError("first message must be hello")
            if hello.get("protocol") != PROTOCOL_VERSION:
                received_protocol = hello.get("protocol")
                raise ProtocolError(
                    f"protocol must be {PROTOCOL_VERSION}, got {received_protocol!r}"
                )
            candidate_token = hello.get("token")
            if not isinstance(candidate_token, str) or not secrets.compare_digest(
                candidate_token, self._token
            ):
                LOGGER.warning("authentication failed from %s", peer_ip)
                await websocket.close(code=4401, reason="authentication failed")
                return
            candidate_name = hello.get("client_name", "unnamed")
            if (
                not isinstance(candidate_name, str)
                or not 1 <= len(candidate_name) <= 64
            ):
                raise ProtocolError("client_name must contain 1-64 characters")
            client_name = candidate_name
            authenticated = True
            await self._send(
                websocket,
                send_lock,
                {
                    "type": "hello",
                    "protocol": PROTOCOL_VERSION,
                    "connection_id": connection_id,
                    "server_monotonic_s": time.monotonic(),
                    "safety": self.config.public_safety_settings(),
                },
            )
            LOGGER.info("authenticated %s from %s", client_name, peer_ip)
            state_task = asyncio.create_task(
                self._state_stream(websocket, send_lock),
                name=f"state-{connection_id}",
            )

            async for raw_message in websocket:
                request: dict[str, Any] | None = None
                try:
                    request = decode_message(raw_message)
                    response = await self._process_request(connection_id, request)
                except ProtocolError as error:
                    response = error_response(request, "invalid_request", str(error))
                except PermissionError as error:
                    response = error_response(request, "permission_denied", str(error))
                except ValueError as error:
                    response = error_response(request, "safety_rejected", str(error))
                except Exception as error:
                    LOGGER.exception("request failed for %s", client_name)
                    response = error_response(
                        request,
                        "controller_error",
                        f"{type(error).__name__}: {error}",
                    )
                if response is not None:
                    await self._send(websocket, send_lock, response)
        except (ProtocolError, asyncio.TimeoutError) as error:
            if authenticated:
                await self._send(
                    websocket,
                    send_lock,
                    error_response(None, "handshake_error", str(error)),
                )
            else:
                with contextlib.suppress(Exception):
                    await websocket.close(code=4400, reason=str(error)[:120])
        except websockets.ConnectionClosed:
            pass
        finally:
            if state_task is not None:
                state_task.cancel()
                with contextlib.suppress(
                    asyncio.CancelledError,
                    websockets.ConnectionClosed,
                ):
                    await state_task
            if authenticated:
                await asyncio.to_thread(self.runtime.release, connection_id)
                LOGGER.info(
                    "disconnected %s from %s; control released", client_name, peer_ip
                )

    async def _process_request(
        self, connection_id: str, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        message_type = request["type"]
        wants_ack = isinstance(request.get("request_id"), str)

        if message_type == "acquire_control":
            acquired = await asyncio.to_thread(self.runtime.acquire, connection_id)
            if not acquired:
                raise PermissionError("another client owns the control lease")
            return ack(request, acquired=True)

        if message_type == "heartbeat":
            await asyncio.to_thread(self.runtime.heartbeat, connection_id)
            return ack(request) if wants_ack else None

        if message_type == "release_control":
            await asyncio.to_thread(self.runtime.release, connection_id)
            return ack(request, released=True)

        if message_type == "velocity":
            frame = request.get("frame", "global")
            if not isinstance(frame, str):
                raise ProtocolError("frame must be local or global")
            await asyncio.to_thread(
                self.runtime.submit_velocity,
                connection_id,
                positive_int(request, "sequence"),
                frame,
                vector3(request, "linear"),
                vector3(request, "angular"),
            )
            return (
                ack(request, accepted_sequence=request["sequence"])
                if wants_ack
                else None
            )

        if message_type == "move_relative":
            await asyncio.to_thread(
                self.runtime.move_relative,
                connection_id,
                vector3(request, "displacement"),
                finite_float(request, "dynamics_factor", default=0.1),
            )
            return ack(
                request,
                accepted=True,
                server_monotonic_s=time.monotonic(),
            )

        if message_type == "move_global":
            absolute = request.get("absolute", False)
            if not isinstance(absolute, bool):
                raise ProtocolError("absolute must be true or false")
            raw_rotation = request.get("rotation_matrix")
            rotation = (
                None if raw_rotation is None else matrix3(request, "rotation_matrix")
            )
            await asyncio.to_thread(
                self.runtime.move_global,
                connection_id,
                vector3(request, "position"),
                absolute=absolute,
                dynamics_factor=finite_float(request, "dynamics_factor", default=0.1),
                rotation_matrix=rotation,
            )
            return ack(
                request,
                accepted=True,
                server_monotonic_s=time.monotonic(),
            )

        if message_type == "stop":
            # Every authenticated observer may stop the robot; no lease required.
            await asyncio.to_thread(self.runtime.stop_all)
            return ack(request, stopped=True)

        if message_type == "recover_errors":
            recovered = await asyncio.to_thread(
                self.runtime.recover_from_errors, connection_id
            )
            return ack(request, recovered=recovered)

        if message_type == "get_state":
            robot = await asyncio.to_thread(self.runtime.state_snapshot)
            return ack(
                request,
                robot=robot,
                bridge=self.runtime.bridge_status(),
            )

        if message_type == "ping":
            return ack(request, server_monotonic_s=time.monotonic())

        raise ProtocolError(f"unsupported message type: {message_type}")

    async def _state_stream(self, websocket: Any, send_lock: asyncio.Lock) -> None:
        period = 1.0 / self.config.state_hz
        sequence = 0
        while True:
            started = time.monotonic()
            try:
                robot = await asyncio.to_thread(self.runtime.state_snapshot)
                message = {
                    "type": "state",
                    "sequence": sequence,
                    "server_monotonic_s": started,
                    "robot": robot,
                    "bridge": self.runtime.bridge_status(),
                }
            except Exception as error:
                message = error_response(
                    None,
                    "state_read_error",
                    f"{type(error).__name__}: {error}",
                )
            await self._send(websocket, send_lock, message)
            sequence += 1
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    @staticmethod
    async def _send(
        websocket: Any,
        send_lock: asyncio.Lock,
        message: dict[str, Any],
    ) -> None:
        async with send_lock:
            await asyncio.wait_for(
                websocket.send(encode_message(message)),
                timeout=0.5,
            )

    @staticmethod
    def _peer_ip(websocket: Any) -> str:
        remote = websocket.remote_address
        if not isinstance(remote, tuple) or not remote:
            raise ProtocolError("cannot determine client address")
        return str(remote[0])


def load_controller(config: ServerConfig, *, dry_run: bool) -> ControllerLike:
    if dry_run:
        return MockFrankaController()
    controller_root = str(Path(config.controller_root).expanduser().resolve())
    if controller_root not in sys.path:
        sys.path.insert(0, controller_root)
    module_name, class_name = config.controller_class.split(":", 1)
    module = importlib.import_module(module_name)
    controller_class = getattr(module, class_name)
    return controller_class(
        host=config.robot_host,
        relative_dynamics_factor=config.relative_dynamics_factor,
        recover_errors=config.recover_errors_on_start,
        command_timeout_ms=config.franky_command_timeout_ms,
        max_linear_speed=config.max_linear_speed,
        max_angular_speed=config.max_angular_speed,
    )


def read_token() -> str:
    token = os.environ.get(TOKEN_ENV, "")
    if len(token) < 16:
        raise RuntimeError(
            f"set {TOKEN_ENV} to a random secret containing at least 16 characters"
        )
    return token


async def serve(config: ServerConfig, *, dry_run: bool = False) -> None:
    controller = load_controller(config, dry_run=dry_run)
    runtime = FrankaRuntime(controller, config)
    bridge = FrankaBridgeServer(runtime, config, read_token())
    mode = "DRY-RUN" if dry_run else "REAL ROBOT"
    LOGGER.warning(
        "starting %s server on %s:%d; allowlist=%s",
        mode,
        config.bind_host,
        config.port,
        ",".join(config.allowed_client_cidrs),
    )
    try:
        async with websockets.serve(
            bridge.handle_connection,
            config.bind_host,
            config.port,
            compression=None,
            ping_interval=2.0,
            ping_timeout=2.0,
            close_timeout=1.0,
            max_size=64 * 1024,
            max_queue=8,
        ):
            await asyncio.Future()
    finally:
        await asyncio.to_thread(runtime.close)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot-side Franka LAN bridge")
    parser.add_argument("--config", type=Path, default=Path("server_config.json"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use a simulated controller and never connect to a robot",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = ServerConfig.from_file(args.config.resolve())
    try:
        asyncio.run(serve(config, dry_run=args.dry_run))
    except KeyboardInterrupt:
        LOGGER.warning("server interrupted; robot stop requested")


if __name__ == "__main__":
    main()
