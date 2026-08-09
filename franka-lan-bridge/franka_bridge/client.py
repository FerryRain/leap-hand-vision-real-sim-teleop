"""Reusable asynchronous client for the Franka LAN bridge."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import socket
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

import websockets

from .protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_message,
    encode_message,
    vector_norm,
)

TOKEN_ENV = "FRANKA_BRIDGE_TOKEN"


class RemoteError(RuntimeError):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        super().__init__(
            f"{response.get('code', 'remote_error')}: "
            f"{response.get('message', 'unknown server error')}"
        )


class FrankaBridgeClient:
    """Full-duplex state and command client suitable for teleoperation loops."""

    def __init__(
        self,
        uri: str,
        token: str | None = None,
        *,
        client_name: str | None = None,
        request_timeout_s: float = 2.0,
    ) -> None:
        self.uri = uri
        self.token = os.environ.get(TOKEN_ENV, "") if token is None else token
        if len(self.token) < 16:
            raise ValueError(
                f"provide token or set {TOKEN_ENV} to at least 16 characters"
            )
        self.client_name = client_name or socket.gethostname()
        self.request_timeout_s = request_timeout_s
        self.websocket: Any | None = None
        self.connection_id: str | None = None
        self.safety: dict[str, Any] = {}
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._state_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=16)
        self._receiver_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._owns_control = False
        self._velocity_sequence = 0

    async def connect(self) -> "FrankaBridgeClient":
        if self.websocket is not None:
            return self
        self.websocket = await websockets.connect(
            self.uri,
            compression=None,
            ping_interval=2.0,
            ping_timeout=2.0,
            close_timeout=1.0,
            max_size=64 * 1024,
            max_queue=8,
        )
        await self._send_raw(
            {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "token": self.token,
                "client_name": self.client_name,
            }
        )
        response = decode_message(
            await asyncio.wait_for(
                self.websocket.recv(), timeout=self.request_timeout_s
            )
        )
        if response.get("type") != "hello":
            await self.websocket.close()
            self.websocket = None
            raise ProtocolError(f"unexpected handshake response: {response}")
        self.connection_id = str(response["connection_id"])
        self.safety = dict(response.get("safety", {}))
        self._receiver_task = asyncio.create_task(
            self._receive_loop(), name="franka-bridge-receiver"
        )
        return self

    async def acquire_control(self) -> None:
        response = await self.request("acquire_control")
        if not response.get("acquired"):
            raise RemoteError(response)
        self._owns_control = True
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="franka-bridge-heartbeat"
        )

    async def release_control(self) -> None:
        await self._cancel_heartbeat()
        if self._owns_control:
            await self.request("release_control")
        self._owns_control = False

    async def send_velocity(
        self,
        linear: Sequence[float],
        angular: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        frame: str = "global",
        wait_ack: bool = False,
    ) -> None:
        if not self._owns_control:
            raise RuntimeError("acquire control before sending velocity")
        normalized_frame = frame.lower()
        if normalized_frame not in {"local", "global"}:
            raise ValueError("frame must be local or global")
        linear_vector = self._vector3(linear, "linear")
        angular_vector = self._vector3(angular, "angular")
        if vector_norm(linear_vector) > float(
            self.safety.get("max_linear_speed", float("inf"))
        ):
            raise ValueError("linear velocity exceeds server-advertised limit")
        if vector_norm(angular_vector) > float(
            self.safety.get("max_angular_speed", float("inf"))
        ):
            raise ValueError("angular velocity exceeds server-advertised limit")
        self._velocity_sequence += 1
        fields = {
            "sequence": self._velocity_sequence,
            "frame": normalized_frame,
            "linear": linear_vector,
            "angular": angular_vector,
        }
        if wait_ack:
            await self.request("velocity", **fields)
        else:
            await self._send_raw({"type": "velocity", **fields})

    async def move_relative(
        self,
        displacement: Sequence[float],
        *,
        dynamics_factor: float = 0.1,
    ) -> dict[str, Any]:
        self._require_control()
        return await self.request(
            "move_relative",
            displacement=[float(item) for item in displacement],
            dynamics_factor=dynamics_factor,
        )

    async def move_global(
        self,
        position: Sequence[float],
        *,
        absolute: bool = False,
        dynamics_factor: float = 0.1,
        rotation_matrix: Sequence[Sequence[float]] | None = None,
    ) -> dict[str, Any]:
        self._require_control()
        fields: dict[str, Any] = {
            "position": [float(item) for item in position],
            "absolute": absolute,
            "dynamics_factor": dynamics_factor,
        }
        if rotation_matrix is not None:
            fields["rotation_matrix"] = self._matrix3(
                rotation_matrix,
                "rotation_matrix",
            )
        return await self.request("move_global", **fields)

    async def stop(self) -> None:
        await self.request("stop")
        self._owns_control = False
        await self._cancel_heartbeat()

    async def recover_errors(self) -> bool:
        self._require_control()
        response = await self.request("recover_errors")
        return bool(response.get("recovered"))

    async def get_state(self) -> dict[str, Any]:
        return await self.request("get_state")

    async def next_state(self, timeout_s: float | None = None) -> dict[str, Any]:
        if timeout_s is None:
            return await self._state_queue.get()
        return await asyncio.wait_for(self._state_queue.get(), timeout=timeout_s)

    async def states(self) -> AsyncIterator[dict[str, Any]]:
        while self.websocket is not None:
            yield await self.next_state()

    async def next_event(self, timeout_s: float | None = None) -> dict[str, Any]:
        if timeout_s is None:
            return await self._event_queue.get()
        return await asyncio.wait_for(self._event_queue.get(), timeout=timeout_s)

    async def request(self, message_type: str, **fields: Any) -> dict[str, Any]:
        self._require_connection()
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send_raw(
                {"type": message_type, "request_id": request_id, **fields}
            )
            response = await asyncio.wait_for(future, timeout=self.request_timeout_s)
        finally:
            self._pending.pop(request_id, None)
        if response.get("type") == "error":
            raise RemoteError(response)
        return response

    async def close(self) -> None:
        await self._cancel_heartbeat()
        if self.websocket is not None:
            if self._owns_control:
                with contextlib.suppress(Exception):
                    await self.stop()
                with contextlib.suppress(Exception):
                    await self.request("release_control")
            self._owns_control = False
            await self.websocket.close()
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver_task
        self._receiver_task = None
        self.websocket = None
        self.connection_id = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("client closed"))
        self._pending.clear()

    async def __aenter__(self) -> "FrankaBridgeClient":
        return await self.connect()

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def _receive_loop(self) -> None:
        assert self.websocket is not None
        try:
            async for raw_message in self.websocket:
                message = decode_message(raw_message)
                request_id = message.get("request_id")
                if isinstance(request_id, str) and request_id in self._pending:
                    future = self._pending[request_id]
                    if not future.done():
                        future.set_result(message)
                elif message.get("type") == "state":
                    if self._state_queue.full():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            self._state_queue.get_nowait()
                    self._state_queue.put_nowait(message)
                else:
                    if self._event_queue.full():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            self._event_queue.get_nowait()
                    self._event_queue.put_nowait(message)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)

    async def _heartbeat_loop(self) -> None:
        lease_ms = float(self.safety.get("lease_timeout_ms", 1000.0))
        period = max(0.05, lease_ms / 3000.0)
        try:
            while self._owns_control:
                await asyncio.sleep(period)
                await self.request("heartbeat")
        except asyncio.CancelledError:
            raise
        except Exception:
            # The server's lease and velocity watchdogs perform the actual stop.
            self._owns_control = False

    async def _cancel_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        self._heartbeat_task = None

    async def _send_raw(self, message: dict[str, Any]) -> None:
        self._require_connection(allow_connecting=message.get("type") == "hello")
        assert self.websocket is not None
        async with self._send_lock:
            await self.websocket.send(encode_message(message))

    def _require_control(self) -> None:
        if not self._owns_control:
            raise RuntimeError("acquire control before commanding motion")

    def _require_connection(self, *, allow_connecting: bool = False) -> None:
        if self.websocket is None and not allow_connecting:
            raise ConnectionError("client is not connected")

    @staticmethod
    def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
        if len(value) != 3:
            raise ValueError(f"{name} must contain three values")
        vector = tuple(float(item) for item in value)
        if not all(math.isfinite(item) for item in vector):
            raise ValueError(f"{name} must contain finite values")
        return vector  # type: ignore[return-value]

    @classmethod
    def _matrix3(
        cls,
        value: Sequence[Sequence[float]],
        name: str,
    ) -> list[list[float]]:
        if len(value) != 3:
            raise ValueError(f"{name} must contain three rows")
        return [list(cls._vector3(row, f"{name} row")) for row in value]
