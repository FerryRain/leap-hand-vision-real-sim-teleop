"""Command-line client for commissioning and observing the bridge."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence

from .client import FrankaBridgeClient


async def run_command(args: argparse.Namespace) -> None:
    async with FrankaBridgeClient(
        args.uri,
        client_name=args.client_name,
        request_timeout_s=args.request_timeout,
    ) as client:
        if args.command == "state":
            state = await client.get_state()
            print(json.dumps(state, indent=2, ensure_ascii=False))
            return
        if args.command == "monitor":
            while True:
                state = await client.next_state(timeout_s=2.0)
                print(json.dumps(state, ensure_ascii=False))
            return
        if args.command == "stop":
            await client.stop()
            print("stop acknowledged")
            return

        await client.acquire_control()
        if args.command == "velocity":
            await _run_velocity(client, args)
        elif args.command == "move-relative":
            response = await client.move_relative(
                args.xyz, dynamics_factor=args.dynamics_factor
            )
            await _wait_for_one_shot(
                client,
                args.motion_timeout,
                float(response["server_monotonic_s"]),
            )
        elif args.command == "move-global":
            response = await client.move_global(
                args.xyz,
                absolute=args.absolute,
                dynamics_factor=args.dynamics_factor,
            )
            await _wait_for_one_shot(
                client,
                args.motion_timeout,
                float(response["server_monotonic_s"]),
            )
        elif args.command == "recover-errors":
            print(f"recovered={await client.recover_errors()}")
        await client.release_control()


async def _run_velocity(client: FrankaBridgeClient, args: argparse.Namespace) -> None:
    period = 1.0 / args.rate
    started = time.monotonic()
    sent = 0
    try:
        while time.monotonic() - started < args.duration:
            iteration = time.monotonic()
            await client.send_velocity(
                args.linear,
                args.angular,
                frame=args.frame,
                wait_ack=sent % 10 == 0,
            )
            sent += 1
            await asyncio.sleep(max(0.0, period - (time.monotonic() - iteration)))
    finally:
        await client.stop()
    print(f"velocity commands sent={sent}; stop acknowledged")


async def _wait_for_one_shot(
    client: FrankaBridgeClient,
    timeout_s: float,
    accepted_at_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = await client.next_state(timeout_s=2.0)
        if float(state.get("server_monotonic_s", 0.0)) < accepted_at_s:
            continue
        active = bool(state.get("bridge", {}).get("one_shot_active"))
        if not active:
            print("motion completed")
            return
    await client.stop()
    raise TimeoutError("motion timeout; stop acknowledged")


def vector3(values: Sequence[str]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three values")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Franka LAN bridge client")
    parser.add_argument(
        "--uri", required=True, help="for example ws://192.168.1.20:8765"
    )
    parser.add_argument("--client-name", default="teleop-client")
    parser.add_argument("--request-timeout", type=float, default=2.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("state", help="request one complete state snapshot")
    subparsers.add_parser("monitor", help="print the state stream as JSON lines")
    subparsers.add_parser("stop", help="stop motion without acquiring control")

    velocity = subparsers.add_parser("velocity", help="send watchdog velocity")
    velocity.add_argument("--frame", choices=("local", "global"), default="global")
    velocity.add_argument("--linear", nargs=3, type=float, required=True)
    velocity.add_argument("--angular", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    velocity.add_argument("--duration", type=float, required=True)
    velocity.add_argument("--rate", type=float, default=30.0)

    relative = subparsers.add_parser("move-relative")
    relative.add_argument("--xyz", nargs=3, type=float, required=True)
    _add_motion_options(relative)

    global_motion = subparsers.add_parser("move-global")
    global_motion.add_argument("--xyz", nargs=3, type=float, required=True)
    global_motion.add_argument("--absolute", action="store_true")
    _add_motion_options(global_motion)

    subparsers.add_parser("recover-errors")
    return parser


def _add_motion_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dynamics-factor", type=float, default=0.1)
    parser.add_argument("--motion-timeout", type=float, default=30.0)


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "duration", 1.0) <= 0.0:
        raise SystemExit("--duration must be positive")
    if getattr(args, "rate", 1.0) <= 0.0:
        raise SystemExit("--rate must be positive")
    asyncio.run(run_command(args))


if __name__ == "__main__":
    main()
