"""Move through four calibrated FR3 points, one SPACE press at a time."""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path
from typing import Any, Protocol

from .client import FrankaBridgeClient
from .terminal_keys import TerminalKeys
from .waypoints import Waypoint, WaypointSet, distance_m, rotation_distance_rad


class KeySource(Protocol):
    def poll(self) -> str | None: ...


class OperatorStop(RuntimeError):
    pass


async def run_sequence(args: argparse.Namespace) -> None:
    waypoints = WaypointSet.load(args.points.resolve())
    target_index = args.start_index - 1

    async with FrankaBridgeClient(
        args.uri,
        client_name=args.client_name,
        request_timeout_s=args.request_timeout,
    ) as client:
        if not bool(client.safety.get("allow_one_shot_motion")):
            raise RuntimeError(
                "server has allow_one_shot_motion=false; validate the workspace "
                "and enable it in server_config.json before playback"
            )
        if not bool(client.safety.get("allow_orientation_motion")):
            raise RuntimeError(
                "server has allow_orientation_motion=false; validate the recorded "
                "orientations and enable it in server_config.json before playback"
            )
        missing_rotation = [
            point.name for point in waypoints.points if point.rotation_matrix is None
        ]
        if missing_rotation:
            raise RuntimeError(
                "waypoint file is missing rotation_matrix for "
                f"{', '.join(missing_rotation)}; update the server pose controller "
                "and recalibrate all four points"
            )
        waypoints.validate_workspace(client.safety)
        maximum_dynamics = float(client.safety.get("max_motion_dynamics_factor", 0.0))
        if args.dynamics_factor > maximum_dynamics:
            raise ValueError(
                f"dynamics factor {args.dynamics_factor} exceeds server limit "
                f"{maximum_dynamics}"
            )

        await client.acquire_control()
        print("Control acquired. Position and calibrated orientation will be planned.")
        print("SPACE: move to next point | E/Q: software stop and exit")
        try:
            with TerminalKeys() as keys:
                while True:
                    target = waypoints.points[target_index]
                    print(
                        f"Ready for {target.name} {target.position_m}; press SPACE.",
                        flush=True,
                    )
                    key = (await keys.wait()).lower()
                    if key in {"q", "e", "\x1b", "\x03"}:
                        await client.stop()
                        print("Software stop acknowledged.")
                        return
                    if key != " ":
                        continue

                    response = await client.move_global(
                        target.position_m,
                        absolute=True,
                        dynamics_factor=args.dynamics_factor,
                        rotation_matrix=target.rotation_matrix,
                    )
                    accepted_at_s = float(response["server_monotonic_s"])
                    print(f"Moving to {target.name}; E/Q stops motion.")
                    position_error, orientation_error = await wait_for_motion(
                        client,
                        target,
                        accepted_at_s=accepted_at_s,
                        timeout_s=args.motion_timeout,
                        arrival_tolerance_m=args.arrival_tolerance,
                        orientation_tolerance_rad=math.radians(
                            args.orientation_tolerance_deg
                        ),
                        keys=keys,
                    )
                    print(
                        f"Reached {target.name}; "
                        f"position error={position_error * 1000:.2f} mm; "
                        f"orientation error={math.degrees(orientation_error):.2f} deg"
                    )

                    target_index += 1
                    if target_index == len(waypoints.points):
                        if args.loop:
                            target_index = 0
                        else:
                            print("P1-P4 sequence completed.")
                            await client.release_control()
                            return
        except OperatorStop:
            print("Software stop acknowledged.")
            return
        except BaseException:
            await client.stop()
            raise


async def wait_for_motion(
    client: FrankaBridgeClient,
    target: Waypoint,
    *,
    accepted_at_s: float,
    timeout_s: float,
    arrival_tolerance_m: float,
    orientation_tolerance_rad: float,
    keys: KeySource,
) -> tuple[float, float]:
    deadline = time.monotonic() + timeout_s
    last_position: tuple[float, float, float] | None = None
    while time.monotonic() < deadline:
        key = keys.poll()
        if key is not None and key.lower() in {"q", "e", "\x1b", "\x03"}:
            await client.stop()
            raise OperatorStop("operator requested software stop")
        try:
            state = await client.next_state(timeout_s=0.05)
        except asyncio.TimeoutError:
            continue
        if float(state.get("server_monotonic_s", 0.0)) < accepted_at_s:
            continue
        last_position = _position_from_state(state)
        active = bool(state.get("bridge", {}).get("one_shot_active"))
        if not active:
            position_error = distance_m(last_position, target.position_m)
            current_rotation = _rotation_from_state(state)
            if target.rotation_matrix is None:
                await client.stop()
                raise RuntimeError(
                    f"{target.name} has no rotation_matrix; stop acknowledged"
                )
            orientation_error = rotation_distance_rad(
                current_rotation,
                target.rotation_matrix,
            )
            if position_error > arrival_tolerance_m:
                await client.stop()
                raise RuntimeError(
                    f"motion ended {position_error * 1000:.2f} mm from "
                    f"{target.name}; stop acknowledged"
                )
            if orientation_error > orientation_tolerance_rad:
                await client.stop()
                raise RuntimeError(
                    f"motion ended {math.degrees(orientation_error):.2f} deg from "
                    f"{target.name} orientation; "
                    "stop acknowledged"
                )
            return position_error, orientation_error

    await client.stop()
    suffix = "" if last_position is None else f"; last position={last_position}"
    raise TimeoutError(f"motion timed out; stop acknowledged{suffix}")


def _position_from_state(state: dict[str, Any]) -> tuple[float, float, float]:
    robot = state.get("robot")
    end_effector = robot.get("end_effector") if isinstance(robot, dict) else None
    raw = end_effector.get("position") if isinstance(end_effector, dict) else None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise RuntimeError("state is missing end_effector.position")
    position = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in position):
        raise RuntimeError("end_effector.position contains non-finite values")
    return position  # type: ignore[return-value]


def _rotation_from_state(
    state: dict[str, Any],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    robot = state.get("robot")
    end_effector = robot.get("end_effector") if isinstance(robot, dict) else None
    raw = (
        end_effector.get("rotation_matrix") if isinstance(end_effector, dict) else None
    )
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise RuntimeError("state is missing end_effector.rotation_matrix")
    rows = tuple(
        tuple(float(item) for item in row)
        for row in raw
        if isinstance(row, (list, tuple)) and len(row) == 3
    )
    if len(rows) != 3 or not all(math.isfinite(item) for row in rows for item in row):
        raise RuntimeError("end_effector.rotation_matrix is not a finite 3x3 matrix")
    return rows  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move to the next calibrated FR3 point on each SPACE press"
    )
    parser.add_argument("--uri", required=True)
    parser.add_argument("--points", type=Path, default=Path("calibrated_points.json"))
    parser.add_argument("--start-index", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--dynamics-factor", type=float, default=0.05)
    parser.add_argument("--motion-timeout", type=float, default=30.0)
    parser.add_argument("--arrival-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--client-name", default="four-point-playback")
    parser.add_argument("--request-timeout", type=float, default=2.0)
    return parser


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_positive(args.dynamics_factor, "--dynamics-factor")
    _validate_positive(args.motion_timeout, "--motion-timeout")
    _validate_positive(args.arrival_tolerance, "--arrival-tolerance")
    _validate_positive(
        args.orientation_tolerance_deg,
        "--orientation-tolerance-deg",
    )
    try:
        asyncio.run(run_sequence(args))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; client shutdown requested a stop.") from None


if __name__ == "__main__":
    main()
