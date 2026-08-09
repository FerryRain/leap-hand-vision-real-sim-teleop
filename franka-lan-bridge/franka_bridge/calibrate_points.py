"""Interactive four-point calibration using live FR3 end-effector state."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from .client import FrankaBridgeClient
from .terminal_keys import TerminalKeys
from .waypoints import REQUIRED_POINT_COUNT, Waypoint, WaypointSet


async def calibrate(args: argparse.Namespace) -> bool:
    points: list[Waypoint] = []
    async with FrankaBridgeClient(
        args.uri,
        client_name=args.client_name,
        request_timeout_s=args.request_timeout,
    ) as client:
        print("Connected. This program only reads state; it does not acquire control.")
        print("SPACE: record point | U: undo | Q/E: cancel")
        print(f"Output: {args.output.resolve()}")
        with TerminalKeys() as keys:
            while len(points) < REQUIRED_POINT_COUNT:
                print(f"Waiting for P{len(points) + 1} ...", flush=True)
                key = (await keys.wait()).lower()
                if key in {"q", "e", "\x1b", "\x03"}:
                    print("Calibration cancelled; no file was written.")
                    return False
                if key in {"u", "\x08", "\x7f"}:
                    if points:
                        removed = points.pop()
                        print(f"Removed {removed.name}.")
                    else:
                        print("No recorded point to remove.")
                    continue
                if key != " ":
                    continue

                response = await client.get_state()
                robot = _robot_from_response(response)
                waypoint = Waypoint.from_robot_state(f"P{len(points) + 1}", robot)
                _validate_position_in_workspace(waypoint, client.safety)
                points.append(waypoint)
                x, y, z = waypoint.position_m
                print(f"Recorded {waypoint.name}: x={x:.6f} y={y:.6f} z={z:.6f} m")

    waypoint_set = WaypointSet.create(points)
    waypoint_set.save(args.output, overwrite=args.overwrite)
    print(f"Saved four points to {args.output.resolve()}")
    return True


def _robot_from_response(response: dict[str, Any]) -> dict[str, Any]:
    robot = response.get("robot")
    if not isinstance(robot, dict):
        raise RuntimeError("server response is missing robot state")
    return robot


def _validate_position_in_workspace(
    point: Waypoint,
    safety: dict[str, Any],
) -> None:
    temporary = WaypointSet.create([point, point, point, point])
    temporary.validate_workspace(safety)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record four FR3 end-effector points with the SPACE key"
    )
    parser.add_argument("--uri", required=True)
    parser.add_argument("--output", type=Path, default=Path("calibrated_points.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--client-name", default="four-point-calibration")
    parser.add_argument("--request-timeout", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        completed = asyncio.run(calibrate(args))
    except KeyboardInterrupt:
        raise SystemExit("Calibration interrupted; no file was written.") from None
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
