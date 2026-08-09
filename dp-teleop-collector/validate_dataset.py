"""Validate raw RGB/depth episodes before training or Zarr export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dp_collector.episode import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a teleoperation dataset.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--fast", action="store_true", help="do not decode every image")
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="ignore rejected and interrupted .partial episodes",
    )
    parser.add_argument(
        "--json", action="store_true", help="print a machine-readable report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_dataset(
        args.dataset_root,
        deep=not args.fast,
        include_partial=not args.accepted_only,
        include_rejected=not args.accepted_only,
    )
    if args.json:
        print(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
    else:
        status = "OK" if report.ok else "INVALID"
        print(
            f"{status}: {len(report.episodes)} episodes, "
            f"{report.num_steps} synchronized steps"
        )
        for error in report.errors:
            print(f"ERROR: {error}")
        for episode in report.episodes:
            marker = "OK" if episode.ok else "INVALID"
            print(
                f"{marker}: {episode.path} "
                f"task={episode.task} steps={episode.num_steps}"
            )
            for warning in episode.warnings:
                print(f"  WARNING: {warning}")
            for error in episode.errors:
                print(f"  ERROR: {error}")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
