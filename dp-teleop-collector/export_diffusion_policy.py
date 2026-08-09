"""Command-line export of recorded episodes to Diffusion Policy Zarr v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dp_collector.exporter import export_to_zarr, summary_as_json
from dp_collector.schema import TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export accepted grasp/release demonstrations to a Zarr v2 dataset."
    )
    parser.add_argument("dataset_root", type=Path, help="collector dataset directory")
    parser.add_argument("output", type=Path, help="new output .zarr directory")
    parser.add_argument("--task", choices=TASKS, help="export only one task")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="include rejected episodes (off by default)",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="drop invalid/stale frames instead of retaining data/valid as a mask",
    )
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "skip image decoding during pre-validation "
            "(images are still decoded for export)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = export_to_zarr(
            args.dataset_root,
            args.output,
            include_rejected=args.include_rejected,
            task=args.task,
            valid_only=args.valid_only,
            deep_validation=not args.fast,
            chunk_length=args.chunk_length,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 2
    print(summary_as_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
