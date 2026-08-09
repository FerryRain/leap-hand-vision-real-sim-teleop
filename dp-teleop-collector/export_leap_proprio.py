"""Export image-free LEAP grasp episodes to Diffusion Policy Zarr v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dp_collector.proprio_exporter import export_proprio_to_zarr  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-length", type=int, default=256)
    args = parser.parse_args()
    summary = export_proprio_to_zarr(
        args.dataset_root,
        args.output,
        include_rejected=args.include_rejected,
        overwrite=args.overwrite,
        chunk_length=args.chunk_length,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
