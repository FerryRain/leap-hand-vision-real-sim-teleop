"""Validate image-free LEAP grasp episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dp_collector.proprio_episode import validate_proprio_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    reports = []
    for status in ("accepted", "rejected", ".partial"):
        parent = root / status
        if not parent.is_dir():
            continue
        for path in sorted(item for item in parent.iterdir() if item.is_dir()):
            report = validate_proprio_episode(path)
            reports.append(
                {
                    "path": str(path),
                    "status": report.status,
                    "num_steps": report.num_steps,
                    "ok": report.ok,
                    "errors": report.errors,
                }
            )
    payload = {
        "dataset_root": str(root),
        "episodes": reports,
        "ok": bool(reports) and all(item["ok"] for item in reports),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
