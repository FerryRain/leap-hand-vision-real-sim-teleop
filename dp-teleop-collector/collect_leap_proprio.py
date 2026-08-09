"""Python entry point for image-free LEAP grasp demonstration collection."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
for path in (REPOSITORY_ROOT,):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from dp_collector.proprio_collector import main  # noqa: E402

if __name__ == "__main__":
    main()
