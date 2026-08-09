"""Make the repository and standalone collector importable during discovery."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The Windows leaptele environment can otherwise stall while NumPy asks its
# BLAS backend for a large worker pool.  Tests only need tiny 3x3 operations.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


COLLECTOR_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = COLLECTOR_ROOT.parent

for path in (REPOSITORY_ROOT, COLLECTOR_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
