#!/usr/bin/env python3
"""V2 entry point for the R2-bound E961 outer evaluation/continuation chain."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_ca1m_tr3d_e961_outer_eval_v1 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
