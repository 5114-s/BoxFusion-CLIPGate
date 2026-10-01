#!/usr/bin/env python3
"""Run the authorized CPU-only terminal-gate-v5-final-R4 chain."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v5_final_r4 import (  # noqa: E402
    PendingR6Inputs, operational_preflight_pending, run_final_gate,
)


def main() -> int:
    try:
        operational_preflight_pending()
        output = run_final_gate()
    except (PendingR6Inputs, FileNotFoundError) as error:
        print(json.dumps({
            "status": "BLOCKED_PENDING", "runtime_ready": False,
            "ground_truth_access": False, "output_created": False,
            "directory_created": False, "gpu_started": False,
            "reason": str(error),
        }, sort_keys=True), file=sys.stderr)
        return 3
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
