#!/usr/bin/env python3
"""Static/operational preflight for E961 terminal-input v5 R2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r2 import (  # noqa: E402
    DEFAULT_CONFIG, PendingOperationalInputs, validate_operational_ready,
    validate_static_config,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--operational", action="store_true")
    return value


def blocked(error: Exception) -> dict[str, object]:
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_operational_preflight.v2",
        "status": "BLOCKED_PENDING",
        "reason": str(error),
        "receipt_or_checkpoint_opened": False,
        "output_created": False,
        "gpu_started": False,
        "ground_truth_access": False,
    }


def main() -> int:
    args = parser().parse_args()
    if not args.operational:
        print(json.dumps(validate_static_config(args.config), indent=2, sort_keys=True))
        return 0
    try:
        ready = validate_operational_ready(args.config)
    except PendingOperationalInputs as error:
        print(json.dumps(blocked(error), indent=2, sort_keys=True), file=sys.stderr)
        return 3
    print(json.dumps({
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_operational_preflight.v2",
        "status": "PASS_OPERATIONAL_READY", "role_count": len(ready.roles),
        "output_created": False, "gpu_started": False, "ground_truth_access": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
