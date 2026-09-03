#!/usr/bin/env python3
"""Static or fail-closed operational preflight for E961 terminal-input R3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r3 import (  # noqa: E402
    DEFAULT_CONFIG, OPERATIONAL_REPORT_SCHEMA, PendingOperationalInputs,
    validate_operational_ready, validate_static_config,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--operational", action="store_true")
    return value


def blocked(error: Exception) -> dict[str, object]:
    return {
        "schema": OPERATIONAL_REPORT_SCHEMA, "status": "BLOCKED_PENDING",
        "reason": str(error), "receipt_or_checkpoint_opened": False,
        "output_created": False, "gpu_started": False, "ground_truth_access": False,
    }


def main() -> int:
    args = parser().parse_args()
    if args.static:
        print(json.dumps(validate_static_config(args.config or DEFAULT_CONFIG), indent=2, sort_keys=True))
        return 0
    try:
        ctx = validate_operational_ready(args.config)
    except PendingOperationalInputs as error:
        # stdout intentionally remains byte-empty for pending operational use.
        print(json.dumps(blocked(error), sort_keys=True), file=sys.stderr)
        return 3
    try:
        print(json.dumps({
            "schema": OPERATIONAL_REPORT_SCHEMA, "status": "PASS_OPERATIONAL_READY",
            "role_count": len(ctx.roles), "output_created": False,
            "gpu_started": False, "ground_truth_access": False,
        }, indent=2, sort_keys=True))
    finally:
        ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
