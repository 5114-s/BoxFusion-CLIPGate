#!/usr/bin/env python3
"""Read-only preflight for the pending CA-1M exploratory gate v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_exploratory_gate_v5 import (  # noqa: E402
    PendingProtocolError,
    static_report,
    validate_ready,
)


DEFAULT_CONFIG = (
    ROOT / "config/ca1m_tr3d_exploratory_gate_xfit_r2_v5_pending.json"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-contract", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.static_contract:
        print(json.dumps(static_report(args.config), indent=2, sort_keys=True))
        return 0
    try:
        validate_ready(args.config)
    except PendingProtocolError as error:
        print(json.dumps({
            "schema": "boxfusion.ca1m_tr3d_exploratory_gate_preflight.v5",
            "ok": False,
            "mode": "operational_preflight",
            "state": "pending",
            "failure_action": "stop_before_opening_candidate_gt_or_output",
            "candidate_or_gt_artifact_opened": False,
            "output_created": False,
            "error": str(error),
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 3
    raise AssertionError("pending v5 unexpectedly became operational")


if __name__ == "__main__":
    raise SystemExit(main())
