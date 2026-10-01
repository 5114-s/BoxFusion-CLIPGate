#!/usr/bin/env python3
"""GT-free preflight for the pending E961 xfit-R2 terminal-v5 input chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_e961_terminal_inputs_v5 import (  # noqa: E402
    DEFAULT_CONFIG,
    PendingE961InputsError,
    static_report,
    validate_operational_ready,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-contract", action="store_true")
    mode.add_argument("--operational-preflight", action="store_true")
    return value


def blocked(error: Exception) -> dict[str, object]:
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_operational_preflight.v1",
        "ok": False,
        "mode": "operational_preflight",
        "state": "pending",
        "error": str(error),
        "failure_action": "stop_before_opening_receipt_checkpoint_candidate_or_output",
        "receipt_opened": False,
        "checkpoint_opened": False,
        "candidate_or_ground_truth_artifact_opened": False,
        "fold1_or_official_validation_path_resolved": False,
        "gpu_started": False,
        "output_created": False,
    }


def main() -> int:
    args = parser().parse_args()
    if args.static_contract:
        print(json.dumps(static_report(args.config), indent=2, sort_keys=True))
        return 0
    try:
        validate_operational_ready(args.config)
    except PendingE961InputsError as error:
        print(json.dumps(blocked(error), indent=2, sort_keys=True), file=sys.stderr)
        return 3
    raise AssertionError("pending input chain unexpectedly became operational")


if __name__ == "__main__":
    raise SystemExit(main())
