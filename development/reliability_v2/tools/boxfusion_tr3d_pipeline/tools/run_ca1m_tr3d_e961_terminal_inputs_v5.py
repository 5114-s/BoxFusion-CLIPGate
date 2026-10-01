#!/usr/bin/env python3
"""Fail-closed runner skeleton for E961 xfit-R2 terminal-v5 inputs.

This pending revision intentionally has no reachable inference/materialization
surface.  Every future P/O/E/M mode first consumes a separately sealed ready
authorization, then four authoritative role receipts, before it may create a
namespace directory.  The current config therefore rejects all operational
modes without opening a receipt or starting a GPU.
"""

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
    ROLE_ORDER,
    static_report,
    validate_operational_ready,
)
from tools.preflight_ca1m_tr3d_e961_terminal_inputs_v5 import blocked  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-contract", action="store_true")
    mode.add_argument("--operational-preflight", action="store_true")
    mode.add_argument("--run-role", choices=ROLE_ORDER)
    mode.add_argument("--seal-exact80", action="store_true")
    value.add_argument("--device", default="cuda:0")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.static_contract:
        report = static_report(args.config)
        report["runner_surface"] = {
            "roles": list(ROLE_ORDER),
            "stage_order": [
                "P_role_anchor_free", "O_cpu_overlay",
                "E_candidate_native", "M_exact80_manifest",
            ],
            "operational_actions_reachable": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    try:
        # This is deliberately before device validation, receipt/path
        # resolution, mkdir, worker construction, or candidate access.
        validate_operational_ready(args.config)
    except PendingE961InputsError as error:
        report = blocked(error)
        report.update({
            "requested_role": args.run_role,
            "requested_seal": bool(args.seal_exact80),
            "device_argument_consumed": False,
        })
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 3
    raise AssertionError("pending runner unexpectedly reached an operational action")


if __name__ == "__main__":
    raise SystemExit(main())
