#!/usr/bin/env python3
"""Run the authorized E961 terminal-input v5 R2 P/O/E/M chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r2 import (  # noqa: E402
    DEFAULT_CONFIG, PendingOperationalInputs, ROLE_ORDER, run_all, run_stage_e,
    run_stage_o, run_stage_p, seal_stage_m, validate_operational_ready,
    validate_static_config,
)
from tools.preflight_ca1m_tr3d_e961_terminal_inputs_v5_r2 import blocked  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--device", default="cuda:0")
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-contract", action="store_true")
    mode.add_argument("--operational-preflight", action="store_true")
    mode.add_argument("--run-p", choices=ROLE_ORDER)
    mode.add_argument("--run-o", choices=ROLE_ORDER)
    mode.add_argument("--run-e", choices=ROLE_ORDER)
    mode.add_argument("--seal-m", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.static_contract:
        print(json.dumps(validate_static_config(args.config), indent=2, sort_keys=True))
        return 0
    try:
        # Always before device use, mkdir, worker construction, or artifact P/O/E/M.
        ctx = validate_operational_ready(args.config)
    except PendingOperationalInputs as error:
        report = blocked(error)
        report["requested_mode"] = next(name for name in (
            "operational_preflight", "run_p", "run_o", "run_e", "seal_m", "run_all"
        ) if getattr(args, name, None))
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 3
    if args.operational_preflight:
        result = {"status": "PASS_OPERATIONAL_READY", "output_created": False, "gpu_started": False}
    elif args.run_p:
        result = run_stage_p(ctx, args.run_p, device=args.device)
    elif args.run_o:
        result = run_stage_o(ctx, args.run_o)
    elif args.run_e:
        result = run_stage_e(ctx, args.run_e)
    elif args.seal_m:
        result = seal_stage_m(ctx)
    else:
        result = run_all(ctx, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
