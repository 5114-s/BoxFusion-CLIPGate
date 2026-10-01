#!/usr/bin/env python3
"""Static or read-only operational preflight for terminal gate v5 final."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v5_final_r4 import (  # noqa: E402
    DEFAULT_PENDING_CONFIG, PendingR6Inputs, load_execution_context,
    operational_preflight_pending, static_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_PENDING_CONFIG)
    parser.add_argument(
        "--mode", choices=("static", "r6", "authorized"), default="static"
    )
    args = parser.parse_args()
    if args.mode == "static":
        print(json.dumps(static_preflight(args.config), indent=2, sort_keys=True))
        return 0
    if args.mode == "r6":
        try:
            operational_preflight_pending(args.config)
        except PendingR6Inputs as error:
            print(json.dumps({
                "status": "BLOCKED_PENDING_R6_EXACT80", "runtime_ready": False,
                "ground_truth_access": False, "output_created": False,
                "directory_created": False, "gpu_started": False,
                "reason": str(error),
            }, sort_keys=True), file=sys.stderr)
            return 3
        print(json.dumps({
            "status": "R6_EXACT80_COMMIT_VALID", "runtime_ready": False,
            "next_required": "seal_scientific_preregistration_then_ready_authorization",
            "ground_truth_access": False, "output_created": False,
            "directory_created": False, "gpu_started": False,
        }, indent=2, sort_keys=True))
        return 0
    context = load_execution_context(require_outputs_absent=True)
    print(json.dumps({
        "status": "PASS_AUTHORIZED_PREFLIGHT", "runtime_ready": True,
        "authorization": str(context.authorization_path),
        "r6_wrapper_sha256": context.r6.wrapper_sha256,
        "candidate_collection_sha256": context.r6.collection_sha256,
        "scene_count": len(context.r6.scene_folds),
        "ground_truth_access": False, "output_created": False,
        "directory_created": False, "gpu_started": False,
        "fold1_access": False, "official_validation_access": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
