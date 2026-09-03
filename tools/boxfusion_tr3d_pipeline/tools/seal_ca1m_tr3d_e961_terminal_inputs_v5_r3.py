#!/usr/bin/env python3
"""Create-only static preregistration or dynamic ready/auth pair for R3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r3 import (  # noqa: E402
    RUN_AUTHORIZATION_PATH, READY_CONFIG_PATH, seal_preregistration,
    seal_preregistration_v1_invalidation, seal_ready_and_authorization, sha256_file,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal-preregistration", action="store_true")
    mode.add_argument("--invalidate-preregistration-v1", action="store_true")
    mode.add_argument("--seal-ready", action="store_true")
    value.add_argument("--outer-receipt", type=Path)
    value.add_argument("--inner-holdout2-receipt", type=Path)
    value.add_argument("--inner-holdout3-receipt", type=Path)
    value.add_argument("--inner-holdout4-receipt", type=Path)
    value.add_argument("--continuation-receipt", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    dynamic = (
        args.outer_receipt, args.inner_holdout2_receipt, args.inner_holdout3_receipt,
        args.inner_holdout4_receipt, args.continuation_receipt,
    )
    if args.invalidate_preregistration_v1:
        if any(item is not None for item in dynamic):
            raise ValueError("static invalidation forbids dynamic receipt arguments")
        path = seal_preregistration_v1_invalidation()
        result = {"status": "INVALIDATED_STATIC_PREREGISTRATION_V1", "path": str(path), "sha256": sha256_file(path)}
    elif args.seal_preregistration:
        if any(item is not None for item in dynamic):
            raise ValueError("static preregistration forbids dynamic receipt arguments")
        path = seal_preregistration()
        result = {"status": "SEALED_STATIC_PREREGISTRATION", "path": str(path), "sha256": sha256_file(path)}
    else:
        if any(item is None for item in dynamic):
            raise ValueError("--seal-ready requires outer, three inner, and continuation receipts")
        ready, authorization = seal_ready_and_authorization(
            outer_receipt=args.outer_receipt,
            inner_holdout2_receipt=args.inner_holdout2_receipt,
            inner_holdout3_receipt=args.inner_holdout3_receipt,
            inner_holdout4_receipt=args.inner_holdout4_receipt,
            continuation_receipt=args.continuation_receipt,
        )
        if ready != READY_CONFIG_PATH or authorization != RUN_AUTHORIZATION_PATH:
            raise RuntimeError("sealer returned noncanonical dynamic paths")
        result = {
            "status": "SEALED_OPERATIONAL_READY", "ready_config": str(ready),
            "ready_config_sha256": sha256_file(ready), "run_authorization": str(authorization),
            "run_authorization_sha256": sha256_file(authorization),
            "gpu_started": False, "ground_truth_access": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
