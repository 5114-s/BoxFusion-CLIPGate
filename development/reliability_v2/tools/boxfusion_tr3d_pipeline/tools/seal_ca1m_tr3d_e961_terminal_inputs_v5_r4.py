#!/usr/bin/env python3
"""Create-only R3 invalidation, R4 preregistration, or R4 ready bundle."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r4 import (  # noqa: E402
    seal_preregistration, seal_r3_invalidation, seal_ready_authorization_bundle,
    sha256_file,
)

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--invalidate-r3", action="store_true")
    mode.add_argument("--seal-preregistration", action="store_true")
    mode.add_argument("--seal-ready", action="store_true")
    for name in ("outer-receipt", "inner-holdout2-receipt", "inner-holdout3-receipt", "inner-holdout4-receipt", "continuation-receipt"):
        value.add_argument(f"--{name}", type=Path)
    return value

def main() -> int:
    args = parser().parse_args()
    dynamic = [args.outer_receipt, args.inner_holdout2_receipt, args.inner_holdout3_receipt, args.inner_holdout4_receipt, args.continuation_receipt]
    if args.invalidate_r3:
        if any(dynamic): raise ValueError("R3 invalidation forbids dynamic receipts")
        path = seal_r3_invalidation(); result = {"status": "INVALIDATED_R3_V2", "path": str(path), "sha256": sha256_file(path)}
    elif args.seal_preregistration:
        if any(dynamic): raise ValueError("static preregistration forbids dynamic receipts")
        path = seal_preregistration(); result = {"status": "SEALED_R4_STATIC_PREREGISTRATION", "path": str(path), "sha256": sha256_file(path)}
    else:
        if any(item is None for item in dynamic): raise ValueError("ready bundle requires all five receipts")
        ready, auth, bundle = seal_ready_authorization_bundle(
            outer_receipt=args.outer_receipt, inner_holdout2_receipt=args.inner_holdout2_receipt,
            inner_holdout3_receipt=args.inner_holdout3_receipt, inner_holdout4_receipt=args.inner_holdout4_receipt,
            continuation_receipt=args.continuation_receipt,
        )
        result = {"status": "SEALED_R4_READY_BUNDLE", "ready": str(ready), "ready_sha256": sha256_file(ready), "authorization": str(auth), "authorization_sha256": sha256_file(auth), "bundle": str(bundle), "bundle_sha256": sha256_file(bundle), "gpu_started": False}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
