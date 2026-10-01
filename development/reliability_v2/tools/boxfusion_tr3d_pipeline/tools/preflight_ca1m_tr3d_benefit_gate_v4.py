#!/usr/bin/env python3
"""Fail-closed preflight/seal runner for the CA-1M terminal benefit gate v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v4 import (  # noqa: E402
    validate_ready,
    validate_static_config,
    write_binding_create_only,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config", type=Path,
        default=ROOT / "config/ca1m_tr3d_benefit_gate_train100_v4.json",
    )
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-contract", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.static_contract:
        path, cfg = validate_static_config(args.config)
        print(json.dumps({
            "ok": True,
            "mode": "static_contract",
            "config": str(path),
            "state": cfg["state"],
            "run_authorized": cfg["run_authorized"],
            "output_created": False,
        }, indent=2, sort_keys=True))
        return 0
    # Both operational modes first validate every prerequisite and every
    # create-only output path.  A pending/malformed chain returns before any
    # parent directory or artifact is created.
    binding = validate_ready(args.config)
    if args.preflight:
        print(json.dumps({
            "ok": True,
            "mode": "preflight",
            "output_created": False,
            "binding": binding,
        }, indent=2, sort_keys=True))
        return 0
    target = Path(str(binding["binding_output"]))
    write_binding_create_only(target, binding)
    print(json.dumps({
        "ok": True,
        "mode": "seal_training_binding",
        "binding": str(target),
        "output_created": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

