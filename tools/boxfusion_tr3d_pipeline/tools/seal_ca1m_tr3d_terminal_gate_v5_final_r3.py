#!/usr/bin/env python3
"""Seal terminal-gate-v5-final science, then its last run authorization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v5_final_r3 import (  # noqa: E402
    PREREGISTRATION_PATH, PROTOCOL_PATH, READY_CONFIG_PATH, RUN_AUTHORIZATION_PATH,
    seal_ready_authorization, seal_scientific_preregistration, seal_static_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("protocol")
    subparsers.add_parser("preregister")
    subparsers.add_parser("authorize")
    args = parser.parse_args()
    if args.action == "protocol":
        print(seal_static_protocol(output_path=PROTOCOL_PATH))
        return 0
    if args.action == "preregister":
        output = seal_scientific_preregistration(
            output_path=PREREGISTRATION_PATH,
        )
        print(output)
        return 0
    ready, authorization = seal_ready_authorization(
        preregistration_path=PREREGISTRATION_PATH,
        ready_path=READY_CONFIG_PATH,
        authorization_path=RUN_AUTHORIZATION_PATH,
    )
    print(ready)
    print(authorization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
