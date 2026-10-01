#!/usr/bin/env python3
"""Permanent tombstone for the invalidated final-R3 execution boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INVALID = (
    ROOT
    / "manifests/ca1m_tr3d_terminal_gate_v5_final_r3/"
    "PREREGISTRATION_PROTOCOL_INVALID.json"
)
INVALID_SHA256 = "9a945949feb53709e257d89b45304fc1799ac89dd01adf7027f8a13368c52fea"
INVALID_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol_invalid.v5.final.r3"
)


def main() -> int:
    data = INVALID.read_bytes()
    value = json.loads(data)
    if (
        hashlib.sha256(data).hexdigest() != INVALID_SHA256
        or value.get("schema") != INVALID_SCHEMA
        or value.get("invalid") is not True
        or value.get("operational_authority") is not False
        or value.get("never_instance_preregistered") is not True
        or value.get("never_ready_authorized") is not True
    ):
        raise PermissionError("final-R3 protocol invalidation differs")
    print(json.dumps({
        "status": "INVALIDATED_FINAL_R3_EXECUTION_BOUNDARY",
        "operational_authority": False,
        "replacement_namespace": value["superseded_by_namespace"],
    }, sort_keys=True), file=sys.stderr)
    return 66


if __name__ == "__main__":
    raise SystemExit(main())
