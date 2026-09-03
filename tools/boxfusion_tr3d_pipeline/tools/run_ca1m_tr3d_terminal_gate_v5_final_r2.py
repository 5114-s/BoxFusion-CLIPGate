#!/usr/bin/env python3
"""Permanent tombstone for the invalidated final-R2 runtime-root protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
INVALID = (
    ROOT
    / "manifests/ca1m_tr3d_terminal_gate_v5_final_r2/"
    "PREREGISTRATION_PROTOCOL_INVALID.json"
)
INVALID_SHA256 = "616ac9db9d48c14fc05628510a5acabbcb7ce7044ee110ea1878f4bfb35e530e"
INVALID_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol_invalid.v5.final.r2"
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
        raise PermissionError("final-R2 protocol invalidation differs")
    print(json.dumps({
        "status": "INVALIDATED_FINAL_R2_RUNTIME_ROOT_TOCTOU",
        "operational_authority": False,
        "replacement_namespace": value["superseded_by_namespace"],
    }, sort_keys=True), file=sys.stderr)
    return 66


if __name__ == "__main__":
    raise SystemExit(main())
