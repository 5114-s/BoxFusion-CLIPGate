#!/usr/bin/env python3
"""Fail-closed tombstone for the independently invalidated R5 runner."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
ROOT = Path(__file__).resolve().parents[1]

INVALID_PATH = ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r5/PREREGISTRATION_INVALID.json"
INVALID_SHA256 = "15d398751a4d3d53a690039a5c2f6152eb46640eff5d38cf7154424ca87042ca"
INVALID_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration_invalid.v5.r5"


def _verify_invalidation() -> dict[str, object]:
    fd = os.open(INVALID_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PermissionError("R5 invalidation is not a single-link regular file")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    value = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest() != INVALID_SHA256
        or value.get("schema") != INVALID_SCHEMA
        or value.get("invalid") is not True
        or value.get("operational_authority") is not False
        or value.get("audit_result") != "CODE_BLOCK"
    ):
        raise PermissionError("R5 invalidation receipt differs")
    return value

def main() -> int:
    try:
        receipt = _verify_invalidation()
        result = {
            "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_runner_tombstone.v5.r5",
            "status": "INVALIDATED_R5_CODE_BLOCK", "exit_code": 66,
            "invalidation_path": str(INVALID_PATH),
            "invalidation_sha256": INVALID_SHA256,
            "superseded_by_namespace": receipt["superseded_by_namespace"],
            "output_created": False, "gpu_started": False, "ground_truth_access": False,
        }
    except Exception as error:
        result = {
            "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_runner_tombstone.v5.r5",
            "status": "INVALIDATED_R5_FAIL_CLOSED", "exit_code": 66,
            "reason": str(error), "output_created": False,
            "gpu_started": False, "ground_truth_access": False,
        }
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 66

if __name__ == "__main__": raise SystemExit(main())
