#!/usr/bin/env python3
"""Fail-closed tombstone for the invalidated terminal-gate final R1 runner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
INVALID_PATH = (
    ROOT
    / "manifests/ca1m_tr3d_terminal_gate_v5_final/PENDING_REVISION_INVALID.json"
)
INVALID_SHA256 = "4bc9467df292a07e1fa34e7597a7bf47a9190299ad47c1d32edb7d5756beb5b1"
INVALID_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_pending_revision_invalid.v5.final"
)


def _verify_invalidation() -> dict[str, object]:
    fd = os.open(INVALID_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PermissionError("terminal-gate final R1 invalidation is not a single-link file")
        chunks: list[bytes] = []
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
        or value.get("never_scientifically_preregistered") is not True
        or value.get("never_ready_authorized") is not True
        or value.get("runtime_namespace_created") is not False
    ):
        raise PermissionError("terminal-gate final R1 invalidation receipt differs")
    return value


def main() -> int:
    try:
        receipt = _verify_invalidation()
        result = {
            "schema": "boxfusion.ca1m_tr3d_terminal_gate_runner_tombstone.v5.final",
            "status": "INVALIDATED_FINAL_R1_STATIC_BLOCK",
            "exit_code": 66,
            "invalidation_path": str(INVALID_PATH),
            "invalidation_sha256": INVALID_SHA256,
            "superseded_by_namespace": receipt["superseded_by_namespace"],
            "output_created": False,
            "gpu_started": False,
            "ground_truth_access": False,
        }
    except Exception as error:
        result = {
            "schema": "boxfusion.ca1m_tr3d_terminal_gate_runner_tombstone.v5.final",
            "status": "INVALIDATED_FINAL_R1_FAIL_CLOSED",
            "exit_code": 66,
            "reason": str(error),
            "output_created": False,
            "gpu_started": False,
            "ground_truth_access": False,
        }
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 66


if __name__ == "__main__":
    raise SystemExit(main())
