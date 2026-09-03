#!/usr/bin/env python3
"""Reject the superseded E961 outer-evaluation protocol V1.

The immutable V1 protocol remains available only as audit evidence. This
legacy entry point deliberately cannot create or validate a formal runtime
authorization; protocol V2 has its own separately named sealer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


INVALID = Path(
    "/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline/manifests/"
    "ca1m_tr3d_e961_outer_dev_eval_v1/PREREGISTRATION_PROTOCOL_V1_INVALID.json"
)
INVALID_SHA256 = "31d39340015df4101725d475310ec09b5daa19751c677ae0d2e51f75ad5ad3d8"


def main() -> int:
    if INVALID.is_symlink() or not INVALID.is_file():
        print("protocol V1 invalidation receipt is missing", file=sys.stderr)
        return 66
    if hashlib.sha256(INVALID.read_bytes()).hexdigest() != INVALID_SHA256:
        print("protocol V1 invalidation receipt drifted", file=sys.stderr)
        return 66
    print(
        "E961 outer evaluation protocol V1 is INVALID/SUPERSEDED; "
        "use the separately reviewed protocol V2 sealer.",
        file=sys.stderr,
    )
    return 66


if __name__ == "__main__":
    raise SystemExit(main())
