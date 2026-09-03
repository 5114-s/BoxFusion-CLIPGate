#!/usr/bin/env python3
"""Read-only preflight for locked-F1/deploy v2 pending static design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_e961_locked_fold1_deploy_v2 import (  # noqa: E402
    DEFAULT_CONFIG,
    validate_pending_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the non-authorizing locked-F1/deploy v2 static design "
            "without resolving F1 or official-validation sources."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--require-sealable",
        action="store_true",
        help="exit 3 while the final L6 protocol/gate/receipt fields are null",
    )
    args = parser.parse_args(argv)
    report = validate_pending_config(args.config)
    print(json.dumps(report, sort_keys=True))
    if args.require_sealable and not report["static_protocol_sealable"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
