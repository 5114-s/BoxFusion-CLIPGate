#!/usr/bin/env python3
"""Verify the frozen B6 identity anchor without creating any artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_b6_manifest import verify_frozen_b6_manifest  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_ROOT / "manifests" / "frozen_b6_full100.json",
    )
    args = parser.parse_args(argv)
    payload = verify_frozen_b6_manifest(args.manifest)
    print(
        json.dumps(
            {
                "schema": "boxfusion.frozen_b6_verification.v1",
                "ok": True,
                "manifest": str(args.manifest.resolve()),
                "scene_count": payload["scene_count"],
                "prediction_tree_sha256": payload[
                    "prediction_tree_sha256"
                ],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "anchor_metrics_percent": payload[
                    "anchor_metrics_percent"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
