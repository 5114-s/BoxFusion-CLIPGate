#!/usr/bin/env python3
"""Verify either a legacy B6 or generic prediction anchor manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import verify_frozen_anchor_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = verify_frozen_anchor_manifest(args.manifest)
    print(json.dumps({
        "schema": "boxfusion.frozen_anchor_verification.v1",
        "ok": True,
        "manifest": str(args.manifest.resolve()),
        "anchor_name": payload["anchor_name"],
        "scene_count": payload["scene_count"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "anchor_metrics_percent": payload["anchor_metrics_percent"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
