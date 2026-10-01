#!/usr/bin/env python3
"""Convert the pinned official ScanNet18 TR3D checkpoint to one-class init."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_foreground_checkpoint import convert_checkpoint


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = ROOT
    manifest = root / "manifests" / "tr3d_official_scannet18_checkpoint.json"
    source = root / "models" / "tr3d_1xb16_scannet-3d-18class.pth"
    output = root / "models" / "tr3d_1xb16_scannet-3d-foreground-init.pth"
    provenance = (
        root / "manifests" / "tr3d_scannet_foreground_init_checkpoint.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=source)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--provenance", type=Path, default=provenance)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=manifest,
        help="manifest whose sha256 must match --source",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    expected_sha = manifest.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("source manifest has no valid sha256")
    record = convert_checkpoint(
        args.source,
        args.output,
        args.provenance,
        expected_source_sha256=expected_sha,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
