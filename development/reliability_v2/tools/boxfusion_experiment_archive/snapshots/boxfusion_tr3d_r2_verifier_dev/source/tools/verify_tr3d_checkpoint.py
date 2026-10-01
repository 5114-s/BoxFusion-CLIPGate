#!/usr/bin/env python3
"""Verify an immutable TR3D checkpoint against its checked-in manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root
        / "manifests"
        / "tr3d_official_scannet18_checkpoint.json",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or root / manifest["filename"]
    actual_size = checkpoint.stat().st_size
    actual_sha = sha256(checkpoint)
    if actual_size != int(manifest["bytes"]):
        raise SystemExit(
            f"checkpoint size mismatch: {actual_size} != {manifest['bytes']}")
    if actual_sha != manifest["sha256"]:
        raise SystemExit(
            f"checkpoint SHA256 mismatch: {actual_sha} != "
            f"{manifest['sha256']}")
    print(f"TR3D checkpoint OK: {checkpoint}")
    print(f"  sha256: {actual_sha}")
    print(f"  bytes: {actual_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
