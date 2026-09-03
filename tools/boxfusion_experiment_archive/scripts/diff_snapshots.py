#!/usr/bin/env python3
"""List source-file additions, removals and changes between two snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(route: str) -> dict[str, dict[str, object]]:
    path = ROOT / "snapshots" / route / "MANIFEST.json"
    if not path.is_file():
        raise SystemExit(f"unknown archived route: {route}")
    return json.loads(path.read_text(encoding="utf-8"))["files"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    left = load(args.left)
    right = load(args.right)
    added = sorted(set(right) - set(left))
    removed = sorted(set(left) - set(right))
    changed = sorted(
        path
        for path in set(left) & set(right)
        if left[path]["sha256"] != right[path]["sha256"]
    )
    for label, paths in (("A", added), ("D", removed), ("M", changed)):
        for path in paths:
            print(f"{label}\t{path}")
    print(
        f"summary: added={len(added)}, removed={len(removed)}, "
        f"changed={len(changed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

