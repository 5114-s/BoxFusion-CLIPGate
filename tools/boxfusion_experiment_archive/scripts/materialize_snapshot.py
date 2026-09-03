#!/usr/bin/env python3
"""Materialize one archived source-only route into a new directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    catalog = json.loads((ROOT / "CATALOG.json").read_text(encoding="utf-8"))
    known = {entry["route"]: entry for entry in catalog["routes"]}
    if args.route not in known:
        parser.error(f"unknown route: {args.route}")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    source = ROOT / known[args.route]["snapshot"] / "source"
    shutil.copytree(source, output, symlinks=False)
    print(f"materialized {args.route} to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

