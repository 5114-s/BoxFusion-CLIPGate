#!/usr/bin/env python3
"""Build or read the immutable producer fingerprint for proposal-cache v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import torch


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"fingerprint input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("entry must be LABEL=PATH")
    return label, Path(raw_path).resolve()


def _compute(entries: list[tuple[str, Path]]) -> str:
    if not entries:
        raise ValueError("at least one fingerprint entry is required")
    labels = [label for label, _ in entries]
    if len(labels) != len(set(labels)):
        raise ValueError("fingerprint labels must be unique")
    payload = {
        "schema": "boxfusion.proposal-cache-producer-fingerprint.v1",
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda),
            "numpy": str(np.__version__),
        },
        "files": [
            {"label": label, "sha256": _sha256(path)}
            for label, path in sorted(entries)
        ],
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _from_index(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"sealed proposal-cache index is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = value.get("producer_fingerprint") if isinstance(value, dict) else None
    if not isinstance(fingerprint, str) or SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError("sealed index has an invalid producer fingerprint")
    return fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    compute = subparsers.add_parser("compute")
    compute.add_argument("--entry", action="append", type=_entry, required=True)
    index = subparsers.add_parser("from-index")
    index.add_argument("--index", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "compute":
        print(_compute(arguments.entry))
    else:
        print(_from_index(arguments.index.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
