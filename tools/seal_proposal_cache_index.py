#!/usr/bin/env python3
"""Seal one v3 cache namespace after every expected scene has finalized."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from boxfusion.proposal_cache import (
    ProposalCache,
    ProposalCacheConfig,
    ProposalCacheError,
)


def _scene_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise ProposalCacheError(f"Scene list is missing: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not row or row.startswith("#") for row in rows):
        raise ProposalCacheError(
            "Scene list must contain one non-empty scene ID per line and no comments"
        )
    if not rows:
        raise ProposalCacheError("Scene list is empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--scene-list", required=True, type=Path)
    arguments = parser.parse_args()
    cache = ProposalCache(
        ProposalCacheConfig(
            mode="record",
            root=arguments.root,
            namespace=arguments.namespace,
        ),
        device=torch.device("cpu"),
    )
    index_path = cache.seal_index(_scene_ids(arguments.scene_list))
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
