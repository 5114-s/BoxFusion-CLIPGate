#!/usr/bin/env python3
"""Fail-closed validation for genuine-TR3D training and observer runs.

The files in ``data/tr3d_scannet`` are trusted local experiment metadata.
This validator intentionally reads their pickle annotations so launch scripts
can reject ScanNet validation leakage, non-foreground labels and stale split
contracts *before* allocating a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


CONTRACT_SCHEMA = "boxfusion.tr3d.scannet_foreground.v1"
_SCENE_RE = re.compile(r"scene[0-9]{4}_[0-9]{2}")


def read_scene_list(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    scenes = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"{path}: scene list is empty or has duplicates")
    invalid = [scene for scene in scenes if _SCENE_RE.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"{path}: invalid scene ids: {invalid[:5]}")
    return scenes


def sha256_lines(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _annotation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; run the matching data export first"
        )
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - trusted local metadata
    if not isinstance(value, dict) or not isinstance(
        value.get("data_list"), list
    ):
        raise ValueError(f"{path}: expected MMDetection3D data_list metadata")
    metainfo = value.get("metainfo", {})
    if (
        tuple(metainfo.get("classes", ())) != ("foreground",)
        or metainfo.get("categories") != {"foreground": 0}
    ):
        raise ValueError(f"{path}: metadata is not class-agnostic foreground")
    rows = value["data_list"]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: annotation data_list is empty or malformed")
    return rows


def _scene_from_row(row: dict[str, Any]) -> str:
    lidar = row.get("lidar_points", {}).get("lidar_path")
    if not isinstance(lidar, str):
        raise ValueError("annotation row has no lidar_points.lidar_path")
    matches = set(_SCENE_RE.findall(lidar))
    if len(matches) != 1:
        raise ValueError(f"cannot identify exactly one scene in {lidar!r}")
    return next(iter(matches))


def _validate_labels(rows: Sequence[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if row.get("coordinate_frame") != "world_unaligned":
            raise ValueError("annotation points are not world_unaligned")
        if row.get("box_coordinate_frame") != "scannet_axis_aligned":
            raise ValueError("annotation boxes are not scannet_axis_aligned")
        axis = row.get("axis_align_matrix")
        if not isinstance(axis, (list, tuple)) or len(axis) != 4:
            raise ValueError("annotation row lacks a 4x4 axis_align_matrix")
        for instance in row.get("instances", ()):
            if int(instance.get("bbox_label_3d", -1)) != 0:
                raise ValueError("annotation contains a non-foreground label")
            count += 1
    return count


def validate_training(
    *,
    contract_path: Path,
    annotation_path: Path,
    expected_split_path: Path,
    prefix: bool,
) -> dict[str, Any]:
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"{contract_path} is missing; prepare ScanNet metadata first"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"{contract_path}: unsupported dataset contract")
    data_root = contract_path.parent
    official_val_path = data_root / "splits" / "official_val.txt"
    forbidden = set(read_scene_list(official_val_path))
    expected = read_scene_list(expected_split_path)
    if forbidden.intersection(expected):
        raise ValueError("training split contains official ScanNet val scenes")
    split_name = expected_split_path.stem
    recorded_sha = contract.get("scene_list_sha256", {}).get(split_name)
    if recorded_sha is not None and recorded_sha != sha256_lines(expected):
        raise ValueError(
            f"{expected_split_path}: split hash disagrees with contract"
        )

    rows = _annotation_rows(annotation_path)
    scenes = tuple(_scene_from_row(row) for row in rows)
    leaked = sorted(set(scenes) & forbidden)
    if leaked:
        raise ValueError(
            "annotation contains forbidden official-val scenes: "
            + ", ".join(leaked[:8])
        )
    if prefix:
        unknown = sorted(set(scenes) - set(expected))
        if unknown:
            raise ValueError(
                "prefix annotation is not a subset of frozen train: "
                + ", ".join(unknown[:8])
            )
        if not all("trajectory_prefix" in row for row in rows):
            raise ValueError("prefix annotation lacks trajectory_prefix records")
    elif set(scenes) != set(expected) or len(scenes) != len(expected):
        raise ValueError(
            "full-scene annotation does not exactly match train split"
        )
    instance_count = _validate_labels(rows)
    return {
        "schema": "boxfusion.tr3d.launch_validation.v1",
        "ok": True,
        "mode": "prefix_train" if prefix else "full_train",
        "contract": str(contract_path.resolve()),
        "annotation": str(annotation_path.resolve()),
        "scene_count": len(set(scenes)),
        "sample_count": len(rows),
        "instance_count": instance_count,
        "official_val_overlap": 0,
        "expected_split_sha256": sha256_lines(expected),
    }


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("full-train", "prefix-train"), required=True
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=root / "data" / "tr3d_scannet" / "DATASET_CONTRACT.json",
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument(
        "--expected-split",
        type=Path,
        default=root / "data" / "tr3d_scannet" / "splits" / "train.txt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_training(
        contract_path=args.contract.resolve(),
        annotation_path=args.annotation.resolve(),
        expected_split_path=args.expected_split.resolve(),
        prefix=args.mode == "prefix-train",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
