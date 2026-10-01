#!/usr/bin/env python3
"""Fail-closed provenance validation for a YiDu A6 AP50 gate.

The validator binds one strict training archive to one runtime checkpoint and
proves that every scene used by the trainer came from the declared ScanNet
training split and not from the forbidden validation split.  It is deliberately
CPU-only and never rewrites either input artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ap50_safety_gate import (  # noqa: E402
    load_ap50_safety_gate,
)
from boxfusion.yidu_local_observer import (  # noqa: E402
    YIDU_GATE_FEATURE_DIM,
    YIDU_GATE_FEATURE_NAMES,
)
from tools.train_ap50_safety_gate import (  # noqa: E402
    TRAINING_SCHEMA,
    load_training_data,
)


SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
REQUIRED_METADATA_KEYS = frozenset(
    {
        "training_schema",
        "training_samples",
        "validation_samples",
        "training_scene_count",
        "validation_scene_count",
        "training_scenes",
        "validation_scenes",
        "training_archives",
    }
)


def _read_scene_list(path: Path, *, role: str) -> frozenset[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{role} scene list not found: {path}")
    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{role} scene list is empty: {path}")
    malformed = [
        row
        for row in rows
        if not SCENE_PATTERN.fullmatch(row)
    ]
    if malformed:
        raise ValueError(
            f"{role} scene list contains malformed scene IDs: "
            + ", ".join(malformed[:4])
        )
    if len(rows) != len(set(rows)):
        raise ValueError(f"{role} scene list contains duplicate scene IDs")
    return frozenset(rows)


def _metadata_integer(
    metadata: Mapping[str, Any],
    name: str,
    *,
    minimum: int,
) -> int:
    value = metadata[name]
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(
            f"checkpoint metadata {name} must be an integer >= {minimum}"
        )
    return int(value)


def _metadata_scene_tuple(
    metadata: Mapping[str, Any],
    name: str,
) -> tuple[str, ...]:
    value = metadata[name]
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise ValueError(
            f"checkpoint metadata {name} must be a non-empty scene list"
        )
    rows = tuple(value)
    if not rows:
        raise ValueError(
            f"checkpoint metadata {name} must be a non-empty scene list"
        )
    if any(
        not isinstance(row, str)
        or not SCENE_PATTERN.fullmatch(row)
        for row in rows
    ):
        raise ValueError(
            f"checkpoint metadata {name} contains malformed scene IDs"
        )
    if len(rows) != len(set(rows)):
        raise ValueError(
            f"checkpoint metadata {name} contains duplicate scene IDs"
        )
    return rows


def validate_yidu_gate_provenance(
    *,
    checkpoint: Path,
    training_archive: Path,
    train_scene_list: Path,
    forbidden_scene_list: Path,
) -> dict[str, object]:
    """Validate one YiDu gate/archive pair and return an auditable summary."""

    train_scenes = _read_scene_list(
        Path(train_scene_list), role="training"
    )
    forbidden_scenes = _read_scene_list(
        Path(forbidden_scene_list), role="forbidden"
    )
    split_overlap = sorted(train_scenes & forbidden_scenes)
    if split_overlap:
        raise ValueError(
            "declared training and forbidden scene lists overlap: "
            + ", ".join(split_overlap[:8])
        )

    data = load_training_data((Path(training_archive),))
    if tuple(data.feature_names) != YIDU_GATE_FEATURE_NAMES:
        raise ValueError(
            "training archive feature schema does not match the fixed "
            f"{YIDU_GATE_FEATURE_DIM}-D YiDu schema"
        )
    archive_scenes = frozenset(
        str(scene_id) for scene_id in data.scene_ids.tolist()
    )
    if not archive_scenes:
        raise ValueError("training archive contains no scene IDs")
    outside_train = sorted(archive_scenes - train_scenes)
    if outside_train:
        raise ValueError(
            "training archive contains scenes outside the declared training "
            "split: " + ", ".join(outside_train[:8])
        )
    forbidden_overlap = sorted(archive_scenes & forbidden_scenes)
    if forbidden_overlap:
        raise ValueError(
            "training archive overlaps forbidden validation scenes: "
            + ", ".join(forbidden_overlap[:8])
        )

    gate = load_ap50_safety_gate(Path(checkpoint))
    if tuple(gate.feature_names) != YIDU_GATE_FEATURE_NAMES:
        raise ValueError(
            "checkpoint feature schema does not match the fixed "
            f"{YIDU_GATE_FEATURE_DIM}-D YiDu schema"
        )
    metadata = gate.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be an object")
    missing = REQUIRED_METADATA_KEYS - set(metadata)
    if missing:
        raise ValueError(
            "checkpoint metadata is missing provenance keys: "
            + ", ".join(sorted(missing))
        )
    if metadata["training_schema"] != TRAINING_SCHEMA:
        raise ValueError("checkpoint metadata training_schema mismatch")
    archive_rows = metadata["training_archives"]
    if (
        not isinstance(archive_rows, Sequence)
        or isinstance(archive_rows, (str, bytes))
        or len(archive_rows) != 1
        or not isinstance(archive_rows[0], Mapping)
    ):
        raise ValueError(
            "checkpoint metadata training_archives must contain exactly "
            "one archive record"
        )
    archive_record = archive_rows[0]
    if set(archive_record) != {"path", "sha256"}:
        raise ValueError(
            "checkpoint metadata training archive record must contain "
            "only path and sha256"
        )
    expected_archive = Path(training_archive).resolve()
    if Path(str(archive_record["path"])).resolve() != expected_archive:
        raise ValueError(
            "checkpoint metadata training archive path mismatch"
        )
    digest = hashlib.sha256()
    with expected_archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if str(archive_record["sha256"]) != digest.hexdigest():
        raise ValueError(
            "checkpoint metadata training archive SHA256 mismatch"
        )

    metadata_training = _metadata_scene_tuple(
        metadata, "training_scenes"
    )
    metadata_validation = _metadata_scene_tuple(
        metadata, "validation_scenes"
    )
    metadata_training_set = frozenset(metadata_training)
    metadata_validation_set = frozenset(metadata_validation)
    internal_overlap = sorted(
        metadata_training_set & metadata_validation_set
    )
    if internal_overlap:
        raise ValueError(
            "checkpoint internal training/validation scenes overlap: "
            + ", ".join(internal_overlap[:8])
        )
    metadata_scene_union = (
        metadata_training_set | metadata_validation_set
    )
    if metadata_scene_union != archive_scenes:
        missing_from_metadata = sorted(
            archive_scenes - metadata_scene_union
        )
        extra_in_metadata = sorted(
            metadata_scene_union - archive_scenes
        )
        raise ValueError(
            "checkpoint metadata scene union disagrees with the training "
            f"archive: missing={missing_from_metadata[:8]}, "
            f"extra={extra_in_metadata[:8]}"
        )
    metadata_outside_train = sorted(metadata_scene_union - train_scenes)
    if metadata_outside_train:
        raise ValueError(
            "checkpoint metadata contains scenes outside the declared "
            "training split: " + ", ".join(metadata_outside_train[:8])
        )
    metadata_forbidden = sorted(
        metadata_scene_union & forbidden_scenes
    )
    if metadata_forbidden:
        raise ValueError(
            "checkpoint metadata overlaps forbidden validation scenes: "
            + ", ".join(metadata_forbidden[:8])
        )

    training_scene_count = _metadata_integer(
        metadata, "training_scene_count", minimum=1
    )
    validation_scene_count = _metadata_integer(
        metadata, "validation_scene_count", minimum=1
    )
    if training_scene_count != len(metadata_training_set):
        raise ValueError(
            "checkpoint metadata training_scene_count disagrees with "
            "training_scenes"
        )
    if validation_scene_count != len(metadata_validation_set):
        raise ValueError(
            "checkpoint metadata validation_scene_count disagrees with "
            "validation_scenes"
        )

    training_samples = _metadata_integer(
        metadata, "training_samples", minimum=1
    )
    validation_samples = _metadata_integer(
        metadata, "validation_samples", minimum=1
    )
    archive_training_samples = int(
        np.count_nonzero(
            np.isin(data.scene_ids, tuple(metadata_training_set))
        )
    )
    archive_validation_samples = int(
        np.count_nonzero(
            np.isin(data.scene_ids, tuple(metadata_validation_set))
        )
    )
    if training_samples != archive_training_samples:
        raise ValueError(
            "checkpoint metadata training_samples disagrees with the "
            "training archive"
        )
    if validation_samples != archive_validation_samples:
        raise ValueError(
            "checkpoint metadata validation_samples disagrees with the "
            "training archive"
        )
    if training_samples + validation_samples != data.sample_count:
        raise ValueError(
            "checkpoint metadata sample counts do not cover the training "
            "archive exactly"
        )

    return {
        "schema": "boxfusion.yidu.gate_provenance_report.v1",
        "valid": True,
        "feature_dim": YIDU_GATE_FEATURE_DIM,
        "archive_samples": data.sample_count,
        "archive_scene_count": len(archive_scenes),
        "internal_training_samples": training_samples,
        "internal_validation_samples": validation_samples,
        "internal_training_scene_count": training_scene_count,
        "internal_validation_scene_count": validation_scene_count,
        "forbidden_overlap": 0,
        "checkpoint": str(Path(checkpoint).resolve()),
        "training_archive": str(Path(training_archive).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-archive", required=True, type=Path)
    parser.add_argument("--train-scene-list", required=True, type=Path)
    parser.add_argument(
        "--forbidden-scene-list", required=True, type=Path
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = validate_yidu_gate_provenance(
        checkpoint=args.checkpoint,
        training_archive=args.training_archive,
        train_scene_list=args.train_scene_list,
        forbidden_scene_list=args.forbidden_scene_list,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
