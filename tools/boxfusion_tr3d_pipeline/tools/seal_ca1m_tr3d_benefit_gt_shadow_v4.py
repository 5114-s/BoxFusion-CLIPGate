#!/usr/bin/env python3
"""Create a byte-identical, read-only fit/dev80 CA-train GT shadow inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v4 import write_binding_create_only  # noqa: E402
from boxfusion.ca1m_tr3d_terminal_v4 import sha256_file  # noqa: E402


SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_gt_shadow_inventory.v1"
FIT_FOLDS = (2, 3, 4)
DEV_FOLDS = (0,)
LOCKED_FOLDS = (1,)


def _regular(path: Path, name: str, *, sealed: bool = False) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    source = path.resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {source}")
    if sealed and source.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be sealed read-only: {source}")
    return source


def _copy_create_only(source: Path, target: Path, expected_sha256: str) -> str:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing GT shadow artifact: {target}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise ValueError(f"copied GT shadow SHA256 differs: {target}")
        temporary.chmod(0o444)
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if target.stat().st_mode & 0o222 or sha256_file(target) != expected_sha256:
        raise RuntimeError(f"published GT shadow identity differs: {target}")
    return expected_sha256


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--oof-sidecar", type=Path, required=True)
    value.add_argument("--source-dataset-manifest", type=Path, required=True)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--receipt", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    oof_path = _regular(args.oof_sidecar, "B6-v2 OOF sidecar", sealed=True)
    source_manifest_path = _regular(
        args.source_dataset_manifest, "B6-v2 source dataset manifest", sealed=True
    )
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    receipt = args.receipt.resolve()
    if output_root.exists() or output_root.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError("refusing existing GT shadow root/receipt")
    source_manifest = json.loads(source_manifest_path.read_text())
    source_rows = {
        str(row["scene_id"]): row for row in source_manifest.get("scenes", ())
    }
    with np.load(oof_path, allow_pickle=False) as archive:
        scenes = np.asarray(archive["scene_ids"]).astype(str)
        folds = np.asarray(archive["fold_ids"], dtype=np.int64)
    scene_fold: dict[str, int] = {}
    for scene in sorted(set(scenes.tolist())):
        values = np.unique(folds[scenes == scene])
        if len(values) != 1:
            raise ValueError(f"OOF scene crosses folds: {scene}")
        scene_fold[scene] = int(values[0])
    selected = tuple(
        scene for scene in sorted(scene_fold)
        if scene_fold[scene] in (*FIT_FOLDS, *DEV_FOLDS)
    )
    if (
        len(scene_fold) != 100
        or len(selected) != 80
        or sum(scene_fold[s] in FIT_FOLDS for s in selected) != 60
        or sum(scene_fold[s] in DEV_FOLDS for s in selected) != 20
        or any(scene_fold[s] in LOCKED_FOLDS for s in selected)
        or set(scene_fold) != set(source_rows)
    ):
        raise ValueError("GT shadow split is not fixed folds234/fold0 with fold1 sealed")

    validated: dict[str, dict[str, Any]] = {}
    for scene in selected:
        row = source_rows[scene]
        box_source = _regular(
            source_root / scene / "derived_train_gt_boxes.npy", f"source GT {scene}"
        )
        manifest_source = _regular(
            source_root / scene / "derived_train_gt_manifest.json",
            f"source GT manifest {scene}",
        )
        box_sha = sha256_file(box_source)
        manifest_sha = sha256_file(manifest_source)
        if (
            int(row.get("fold_id", -1)) != scene_fold[scene]
            or box_sha != row.get("derived_gt_sha256")
            or manifest_sha != (row.get("derived_gt_manifest") or {}).get("sha256")
            or Path(str((row.get("derived_gt_manifest") or {}).get("path", ""))).resolve()
            != manifest_source
        ):
            raise ValueError(f"source B6 dataset GT identity differs: {scene}")
        validated[scene] = {
            "fold_id": scene_fold[scene],
            "source_box": box_source,
            "source_manifest": manifest_source,
            "box_sha256": box_sha,
            "manifest_sha256": manifest_sha,
            "source_box_mode": oct(box_source.stat().st_mode & 0o777),
            "source_manifest_mode": oct(manifest_source.stat().st_mode & 0o777),
        }

    output_root.mkdir(parents=True, exist_ok=False)
    published: dict[str, dict[str, Any]] = {}
    for scene in selected:
        source = validated[scene]
        target_dir = output_root / scene
        target_dir.mkdir()
        box_target = target_dir / "derived_train_gt_boxes.npy"
        manifest_target = target_dir / "derived_train_gt_manifest.json"
        _copy_create_only(source["source_box"], box_target, source["box_sha256"])
        _copy_create_only(
            source["source_manifest"], manifest_target, source["manifest_sha256"]
        )
        published[scene] = {
            "fold_id": source["fold_id"],
            "box": {
                "path": str(box_target), "sha256": source["box_sha256"],
                "source_path": str(source["source_box"]),
                "source_mode": source["source_box_mode"], "mode": "0o444",
            },
            "manifest": {
                "path": str(manifest_target), "sha256": source["manifest_sha256"],
                "source_path": str(source["source_manifest"]),
                "source_mode": source["source_manifest_mode"], "mode": "0o444",
            },
        }
    actual = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*") if path.is_file() and not path.is_symlink()
    }
    expected = {
        f"{scene}/{name}" for scene in selected
        for name in ("derived_train_gt_boxes.npy", "derived_train_gt_manifest.json")
    }
    if actual != expected:
        raise RuntimeError("GT shadow root is not exact selected80/two-files-per-scene")
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "create_only": True,
        "train_only": True,
        "scene_count": 80,
        "file_count": 160,
        "fit_fold_ids": list(FIT_FOLDS),
        "threshold_dev_fold_ids": list(DEV_FOLDS),
        "locked_internal_fold_ids": list(LOCKED_FOLDS),
        "fit_scene_count": 60,
        "threshold_dev_scene_count": 20,
        "locked_internal_scene_count_accessed": 0,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "gt_array_content_loaded": False,
        "opaque_source_bytes_hashed_and_copied": True,
        "source_bytes_mutated": False,
        "shadow_files_read_only": True,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_dataset_manifest": {
            "path": str(source_manifest_path), "sha256": sha256_file(source_manifest_path),
        },
        "oof_sidecar": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
        "scenes": published,
        "inventory_sha256": hashlib.sha256(
            json.dumps(published, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    write_binding_create_only(receipt, payload)
    print(json.dumps({
        "complete": True, "scene_count": 80, "file_count": 160,
        "gt_array_content_loaded": False, "receipt": str(receipt),
        "receipt_sha256": sha256_file(receipt), "output_root": str(output_root),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
