#!/usr/bin/env python3
"""Build a leakage-safe K=5 dataset for the SGCDet-inspired local refiner.

The GT matching and AP50-aware target construction deliberately remain owned
by :mod:`tools.build_oriented_refiner_dataset`.  This builder first invokes
the already audited strict B5 -> joint-local join, then converts only the
runtime-exact K=5 tensors into a small, pickle-free sparse-refiner schema.

This two-step design is intentional: reimplementing the result-index/GT join
here would create a second, subtly different source of supervision.  Every
row is still joined by ``(scene_id, result_index)`` and every validation scene
is rejected before an archive is written.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.joint_local_head import (
    JOINT_LOCAL_HEAD_COORDINATE_FRAME,
    JOINT_LOCAL_HEAD_INPUT_SCHEMA,
    JOINT_VIEW_FEATURE_NAMES,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from tools.build_joint_local_dataset import (
    EXPECTED_TOP_K_VIEWS,
    JOINT_LOCAL_DATASET_FORMAT_VERSION,
    JOINT_LOCAL_DATASET_SCHEMA,
    JOINT_METADATA_KEYS,
    JOINT_SAMPLE_KEYS,
    JointDatasetBuildConfig,
    build_joint_local_dataset,
)
from tools.build_oriented_refiner_dataset import read_scene_ids
from tools.train_joint_local_head import load_joint_local_dataset


DATASET_SCHEMA = "boxfusion.sgcdet_sparse_refiner_dataset"
DATASET_FORMAT_VERSION = 1
INPUT_SCHEMA = "sgcdet_local_sparse_k5_p128_v1"
OBJECTIVE = "ap50_sparse_occupancy_residual_quality"
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")

SAMPLE_KEYS = frozenset(
    {
        "points_local",
        "point_mask",
        "view_features",
        "view_mask",
        "local_boxes",
        "quality_features",
        "target_residual",
        "geometry_mask",
        "scene_ids",
        "result_indices",
        "track_ids",
        "matched_gt_index",
        "baseline_iou",
        "target_iou",
        "iou_gain",
        "cross_iou50",
        "ap50_weight",
        "runtime_eligible",
        "identity_tp50",
        "candidate_oracle_tp50",
        "aligned_basis",
        "original_aligned_center",
        "matched_gt_box",
    }
)

METADATA_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "input_schema",
        "objective",
        "coordinate_frame",
        "top_k_views",
        "points_per_view",
        "sample_count",
        "quality_feature_names",
        "view_feature_names",
        "max_center_fraction",
        "max_log_dimension_residual",
        "source_joint_dataset_schema",
        "source_joint_dataset_format_version",
        "source_joint_dataset_sha256",
        "source_b5_dataset_sha256",
        "forbidden_scene_count",
        "forbidden_scene_sha256",
        "training_scene_count",
        "training_scene_sha256",
        "diagnostic_scene_count",
        "diagnostic_scene_sha256",
        "strict_k5_diagnostics",
    }
)


@dataclass(frozen=True)
class SparseDatasetBuildConfig:
    b5_dataset: Path
    diagnostics_root: Path
    forbidden_scene_list: Path
    output: Path

    def validated(self) -> "SparseDatasetBuildConfig":
        b5 = Path(self.b5_dataset)
        diagnostics = Path(self.diagnostics_root)
        forbidden = Path(self.forbidden_scene_list)
        output = Path(self.output)
        if not b5.is_file() or b5.suffix.lower() != ".npz":
            raise FileNotFoundError(f"strict B5-v2 dataset is absent: {b5}")
        if not diagnostics.is_dir():
            raise FileNotFoundError(
                f"K=5 diagnostics directory is absent: {diagnostics}"
            )
        if not forbidden.is_file():
            raise FileNotFoundError(
                f"forbidden validation-scene list is absent: {forbidden}"
            )
        if output.suffix.lower() != ".npz":
            raise ValueError("sparse-refiner dataset must end in .npz")
        if output.resolve() == b5.resolve():
            raise ValueError("output must not overwrite the strict B5 source")
        return self


@dataclass(frozen=True)
class SparseDatasetBuildSummary:
    scenes: int
    samples: int
    geometry_positives: int
    cross_iou50: int
    source_joint_dataset_sha256: str
    output: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_sha256(scene_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(set(scene_ids))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.hasobject:
        raise TypeError(f"{name} must be a safe scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str):
        raise TypeError(f"{name} must be a string")
    return item


def _scalar_integer(value: Any, name: str) -> int:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be an integer scalar")
    return int(array)


def _safe_npz(path: Path, expected: frozenset[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != set(expected):
                raise ValueError(
                    f"{path} schema keys are invalid: "
                    f"missing={sorted(set(expected) - keys)}, "
                    f"unexpected={sorted(keys - set(expected))}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(f"{path} contains forbidden object arrays") from error
        raise
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError(f"{path} contains forbidden object dtype")
    return arrays


def convert_joint_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    joint_sha256: str,
    forbidden_scenes: Sequence[str],
) -> dict[str, np.ndarray]:
    """Convert a validated joint archive into the sparse-refiner schema."""

    scene_ids = np.asarray(arrays["scene_ids"])
    if scene_ids.dtype.hasobject or scene_ids.dtype.kind not in {"U", "S"}:
        raise TypeError("scene_ids must be a safe string array")
    scene_ids = scene_ids.astype(np.str_)
    scenes = sorted(np.unique(scene_ids).tolist())
    if len(scenes) < 2 or any(SCENE_PATTERN.fullmatch(s) is None for s in scenes):
        raise ValueError("training archive must contain at least two valid scenes")
    leaked = sorted(set(scenes) & set(forbidden_scenes))
    if leaked:
        raise ValueError(
            "training archive overlaps forbidden validation scenes: "
            f"{leaked[:5]}"
        )

    points = np.asarray(arrays["joint_points_local"])
    point_mask = np.asarray(arrays["joint_point_mask"])
    if points.dtype != np.float32 or points.ndim != 4:
        raise TypeError("joint_points_local must be float32 [N,5,128,3]")
    n = int(points.shape[0])
    if points.shape[1:] != (EXPECTED_TOP_K_VIEWS, 128, 3):
        raise ValueError("SGCDet sparse refiner requires exact K=5/P=128")
    if point_mask.shape != points.shape[:-1] or point_mask.dtype != np.bool_:
        raise TypeError("joint_point_mask must be Boolean [N,5,128]")
    if not np.isfinite(points).all() or not np.all(points[~point_mask] == 0.0):
        raise ValueError("point tensors or padding are invalid")

    field_map = {
        "points_local": "joint_points_local",
        "point_mask": "joint_point_mask",
        "view_features": "joint_view_features",
        "view_mask": "joint_view_mask",
        "local_boxes": "joint_local_boxes",
        "quality_features": "joint_quality_features",
        "target_residual": "target_residual",
        "geometry_mask": "geometry_mask",
        "scene_ids": "scene_ids",
        "result_indices": "result_indices",
        "track_ids": "track_ids",
        "matched_gt_index": "matched_gt_index",
        "baseline_iou": "original_iou",
        "target_iou": "refined_iou",
        "iou_gain": "iou_gain",
        "cross_iou50": "cross_iou50",
        "ap50_weight": "ap50_weight",
        "runtime_eligible": "runtime_eligible",
        "identity_tp50": "identity_tp50",
        "candidate_oracle_tp50": "candidate_oracle_tp50",
        "aligned_basis": "aligned_basis",
        "original_aligned_center": "original_aligned_center",
        "matched_gt_box": "matched_gt_box",
    }
    output = {
        target: np.ascontiguousarray(np.asarray(arrays[source]).copy())
        for target, source in field_map.items()
    }
    if any(value.shape[0] != n for value in output.values()):
        raise ValueError("sample arrays disagree on their first dimension")
    if not output["view_mask"].any(axis=1).all():
        raise ValueError("every retained sample must have a valid K=5 view")
    if not np.array_equal(
        output["view_mask"], output["point_mask"].any(axis=2)
    ):
        raise ValueError("view masks disagree with point masks")
    if not output["geometry_mask"].any() or output["geometry_mask"].all():
        raise ValueError("dataset needs geometry positives and negatives")
    if not np.array_equal(
        output["cross_iou50"],
        output["runtime_eligible"]
        & ~output["identity_tp50"]
        & output["candidate_oracle_tp50"],
    ):
        raise ValueError("cross_iou50 labels violate strict AP50 provenance")
    for scene in scenes:
        rows = scene_ids == scene
        if len(np.unique(output["result_indices"][rows])) != int(rows.sum()):
            raise ValueError(f"{scene}: duplicate result_indices")

    forbidden = sorted(set(str(scene) for scene in forbidden_scenes))
    output.update(
        {
            "schema": np.asarray(DATASET_SCHEMA),
            "format_version": np.asarray(DATASET_FORMAT_VERSION, dtype=np.int64),
            "input_schema": np.asarray(INPUT_SCHEMA),
            "objective": np.asarray(OBJECTIVE),
            "coordinate_frame": np.asarray(JOINT_LOCAL_HEAD_COORDINATE_FRAME),
            "top_k_views": np.asarray(EXPECTED_TOP_K_VIEWS, dtype=np.int64),
            "points_per_view": np.asarray(128, dtype=np.int64),
            "sample_count": np.asarray(n, dtype=np.int64),
            "quality_feature_names": np.asarray(QUALITY_FEATURE_NAMES, dtype=np.str_),
            "view_feature_names": np.asarray(JOINT_VIEW_FEATURE_NAMES, dtype=np.str_),
            "max_center_fraction": np.asarray(
                arrays["max_center_fraction"], dtype=np.float32
            ),
            "max_log_dimension_residual": np.asarray(
                arrays["max_log_dimension_residual"], dtype=np.float32
            ),
            "source_joint_dataset_schema": np.asarray(
                JOINT_LOCAL_DATASET_SCHEMA
            ),
            "source_joint_dataset_format_version": np.asarray(
                JOINT_LOCAL_DATASET_FORMAT_VERSION, dtype=np.int64
            ),
            "source_joint_dataset_sha256": np.asarray(joint_sha256),
            "source_b5_dataset_sha256": np.asarray(
                _scalar_string(
                    arrays["source_dataset_sha256"], "source_dataset_sha256"
                )
            ),
            "forbidden_scene_count": np.asarray(len(forbidden), dtype=np.int64),
            "forbidden_scene_sha256": np.asarray(_scene_sha256(forbidden)),
            "training_scene_count": np.asarray(len(scenes), dtype=np.int64),
            "training_scene_sha256": np.asarray(_scene_sha256(scenes)),
            "diagnostic_scene_count": np.asarray(len(scenes), dtype=np.int64),
            "diagnostic_scene_sha256": np.asarray(_scene_sha256(scenes)),
            "strict_k5_diagnostics": np.asarray(True, dtype=np.bool_),
        }
    )
    if set(output) != SAMPLE_KEYS | METADATA_KEYS:
        raise RuntimeError("internal sparse-refiner schema is incomplete")
    for name, value in output.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(f"{name} is not a safe NumPy array")
    return output


def build_sgcdet_sparse_refiner_dataset(
    config: SparseDatasetBuildConfig,
) -> SparseDatasetBuildSummary:
    config = config.validated()
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forbidden = read_scene_ids(Path(config.forbidden_scene_list))

    with tempfile.TemporaryDirectory(
        prefix="sgcdet_sparse_join_", dir=str(output.parent)
    ) as temporary_directory:
        joint_path = Path(temporary_directory) / "joint_source.npz"
        build_joint_local_dataset(
            JointDatasetBuildConfig(
                b5_dataset=Path(config.b5_dataset),
                diagnostics_root=Path(config.diagnostics_root),
                forbidden_scene_list=Path(config.forbidden_scene_list),
                output=joint_path,
            )
        )
        # Exercise the canonical strict loader before reducing the schema.
        load_joint_local_dataset(joint_path)
        joint_sha256 = _sha256_file(joint_path)
        arrays = _safe_npz(joint_path, JOINT_SAMPLE_KEYS | JOINT_METADATA_KEYS)
        converted = convert_joint_arrays(
            arrays,
            joint_sha256=joint_sha256,
            forbidden_scenes=forbidden,
        )

    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **converted)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    # Reload without pickle so filesystem corruption cannot be reported as a
    # successful build.
    _safe_npz(output, SAMPLE_KEYS | METADATA_KEYS)
    scenes = sorted(np.unique(converted["scene_ids"]).tolist())
    return SparseDatasetBuildSummary(
        scenes=len(scenes),
        samples=int(len(converted["scene_ids"])),
        geometry_positives=int(np.count_nonzero(converted["geometry_mask"])),
        cross_iou50=int(np.count_nonzero(converted["cross_iou50"])),
        source_joint_dataset_sha256=joint_sha256,
        output=output,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b5-dataset", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--forbidden-scene-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    summary = build_sgcdet_sparse_refiner_dataset(
        SparseDatasetBuildConfig(
            b5_dataset=arguments.b5_dataset,
            diagnostics_root=arguments.diagnostics_root,
            forbidden_scene_list=arguments.forbidden_scene_list,
            output=arguments.output,
        )
    )
    print(
        "Built SGCDet-inspired sparse-refiner dataset: "
        f"scenes={summary.scenes}, samples={summary.samples}, "
        f"geometry_positives={summary.geometry_positives}, "
        f"cross_iou50={summary.cross_iou50}, output={summary.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
