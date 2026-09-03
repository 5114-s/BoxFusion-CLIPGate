#!/usr/bin/env python3
"""Build train-only AP50 gate supervision from TriFusion candidates.

For each valid geometry candidate this tool selects the ground-truth box that
best matches the *original* exported prediction and records the original and
candidate IoU to that same target.  Holding the target fixed prevents a
candidate from being rewarded for jumping to a neighbouring object.

The output is the exact strict schema accepted by
``tools/train_ap50_safety_gate.py``.  A forbidden validation scene list is
mandatory; any overlap aborts before an output file is written.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_fused_oracle import (  # noqa: E402
    center_size_to_minmax,
    corners_to_minmax,
    load_axis_alignment,
    load_scene_predictions,
    pairwise_aabb_iou,
    read_scene_ids,
    transform_corners,
)
from tools.build_trifusion_geometry_candidates import (  # noqa: E402
    OUTPUT_SUFFIX,
)
from tools.report_trifusion_oracles import (  # noqa: E402
    load_geometry_candidates,
)
from tools.train_ap50_safety_gate import (  # noqa: E402
    TRAINING_FORMAT_VERSION,
    TRAINING_SCHEMA,
)


def _string_vector(
    value: np.ndarray, *, name: str, rows: int, path: Path
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.hasobject:
        raise ValueError(
            f"{path}: {name} must have non-object shape [{rows}]"
        )
    output = tuple(str(item) for item in array.tolist())
    if any(not item for item in output):
        raise ValueError(f"{path}: {name} cannot contain empty strings")
    return output


def _load_candidate_features(
    path: Path, *, expected_rows: int
) -> tuple[tuple[str, ...], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"candidate_feature_names", "candidate_features"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{path}: missing gate feature fields {sorted(missing)}"
            )
        features = np.asarray(
            archive["candidate_features"], dtype=np.float32
        )
        names_array = np.asarray(archive["candidate_feature_names"])
    if (
        names_array.ndim != 1
        or names_array.dtype.hasobject
        or len(names_array) < 1
    ):
        raise ValueError(f"{path}: invalid candidate_feature_names")
    names = _string_vector(
        names_array,
        name="candidate_feature_names",
        rows=len(names_array),
        path=path,
    )
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: candidate feature names are not unique")
    if (
        features.shape != (expected_rows, len(names))
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            f"{path}: candidate_features must have finite shape "
            f"[{expected_rows},{len(names)}]"
        )
    return names, features


def build_gate_training_archive(
    *,
    geometry_root: Path,
    prediction_root: Path,
    scene_list: Path,
    forbidden_scene_list: Path,
    gt_root: Path,
    scan_root: Path,
    output: Path,
    include_unverified: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    scenes = read_scene_ids(scene_list)
    forbidden = set(read_scene_ids(forbidden_scene_list))
    overlap = sorted(set(scenes) & forbidden)
    if overlap:
        raise ValueError(
            "training scene list overlaps forbidden validation scenes: "
            + ", ".join(overlap[:8])
        )
    if not scenes:
        raise ValueError(f"training scene list is empty: {scene_list}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite training data: {output}")

    collected_features = []
    collected_original_iou = []
    collected_candidate_iou = []
    collected_scene_ids = []
    feature_names: tuple[str, ...] | None = None
    candidate_rows = valid_rows = verified_rows = 0

    for scene_id in scenes:
        artifact_path = geometry_root / f"{scene_id}{OUTPUT_SUFFIX}"
        geometry = load_geometry_candidates(
            artifact_path, expected_scene_id=scene_id
        )
        names, features = _load_candidate_features(
            artifact_path, expected_rows=len(geometry.candidate_corners)
        )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"{artifact_path}: candidate feature schema changed"
            )
        prediction_corners, _ = load_scene_predictions(
            prediction_root / f"{scene_id}_boxes.pkl"
        )
        if (
            len(geometry.prediction_indices)
            and int(np.max(geometry.prediction_indices))
            >= len(prediction_corners)
        ):
            raise ValueError(
                f"{scene_id}: geometry prediction index is out of range"
            )
        paired = prediction_corners[geometry.prediction_indices]
        if not np.allclose(
            paired,
            geometry.original_corners,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"{scene_id}: geometry original/export corners disagree"
            )

        transform = load_axis_alignment(scan_root, scene_id)
        prediction_minmax = corners_to_minmax(
            transform_corners(prediction_corners, transform)
        )
        candidate_minmax = corners_to_minmax(
            transform_corners(geometry.candidate_corners, transform)
        )
        gt_payload = np.load(
            gt_root / f"{scene_id}_bbox.npy", allow_pickle=False
        )
        gt_minmax = center_size_to_minmax(gt_payload)
        original_matrix = pairwise_aabb_iou(
            prediction_minmax, gt_minmax
        )
        candidate_matrix = pairwise_aabb_iou(
            candidate_minmax, gt_minmax
        )

        for prediction_row, prediction_index in enumerate(
            geometry.prediction_indices.tolist()
        ):
            start = int(geometry.candidate_offsets[prediction_row])
            stop = int(geometry.candidate_offsets[prediction_row + 1])
            for candidate_index in range(start, stop):
                candidate_rows += 1
                if not bool(geometry.candidate_valid[candidate_index]):
                    continue
                valid_rows += 1
                if bool(geometry.candidate_verified[candidate_index]):
                    verified_rows += 1
                elif not include_unverified:
                    continue

                if len(gt_minmax) == 0:
                    original_iou = 0.0
                    candidate_iou = 0.0
                else:
                    # The original prediction fixes the object identity.
                    target_index = int(
                        np.argmax(original_matrix[prediction_index])
                    )
                    original_iou = float(
                        original_matrix[prediction_index, target_index]
                    )
                    candidate_iou = float(
                        candidate_matrix[candidate_index, target_index]
                    )
                collected_features.append(features[candidate_index])
                collected_original_iou.append(original_iou)
                collected_candidate_iou.append(candidate_iou)
                collected_scene_ids.append(scene_id)

    if feature_names is None or not collected_features:
        raise ValueError("no valid TriFusion candidates were collected")
    matrix = np.stack(collected_features).astype(np.float32)
    original = np.asarray(collected_original_iou, dtype=np.float32)
    candidate = np.asarray(collected_candidate_iou, dtype=np.float32)
    scene_width = max(len(scene) for scene in collected_scene_ids)
    scene_ids = np.asarray(
        collected_scene_ids, dtype=f"<U{scene_width}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(TRAINING_SCHEMA),
        format_version=np.asarray(
            TRAINING_FORMAT_VERSION, dtype=np.int64
        ),
        feature_names=np.asarray(feature_names, dtype="<U96"),
        gate_features=matrix,
        original_iou=original,
        candidate_iou=candidate,
        scene_ids=scene_ids,
    )
    delta = candidate - original
    return {
        "schema": TRAINING_SCHEMA,
        "format_version": TRAINING_FORMAT_VERSION,
        "scenes": len(set(collected_scene_ids)),
        "candidate_rows": candidate_rows,
        "valid_rows": valid_rows,
        "verified_rows": verified_rows,
        "training_rows": len(matrix),
        "feature_dim": matrix.shape[1],
        "improved": int(np.sum(delta > 1e-6)),
        "harmed": int(np.sum(delta < -1e-6)),
        "cross_iou25_up": int(
            np.sum((original < 0.25) & (candidate >= 0.25))
        ),
        "cross_iou50_up": int(
            np.sum((original < 0.50) & (candidate >= 0.50))
        ),
        "output": str(output),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--forbidden-scene-list", type=Path, required=True
    )
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Exclude valid candidates which failed the inference hard gate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = build_gate_training_archive(
        geometry_root=args.geometry_root,
        prediction_root=args.prediction_root,
        scene_list=args.scene_list,
        forbidden_scene_list=args.forbidden_scene_list,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        output=args.output,
        include_unverified=not args.verified_only,
        overwrite=args.overwrite,
    )
    import json

    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
