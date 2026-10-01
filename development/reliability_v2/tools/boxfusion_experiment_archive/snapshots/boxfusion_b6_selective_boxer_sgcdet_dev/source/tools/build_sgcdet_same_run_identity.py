#!/usr/bin/env python3
"""Build a strict pre-geometry counterfactual from one active SGCDet run."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SCHEMA = "boxfusion.sgcdet_same_run_identity.v1"


def _scene_ids(path: Path) -> Tuple[str, ...]:
    scenes = tuple(
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split() and not line.lstrip().startswith("#")
    )
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"scene list is empty or contains duplicates: {path}")
    return scenes


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "q50": 0.0, "q90": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_min, left_max = left.min(axis=0), left.max(axis=0)
    right_min, right_max = right.min(axis=0), right.max(axis=0)
    intersection = np.maximum(
        np.minimum(left_max, right_max) - np.maximum(left_min, right_min),
        0.0,
    )
    intersection_volume = float(np.prod(intersection))
    left_volume = float(np.prod(np.maximum(left_max - left_min, 0.0)))
    right_volume = float(np.prod(np.maximum(right_max - right_min, 0.0)))
    union = left_volume + right_volume - intersection_volume
    return intersection_volume / union if union > 0.0 else 0.0


def build(
    active_root: Path,
    diagnostics_root: Path,
    scene_list: Path,
    output_root: Path,
) -> Dict[str, object]:
    scenes = _scene_ids(scene_list)
    expected_names = {f"{scene}_boxes.pkl" for scene in scenes}
    active_names = {path.name for path in active_root.glob("*_boxes.pkl")}
    diagnostic_names = {path.name for path in diagnostics_root.glob("*_tracks.npz")}
    expected_diagnostics = {f"{scene}_tracks.npz" for scene in scenes}
    if active_names != expected_names:
        raise ValueError(
            "active prediction set mismatch: "
            f"missing={sorted(expected_names - active_names)}, "
            f"extra={sorted(active_names - expected_names)}"
        )
    if diagnostic_names != expected_diagnostics:
        raise ValueError(
            "diagnostic set mismatch: "
            f"missing={sorted(expected_diagnostics - diagnostic_names)}, "
            f"extra={sorted(diagnostic_names - expected_diagnostics)}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    existing = tuple(output_root.glob("*_boxes.pkl"))
    if existing:
        raise FileExistsError(
            f"counterfactual output already contains predictions: {output_root}"
        )

    total_rows = changed_rows = mapped_rows = 0
    changed_scenes = 0
    center_shifts: List[float] = []
    volume_ratios: List[float] = []
    pre_post_ious: List[float] = []
    per_scene: Dict[str, Dict[str, int]] = {}

    for scene in scenes:
        prediction_path = active_root / f"{scene}_boxes.pkl"
        diagnostic_path = diagnostics_root / f"{scene}_tracks.npz"
        with prediction_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError(f"{prediction_path}: expected one-scene outer list")
        rows = payload[0]
        if not isinstance(rows, list):
            raise ValueError(f"{prediction_path}: rows must be a list")

        with np.load(diagnostic_path, allow_pickle=False) as data:
            required = {
                "output_geometry_schema",
                "output_pre_geometry_corners",
                "output_post_geometry_corners",
                "output_refit_applied",
                "result_indices",
                "sparse_accepted",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"{diagnostic_path}: missing arrays {missing}")
            if str(np.asarray(data["output_geometry_schema"]).item()) != (
                "boxfusion.full_output_geometry_prepost.v1"
            ):
                raise ValueError(f"{diagnostic_path}: unexpected geometry schema")
            pre = np.asarray(data["output_pre_geometry_corners"])
            post = np.asarray(data["output_post_geometry_corners"])
            applied = np.asarray(data["output_refit_applied"], dtype=bool)
            result_indices = np.asarray(data["result_indices"], dtype=np.int64)
            sparse_accepted = np.asarray(data["sparse_accepted"], dtype=bool)

        expected_shape = (len(rows), 8, 3)
        if pre.shape != expected_shape or post.shape != expected_shape:
            raise ValueError(
                f"{diagnostic_path}: pre/post shape does not match {expected_shape}"
            )
        if pre.dtype != np.float32 or post.dtype != np.float32:
            raise ValueError(f"{diagnostic_path}: pre/post corners must be float32")
        if applied.shape != (len(rows),):
            raise ValueError(f"{diagnostic_path}: output_refit_applied shape mismatch")
        if result_indices.shape != sparse_accepted.shape:
            raise ValueError(f"{diagnostic_path}: sparse accepted mapping mismatch")
        if result_indices.size and (
            int(result_indices.min()) < 0
            or int(result_indices.max()) >= len(rows)
            or np.unique(result_indices).size != result_indices.size
        ):
            raise ValueError(f"{diagnostic_path}: invalid result_indices")

        changed = np.any(pre != post, axis=(1, 2))
        mapped_accepted = np.zeros(len(rows), dtype=bool)
        mapped_accepted[result_indices] = sparse_accepted
        if not np.array_equal(changed, applied):
            raise ValueError(f"{diagnostic_path}: changed rows != output_refit_applied")
        if not np.array_equal(changed, mapped_accepted):
            raise ValueError(f"{diagnostic_path}: changed rows != mapped sparse_accepted")

        identity_rows = []
        for index, row in enumerate(rows):
            if not isinstance(row, tuple) or len(row) != 3:
                raise ValueError(f"{prediction_path}: invalid row {index}")
            label, corners, score = row
            exported = np.asarray(corners)
            if exported.dtype != np.float32 or exported.shape != (8, 3):
                raise ValueError(f"{prediction_path}: invalid corners at row {index}")
            if not np.array_equal(exported, post[index]):
                raise ValueError(
                    f"{prediction_path}: exported row {index} != diagnostic post"
                )
            identity_rows.append(
                (int(label), pre[index].copy(), float(score))
            )
            if changed[index]:
                pre_min, pre_max = pre[index].min(axis=0), pre[index].max(axis=0)
                post_min, post_max = post[index].min(axis=0), post[index].max(axis=0)
                pre_center = 0.5 * (pre_min + pre_max)
                post_center = 0.5 * (post_min + post_max)
                center_shifts.append(float(np.linalg.norm(post_center - pre_center)))
                pre_volume = float(np.prod(np.maximum(pre_max - pre_min, 0.0)))
                post_volume = float(np.prod(np.maximum(post_max - post_min, 0.0)))
                volume_ratios.append(post_volume / max(pre_volume, 1.0e-12))
                pre_post_ious.append(_aabb_iou(pre[index], post[index]))

        with (output_root / prediction_path.name).open("wb") as handle:
            pickle.dump([identity_rows], handle, protocol=pickle.HIGHEST_PROTOCOL)

        scene_changed = int(changed.sum())
        total_rows += len(rows)
        mapped_rows += int(result_indices.size)
        changed_rows += scene_changed
        changed_scenes += int(scene_changed > 0)
        per_scene[scene] = {
            "rows": len(rows),
            "mapped_rows": int(result_indices.size),
            "changed_rows": scene_changed,
        }

    report: Dict[str, object] = {
        "schema": SCHEMA,
        "active_prediction_root": str(active_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        "scene_list": str(scene_list.resolve()),
        "counterfactual_root": str(output_root.resolve()),
        "scenes": len(scenes),
        "total_rows": total_rows,
        "mapped_rows": mapped_rows,
        "changed_rows": changed_rows,
        "changed_scenes": changed_scenes,
        "center_shift_m": _quantiles(center_shifts),
        "volume_ratio": _quantiles(volume_ratios),
        "pre_post_aabb_iou": _quantiles(pre_post_ious),
        "per_scene": per_scene,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-pred-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.active_pred_root,
        args.diagnostics_root,
        args.scene_list,
        args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
