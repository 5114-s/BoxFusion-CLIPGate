#!/usr/bin/env python3
"""Independent no-GT audit for terminal-p100 R3 same-run outputs."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping

import numpy as np


SCHEMA = "boxfusion.tr3d_r3_terminal_active_audit.v2"
DIAGNOSTIC_SCHEMA = "boxfusion.tr3d_r3_terminal_active_scene.v1"
FROZEN_MANIFEST_SCHEMA = "boxfusion.frozen_anchor_manifest.v1"
SHADOW_MANIFEST_SCHEMA = "boxfusion.tr3d_r3_shadow_active_manifest.v1"
EXPECTED_FROZEN_MANIFEST_SHA256 = (
    "327b0cfb07265db04db3af2f631e27e1165a65c9367a9db1d09a31299911342e"
)
EXPECTED_SHADOW_MANIFEST_SHA256 = (
    "2cef1b228bab9df99b203e76c0e72cf14ae4e61ce6519a7e38c12e652a160f56"
)
EXPECTED_FROZEN_TREE_SHA256 = (
    "fe10ee44a56bc5160a606cc8f6d68c90ed08775874130c5e7840e7e184b74e17"
)
EXPECTED_SHADOW_TREE_SHA256 = (
    "1159d87568ae957cce97c534efd7c12f37217a11d1a55009343d3f08672403e2"
)
EXPECTED_PREFIX_MANIFEST_SHA256 = (
    "d2599338c8fe70a74d2ccf062875177240f4ac296af324cbb9796251a9350e54"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
)
EXPECTED_CONFIG_SHA256 = (
    "709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785"
)
FROZEN_NEAR_IOU = 0.15


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path, scenes: tuple[str, ...]) -> str:
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    observed = {
        path.name
        for path in root.glob("scene*_boxes.pkl")
        if path.is_file() and not path.is_symlink()
    }
    if observed != expected:
        raise ValueError(
            f"{root}: reference prediction set mismatch; "
            f"missing={sorted(expected-observed)[:5]}, "
            f"extra={sorted(observed-expected)[:5]}"
        )
    rows = {name: _file_sha(root / name) for name in sorted(expected)}
    digest = hashlib.sha256()
    for relative, value in sorted(rows.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _verify_reference_lineage(
    *,
    frozen_root: Path,
    shadow_root: Path,
    same_run_baseline_root: Path,
    active_root: Path,
    frozen_manifest: Path,
    shadow_manifest: Path,
    requested_scenes: tuple[str, ...],
) -> dict[str, Any]:
    roots = tuple(
        path.resolve()
        for path in (
            frozen_root,
            shadow_root,
            same_run_baseline_root,
            active_root,
        )
    )
    if len(set(roots)) != 4:
        raise ValueError(
            "frozen, shadow, same-run baseline, and active roots must be "
            "distinct"
        )
    if _file_sha(frozen_manifest) != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen G0 manifest SHA256 mismatch")
    if _file_sha(shadow_manifest) != EXPECTED_SHADOW_MANIFEST_SHA256:
        raise ValueError("shadow-gold manifest SHA256 mismatch")
    frozen = _load_json(frozen_manifest)
    shadow = _load_json(shadow_manifest)
    if frozen.get("schema") != FROZEN_MANIFEST_SCHEMA:
        raise ValueError("unexpected frozen G0 manifest schema")
    if shadow.get("schema") != SHADOW_MANIFEST_SCHEMA:
        raise ValueError("unexpected shadow-gold manifest schema")
    frozen_scenes = tuple(str(value) for value in frozen.get("scene_ids", ()))
    if len(frozen_scenes) != 100 or len(set(frozen_scenes)) != 100:
        raise ValueError("frozen G0 manifest must describe exactly 100 scenes")
    if not set(requested_scenes).issubset(frozen_scenes):
        raise ValueError("requested audit scenes are absent from frozen G0")
    if Path(str(frozen.get("reference_result_root", ""))).resolve() != roots[0]:
        raise ValueError("frozen G0 root disagrees with its manifest")
    if Path(str(shadow.get("output_root", ""))).resolve() != roots[1]:
        raise ValueError("shadow-gold root disagrees with its manifest")
    if not (
        shadow.get("complete") is True
        and shadow.get("shadow_only") is True
        and shadow.get("ground_truth_access") is False
    ):
        raise ValueError("shadow-gold manifest contract mismatch")
    frozen_tree = _tree_hash(roots[0], frozen_scenes)
    shadow_tree = _tree_hash(roots[1], frozen_scenes)
    if (
        frozen_tree != EXPECTED_FROZEN_TREE_SHA256
        or frozen.get("prediction_tree_sha256") != frozen_tree
    ):
        raise ValueError("frozen G0 prediction tree SHA256 mismatch")
    if (
        shadow_tree != EXPECTED_SHADOW_TREE_SHA256
        or shadow.get("output_prediction_tree_sha256") != shadow_tree
    ):
        raise ValueError("shadow-gold prediction tree SHA256 mismatch")
    return {
        "frozen_manifest_sha256": EXPECTED_FROZEN_MANIFEST_SHA256,
        "shadow_manifest_sha256": EXPECTED_SHADOW_MANIFEST_SHA256,
        "frozen_prediction_tree_sha256": frozen_tree,
        "shadow_prediction_tree_sha256": shadow_tree,
    }


def _array_sha(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_prediction(path: Path) -> list | tuple:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local experiment artifact
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 1
        or not isinstance(payload[0], (list, tuple))
    ):
        raise ValueError(f"{path}: malformed prediction payload")
    for index, row in enumerate(payload[0]):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{path}: malformed row {index}")
        corners = np.asarray(row[1])
        score = float(row[2])
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f"{path}: malformed corners {index}")
        if not np.isfinite(score):
            raise ValueError(f"{path}: malformed score {index}")
    return payload


def _geometry(payload: list | tuple) -> np.ndarray:
    rows = payload[0]
    if not rows:
        return np.empty((0, 8, 3), dtype=np.float32)
    values = [np.asarray(row[1]) for row in rows]
    dtype = values[0].dtype
    if any(value.dtype != dtype for value in values):
        raise ValueError("prediction geometry rows do not share one dtype")
    return np.ascontiguousarray(np.stack(values))


def _same_array(left: Any, right: Any) -> bool:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    return bool(
        type(left) is type(right)
        and lhs.dtype == rhs.dtype
        and lhs.shape == rhs.shape
        and lhs.strides == rhs.strides
        and lhs.flags.c_contiguous == rhs.flags.c_contiguous
        and lhs.flags.f_contiguous == rhs.flags.f_contiguous
        and lhs.tobytes(order="A") == rhs.tobytes(order="A")
    )


def _same_scalar(left: Any, right: Any) -> bool:
    return bool(
        type(left) is type(right)
        and pickle.dumps(left, protocol=5) == pickle.dumps(right, protocol=5)
    )


def _npz_text(values: Mapping[str, np.ndarray], name: str) -> str:
    if name not in values:
        raise ValueError(f"parent cache is missing {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"parent cache {name} must be a string scalar")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"parent cache {name} must be text")
    return result


def _npz_scalar(
    values: Mapping[str, np.ndarray], name: str, dtype: np.dtype
) -> Any:
    if name not in values:
        raise ValueError(f"parent cache is missing {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype != np.dtype(dtype):
        raise ValueError(
            f"parent cache {name} must be a {np.dtype(dtype)} scalar"
        )
    return value.item()


def _load_parent_cache_for_audit(path: Path) -> dict[str, Any]:
    """Load one parent cache from a single immutable byte snapshot."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    encoded = path.read_bytes()
    cache_sha256 = hashlib.sha256(encoded).hexdigest()
    with np.load(BytesIO(encoded), allow_pickle=False) as archive:
        values = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError(f"{path}: parent cache contains object arrays")
    if _npz_text(values, "schema") != "boxfusion.tr3d_residual_cache.v1":
        raise ValueError(f"{path}: unexpected parent cache schema")
    scene_id = _npz_text(values, "scene_id")
    prefix_id = _npz_text(values, "prefix_id")
    prefix_fraction = float(
        _npz_scalar(values, "prefix_fraction", np.float64)
    )
    checkpoint_sha256 = _npz_text(values, "checkpoint_sha256")
    config_sha256 = _npz_text(values, "config_sha256")
    source_scene_sha256 = _npz_text(values, "source_scene_sha256")
    proposal_ids = np.asarray(values.get("proposal_ids"))
    corners = np.asarray(values.get("corners_world"))
    scores = np.asarray(values.get("scores_3d"))
    count = len(proposal_ids) if proposal_ids.ndim == 1 else -1
    if (
        count < 0
        or proposal_ids.dtype != np.dtype(np.int64)
        or len(np.unique(proposal_ids)) != count
        or np.any(proposal_ids < 0)
        or corners.dtype != np.dtype(np.float32)
        or corners.shape != (count, 8, 3)
        or not np.isfinite(corners).all()
        or scores.dtype != np.dtype(np.float32)
        or scores.shape != (count,)
        or not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
    ):
        raise ValueError(f"{path}: malformed parent proposal arrays")
    if count and np.any(np.ptp(corners, axis=1) <= 0.0):
        raise ValueError(f"{path}: parent proposal has non-positive extent")
    num_input_points = int(
        _npz_scalar(values, "num_input_points", np.int64)
    )
    return {
        "cache_sha256": cache_sha256,
        "scene_id": scene_id,
        "prefix_id": prefix_id,
        "prefix_fraction": prefix_fraction,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "source_scene_sha256": source_scene_sha256,
        "num_input_points": num_input_points,
        "proposal_ids": proposal_ids,
        "corners_world": corners,
        "scores_3d": scores,
    }


def _prediction_drift(reference: list | tuple, observed: list | tuple) -> dict[str, Any]:
    """Describe historical-reference drift without declaring it unsafe."""

    reference_rows = reference[0]
    observed_rows = observed[0]
    aligned = min(len(reference_rows), len(observed_rows))
    row_type_drift = 0
    label_drift = 0
    score_drift = 0
    geometry_drift = 0
    score_max_abs_delta = 0.0
    for index in range(aligned):
        left = reference_rows[index]
        right = observed_rows[index]
        row_type_drift += int(type(left) is not type(right))
        label_drift += int(not _same_scalar(left[0], right[0]))
        score_drift += int(not _same_scalar(left[2], right[2]))
        geometry_drift += int(not _same_array(left[1], right[1]))
        score_max_abs_delta = max(
            score_max_abs_delta,
            abs(float(left[2]) - float(right[2])),
        )
    return {
        "reference_count": len(reference_rows),
        "observed_count": len(observed_rows),
        "count_delta": len(observed_rows) - len(reference_rows),
        "outer_container_type_equal": type(reference) is type(observed),
        "batch_container_type_equal": type(reference_rows) is type(observed_rows),
        "aligned_rows": aligned,
        "row_container_type_drift": row_type_drift,
        "label_drift_rows": label_drift,
        "score_drift_rows": score_drift,
        "geometry_drift_rows": geometry_drift,
        "score_max_abs_delta": score_max_abs_delta,
    }


def _accumulate_drift(total: dict[str, Any], scene: Mapping[str, Any]) -> None:
    total["reference_rows"] += int(scene["reference_count"])
    total["observed_rows"] += int(scene["observed_count"])
    total["count_abs_delta"] += abs(int(scene["count_delta"]))
    for name in (
        "row_container_type_drift",
        "label_drift_rows",
        "score_drift_rows",
        "geometry_drift_rows",
    ):
        total[name] += int(scene[name])
    total["outer_container_type_drift_scenes"] += int(
        not scene["outer_container_type_equal"]
    )
    total["batch_container_type_drift_scenes"] += int(
        not scene["batch_container_type_equal"]
    )
    total["score_max_abs_delta"] = max(
        float(total["score_max_abs_delta"]),
        float(scene["score_max_abs_delta"]),
    )


def _empty_drift_total() -> dict[str, Any]:
    return {
        "reference_rows": 0,
        "observed_rows": 0,
        "count_abs_delta": 0,
        "outer_container_type_drift_scenes": 0,
        "batch_container_type_drift_scenes": 0,
        "row_container_type_drift": 0,
        "label_drift_rows": 0,
        "score_drift_rows": 0,
        "geometry_drift_rows": 0,
        "score_max_abs_delta": 0.0,
    }


def _aligned_minmax(corners_world: Any, axis_alignment: Any) -> np.ndarray:
    corners = np.asarray(corners_world, dtype=np.float64)
    matrix = np.asarray(axis_alignment, dtype=np.float64)
    if (
        corners.ndim != 3
        or corners.shape[1:] != (8, 3)
        or not np.isfinite(corners).all()
        or matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError("invalid geometry/axis alignment for R3 recomputation")
    aligned = corners @ matrix[:3, :3].T + matrix[None, None, :3, 3]
    if not len(aligned):
        return np.empty((0, 6), dtype=np.float64)
    result = np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)
    if np.any(result[:, 3:] <= result[:, :3]):
        raise ValueError("R3 recomputation received non-positive AABBs")
    return result


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    overlap = np.maximum(
        np.minimum(left[:, None, 3:], right[None, :, 3:])
        - np.maximum(left[:, None, :3], right[None, :, :3]),
        0.0,
    )
    intersection = np.prod(overlap, axis=2)
    left_volume = np.prod(left[:, 3:] - left[:, :3], axis=1)
    right_volume = np.prod(right[:, 3:] - right[:, :3], axis=1)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _recompute_frozen_selections(
    *,
    baseline_geometry: np.ndarray,
    baseline_scores: np.ndarray,
    parent: Mapping[str, Any],
    axis_alignment: Any,
) -> tuple[dict[str, Any], ...]:
    """Independently reproduce the frozen IoU/tie-break/score R3 rule."""

    candidate_boxes = _aligned_minmax(
        parent["corners_world"], axis_alignment
    )
    anchor_boxes = _aligned_minmax(baseline_geometry, axis_alignment)
    if not len(candidate_boxes) or not len(anchor_boxes):
        return ()
    iou = _pairwise_iou(candidate_boxes, anchor_boxes)
    candidate_centres = (candidate_boxes[:, :3] + candidate_boxes[:, 3:]) * 0.5
    anchor_centres = (anchor_boxes[:, :3] + anchor_boxes[:, 3:]) * 0.5
    distances = np.linalg.norm(
        candidate_centres[:, None] - anchor_centres[None, :, :], axis=2
    )
    best_iou = iou.max(axis=1)
    associated = np.empty(len(candidate_boxes), dtype=np.int64)
    for row in range(len(candidate_boxes)):
        tied = np.flatnonzero(iou[row] == best_iou[row])
        associated[row] = int(
            tied[int(np.argmin(distances[row, tied]))]
        )
    near_rows = np.flatnonzero(best_iou > FROZEN_NEAR_IOU)
    near_anchors = associated[near_rows]
    selections: list[dict[str, Any]] = []
    for anchor in np.unique(near_anchors):
        parent_rows = near_rows[np.flatnonzero(near_anchors == anchor)]
        order = np.lexsort(
            (
                parent["proposal_ids"][parent_rows],
                -parent["scores_3d"][parent_rows],
            )
        )
        proposal_row = int(parent_rows[int(order[0])])
        tr3d_score = float(parent["scores_3d"][proposal_row])
        anchor_index = int(anchor)
        anchor_score = float(baseline_scores[anchor_index])
        if tr3d_score <= anchor_score:
            continue
        candidate = parent["corners_world"][proposal_row]
        selections.append(
            {
                "anchor_index": anchor_index,
                "proposal_row": proposal_row,
                "proposal_id": int(parent["proposal_ids"][proposal_row]),
                "tr3d_score": tr3d_score,
                "anchor_score": anchor_score,
                "anchor_iou": float(best_iou[proposal_row]),
                "geometry_changed": not _same_array(
                    baseline_geometry[anchor_index], candidate
                ),
            }
        )
    return tuple(selections)


def _read_scenes(path: Path) -> tuple[str, ...]:
    scenes = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("scene list is empty or contains duplicates")
    return scenes


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _prefix_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    if _file_sha(path) != EXPECTED_PREFIX_MANIFEST_SHA256:
        raise ValueError("terminal prefix manifest SHA256 mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        scene = str(row.get("scene_id", ""))
        if not scene or scene in result:
            raise ValueError(f"{path}:{line_number}: invalid/duplicate scene")
        result[scene] = row
    return result


def _canonical_row_sha(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit(
    *,
    scene_list: Path,
    frozen_root: Path,
    same_run_baseline_root: Path,
    active_root: Path,
    shadow_root: Path,
    diagnostics_root: Path,
    frozen_manifest: Path,
    shadow_manifest: Path,
    prefix_manifest: Path,
    parent_cache_root: Path,
) -> dict[str, Any]:
    scenes = _read_scenes(scene_list)
    lineage = _verify_reference_lineage(
        frozen_root=frozen_root,
        shadow_root=shadow_root,
        same_run_baseline_root=same_run_baseline_root,
        active_root=active_root,
        frozen_manifest=frozen_manifest,
        shadow_manifest=shadow_manifest,
        requested_scenes=scenes,
    )
    prefixes = _prefix_rows(prefix_manifest)
    issues: list[str] = []
    rows = 0
    changed_rows = 0
    selected_rows = 0
    cached_model_runtime: list[float] = []
    replay_runtime: list[float] = []
    baseline_frozen_drift_total = _empty_drift_total()
    active_shadow_drift_total = _empty_drift_total()
    per_scene: dict[str, Any] = {}
    for scene in scenes:
        frozen_path = frozen_root / f"{scene}_boxes.pkl"
        baseline_path = same_run_baseline_root / f"{scene}_boxes.pkl"
        active_path = active_root / f"{scene}_boxes.pkl"
        shadow_path = shadow_root / f"{scene}_boxes.pkl"
        diagnostic_path = diagnostics_root / f"{scene}_tr3d_terminal.json"
        try:
            frozen = _load_prediction(frozen_path)
            baseline = _load_prediction(baseline_path)
            active = _load_prediction(active_path)
            shadow = _load_prediction(shadow_path)
            diagnostic = _load_json(diagnostic_path)
        except Exception as error:
            issues.append(f"{scene}: {type(error).__name__}: {error}")
            continue
        if diagnostic.get("schema") != DIAGNOSTIC_SCHEMA:
            issues.append(f"{scene}: diagnostic schema mismatch")
        prefix = prefixes.get(scene)
        if prefix is None:
            issues.append(f"{scene}: terminal prefix row is absent")
            continue
        prefix_id = str(prefix.get("tag") or prefix.get("prefix_id") or "")
        point_path = Path(str(prefix.get("point_path", "")))
        parent_path = parent_cache_root / scene / f"{prefix_id}.npz"
        try:
            point_sha = _file_sha(point_path)
            parent = _load_parent_cache_for_audit(parent_path)
        except Exception as error:
            issues.append(f"{scene}: lineage file error: {error}")
            continue
        observed_timestamps = tuple(
            int(value) for value in diagnostic.get(
                "observed_source_timestamps", ()
            )
        )
        expected_timestamps = tuple(
            int(value) for value in prefix.get("used_source_timestamps", ())
        )
        lineage_checks = {
            "prefix_id": diagnostic.get("prefix_id") == prefix_id,
            "terminal_timestamp": diagnostic.get("current_source_timestamp")
            == int(prefix.get("last_source_timestamp", -1)),
            "observed_schedule": observed_timestamps == expected_timestamps,
            "manifest_row": diagnostic.get("manifest_row_sha256")
            == _canonical_row_sha(prefix),
            "manifest_path": Path(
                str(diagnostic.get("manifest_path", ""))
            ).resolve()
            == prefix_manifest.resolve(),
            "parent_cache_root": Path(
                str(diagnostic.get("parent_cache_root", ""))
            ).resolve()
            == parent_cache_root.resolve(),
            "parent_cache": diagnostic.get("cache_sha256")
            == parent["cache_sha256"],
            "source_point": diagnostic.get("source_point_sha256") == point_sha,
            "parent_source_point": parent["source_scene_sha256"] == point_sha,
            "checkpoint": diagnostic.get("checkpoint_sha256")
            == EXPECTED_CHECKPOINT_SHA256
            == parent["checkpoint_sha256"],
            "config": diagnostic.get("config_sha256")
            == EXPECTED_CONFIG_SHA256
            == parent["config_sha256"],
            "parent_scene": parent["scene_id"] == scene,
            "parent_prefix": parent["prefix_id"] == prefix_id,
            "parent_fraction": float(parent["prefix_fraction"])
            == float(prefix.get("fraction", -1.0)),
            "parent_point_count": int(parent["num_input_points"])
            == int(prefix.get("point_count", -1)),
            "parent_proposal_count": diagnostic.get("parent_proposal_count")
            == len(parent["proposal_ids"]),
        }
        for name, ok in lineage_checks.items():
            if not ok:
                issues.append(f"{scene}: {name} lineage mismatch")

        baseline_frozen_drift = _prediction_drift(frozen, baseline)
        active_shadow_drift = _prediction_drift(shadow, active)
        _accumulate_drift(
            baseline_frozen_drift_total, baseline_frozen_drift
        )
        _accumulate_drift(active_shadow_drift_total, active_shadow_drift)

        baseline_rows = baseline[0]
        active_rows = active[0]
        protected_ok = True
        if type(baseline) is not type(active):
            protected_ok = False
            issues.append(f"{scene}: outer prediction container type changed")
        if type(baseline_rows) is not type(active_rows):
            protected_ok = False
            issues.append(f"{scene}: batch prediction container type changed")
        if len(baseline_rows) != len(active_rows):
            protected_ok = False
            issues.append(
                f"{scene}: same-run prediction count changed "
                f"({len(baseline_rows)} != {len(active_rows)})"
            )
            per_scene[scene] = {
                "baseline_rows": len(baseline_rows),
                "active_rows": len(active_rows),
                "selected": diagnostic.get("selected_count"),
                "changed": diagnostic.get("changed_count"),
                "protected_ok": False,
                "historical_drift": {
                    "baseline_vs_frozen": baseline_frozen_drift,
                    "active_vs_shadow": active_shadow_drift,
                },
            }
            continue
        rows += len(active_rows)

        baseline_geometry = _geometry(baseline)
        active_geometry = _geometry(active)
        baseline_scores = np.asarray(
            [float(row[2]) for row in baseline_rows], dtype=np.float64
        )
        if diagnostic.get("input_geometry_sha256") != _array_sha(
            baseline_geometry
        ):
            issues.append(
                f"{scene}: diagnostic input geometry is not same-run baseline"
            )
        if diagnostic.get("input_scores_sha256") != _array_sha(
            baseline_scores
        ):
            issues.append(
                f"{scene}: diagnostic input scores are not same-run baseline"
            )
        input_row_hashes = tuple(
            str(value)
            for value in diagnostic.get("input_row_geometry_sha256", ())
        )
        if len(input_row_hashes) != len(active_rows):
            issues.append(f"{scene}: input row geometry hashes are incomplete")
            input_row_hashes = tuple("" for _ in active_rows)

        try:
            expected_selections = _recompute_frozen_selections(
                baseline_geometry=baseline_geometry,
                baseline_scores=baseline_scores,
                parent=parent,
                axis_alignment=prefix.get("axis_align_matrix"),
            )
        except Exception as error:
            issues.append(
                f"{scene}: frozen R3 selection recomputation failed: {error}"
            )
            expected_selections = ()

        raw_selections = diagnostic.get("selections", ())
        if not isinstance(raw_selections, (list, tuple)):
            issues.append(f"{scene}: selections must be a sequence")
            raw_selections = ()
        selections_by_anchor: dict[int, Mapping[str, Any]] = {}
        selected_parent_rows: set[int] = set()
        for selection_index, item in enumerate(raw_selections):
            if not isinstance(item, Mapping):
                issues.append(
                    f"{scene}: selection {selection_index} is not an object"
                )
                continue
            anchor = item.get("anchor_index")
            proposal_row = item.get("proposal_row")
            if (
                isinstance(anchor, bool)
                or not isinstance(anchor, int)
                or isinstance(proposal_row, bool)
                or not isinstance(proposal_row, int)
            ):
                issues.append(
                    f"{scene}: selection {selection_index} has invalid indices"
                )
                continue
            if anchor < 0 or anchor >= len(baseline_rows):
                issues.append(
                    f"{scene}: selection {selection_index} anchor is out of range"
                )
                continue
            if proposal_row < 0 or proposal_row >= len(parent["proposal_ids"]):
                issues.append(
                    f"{scene}: selection {selection_index} proposal is out of range"
                )
                continue
            if anchor in selections_by_anchor:
                issues.append(f"{scene}: duplicate selected anchor {anchor}")
                continue
            if proposal_row in selected_parent_rows:
                issues.append(
                    f"{scene}: duplicate selected parent row {proposal_row}"
                )
                continue
            selections_by_anchor[anchor] = item
            selected_parent_rows.add(proposal_row)

            expected_proposal_id = int(parent["proposal_ids"][proposal_row])
            expected_tr3d_score = float(parent["scores_3d"][proposal_row])
            if item.get("proposal_id") != expected_proposal_id:
                issues.append(
                    f"{scene}: selection {selection_index} proposal_id mismatch"
                )
            try:
                observed_tr3d_score = float(item.get("tr3d_score"))
                observed_anchor_score = float(item.get("anchor_score"))
            except (TypeError, ValueError):
                observed_tr3d_score = observed_anchor_score = float("nan")
            if observed_tr3d_score != expected_tr3d_score:
                issues.append(
                    f"{scene}: selection {selection_index} TR3D score mismatch"
                )
            if observed_anchor_score != float(baseline_scores[anchor]):
                issues.append(
                    f"{scene}: selection {selection_index} anchor score mismatch"
                )
            if not expected_tr3d_score > float(baseline_scores[anchor]):
                issues.append(
                    f"{scene}: selection {selection_index} violates score gate"
                )

        try:
            diagnostic_selected_count = int(
                diagnostic.get("selected_count", -1)
            )
        except (TypeError, ValueError):
            diagnostic_selected_count = -1
        if diagnostic_selected_count != len(raw_selections):
            issues.append(f"{scene}: diagnostic selected count mismatch")
        if len(selections_by_anchor) != len(raw_selections):
            issues.append(f"{scene}: one or more selections are invalid")
        if len(raw_selections) != len(expected_selections):
            issues.append(
                f"{scene}: selections disagree with independently recomputed "
                "frozen R3 rule"
            )
        for selection_index, expected in enumerate(expected_selections):
            if selection_index >= len(raw_selections):
                break
            observed = raw_selections[selection_index]
            if not isinstance(observed, Mapping):
                continue
            for name in ("anchor_index", "proposal_row", "proposal_id"):
                if observed.get(name) != expected[name]:
                    issues.append(
                        f"{scene}: selection {selection_index} frozen-rule "
                        f"{name} mismatch"
                    )
            for name in ("tr3d_score", "anchor_score", "anchor_iou"):
                try:
                    equal = float(observed.get(name)) == float(expected[name])
                except (TypeError, ValueError):
                    equal = False
                if not equal:
                    issues.append(
                        f"{scene}: selection {selection_index} frozen-rule "
                        f"{name} mismatch"
                    )
            if observed.get("geometry_changed") is not expected[
                "geometry_changed"
            ]:
                issues.append(
                    f"{scene}: selection {selection_index} frozen-rule "
                    "geometry_changed mismatch"
                )

        same_run_scene_changed = 0
        selection_changed_flags = 0
        for index, (base, output) in enumerate(zip(baseline_rows, active_rows)):
            if type(base) is not type(output):
                protected_ok = False
                issues.append(f"{scene}:{index}: row container type changed")
            if not _same_scalar(base[0], output[0]):
                protected_ok = False
                issues.append(f"{scene}:{index}: label changed")
            if not _same_scalar(base[2], output[2]):
                protected_ok = False
                issues.append(f"{scene}:{index}: score changed")
            if type(base[1]) is not type(output[1]):
                protected_ok = False
                issues.append(
                    f"{scene}:{index}: geometry container type changed"
                )
            if input_row_hashes[index] != _array_sha(np.asarray(base[1])):
                issues.append(
                    f"{scene}:{index}: diagnostic row input is not baseline"
                )
            changed = not _same_array(base[1], output[1])
            same_run_scene_changed += int(changed)
            selection = selections_by_anchor.get(index)
            if selection is None:
                if changed:
                    protected_ok = False
                    issues.append(
                        f"{scene}:{index}: unselected same-run geometry changed"
                    )
                continue
            proposal_row = int(selection["proposal_row"])
            candidate = parent["corners_world"][proposal_row]
            if not _same_array(output[1], candidate):
                protected_ok = False
                issues.append(
                    f"{scene}:{index}: selected geometry differs from parent "
                    f"proposal row {proposal_row}"
                )
            declared_changed = selection.get("geometry_changed")
            if type(declared_changed) is not bool or declared_changed != changed:
                issues.append(
                    f"{scene}:{index}: selection geometry_changed mismatch"
                )
            selection_changed_flags += int(declared_changed is True)

        if diagnostic.get("output_geometry_sha256") != _array_sha(active_geometry):
            issues.append(f"{scene}: diagnostic/output geometry hash mismatch")
        if diagnostic.get("prediction_count") != len(baseline_rows):
            issues.append(f"{scene}: diagnostic prediction count mismatch")
        try:
            diagnostic_changed_count = int(
                diagnostic.get("changed_count", -1)
            )
        except (TypeError, ValueError):
            diagnostic_changed_count = -1
        if (
            diagnostic_changed_count != same_run_scene_changed
            or diagnostic_changed_count != selection_changed_flags
        ):
            issues.append(f"{scene}: diagnostic changed count mismatch")
        if not diagnostic.get(
            "labels_scores_order_count_unchanged_by_construction", False
        ):
            issues.append(f"{scene}: protected-field contract absent")
        if diagnostic.get("ground_truth_access") is not False:
            issues.append(f"{scene}: inference diagnostic does not deny GT access")
        if diagnostic.get("provider_mode") != "immutable_parent_cache_replay":
            issues.append(f"{scene}: unexpected provider mode")
        selected = diagnostic_selected_count
        changed = diagnostic_changed_count
        try:
            model_s = float(
                diagnostic.get("cache_model_runtime_s", float("nan"))
            )
            replay_s = float(
                diagnostic.get("replay_total_s", float("nan"))
            )
        except (TypeError, ValueError):
            model_s = replay_s = float("nan")
        if selected < 0 or changed < 0 or not np.isfinite(model_s + replay_s):
            issues.append(f"{scene}: invalid count/runtime diagnostic")
        else:
            selected_rows += selected
            changed_rows += changed
            cached_model_runtime.append(model_s)
            replay_runtime.append(replay_s)
        per_scene[scene] = {
            "rows": len(active_rows),
            "selected": selected,
            "changed": changed,
            "protected_ok": protected_ok,
            "same_run_baseline_authoritative": True,
            "historical_drift": {
                "baseline_vs_frozen": baseline_frozen_drift,
                "active_vs_shadow": active_shadow_drift,
            },
        }
    model_array = np.asarray(cached_model_runtime, dtype=np.float64)
    replay_array = np.asarray(replay_runtime, dtype=np.float64)
    return {
        "schema": SCHEMA,
        "ok": not issues and len(per_scene) == len(scenes),
        "scene_count": len(scenes),
        "audited_scene_count": len(per_scene),
        "prediction_rows": rows,
        "selected_rows": selected_rows,
        "changed_rows": changed_rows,
        "same_run_baseline_root": str(same_run_baseline_root.resolve()),
        "historical_reference_drift": {
            "authoritative_for_safety": False,
            "baseline_vs_frozen": baseline_frozen_drift_total,
            "active_vs_shadow": active_shadow_drift_total,
        },
        "issues": issues,
        "reference_lineage": lineage,
        "prefix_manifest_sha256": EXPECTED_PREFIX_MANIFEST_SHA256,
        "contracts": {
            "auditor_ground_truth_access": False,
            "inference_diagnostics_deny_ground_truth_access": not any(
                "does not deny GT access" in issue for issue in issues
            ),
            "r3_same_run_labels_scores_order_count_exact": not issues,
            "r3_same_run_geometry_safety_exact": not issues,
            "historical_frozen_shadow_drift_is_non_authoritative": True,
            "provider_mode": "immutable_parent_cache_replay",
            "live_tr3d_latency_authoritative": False,
        },
        "runtime": {
            "cached_model_median_ms": (
                None if not len(model_array) else float(np.median(model_array) * 1000)
            ),
            "cached_model_p95_ms": (
                None if not len(model_array) else float(np.quantile(model_array, 0.95) * 1000)
            ),
            "replay_median_ms": (
                None if not len(replay_array) else float(np.median(replay_array) * 1000)
            ),
            "replay_p95_ms": (
                None if not len(replay_array) else float(np.quantile(replay_array, 0.95) * 1000)
            ),
        },
        "per_scene": per_scene,
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"audit report already exists: {path}") from error
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--same-run-baseline-root", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--shadow-manifest", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        scene_list=args.scene_list.resolve(),
        frozen_root=args.frozen_root.resolve(),
        same_run_baseline_root=args.same_run_baseline_root.resolve(),
        active_root=args.active_root.resolve(),
        shadow_root=args.shadow_root.resolve(),
        diagnostics_root=args.diagnostics_root.resolve(),
        frozen_manifest=args.frozen_manifest.resolve(),
        shadow_manifest=args.shadow_manifest.resolve(),
        prefix_manifest=args.prefix_manifest.resolve(),
        parent_cache_root=args.parent_cache_root.resolve(),
    )
    _write_create_only(args.report.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
