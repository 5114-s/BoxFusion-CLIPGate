#!/usr/bin/env python3
"""Report class-agnostic ScanNet recall of missing-track Mask Graph boxes.

The online pipeline stores Mask Graph components as world-frame AABBs in each
``<scene>_tracks.npz`` diagnostic file.  ScanNet ground truth is axis aligned,
so this tool expands every graph AABB to eight corners, applies the scene
``axisAlignment`` transform, and encloses the transformed corners with a new
axis-aligned box before computing IoU.

This is an offline diagnostic only.  It intentionally depends on just NumPy
and the Python standard library and must not be imported by online inference.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_THRESHOLDS = (0.15, 0.25, 0.50)
DIAGNOSTIC_SUFFIX = "_tracks.npz"
PREDICTION_SUFFIX = "_boxes.pkl"
LIVE_STATES = frozenset(("active", "archived"))
KNOWN_STATES = frozenset(
    ("active", "archived", "absorbed", "discarded", "expired")
)
GRAPH_FIELDS = (
    "graph_component_track_ids",
    "graph_component_states",
    "graph_component_boxes",
    "graph_component_track_confirmed",
    "graph_component_confirmed",
)

_CORNER_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def validate_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("at least one IoU threshold is required")
    if (
        not np.isfinite(np.asarray(thresholds)).all()
        or any(value <= 0.0 or value > 1.0 for value in thresholds)
    ):
        raise ValueError("IoU thresholds must be finite and in (0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("IoU thresholds must be unique")
    return thresholds


def read_scene_ids(path: str | Path) -> list[str]:
    scene_list = Path(path)
    if not scene_list.is_file():
        raise FileNotFoundError(scene_list)
    scenes = [
        line.strip()
        for line in scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scenes:
        raise ValueError(f"no scene ids in {scene_list}")
    duplicates = sorted(
        scene for scene, count in Counter(scenes).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"duplicate scene ids in {scene_list}: {duplicates}"
        )
    return scenes


def load_axis_alignment(scans_root: str | Path, scene_id: str) -> np.ndarray:
    metadata = Path(scans_root) / scene_id / f"{scene_id}.txt"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    values: np.ndarray | None = None
    for line in metadata.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("axisAlignment"):
            if "=" not in stripped:
                raise ValueError(f"malformed axisAlignment in {metadata}")
            values = np.fromstring(stripped.split("=", 1)[1], sep=" ")
            break
    if values is None or values.size != 16 or not np.isfinite(values).all():
        raise ValueError(
            f"invalid or missing axisAlignment in {metadata}"
        )
    transform = values.reshape(4, 4).astype(np.float64, copy=False)
    if not np.allclose(
        transform[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    ):
        raise ValueError(f"axisAlignment is not homogeneous in {metadata}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"axisAlignment is not rigid in {metadata}")
    if not np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=2e-3):
        raise ValueError(
            f"axisAlignment rotation is singular in {metadata}"
        )
    return transform


def load_gt_boxes(gt_root: str | Path, scene_id: str) -> np.ndarray:
    path = Path(gt_root) / f"{scene_id}_bbox.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = np.load(path, allow_pickle=False)
    if payload.ndim != 2 or payload.shape[1] < 6:
        raise ValueError(f"GT boxes in {path} must have shape [N, >=6]")
    boxes = np.asarray(payload[:, :6], dtype=np.float64)
    if (
        not np.isfinite(boxes).all()
        or (len(boxes) and np.any(boxes[:, 3:6] <= 0.0))
    ):
        raise ValueError(f"GT boxes in {path} are invalid")
    return boxes


def center_size_to_corners(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(
            f"center/size boxes must have shape [N,6], got {values.shape}"
        )
    if (
        not np.isfinite(values).all()
        or (len(values) and np.any(values[:, 3:6] <= 0.0))
    ):
        raise ValueError("center/size boxes contain invalid values")
    return (
        values[:, None, :3]
        + _CORNER_SIGNS[None] * (0.5 * values[:, None, 3:6])
    )


def transform_corners(
    corners: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError(
            f"corners must have shape [N,8,3], got {values.shape}"
        )
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape [4,4]")
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise ValueError("corners and transform must be finite")
    return (
        values @ matrix[:3, :3].T
        + matrix[None, None, :3, 3]
    )


def corners_to_minmax(corners: np.ndarray) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError(
            f"corners must have shape [N,8,3], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("corners contain non-finite values")
    if not len(values):
        return np.empty((0, 6), dtype=np.float64)
    return np.concatenate(
        (values.min(axis=1), values.max(axis=1)), axis=1
    )


def center_size_to_minmax(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(
            f"center/size boxes must have shape [N,6], got {values.shape}"
        )
    if (
        not np.isfinite(values).all()
        or (len(values) and np.any(values[:, 3:6] <= 0.0))
    ):
        raise ValueError("center/size boxes contain invalid values")
    half = values[:, 3:6] * 0.5
    return np.concatenate(
        (values[:, :3] - half, values[:, :3] + half), axis=1
    )


def pairwise_aabb_iou(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pred = np.asarray(predictions, dtype=np.float64)
    gt = np.asarray(targets, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 6:
        raise ValueError(f"predictions must have shape [N,6], got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 6:
        raise ValueError(f"targets must have shape [M,6], got {gt.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("IoU boxes must be finite")
    if not len(pred) or not len(gt):
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    if np.any(pred[:, 3:] <= pred[:, :3]) or np.any(
        gt[:, 3:] <= gt[:, :3]
    ):
        raise ValueError("IoU boxes must have positive extents")
    intersection_min = np.maximum(pred[:, None, :3], gt[None, :, :3])
    intersection_max = np.minimum(pred[:, None, 3:], gt[None, :, 3:])
    intersection_size = np.maximum(
        intersection_max - intersection_min, 0.0
    )
    intersection = np.prod(intersection_size, axis=2)
    pred_volume = np.prod(pred[:, 3:] - pred[:, :3], axis=1)
    gt_volume = np.prod(gt[:, 3:] - gt[:, :3], axis=1)
    union = pred_volume[:, None] + gt_volume[None] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _read_scalar_scene_id(payload: Mapping[str, np.ndarray], path: Path) -> str:
    if "scene_id" not in payload:
        raise ValueError(f"missing scene_id in {path}")
    value = np.asarray(payload["scene_id"])
    if value.shape != ():
        raise ValueError(f"scene_id in {path} must be a scalar")
    item = value.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"scene_id in {path} must be a non-empty string")
    return item


def load_graph_components(
    diagnostics_root: str | Path, scene_id: str
) -> dict[str, np.ndarray]:
    path = Path(diagnostics_root) / f"{scene_id}{DIAGNOSTIC_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        stored_scene = _read_scalar_scene_id(payload, path)
        if stored_scene != scene_id:
            raise ValueError(
                f"scene mismatch in {path}: expected {scene_id}, "
                f"found {stored_scene}"
            )
        missing = [name for name in GRAPH_FIELDS if name not in payload]
        if missing:
            raise ValueError(
                f"missing Mask Graph fields in {path}: {missing}"
            )
        track_ids_raw = np.asarray(
            payload["graph_component_track_ids"]
        )
        if track_ids_raw.dtype.kind not in "iu":
            raise ValueError(f"graph track ids in {path} must be integers")
        track_ids = track_ids_raw.astype(np.int64, copy=False)
        states = np.asarray(payload["graph_component_states"])
        boxes = np.asarray(
            payload["graph_component_boxes"], dtype=np.float64
        )
        track_confirmed_raw = np.asarray(
            payload["graph_component_track_confirmed"]
        )
        graph_confirmed_raw = np.asarray(
            payload["graph_component_confirmed"]
        )

    if track_ids.ndim != 1:
        raise ValueError(f"graph track ids in {path} must have shape [N]")
    count = len(track_ids)
    if states.ndim != 1 or len(states) != count:
        raise ValueError(f"graph states in {path} must have shape [{count}]")
    if boxes.shape != (count, 6):
        raise ValueError(
            f"graph boxes in {path} must have shape [{count},6]"
        )
    for name, values in (
        ("track confirmation", track_confirmed_raw),
        ("graph confirmation", graph_confirmed_raw),
    ):
        if values.ndim != 1 or len(values) != count:
            raise ValueError(
                f"graph {name} in {path} must have shape [{count}]"
            )
        if values.dtype.kind != "b":
            raise ValueError(f"graph {name} in {path} must be boolean")
    if len(set(track_ids.tolist())) != count or np.any(track_ids < 0):
        raise ValueError(
            f"graph track ids in {path} must be unique non-negative integers"
        )
    state_strings = states.astype(np.str_)
    unknown = sorted(set(state_strings.tolist()) - KNOWN_STATES)
    if unknown:
        raise ValueError(f"unknown graph states in {path}: {unknown}")
    if (
        not np.isfinite(boxes).all()
        or (count and np.any(boxes[:, 3:6] <= 0.0))
    ):
        raise ValueError(f"graph boxes in {path} are invalid")
    return {
        "track_ids": track_ids,
        "states": state_strings,
        "boxes": boxes,
        "track_confirmed": track_confirmed_raw.astype(bool, copy=False),
        "graph_confirmed": graph_confirmed_raw.astype(bool, copy=False),
    }


def load_baseline_corners(
    pred_root: str | Path, scene_id: str
) -> np.ndarray:
    path = Path(pred_root) / f"{scene_id}{PREDICTION_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(
            f"predictions in {path} must contain exactly one scene batch"
        )
    detections = payload[0]
    if not isinstance(detections, (list, tuple)):
        raise ValueError(f"detections in {path} must be a sequence")
    corners = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, (list, tuple)) or len(detection) < 3:
            raise ValueError(
                f"detection {index} in {path} must contain label/corners/score"
            )
        value = np.asarray(detection[1], dtype=np.float64)
        if value.shape != (8, 3) or not np.isfinite(value).all():
            raise ValueError(
                f"detection {index} corners in {path} must be finite [8,3]"
            )
        score = float(detection[2])
        if not np.isfinite(score):
            raise ValueError(
                f"detection {index} score in {path} is not finite"
            )
        corners.append(value)
    if not corners:
        return np.empty((0, 8, 3), dtype=np.float64)
    return np.stack(corners)


def _selection_masks(components: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    states = components["states"]
    graph_confirmed = components["graph_confirmed"]
    track_confirmed = components["track_confirmed"]
    live = np.isin(states, tuple(sorted(LIVE_STATES)))
    output_confirmed = graph_confirmed & track_confirmed
    return {
        "all": np.ones(len(states), dtype=bool),
        "confirmed": graph_confirmed,
        "confirmed_live": output_confirmed & live,
        "confirmed_active": output_confirmed & (states == "active"),
        "confirmed_archived": output_confirmed & (states == "archived"),
    }


def _scene_recall_counts(
    predictions: np.ndarray,
    gt: np.ndarray,
    thresholds: Sequence[float],
) -> dict[float, int]:
    iou = pairwise_aabb_iou(predictions, gt)
    maximum = (
        iou.max(axis=0)
        if len(predictions)
        else np.zeros(len(gt), dtype=np.float64)
    )
    return {
        float(threshold): int(np.count_nonzero(maximum >= threshold))
        for threshold in thresholds
    }


def _empty_accumulator(thresholds: Sequence[float]) -> dict[str, Any]:
    return {
        "proposal_count": 0,
        "ground_truth_count": 0,
        "matched": {float(value): 0 for value in thresholds},
    }


def _update_accumulator(
    accumulator: dict[str, Any],
    proposal_count: int,
    ground_truth_count: int,
    matched: Mapping[float, int],
) -> None:
    accumulator["proposal_count"] += int(proposal_count)
    accumulator["ground_truth_count"] += int(ground_truth_count)
    for threshold, count in matched.items():
        accumulator["matched"][float(threshold)] += int(count)


def _render_accumulator(
    accumulator: Mapping[str, Any], thresholds: Sequence[float]
) -> dict[str, Any]:
    gt_count = int(accumulator["ground_truth_count"])
    reports = {}
    for threshold in thresholds:
        matched = int(accumulator["matched"][float(threshold)])
        reports[_threshold_key(threshold)] = {
            "matched_ground_truth": matched,
            "ground_truth": gt_count,
            "recall": (
                float(matched / gt_count) if gt_count > 0 else 0.0
            ),
        }
    return {
        "proposal_count": int(accumulator["proposal_count"]),
        "ground_truth_count": gt_count,
        "thresholds": reports,
    }


def _render_increment(
    baseline: Mapping[str, Any],
    combined: Mapping[str, Any],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    rendered = {}
    for threshold in thresholds:
        key = _threshold_key(threshold)
        base_row = baseline["thresholds"][key]
        combined_row = combined["thresholds"][key]
        rendered[key] = {
            "incremental_matched_ground_truth": int(
                combined_row["matched_ground_truth"]
                - base_row["matched_ground_truth"]
            ),
            "recall_gain": float(
                combined_row["recall"] - base_row["recall"]
            ),
        }
    return rendered


def build_report(
    *,
    diagnostics_root: str | Path,
    gt_root: str | Path,
    scans_root: str | Path,
    scene_list: str | Path,
    pred_root: str | Path | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build an aggregate and per-scene Mask Graph proposal-recall report."""

    thresholds = validate_thresholds(thresholds)
    scenes = read_scene_ids(scene_list)
    for root, label in (
        (diagnostics_root, "diagnostics root"),
        (gt_root, "GT root"),
        (scans_root, "scans root"),
    ):
        if not Path(root).is_dir():
            raise FileNotFoundError(f"{label} does not exist: {root}")
    if pred_root is not None and not Path(pred_root).is_dir():
        raise FileNotFoundError(
            f"prediction root does not exist: {pred_root}"
        )

    group_names = (
        "all",
        "confirmed",
        "confirmed_live",
        "confirmed_active",
        "confirmed_archived",
    )
    aggregate = {
        name: _empty_accumulator(thresholds) for name in group_names
    }
    baseline_aggregate = (
        _empty_accumulator(thresholds) if pred_root is not None else None
    )
    combined_aggregate = (
        {
            "confirmed": _empty_accumulator(thresholds),
            "confirmed_live": _empty_accumulator(thresholds),
        }
        if pred_root is not None
        else None
    )
    total_state_counts: Counter[str] = Counter()
    per_scene: dict[str, Any] = {}

    for scene_id in scenes:
        components = load_graph_components(diagnostics_root, scene_id)
        transform = load_axis_alignment(scans_root, scene_id)
        gt = center_size_to_minmax(load_gt_boxes(gt_root, scene_id))
        graph_world_corners = center_size_to_corners(components["boxes"])
        graph_aligned = corners_to_minmax(
            transform_corners(graph_world_corners, transform)
        )
        masks = _selection_masks(components)
        state_counts = Counter(components["states"].tolist())
        total_state_counts.update(state_counts)
        scene_groups: dict[str, Any] = {}
        for name in group_names:
            selected = graph_aligned[masks[name]]
            matched = _scene_recall_counts(selected, gt, thresholds)
            _update_accumulator(
                aggregate[name], len(selected), len(gt), matched
            )
            local = _empty_accumulator(thresholds)
            _update_accumulator(local, len(selected), len(gt), matched)
            scene_groups[name] = _render_accumulator(local, thresholds)

        scene_row: dict[str, Any] = {
            "state_counts": dict(sorted(state_counts.items())),
            "graph": scene_groups,
        }
        if pred_root is not None:
            baseline_corners = load_baseline_corners(pred_root, scene_id)
            baseline = corners_to_minmax(
                transform_corners(baseline_corners, transform)
            )
            baseline_matched = _scene_recall_counts(
                baseline, gt, thresholds
            )
            assert baseline_aggregate is not None
            assert combined_aggregate is not None
            _update_accumulator(
                baseline_aggregate,
                len(baseline),
                len(gt),
                baseline_matched,
            )
            local_baseline = _empty_accumulator(thresholds)
            _update_accumulator(
                local_baseline,
                len(baseline),
                len(gt),
                baseline_matched,
            )
            rendered_baseline = _render_accumulator(
                local_baseline, thresholds
            )
            scene_row["baseline"] = rendered_baseline
            scene_row["baseline_plus_graph"] = {}
            for name in ("confirmed", "confirmed_live"):
                selected = graph_aligned[masks[name]]
                combined = np.concatenate((baseline, selected), axis=0)
                matched = _scene_recall_counts(combined, gt, thresholds)
                _update_accumulator(
                    combined_aggregate[name],
                    len(combined),
                    len(gt),
                    matched,
                )
                local = _empty_accumulator(thresholds)
                _update_accumulator(
                    local, len(combined), len(gt), matched
                )
                rendered = _render_accumulator(local, thresholds)
                rendered["increment_vs_baseline"] = _render_increment(
                    rendered_baseline, rendered, thresholds
                )
                scene_row["baseline_plus_graph"][name] = rendered
        per_scene[scene_id] = scene_row

    rendered_graph = {
        name: _render_accumulator(aggregate[name], thresholds)
        for name in group_names
    }
    report: dict[str, Any] = {
        "schema": "mask_graph_proposal_recall_v1",
        "scene_count": len(scenes),
        "scenes": scenes,
        "thresholds": [float(value) for value in thresholds],
        "selection_definitions": {
            "all": "all graph components in diagnostics",
            "confirmed": "graph_component_confirmed",
            "confirmed_live": (
                "graph_component_confirmed AND "
                "graph_component_track_confirmed AND state in "
                "{active,archived}"
            ),
            "confirmed_active": (
                "confirmed_live restricted to state=active"
            ),
            "confirmed_archived": (
                "confirmed_live restricted to state=archived"
            ),
        },
        "state_counts": dict(sorted(total_state_counts.items())),
        "graph": rendered_graph,
        "per_scene": per_scene,
    }
    if pred_root is not None:
        assert baseline_aggregate is not None
        assert combined_aggregate is not None
        rendered_baseline = _render_accumulator(
            baseline_aggregate, thresholds
        )
        report["baseline"] = rendered_baseline
        report["baseline_plus_graph"] = {}
        for name in ("confirmed", "confirmed_live"):
            rendered = _render_accumulator(
                combined_aggregate[name], thresholds
            )
            rendered["increment_vs_baseline"] = _render_increment(
                rendered_baseline, rendered, thresholds
            )
            report["baseline_plus_graph"][name] = rendered
    return report


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        return validate_thresholds(
            item.strip() for item in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure class-agnostic ScanNet proposal recall of Mask Graph "
            "components stored in online-refinement diagnostics."
        )
    )
    parser.add_argument(
        "--diagnostics-root", type=Path, required=True
    )
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--pred-root",
        type=Path,
        help=(
            "Optional baseline prediction root; enables baseline versus "
            "baseline+confirmed incremental-recall reports."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="comma-separated IoU thresholds (default: 0.15,0.25,0.50)",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        type=Path,
        help="optional path for the rendered JSON report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        diagnostics_root=args.diagnostics_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        scene_list=args.scene_list,
        pred_root=args.pred_root,
        thresholds=args.thresholds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
