#!/usr/bin/env python3
"""Offline oracle audit for cross-lifecycle Mask Graph fragment stitching.

This tool never participates in online inference.  It reads stored graph
diagnostics, frozen predictions, ScanNet ground truth, and axis-alignment
metadata to answer two questions:

1. how many same-label cross-track clusters are induced by geometry rules;
2. how many *new* ground-truth objects those clusters can cover beyond a
   frozen baseline.

The ``recommended_or`` row reuses the exact observer runtime candidate
builder, then applies the frozen C1 extent/global/output-track structural
gates.  All sweep rows, including ``strict_and``, are explicitly exploratory
legacy connected-component analyses and must not be interpreted as deployable
runtime behavior.  Candidate boxes are appended at a deliberately low score
for a ranked-AP simulation; no prediction or runtime file is changed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.fragment_stitch import (
    build_fragment_stitch_candidates,
    resolve_fragment_stitch_config,
)
from boxfusion.online_refinement import (
    _axis_overlap_containment,
    aabb_iou,
    bev_iou_and_containment,
    supplemental_extent_is_valid,
)
from tools.analyze_fused_oracle import (
    maximum_matches,
    ranked_metrics,
    score_scene,
)
from tools.report_mask_graph_recall import (
    center_size_to_corners,
    center_size_to_minmax,
    corners_to_minmax,
    load_axis_alignment,
    load_baseline_corners,
    load_gt_boxes,
    pairwise_aabb_iou,
    read_scene_ids,
    transform_corners,
)


DEFAULT_THRESHOLDS = (0.15, 0.25, 0.50)
LIVE_STATES = frozenset(("active", "archived"))
REQUIRED_FIELDS = (
    "graph_component_track_ids",
    "graph_component_states",
    "graph_component_event_frames",
    "graph_component_boxes",
    "graph_component_view_counts",
    "graph_component_node_counts",
    "graph_component_edge_counts",
    "graph_component_unique_frame_counts",
    "graph_component_confirmed",
    "graph_component_mean_detector_score",
    "graph_component_labels",
    "graph_component_memory_geometry_points",
    "boxes",
    "track_ids",
    "output_is_supplemental",
)
OPTIONAL_FIELDS = ("fragment_stitch_config_json",)

C1_EXTENT_CONFIG: Mapping[str, Any] = {
    "class_aware_extent": True,
    "planar_extent_labels": ("door", "window"),
    "planar_min_extent": 0.04,
    "planar_middle_extent": 0.50,
    "planar_max_extent": 0.50,
    "small_extent_labels": ("sink",),
    "small_min_extent": 0.12,
    "small_middle_extent": 0.20,
    "small_max_extent": 0.30,
}


@dataclass(frozen=True)
class FragmentRule:
    name: str
    mode: str
    minimum_iou: float = 0.0
    minimum_containment: float = 0.0
    maximum_center_distance: float = float("inf")

    def edges(
        self,
        iou: np.ndarray,
        containment: np.ndarray,
        center_distance: np.ndarray,
    ) -> np.ndarray:
        iou_gate = iou >= self.minimum_iou
        containment_gate = (
            (containment >= self.minimum_containment)
            & (center_distance <= self.maximum_center_distance)
        )
        if self.mode == "iou":
            return iou_gate
        if self.mode == "center":
            return center_distance <= self.maximum_center_distance
        if self.mode == "containment":
            return containment_gate
        if self.mode == "or":
            return iou_gate | containment_gate
        if self.mode == "and":
            return iou_gate & containment_gate
        raise ValueError(f"unknown fragment rule mode: {self.mode}")


def pairwise_fragment_geometry(
    boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("fragment boxes must have shape [N,6]")
    if (
        not np.isfinite(values).all()
        or (len(values) and np.any(values[:, 3:6] <= 0.0))
    ):
        raise ValueError("fragment boxes must be finite and positive")
    minimum = values[:, :3] - 0.5 * values[:, 3:6]
    maximum = values[:, :3] + 0.5 * values[:, 3:6]
    volume = np.prod(values[:, 3:6], axis=1)
    overlap = np.maximum(
        np.minimum(maximum[:, None], maximum[None])
        - np.maximum(minimum[:, None], minimum[None]),
        0.0,
    )
    intersection = np.prod(overlap, axis=2)
    union = volume[:, None] + volume[None] - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    smaller = np.minimum(volume[:, None], volume[None])
    containment = np.divide(
        intersection,
        smaller,
        out=np.zeros_like(intersection),
        where=smaller > 0.0,
    )
    distance = np.linalg.norm(
        values[:, None, :3] - values[None, :, :3],
        axis=2,
    )
    return iou, containment, distance


def connected_components(edges: np.ndarray) -> list[list[int]]:
    values = np.asarray(edges, dtype=bool)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("edge matrix must be square")
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in zip(*np.where(np.triu(values, 1))):
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    groups: dict[int, list[int]] = {}
    for index in range(len(values)):
        groups.setdefault(find(index), []).append(index)
    return [members for members in groups.values() if len(members) >= 2]


def _load_fragments(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in REQUIRED_FIELDS if name not in payload]
        if missing:
            raise ValueError(f"missing fragment fields in {path}: {missing}")
        return {
            name: np.asarray(payload[name]).copy()
            for name in REQUIRED_FIELDS + OPTIONAL_FIELDS
            if name in payload
        }


def resolve_fragment_stitch_config_provenance(
    fragments: Mapping[str, np.ndarray],
    *,
    minimum_frame_gap: int,
) -> dict[str, Any]:
    """Resolve one scene's recorded runtime config or legacy fallback."""

    if isinstance(minimum_frame_gap, (bool, np.bool_)) or not isinstance(
        minimum_frame_gap, (int, np.integer)
    ):
        raise ValueError("minimum_frame_gap must be an integer")
    minimum_frame_gap = int(minimum_frame_gap)
    if minimum_frame_gap < 1:
        raise ValueError("minimum_frame_gap must be positive")

    if "fragment_stitch_config_json" not in fragments:
        effective = resolve_fragment_stitch_config(
            {
                "enabled": True,
                "minimum_event_frame_separation": minimum_frame_gap,
            }
        )
        return {
            "config_source": "legacy_default",
            "effective_config": effective,
        }

    encoded = np.asarray(fragments["fragment_stitch_config_json"])
    if encoded.shape != ():
        raise ValueError(
            "fragment_stitch_config_json must be a scalar JSON string"
        )
    value = encoded.item()
    if not isinstance(value, str):
        raise ValueError(
            "fragment_stitch_config_json must contain a JSON string"
        )
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            "fragment_stitch_config_json is not valid JSON"
        ) from error
    if not isinstance(decoded, Mapping):
        raise ValueError(
            "fragment_stitch_config_json must decode to a mapping"
        )
    effective = resolve_fragment_stitch_config(decoded)
    recorded_gap = int(effective["minimum_event_frame_separation"])
    if recorded_gap != minimum_frame_gap:
        raise ValueError(
            "CLI minimum-frame-gap conflicts with recorded "
            "fragment_stitch config: "
            f"CLI={minimum_frame_gap}, recorded={recorded_gap}"
        )
    return {
        "config_source": "diagnostic_npz",
        "effective_config": effective,
    }


def resolve_consistent_fragment_stitch_config(
    scene_fragments: Sequence[tuple[str, Mapping[str, np.ndarray]]],
    *,
    minimum_frame_gap: int,
) -> dict[str, Any]:
    """Resolve provenance and reject effective-config drift across scenes."""

    if not scene_fragments:
        raise ValueError(
            "cannot resolve fragment_stitch config from an empty scene list"
        )
    resolved_rows = []
    for scene, fragments in scene_fragments:
        provenance = resolve_fragment_stitch_config_provenance(
            fragments,
            minimum_frame_gap=minimum_frame_gap,
        )
        canonical = json.dumps(
            provenance["effective_config"],
            sort_keys=True,
            separators=(",", ":"),
        )
        resolved_rows.append((str(scene), provenance, canonical))

    reference_scene, reference, reference_canonical = resolved_rows[0]
    for scene, _, canonical in resolved_rows[1:]:
        if canonical != reference_canonical:
            raise ValueError(
                "inconsistent fragment_stitch configs across scenes: "
                f"{reference_scene} differs from {scene}"
            )
    sources = {
        row[1]["config_source"] for row in resolved_rows
    }
    config_source = (
        next(iter(sources))
        if len(sources) == 1
        else "mixed_diagnostic_npz_and_legacy_default"
    )
    return {
        "config_source": config_source,
        "effective_config": dict(reference["effective_config"]),
    }


def _load_detections(path: Path) -> list[tuple[Any, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(f"invalid prediction payload: {path}")
    return list(payload[0])


def _corners_to_world_boxes(
    detections: Sequence[tuple[Any, np.ndarray, float]],
) -> np.ndarray:
    boxes = []
    for _, corners, _ in detections:
        values = np.asarray(corners, dtype=np.float64)
        if values.shape != (8, 3) or not np.isfinite(values).all():
            raise ValueError("prediction corners must be finite [8,3]")
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        boxes.append(
            np.concatenate(((minimum + maximum) * 0.5, maximum - minimum))
        )
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 6)


def _duplicates_frozen_global(
    candidate: np.ndarray,
    baseline_world_boxes: np.ndarray,
    *,
    final_minimum_extent: float,
) -> bool:
    eligible = baseline_world_boxes[
        np.all(
            baseline_world_boxes[:, 3:6] >= final_minimum_extent,
            axis=1,
        )
    ]
    for reference in eligible:
        if (
            aabb_iou(
                candidate[:3],
                candidate[3:6],
                reference[:3],
                reference[3:6],
            )
            >= 0.25
        ):
            return True
        bev_iou, bev_containment = bev_iou_and_containment(
            candidate, reference
        )
        z_containment = _axis_overlap_containment(
            candidate, reference, axis=2
        )
        if (
            bev_iou >= 0.50
            and bev_containment >= 0.80
            and z_containment >= 0.25
        ):
            return True
    return False


def _build_edges(
    fragments: Mapping[str, np.ndarray],
    rule: FragmentRule,
    *,
    minimum_frame_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(fragments["graph_component_boxes"], dtype=np.float64)
    labels = np.asarray(fragments["graph_component_labels"]).astype(str)
    frames = np.asarray(
        fragments["graph_component_event_frames"], dtype=np.int64
    )
    iou, containment, distance = pairwise_fragment_geometry(boxes)
    edges = (
        rule.edges(iou, containment, distance)
        & (labels[:, None] == labels[None])
        & (np.abs(frames[:, None] - frames[None]) >= minimum_frame_gap)
    )
    np.fill_diagonal(edges, False)
    return edges, iou


def _rank_anchor(
    members: Sequence[int],
    fragments: Mapping[str, np.ndarray],
    iou: np.ndarray,
    *,
    structural: bool,
    default_minimum_extent: float,
) -> int | None:
    boxes = np.asarray(fragments["graph_component_boxes"], dtype=np.float64)
    labels = np.asarray(fragments["graph_component_labels"]).astype(str)
    views = np.asarray(
        fragments["graph_component_view_counts"], dtype=np.int64
    )
    unique_frames = np.asarray(
        fragments["graph_component_unique_frame_counts"], dtype=np.int64
    )
    points = np.asarray(
        fragments["graph_component_memory_geometry_points"], dtype=np.int64
    )
    scores = np.asarray(
        fragments["graph_component_mean_detector_score"], dtype=np.float64
    )
    track_ids = np.asarray(
        fragments["graph_component_track_ids"], dtype=np.int64
    )
    ranked = []
    for index in members:
        if structural and not supplemental_extent_is_valid(
            boxes[index, 3:6],
            labels[index],
            C1_EXTENT_CONFIG,
            default_minimum_extent=default_minimum_extent,
        ):
            continue
        ranked.append(
            (
                float(iou[index, members].sum()),
                int(unique_frames[index]),
                int(views[index]),
                int(points[index]),
                float(scores[index]),
                -int(track_ids[index]),
                int(index),
            )
        )
    return None if not ranked else int(max(ranked)[-1])


def _runtime_fragment_snapshots(
    fragments: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Convert frozen diagnostic columns to the exact runtime input schema."""

    columns = {
        "track_id": np.asarray(
            fragments["graph_component_track_ids"], dtype=np.int64
        ),
        "lifecycle_state": np.asarray(
            fragments["graph_component_states"]
        ).astype(str),
        "event_frame": np.asarray(
            fragments["graph_component_event_frames"], dtype=np.int64
        ),
        "box": np.asarray(
            fragments["graph_component_boxes"], dtype=np.float32
        ),
        "view_count": np.asarray(
            fragments["graph_component_view_counts"], dtype=np.int64
        ),
        "node_count": np.asarray(
            fragments["graph_component_node_counts"], dtype=np.int64
        ),
        "edge_count": np.asarray(
            fragments["graph_component_edge_counts"], dtype=np.int64
        ),
        "memory_geometry_points": np.asarray(
            fragments["graph_component_memory_geometry_points"],
            dtype=np.int64,
        ),
        "mean_detector_score": np.asarray(
            fragments["graph_component_mean_detector_score"],
            dtype=np.float64,
        ),
        "label": np.asarray(
            fragments["graph_component_labels"]
        ).astype(str),
        "graph_confirmed": np.asarray(
            fragments["graph_component_confirmed"], dtype=bool
        ),
    }
    row_count = len(columns["track_id"])
    for name, values in columns.items():
        if name == "box":
            if values.shape != (row_count, 6):
                raise ValueError(
                    "graph_component_boxes must align as [N,6]"
                )
        elif values.shape != (row_count,):
            raise ValueError(
                f"fragment diagnostic column {name} must align as [N]"
            )

    return [
        {
            name: (
                values[index].copy()
                if name == "box"
                else values[index].item()
            )
            for name, values in columns.items()
        }
        for index in range(row_count)
    ]


def select_recommended_clusters(
    fragments: Mapping[str, np.ndarray],
    *,
    baseline_world_boxes: np.ndarray,
    minimum_frame_gap: int = 5,
    minimum_cluster_max_score: float = 0.85,
    minimum_cluster_mean_score: float = 0.70,
    final_minimum_extent: float = 0.40,
    fragment_stitch_config: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Apply the exact runtime anchor-clique, then frozen C1 structure gates."""

    if fragment_stitch_config is None:
        runtime_config = resolve_fragment_stitch_config(
            {
                "enabled": True,
                "minimum_max_detector_score": (
                    minimum_cluster_max_score
                ),
                "minimum_mean_detector_score": (
                    minimum_cluster_mean_score
                ),
                "minimum_event_frame_separation": minimum_frame_gap,
            }
        )
    else:
        runtime_config = resolve_fragment_stitch_config(
            fragment_stitch_config
        )
        recorded_gap = int(
            runtime_config["minimum_event_frame_separation"]
        )
        if recorded_gap != int(minimum_frame_gap):
            raise ValueError(
                "minimum_frame_gap conflicts with effective "
                "fragment_stitch config"
            )
    # Match the observer controller's fail-open row contract: an invalid
    # low-point/empty-memory snapshot is ignored without discarding otherwise
    # valid candidates from the scene.
    valid_snapshots = []
    for snapshot in _runtime_fragment_snapshots(fragments):
        try:
            build_fragment_stitch_candidates([snapshot], runtime_config)
        except Exception:
            continue
        valid_snapshots.append(snapshot)
    runtime_candidates = build_fragment_stitch_candidates(
        valid_snapshots, runtime_config
    )

    boxes = np.asarray(
        fragments["graph_component_boxes"], dtype=np.float64
    )
    states = np.asarray(fragments["graph_component_states"]).astype(str)
    frames = np.asarray(
        fragments["graph_component_event_frames"], dtype=np.int64
    )
    scores = np.asarray(
        fragments["graph_component_mean_detector_score"], dtype=np.float64
    )
    views = np.asarray(
        fragments["graph_component_view_counts"], dtype=np.int64
    )
    track_ids = np.asarray(
        fragments["graph_component_track_ids"], dtype=np.int64
    )
    index_by_track_id = {
        int(track_id): int(index)
        for index, track_id in enumerate(track_ids)
    }
    if len(index_by_track_id) != len(track_ids):
        raise ValueError("fragment diagnostic track IDs must be unique")

    existing_supplemental = {
        int(-stable_id - 1)
        for stable_id in np.asarray(fragments["track_ids"], dtype=np.int64)[
            np.asarray(fragments["output_is_supplemental"], dtype=bool)
        ]
    }
    baseline = np.asarray(baseline_world_boxes, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape[1:] != (6,):
        raise ValueError("baseline world boxes must have shape [N,6]")
    if not np.isfinite(baseline).all():
        raise ValueError("baseline world boxes must be finite")

    selected: list[dict[str, Any]] = []
    for candidate in runtime_candidates:
        # These are the C1 gates applied *after* the runtime identity/quality
        # contract.  They cannot change the runtime partition or anchor.
        if existing_supplemental.intersection(candidate.track_ids):
            continue
        if not supplemental_extent_is_valid(
            candidate.box[3:6],
            candidate.label,
            C1_EXTENT_CONFIG,
            default_minimum_extent=final_minimum_extent,
        ):
            continue
        if _duplicates_frozen_global(
            candidate.box,
            baseline,
            final_minimum_extent=final_minimum_extent,
        ):
            continue

        member_indices = [
            index_by_track_id[int(track_id)]
            for track_id in candidate.track_ids
        ]
        anchor_index = index_by_track_id[
            int(candidate.representative_track_id)
        ]
        selected.append(
            {
                "anchor_index": anchor_index,
                "anchor_track_id": int(
                    candidate.representative_track_id
                ),
                "anchor_box": np.asarray(
                    candidate.box, dtype=np.float64
                ).copy(),
                "label": candidate.label,
                "member_indices": member_indices,
                "member_track_ids": [
                    int(value) for value in candidate.track_ids
                ],
                "member_states": [
                    str(states[index]) for index in member_indices
                ],
                "member_event_frames": [
                    int(frames[index]) for index in member_indices
                ],
                "member_scores": [
                    float(scores[index]) for index in member_indices
                ],
                "member_views": [
                    int(views[index]) for index in member_indices
                ],
            }
        )
    return selected


def select_clusters(
    fragments: Mapping[str, np.ndarray],
    rule: FragmentRule,
    *,
    structural: bool,
    baseline_world_boxes: np.ndarray | None = None,
    minimum_frame_gap: int = 5,
    minimum_cluster_max_score: float = 0.85,
    minimum_cluster_mean_score: float = 0.70,
    final_minimum_extent: float = 0.40,
) -> list[dict[str, Any]]:
    edges, iou = _build_edges(
        fragments, rule, minimum_frame_gap=minimum_frame_gap
    )
    boxes = np.asarray(fragments["graph_component_boxes"], dtype=np.float64)
    labels = np.asarray(fragments["graph_component_labels"]).astype(str)
    states = np.asarray(fragments["graph_component_states"]).astype(str)
    frames = np.asarray(
        fragments["graph_component_event_frames"], dtype=np.int64
    )
    scores = np.asarray(
        fragments["graph_component_mean_detector_score"], dtype=np.float64
    )
    views = np.asarray(
        fragments["graph_component_view_counts"], dtype=np.int64
    )
    track_ids = np.asarray(
        fragments["graph_component_track_ids"], dtype=np.int64
    )
    existing_supplemental = {
        int(-stable_id - 1)
        for stable_id in np.asarray(fragments["track_ids"], dtype=np.int64)[
            np.asarray(fragments["output_is_supplemental"], dtype=bool)
        ]
    }

    selected: list[dict[str, Any]] = []
    for component in connected_components(edges):
        if structural:
            if existing_supplemental.intersection(
                int(track_ids[index]) for index in component
            ):
                continue
            if not any(states[index] in LIVE_STATES for index in component):
                continue
            if (
                float(np.max(scores[component]))
                < minimum_cluster_max_score
                or float(np.mean(scores[component]))
                < minimum_cluster_mean_score
            ):
                continue
        anchor = _rank_anchor(
            component,
            fragments,
            iou,
            structural=structural,
            default_minimum_extent=final_minimum_extent,
        )
        if anchor is None:
            continue
        # Legacy exploratory pruning retained for historical sweeps.  This is
        # not the runtime anchor-clique contract and can still group mutually
        # incompatible direct neighbors around one medoid.
        members = [
            index
            for index in component
            if index == anchor or bool(edges[anchor, index])
        ]
        if len(members) < 2:
            continue
        if structural and (
            float(np.max(scores[members])) < minimum_cluster_max_score
            or float(np.mean(scores[members])) < minimum_cluster_mean_score
        ):
            continue
        if structural:
            if baseline_world_boxes is None:
                raise ValueError(
                    "structural selection requires baseline world boxes"
                )
            if _duplicates_frozen_global(
                boxes[anchor],
                baseline_world_boxes,
                final_minimum_extent=final_minimum_extent,
            ):
                continue
        selected.append(
            {
                "anchor_index": int(anchor),
                "anchor_track_id": int(track_ids[anchor]),
                "anchor_box": boxes[anchor].copy(),
                "label": str(labels[anchor]),
                "member_indices": [int(index) for index in members],
                "member_track_ids": [
                    int(track_ids[index]) for index in members
                ],
                "member_states": [str(states[index]) for index in members],
                "member_event_frames": [
                    int(frames[index]) for index in members
                ],
                "member_scores": [
                    float(scores[index]) for index in members
                ],
                "member_views": [int(views[index]) for index in members],
            }
        )
    return selected


def _render_rule(
    *,
    rule: FragmentRule,
    scenes: Sequence[str],
    diagnostics_root: Path,
    prediction_root: Path,
    gt_root: Path,
    scans_root: Path,
    thresholds: Sequence[float],
    structural: bool,
    supplemental_score: float,
    minimum_frame_gap: int,
    runtime_recommended: bool,
    fragment_stitch_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_count = 0
    ground_truth_count = 0
    threshold_records = {
        float(value): {"real": [], "all_tp": 0, "novel_tp": 0}
        for value in thresholds
    }
    manifest: list[dict[str, Any]] = []

    for scene in scenes:
        fragments = _load_fragments(
            diagnostics_root / f"{scene}_tracks.npz"
        )
        detections = _load_detections(
            prediction_root / f"{scene}_boxes.pkl"
        )
        baseline_world = _corners_to_world_boxes(detections)
        if runtime_recommended:
            clusters = select_recommended_clusters(
                fragments,
                baseline_world_boxes=baseline_world,
                minimum_frame_gap=minimum_frame_gap,
                fragment_stitch_config=fragment_stitch_provenance[
                    "effective_config"
                ],
            )
        else:
            clusters = select_clusters(
                fragments,
                rule,
                structural=structural,
                baseline_world_boxes=baseline_world,
                minimum_frame_gap=minimum_frame_gap,
            )
        transform = load_axis_alignment(scans_root, scene)
        baseline_corners = load_baseline_corners(prediction_root, scene)
        baseline_aligned = corners_to_minmax(
            transform_corners(baseline_corners, transform)
        )
        gt = center_size_to_minmax(load_gt_boxes(gt_root, scene))
        candidate_world = np.asarray(
            [row["anchor_box"] for row in clusters],
            dtype=np.float64,
        ).reshape(-1, 6)
        candidate_corners = center_size_to_corners(candidate_world)
        candidate_aligned = corners_to_minmax(
            transform_corners(candidate_corners, transform)
        )
        baseline_iou = pairwise_aabb_iou(baseline_aligned, gt)
        candidate_iou = pairwise_aabb_iou(candidate_aligned, gt)

        baseline_scores = np.asarray(
            [float(item[2]) for item in detections], dtype=np.float64
        )
        combined_iou = np.concatenate(
            (baseline_iou, candidate_iou), axis=0
        )
        combined_scores = np.concatenate(
            (
                baseline_scores,
                np.full(len(clusters), supplemental_score),
            )
        )
        for threshold in thresholds:
            threshold = float(threshold)
            records, _, _ = score_scene(
                combined_iou, combined_scores, threshold
            )
            threshold_records[threshold]["real"].extend(records)
            threshold_records[threshold]["all_tp"] += len(
                maximum_matches(candidate_iou, threshold)[0]
            )
            baseline_covered = (
                np.max(baseline_iou, axis=0) >= threshold
                if len(baseline_iou)
                else np.zeros(len(gt), dtype=bool)
            )
            threshold_records[threshold]["novel_tp"] += len(
                maximum_matches(
                    candidate_iou[:, ~baseline_covered], threshold
                )[0]
            )

        for row, overlaps in zip(clusters, candidate_iou):
            rendered = {
                key: value
                for key, value in row.items()
                if key not in {"anchor_index", "member_indices", "anchor_box"}
            }
            rendered["scene"] = scene
            rendered["anchor_box"] = row["anchor_box"].tolist()
            rendered["best_gt_iou"] = float(
                np.max(overlaps, initial=0.0)
            )
            manifest.append(rendered)
        candidate_count += len(clusters)
        ground_truth_count += len(gt)

    threshold_report = {}
    for threshold in thresholds:
        threshold = float(threshold)
        records = threshold_records[threshold]
        ap, recall, precision = ranked_metrics(
            records["real"], ground_truth_count
        )
        threshold_report[f"{threshold:.2f}"] = {
            "all_candidate_tp": int(records["all_tp"]),
            "novel_candidate_tp": int(records["novel_tp"]),
            "novel_candidate_precision": (
                float(records["novel_tp"] / candidate_count)
                if candidate_count
                else 0.0
            ),
            "novel_recall_gain": float(
                records["novel_tp"] / ground_truth_count
            ),
            "ranked_average_precision": float(ap),
            "ranked_recall": float(recall),
            "ranked_final_precision": float(precision),
        }
    rendered_rule = {
        "selection_contract": (
            "runtime_anchor_clique_then_c1_structural_gates"
            if runtime_recommended
            else "exploratory_legacy_connected_components"
        ),
        "exploratory_only": not runtime_recommended,
        "candidate_count": int(candidate_count),
        "ground_truth_count": int(ground_truth_count),
        "thresholds": threshold_report,
        "candidates": manifest,
    }
    if runtime_recommended:
        rendered_rule["config_source"] = fragment_stitch_provenance[
            "config_source"
        ]
        rendered_rule["effective_config"] = dict(
            fragment_stitch_provenance["effective_config"]
        )
    return rendered_rule


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scene_ids(args.scene_list)
    fragment_stitch_provenance = resolve_consistent_fragment_stitch_config(
        [
            (
                scene,
                _load_fragments(
                    args.diagnostics_root / f"{scene}_tracks.npz"
                ),
            )
            for scene in scenes
        ],
        minimum_frame_gap=args.minimum_frame_gap,
    )
    sweep = [
        FragmentRule(f"iou_{value:.2f}", "iou", minimum_iou=value)
        for value in (0.05, 0.10, 0.15, 0.25, 0.50)
    ]
    sweep += [
        FragmentRule(
            f"center_{value:.2f}",
            "center",
            maximum_center_distance=value,
        )
        for value in (0.15, 0.25, 0.40)
    ]
    sweep += [
        FragmentRule(
            f"containment_{containment:.2f}_center_{distance:.2f}",
            "containment",
            minimum_containment=containment,
            maximum_center_distance=distance,
        )
        for containment in (0.50, 0.70, 0.85)
        for distance in (0.15, 0.25, 0.40)
    ]
    rules = {
        rule.name: _render_rule(
            rule=rule,
            scenes=scenes,
            diagnostics_root=args.diagnostics_root,
            prediction_root=args.prediction_root,
            gt_root=args.gt_root,
            scans_root=args.scans_root,
            thresholds=args.thresholds,
            structural=False,
            supplemental_score=args.supplemental_score,
            minimum_frame_gap=args.minimum_frame_gap,
            runtime_recommended=False,
            fragment_stitch_provenance=fragment_stitch_provenance,
        )
        for rule in sweep
    }
    selected = (
        FragmentRule(
            "recommended_or",
            "or",
            minimum_iou=0.40,
            minimum_containment=0.60,
            maximum_center_distance=0.25,
        ),
        FragmentRule(
            "strict_and",
            "and",
            minimum_iou=0.45,
            minimum_containment=0.80,
            maximum_center_distance=float("inf"),
        ),
    )
    for rule in selected:
        rules[rule.name] = _render_rule(
            rule=rule,
            scenes=scenes,
            diagnostics_root=args.diagnostics_root,
            prediction_root=args.prediction_root,
            gt_root=args.gt_root,
            scans_root=args.scans_root,
            thresholds=args.thresholds,
            structural=True,
            supplemental_score=args.supplemental_score,
            minimum_frame_gap=args.minimum_frame_gap,
            runtime_recommended=(rule.name == "recommended_or"),
            fragment_stitch_provenance=fragment_stitch_provenance,
        )
    return {
        "schema": "fragment_stitch_offline_ablation_v3",
        "scene_count": len(scenes),
        "scenes": scenes,
        "supplemental_score": float(args.supplemental_score),
        "minimum_frame_gap": int(args.minimum_frame_gap),
        "fragment_stitch_config": {
            "config_source": fragment_stitch_provenance[
                "config_source"
            ],
            "effective_config": dict(
                fragment_stitch_provenance["effective_config"]
            ),
        },
        "deployable_contract_rule": "recommended_or",
        "exploratory_only_rules": [
            rule.name for rule in sweep
        ]
        + ["strict_and"],
        "structural_contract": {
            "same_normalized_label": True,
            "requires_live_member": True,
            "exclude_graph_confirmed_cluster": True,
            "minimum_cluster_max_score": 0.85,
            "minimum_cluster_mean_score": 0.70,
            "c1_extent_policy": True,
            "c1_global_duplicate_gates": True,
            "exclude_existing_supplemental_track": True,
            "runtime_candidate_builder_reused": True,
            "anchor_clique_partition": True,
        },
        "rules": rules,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=lambda value: tuple(
            float(item.strip()) for item in value.split(",")
        ),
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--supplemental-score", type=float, default=0.051)
    parser.add_argument("--minimum-frame-gap", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 0.0 < args.supplemental_score <= 1.0:
        parser.error("--supplemental-score must be in (0,1]")
    if args.minimum_frame_gap < 1:
        parser.error("--minimum-frame-gap must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
