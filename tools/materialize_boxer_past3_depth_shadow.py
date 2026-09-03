#!/usr/bin/env python3
"""Seal depth-qualified, frozen-geometry Boxer-Past3 shadow receipts.

The input is the geometry-only OWLv2+Boxer candidate sidecar.  Per-frame OBBs
are associated by :mod:`boxfusion.boxer_past3_receipt`; that tracker freezes an
immutable geometry/provenance receipt at the first stable third-view event.
This tool can then accumulate at most five past/current matched evidence
attempts for that *same* frozen OBB.  A receipt becomes depth-qualified only
when one weakly connected support component contains at least three distinct
frames and at least two causal (earlier -> later) depth-consistent edges.

The tool has no GT, label, CLIP, training, or native-mutation input.  Native T05
boxes are opened only after causal confirmation for terminal duplicate
suppression.  All candidates remain counterfactual (``birth=false``).
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from boxfusion.boxer_past3_receipt import (  # noqa: E402
    BoxerObservation,
    BoxerPast3Receipt,
    BoxerPast3ReceiptTracker,
    SCHEMA as RECEIPT_SCHEMA,
)
from tools.boxfusion_tr3d_pipeline.boxfusion.depth_guide_geometry import (  # noqa: E402
    DEPTH_ALPHA,
    MIN_GUIDE_POINTS,
    project_guide_metrics,
    sample_depth_guide_points,
)
from tools.materialize_boxer_past3_shadow import (  # noqa: E402
    INPUT_SCHEMA,
    ShadowError,
    _array_content_sha256,
    _load_prediction,
    _obb_corners,
    _regular_file,
    _sha256,
    _validate_arrays,
    _validate_input_manifest,
    _write_deterministic_npz,
    _write_json_exclusive,
)


SCHEMA = "boxfusion.boxer_past3_depth_shadow.v1"

MAX_EVIDENCE_ATTEMPTS = 5
# Receipt association retains a track for ten missed keyframes.  Current plus
# that exact committed past is the entire RGB-D ring; arbitrary historical
# disk re-reads are forbidden.
MAX_DEPTH_FRAME_CACHE = 11
MIN_COMPONENT_NODES = 3
MIN_COMPONENT_SUPPORT_EDGES = 2
MIN_CAMERA_BASELINE_M = 0.15
MIN_VIEW_RAY_SPAN_DEG = 10.0
MIN_FORWARD_VISIBILITY = 0.30
MIN_BACKWARD_CONTAINMENT = 0.90
NATIVE_NOVELTY_IOU = 0.10
SELF_NMS_IOU = 0.25
MAX_OUTPUTS_PER_SCENE = 6
DEPTH_SCALE_METERS_PER_UNIT = 0.001

_ALLOWED_SCHEDULE_SCHEMAS = {
    "boxfusion.cutr_postfilter_cache.v2",
    "boxfusion.cutr_postfilter_cache.v3",
    "boxfusion.t05_keyframe_schedule.v1",
}


@dataclass(frozen=True)
class _FrameData:
    frame_id: int
    depth_m: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    camera_center: np.ndarray


@dataclass(frozen=True)
class _DepthNode:
    frame_id: int
    source_row: int
    guide_points_world: np.ndarray
    camera_center: np.ndarray


@dataclass(frozen=True)
class _DepthAttempt:
    frame_id: int
    source_row: int
    node: Optional[_DepthNode]
    failure_reason: Optional[str]


class FrameAccessError(ShadowError):
    """Raised when depth evidence requests future or evicted stream state."""


def _load_schedule(
    schedule_root: Path,
    scene: str,
    scene_ledger: Mapping[str, Any],
) -> tuple[tuple[int, ...], str, str]:
    path = schedule_root / scene / "manifest.json"
    manifest = json.loads(_regular_file(path, "sealed T05 schedule").read_text())
    if not isinstance(manifest, dict):
        raise ShadowError(f"sealed T05 schedule is not an object for {scene}")
    schema = manifest.get("schema")
    if schema not in _ALLOWED_SCHEDULE_SCHEMAS:
        raise ShadowError(f"unexpected T05 schedule schema for {scene}: {schema!r}")
    if manifest.get("scene_id") != scene:
        raise ShadowError(f"T05 schedule scene mismatch for {scene}")
    if scene_ledger.get("sealed_schedule_manifest_sha256") != _sha256(path):
        raise ShadowError(f"T05 schedule hash differs from Boxer seal for {scene}")
    raw = manifest.get("recorded_frame_ids")
    count = manifest.get("record_count")
    if (
        not isinstance(raw, list)
        or not raw
        or any(type(value) is not int or value < 0 for value in raw)
        or raw != sorted(raw)
        or len(set(raw)) != len(raw)
        or count != len(raw)
    ):
        raise ShadowError(f"invalid recorded T05 schedule for {scene}")
    invalid = scene_ledger.get("sealed_schedule_invalid_pose_frame_ids_excluded")
    if not isinstance(invalid, list) or any(type(value) is not int for value in invalid):
        raise ShadowError(f"invalid pose-abstention ledger for {scene}")
    mode = scene_ledger.get("sealed_schedule_mode")
    if mode == "valid_recorded_frames":
        invalid_set = set(invalid)
        schedule = tuple(value for value in raw if value not in invalid_set)
    elif mode == "legacy_record_count":
        if invalid:
            raise ShadowError(f"legacy schedule unexpectedly excludes poses for {scene}")
        schedule = tuple(raw)
    else:
        raise ShadowError(f"unknown sealed schedule mode for {scene}: {mode!r}")
    if len(schedule) != scene_ledger.get("sealed_schedule_frame_count"):
        raise ShadowError(f"sealed schedule count mismatch for {scene}")
    namespace = manifest.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ShadowError(f"invalid sealed schedule namespace for {scene}")
    return schedule, namespace, str(schema)


class _SceneFrameStore:
    """Strict sequential current-plus-TTL ring for one online RGB-D stream."""

    def __init__(self, scene_root: Path, scene: str, allowed_frames: Sequence[int]):
        self.scene = scene
        self.frames_root = scene_root / scene / "frames"
        self.allowed_frame_sequence = tuple(int(value) for value in allowed_frames)
        self.allowed_frames = frozenset(self.allowed_frame_sequence)
        intrinsic_path = self.frames_root / "intrinsic" / "intrinsic_depth.txt"
        _regular_file(intrinsic_path, "ScanNet depth intrinsics")
        try:
            K = np.loadtxt(intrinsic_path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise ShadowError(f"invalid depth intrinsics: {intrinsic_path}") from error
        if K.shape == (4, 4):
            K = K[:3, :3]
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ShadowError(f"depth intrinsics must be finite [3,3]/[4,4]: {intrinsic_path}")
        self.K = np.ascontiguousarray(K)
        self.intrinsic_path = intrinsic_path
        self.intrinsic_sha256 = _sha256(intrinsic_path)
        self._cache: OrderedDict[int, _FrameData] = OrderedDict()
        self._next_advance_index = 0
        self._last_advanced_frame: Optional[int] = None
        self.pose_sha256: dict[str, str] = {}
        self.depth_sha256: dict[str, str] = {}
        self.peak_cached_frames = 0
        self.cache_hits = 0
        self.frames_advanced = 0

    def advance(self, frame_id: int) -> _FrameData:
        """Load exactly the next scheduled frame, including empty keyframes."""

        frame_id = int(frame_id)
        if self._next_advance_index >= len(self.allowed_frame_sequence):
            raise FrameAccessError(f"RGB-D stream already ended for {self.scene}")
        expected = self.allowed_frame_sequence[self._next_advance_index]
        if frame_id != expected:
            raise FrameAccessError(
                f"RGB-D advance must be sequential in {self.scene}: "
                f"expected={expected}, actual={frame_id}"
            )
        pose_path = self.frames_root / "pose" / f"{frame_id}.txt"
        depth_path = self.frames_root / "depth" / f"{frame_id}.png"
        _regular_file(pose_path, "sealed-frame camera pose")
        _regular_file(depth_path, "sealed-frame depth image")
        try:
            pose = np.loadtxt(pose_path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise ShadowError(f"invalid camera pose: {pose_path}") from error
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or np.max(np.abs(pose[3] - [0.0, 0.0, 0.0, 1.0])) > 1e-5
        ):
            raise ShadowError(f"invalid finite homogeneous camera pose: {pose_path}")
        depth_raw = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None or depth_raw.ndim != 2 or depth_raw.dtype.kind not in "ui":
            raise ShadowError(f"depth image must be an integer single-channel PNG: {depth_path}")
        depth_m = np.ascontiguousarray(depth_raw, dtype=np.float64)
        depth_m *= DEPTH_SCALE_METERS_PER_UNIT
        value = _FrameData(
            frame_id=frame_id,
            depth_m=depth_m,
            K=self.K,
            T_wc=np.ascontiguousarray(pose),
            camera_center=np.ascontiguousarray(pose[:3, 3]),
        )
        self.pose_sha256[str(frame_id)] = _sha256(pose_path)
        self.depth_sha256[str(frame_id)] = _sha256(depth_path)
        self._cache[frame_id] = value
        while len(self._cache) > MAX_DEPTH_FRAME_CACHE:
            self._cache.popitem(last=False)
        self.peak_cached_frames = max(self.peak_cached_frames, len(self._cache))
        self._next_advance_index += 1
        self._last_advanced_frame = frame_id
        self.frames_advanced += 1
        return value

    def get(self, frame_id: int) -> _FrameData:
        """Read only committed past/current ring state; never touch disk."""

        frame_id = int(frame_id)
        if frame_id not in self.allowed_frames:
            raise FrameAccessError(f"off-schedule RGB-D access in {self.scene}: {frame_id}")
        if frame_id in self._cache:
            self.cache_hits += 1
            return self._cache[frame_id]
        if self._last_advanced_frame is None or frame_id > self._last_advanced_frame:
            raise FrameAccessError(f"future RGB-D access in {self.scene}: {frame_id}")
        raise FrameAccessError(f"evicted RGB-D access in {self.scene}: {frame_id}")


def _projected_obb_box(corners_world: np.ndarray, frame: _FrameData) -> np.ndarray:
    try:
        T_cw = np.linalg.inv(frame.T_wc)
    except np.linalg.LinAlgError as error:
        raise ShadowError("camera pose is not invertible") from error
    camera = corners_world @ T_cw[:3, :3].T + T_cw[:3, 3]
    if np.any(camera[:, 2] <= 1e-3):
        raise ShadowError("a depth node unexpectedly crosses the camera near plane")
    pixels_h = camera @ frame.K.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    box = np.asarray(
        [pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()],
        dtype=np.float64,
    )
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ShadowError("frozen OBB has an invalid image projection")
    return box


def _make_attempt(
    *,
    frame_id: int,
    source_row: int,
    corners_world: np.ndarray,
    frames: _SceneFrameStore,
) -> _DepthAttempt:
    try:
        frame = frames.get(frame_id)
    except FrameAccessError as error:
        return _DepthAttempt(frame_id, source_row, None, str(error))
    height, width = frame.depth_m.shape
    sample = sample_depth_guide_points(
        frame.depth_m,
        frame.K,
        frame.T_wc,
        np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float64),
        corners_world,
    )
    if sample is None:
        return _DepthAttempt(frame_id, source_row, None, "guide_fewer_than_16_or_unprojectable")
    if not MIN_GUIDE_POINTS <= len(sample.points_world) <= 64:
        raise ShadowError("depth utility returned an out-of-contract guide size")
    node = _DepthNode(
        frame_id=frame_id,
        source_row=source_row,
        guide_points_world=sample.points_world,
        camera_center=frame.camera_center,
    )
    return _DepthAttempt(frame_id, source_row, node, None)


def _edge_view_geometry(
    left_camera: np.ndarray,
    right_camera: np.ndarray,
    object_center: np.ndarray,
) -> tuple[float, float]:
    baseline = float(np.linalg.norm(left_camera - right_camera))
    left_ray = object_center - left_camera
    right_ray = object_center - right_camera
    left_norm = float(np.linalg.norm(left_ray))
    right_norm = float(np.linalg.norm(right_ray))
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return baseline, 0.0
    cosine = float(np.dot(left_ray, right_ray) / (left_norm * right_norm))
    span = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return baseline, span


def _components(nodes: Sequence[_DepthNode], edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame_ids = [node.frame_id for node in nodes]
    adjacency = {frame_id: set() for frame_id in frame_ids}
    support_edges = [row for row in edges if row["support"]]
    for row in support_edges:
        left = int(row["source_frame_id"])
        right = int(row["target_frame_id"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    results = []
    visited: set[int] = set()
    for root in sorted(frame_ids):
        if root in visited:
            continue
        stack = [root]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        visited.update(component)
        component_edges = [
            row
            for row in support_edges
            if int(row["source_frame_id"]) in component
            and int(row["target_frame_id"]) in component
        ]
        results.append(
            {
                "frame_ids": sorted(component),
                "support_edge_count": len(component_edges),
                "support_edges": component_edges,
            }
        )
    return results


def _choose_qualifying_component(
    components: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Choose one fixed component; disconnected global totals never qualify."""

    eligible = [
        row
        for row in components
        if len(row["frame_ids"]) >= MIN_COMPONENT_NODES
        and row["support_edge_count"] >= MIN_COMPONENT_SUPPORT_EDGES
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            -row["support_edge_count"],
            -len(row["frame_ids"]),
            tuple(row["frame_ids"]),
        ),
    )[0]


def _evaluate_graph(
    attempts: Sequence[_DepthAttempt],
    corners_world: np.ndarray,
    frames: _SceneFrameStore,
    prior_edges: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    nodes = tuple(
        sorted(
            (row.node for row in attempts if row.node is not None),
            key=lambda row: (row.frame_id, row.source_row),
        )
    )
    if len({row.frame_id for row in nodes}) != len(nodes):
        raise ShadowError("depth evidence contains duplicate frames")
    object_center = np.asarray(corners_world, dtype=np.float64).mean(axis=0)
    cached_edges = {
        (int(row["source_frame_id"]), int(row["target_frame_id"])): dict(row)
        for row in prior_edges
    }
    edges: list[dict[str, Any]] = []
    for source_index, source in enumerate(nodes):
        for target in nodes[source_index + 1 :]:
            edge_key = (source.frame_id, target.frame_id)
            if edge_key in cached_edges:
                edges.append(cached_edges[edge_key])
                continue
            # A never-before-seen pair can only target the current node.  The
            # target frame is therefore still in the strict online ring.  Old
            # pair metrics are retained as small scalars, never recomputed by
            # reloading an evicted RGB-D frame.
            target_frame = frames.get(target.frame_id)
            proposal_box = _projected_obb_box(corners_world, target_frame)
            metrics = project_guide_metrics(
                source.guide_points_world,
                target_frame.depth_m,
                target_frame.K,
                target_frame.T_wc,
                proposal_box_xyxy=proposal_box,
                alpha=DEPTH_ALPHA,
            )
            if metrics.v_b is None or metrics.d_b is None or metrics.affinity_a is None:
                raise ShadowError("depth utility omitted requested backward metrics")
            baseline, ray_span = _edge_view_geometry(
                source.camera_center, target.camera_center, object_center
            )
            independent = (
                baseline >= MIN_CAMERA_BASELINE_M
                and ray_span >= MIN_VIEW_RAY_SPAN_DEG
            )
            support = (
                independent
                and metrics.v_f > MIN_FORWARD_VISIBILITY
                and metrics.v_b > MIN_BACKWARD_CONTAINMENT
            )
            edges.append(
                {
                    "source_frame_id": source.frame_id,
                    "target_frame_id": target.frame_id,
                    "source_row": source.source_row,
                    "target_source_row": target.source_row,
                    "source_guide_points": len(source.guide_points_world),
                    "camera_baseline_m": baseline,
                    "view_ray_span_deg": ray_span,
                    "independent_view": independent,
                    "v_f": metrics.v_f,
                    "v_b": metrics.v_b,
                    "d_f_diagnostic_only": metrics.d_f,
                    "d_b_diagnostic_only": metrics.d_b,
                    "q_f_diagnostic_only": metrics.q_f,
                    "affinity_diagnostic_only": metrics.affinity_a,
                    "support": support,
                }
            )
    components = _components(nodes, edges)
    chosen = _choose_qualifying_component(components)
    graph = {
        "attempt_count": len(attempts),
        "attempts": [
            {
                "frame_id": row.frame_id,
                "source_row": row.source_row,
                "guide_points": 0 if row.node is None else len(row.node.guide_points_world),
                "valid_node": row.node is not None,
                "failure_reason": row.failure_reason,
            }
            for row in attempts
        ],
        "node_count": len(nodes),
        "node_frame_ids": [row.frame_id for row in nodes],
        "edges": edges,
        "support_edge_count_total": sum(int(row["support"]) for row in edges),
        "weak_components": [
            {
                "frame_ids": row["frame_ids"],
                "support_edge_count": row["support_edge_count"],
            }
            for row in components
        ],
        "passes": chosen is not None,
        "qualifying_component": chosen,
    }
    return graph, tuple(edges)


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_min, left_max = left.min(axis=0), left.max(axis=0)
    right_min, right_max = right.min(axis=0), right.max(axis=0)
    intersection = float(
        np.prod(np.maximum(np.minimum(left_max, right_max) - np.maximum(left_min, right_min), 0.0))
    )
    left_volume = float(np.prod(left_max - left_min))
    right_volume = float(np.prod(right_max - right_min))
    union = left_volume + right_volume - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _terminal_filter(
    qualified: Sequence[dict[str, Any]],
    native_corners: np.ndarray,
    native_scores: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    novelty_eligible = []
    native_rejected = []
    for row in qualified:
        max_native_iou = max(
            (_aabb_iou(row["corners_world_array"], box) for box in native_corners),
            default=0.0,
        )
        row = dict(row)
        row["max_terminal_native_aabb_iou"] = max_native_iou
        if max_native_iou >= NATIVE_NOVELTY_IOU:
            native_rejected.append(int(row["track_id"]))
        else:
            novelty_eligible.append(row)
    ranked = sorted(
        novelty_eligible,
        key=lambda row: (
            -float(row["receipt"]["raw_mean_score"]),
            -int(row["qualification"]["node_count"]),
            int(row["track_id"]),
        ),
    )
    kept: list[dict[str, Any]] = []
    nms_rejected = []
    for row in ranked:
        if any(
            _aabb_iou(row["corners_world_array"], other["corners_world_array"])
            >= SELF_NMS_IOU
            for other in kept
        ):
            nms_rejected.append(int(row["track_id"]))
        else:
            kept.append(row)
    cap_rejected = [int(row["track_id"]) for row in kept[MAX_OUTPUTS_PER_SCENE:]]
    outputs = kept[:MAX_OUTPUTS_PER_SCENE]
    minimum_native_score = float(np.min(native_scores)) if len(native_scores) else None
    for row in outputs:
        raw_mean = float(row["receipt"]["raw_mean_score"])
        cap = raw_mean if minimum_native_score is None else float(np.nextafter(minimum_native_score, 0.0))
        row["appended_score_diagnostic_only"] = min(raw_mean, cap)
    return outputs, {
        "native_overlap_rejected_track_ids": native_rejected,
        "self_nms_rejected_track_ids": nms_rejected,
        "output_cap_rejected_track_ids": cap_rejected,
    }


def materialize_boxer_past3_depth_shadow(
    *,
    input_json: Path,
    input_npz: Path,
    baseline_root: Path,
    schedule_root: Path,
    scene_rgbd_root: Path,
    preregistration: Path,
    output_json: Path,
    output_npz: Path,
) -> dict[str, Any]:
    """Materialize one deterministic S1 shadow sidecar; GT is not an API input."""

    input_json = input_json.resolve()
    input_npz = input_npz.resolve()
    baseline_root = baseline_root.resolve()
    schedule_root = schedule_root.resolve()
    scene_rgbd_root = scene_rgbd_root.resolve()
    preregistration = preregistration.resolve()
    output_json = output_json.resolve()
    output_npz = output_npz.resolve()
    if output_json == output_npz or output_json.parent != output_npz.parent:
        raise ShadowError("output JSON and NPZ must be distinct files in one directory")
    if output_json.exists() or output_json.is_symlink():
        raise ShadowError(f"refusing to overwrite shadow JSON: {output_json}")
    if output_npz.exists() or output_npz.is_symlink():
        raise ShadowError(f"refusing to overwrite shadow NPZ: {output_npz}")
    if not baseline_root.is_dir() or not schedule_root.is_dir() or not scene_rgbd_root.is_dir():
        raise ShadowError("baseline, sealed schedule and ScanNet RGB-D roots must be directories")
    _regular_file(preregistration, "Boxer-Past3 S1 preregistration")

    input_manifest, scenes = _validate_input_manifest(input_json, input_npz)
    arrays = _validate_arrays(input_npz, scenes, int(input_manifest["per_view_candidate_count"]))
    receipt_source = _REPOSITORY_ROOT / "boxfusion" / "boxer_past3_receipt.py"
    depth_source = _REPOSITORY_ROOT / "tools" / "boxfusion_tr3d_pipeline" / "boxfusion" / "depth_guide_geometry.py"
    _regular_file(receipt_source, "Boxer-Past3 receipt source")
    _regular_file(depth_source, "depth-guide source")

    native_before: dict[str, str] = {}
    native_after: dict[str, str] = {}
    schedule_hashes: dict[str, str] = {}
    scene_reports: dict[str, Any] = {}
    accepted_rows: list[dict[str, Any]] = []
    schedule_namespace: Optional[str] = None
    schedule_schemas: set[str] = set()
    frame_times_ms: list[float] = []

    for scene_index, scene in enumerate(scenes):
        ledger = input_manifest["scenes"][scene_index]
        schedule, namespace, schedule_schema = _load_schedule(schedule_root, scene, ledger)
        schedule_schemas.add(schedule_schema)
        if schedule_namespace is None:
            schedule_namespace = namespace
        elif schedule_namespace != namespace:
            raise ShadowError("sealed T05 schedule namespace changes across scenes")
        schedule_hashes[scene] = _sha256(schedule_root / scene / "manifest.json")
        prediction_path = baseline_root / f"{scene}_boxes.pkl"
        native_before[scene] = _sha256(_regular_file(prediction_path, "T05 prediction"))
        native_corners, native_scores = _load_prediction(prediction_path)
        positions = np.flatnonzero(arrays["per_view_scene_index"] == scene_index)
        observed_frames = set(int(value) for value in arrays["per_view_frame_id"][positions])
        if not observed_frames.issubset(schedule):
            raise ShadowError(f"off-schedule Boxer candidate reached S1 tracker for {scene}")

        frame_store = _SceneFrameStore(scene_rgbd_root, scene, schedule)
        tracker = BoxerPast3ReceiptTracker()
        receipts: dict[int, BoxerPast3Receipt] = {}
        attempts: dict[int, list[_DepthAttempt]] = {}
        last_graph: dict[int, dict[str, Any]] = {}
        edge_cache: dict[int, tuple[dict[str, Any], ...]] = {}
        qualifications: dict[int, dict[str, Any]] = {}

        for frame_id in schedule:
            started = time.perf_counter()
            # Advance even when Boxer emitted no row.  This makes TTL, RGB-D
            # availability, and causal time share the exact sealed schedule.
            frame_store.advance(frame_id)
            frame_positions = positions[arrays["per_view_frame_id"][positions] == frame_id]
            observations = [
                BoxerObservation(
                    frame_id=frame_id,
                    source_row=int(arrays["per_view_source_row"][row]),
                    score=float(arrays["per_view_source_score"][row]),
                    corners=_obb_corners(
                        arrays["per_view_center_world"][row],
                        arrays["per_view_extent_xyz"][row],
                        arrays["per_view_quaternion_wxyz"][row],
                    ),
                )
                for row in frame_positions
            ]
            query = tracker.query(frame_id, observations)
            commit = tracker.commit(query)
            if not commit.audit_complete:
                raise ShadowError(f"bounded receipt tracker capacity was exceeded for {scene}")

            for receipt in commit.newly_frozen_receipts:
                track_id = int(receipt.track_id)
                if track_id in receipts:
                    raise ShadowError("one track froze more than one receipt")
                receipts[track_id] = receipt
                rows = [
                    _make_attempt(
                        frame_id=evidence_frame,
                        source_row=evidence_row,
                        corners_world=receipt.corners,
                        frames=frame_store,
                    )
                    for evidence_frame, evidence_row in zip(
                        receipt.evidence_frame_ids, receipt.evidence_source_rows
                    )
                ]
                attempts[track_id] = rows[-MAX_EVIDENCE_ATTEMPTS:]
                graph, edge_cache[track_id] = _evaluate_graph(
                    attempts[track_id], receipt.corners, frame_store
                )
                last_graph[track_id] = graph
                if graph["passes"]:
                    qualifications[track_id] = {
                        "qualification_frame_id": frame_id,
                        **graph,
                    }

            # Once a receipt exists, later matched observations may extend the
            # bounded evidence window.  The confirmation-frame row is already
            # in the immutable receipt, so it is not inserted twice.
            for assignment in commit.assignments:
                track_id = int(assignment.track_id)
                if assignment.action != "matched":
                    continue
                if track_id not in receipts or track_id in qualifications:
                    continue
                if any(row.frame_id == frame_id for row in attempts[track_id]):
                    continue
                receipt = receipts[track_id]
                attempts[track_id].append(
                    _make_attempt(
                        frame_id=frame_id,
                        source_row=int(assignment.source_row),
                        corners_world=receipt.corners,
                        frames=frame_store,
                    )
                )
                attempts[track_id] = attempts[track_id][-MAX_EVIDENCE_ATTEMPTS:]
                graph, edge_cache[track_id] = _evaluate_graph(
                    attempts[track_id],
                    receipt.corners,
                    frame_store,
                    edge_cache.get(track_id, ()),
                )
                last_graph[track_id] = graph
                if graph["passes"]:
                    qualifications[track_id] = {
                        "qualification_frame_id": frame_id,
                        **graph,
                    }
            frame_times_ms.append((time.perf_counter() - started) * 1000.0)

        summary = tracker.summary()
        if summary.get("schema") != RECEIPT_SCHEMA or summary.get("gt_access") is not False:
            raise ShadowError("receipt tracker violated its no-GT contract")
        if not summary.get("audit_complete"):
            raise ShadowError(f"incomplete bounded receipt tracker audit for {scene}")

        qualified_rows = [
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "track_id": track_id,
                "corners_world_array": receipt.corners,
                "receipt": receipt.to_json_dict(),
                "qualification": qualifications[track_id],
            }
            for track_id, receipt in sorted(receipts.items())
            if track_id in qualifications
        ]
        outputs, terminal = _terminal_filter(qualified_rows, native_corners, native_scores)
        accepted_scene = []
        for row in outputs:
            public = {
                "scene_id": scene,
                "scene_index": scene_index,
                "track_id": int(row["track_id"]),
                "confirmation_frame_id": int(row["receipt"]["confirmation_frame_id"]),
                "qualification_frame_id": int(row["qualification"]["qualification_frame_id"]),
                "receipt_evidence_frame_ids": list(row["receipt"]["evidence_frame_ids"]),
                "receipt_evidence_source_rows": list(row["receipt"]["evidence_source_rows"]),
                "raw_mean_score": float(row["receipt"]["raw_mean_score"]),
                "appended_score_diagnostic_only": float(row["appended_score_diagnostic_only"]),
                "median_pairwise_aabb_iou": float(row["receipt"]["median_pairwise_aabb_iou"]),
                "center_rms_m": float(row["receipt"]["center_rms_m"]),
                "max_terminal_native_aabb_iou": float(row["max_terminal_native_aabb_iou"]),
                "corners_world": row["corners_world_array"].tolist(),
                "depth_qualification": row["qualification"],
            }
            accepted_scene.append(public)
            accepted_rows.append(public)

        receipt_diagnostics = {}
        for track_id, receipt in sorted(receipts.items()):
            receipt_diagnostics[str(track_id)] = {
                "receipt": receipt.to_json_dict(),
                "depth_qualified": track_id in qualifications,
                "qualification": qualifications.get(track_id),
                "last_bounded_graph": last_graph.get(track_id),
            }
        scene_reports[scene] = {
            "raw_per_view_candidates": int(len(positions)),
            "processed_keyframes": len(schedule),
            "nonempty_candidate_keyframes": len(observed_frames),
            "zero_candidate_keyframes": len(schedule) - len(observed_frames),
            "native_terminal_predictions": len(native_corners),
            "frozen_s0_receipts": len(receipts),
            "depth_qualified_before_terminal": len(qualified_rows),
            "terminal_accepted_candidates": len(accepted_scene),
            "terminal_rejections": terminal,
            "accepted_candidates": accepted_scene,
            "receipt_diagnostics": receipt_diagnostics,
            "tracker_summary": summary,
            "depth_frame_store": {
                "max_cached_frames": MAX_DEPTH_FRAME_CACHE,
                "peak_cached_frames": frame_store.peak_cached_frames,
                "cache_hits": frame_store.cache_hits,
                "frames_advanced": frame_store.frames_advanced,
                "zero_candidate_frames_advanced": len(schedule) - len(observed_frames),
                "arbitrary_historical_reload": False,
                "future_access_allowed": False,
            },
            "intrinsic_sha256": frame_store.intrinsic_sha256,
            "pose_sha256": dict(sorted(frame_store.pose_sha256.items(), key=lambda item: int(item[0]))),
            "depth_sha256": dict(sorted(frame_store.depth_sha256.items(), key=lambda item: int(item[0]))),
        }

    native_after = {scene: _sha256(baseline_root / f"{scene}_boxes.pkl") for scene in scenes}
    if native_after != native_before:
        raise ShadowError("native T05 predictions changed during S1 shadow materialization")

    receipt_offsets = [0]
    receipt_frames: list[int] = []
    receipt_rows: list[int] = []
    node_offsets = [0]
    node_frames: list[int] = []
    node_rows: list[int] = []
    node_guide_counts: list[int] = []
    edge_offsets = [0]
    edge_source_frames: list[int] = []
    edge_target_frames: list[int] = []
    edge_vf: list[float] = []
    edge_vb: list[float] = []
    edge_baseline: list[float] = []
    edge_ray_span: list[float] = []
    for row in accepted_rows:
        receipt_frames.extend(row["receipt_evidence_frame_ids"])
        receipt_rows.extend(row["receipt_evidence_source_rows"])
        receipt_offsets.append(len(receipt_frames))
        qualification = row["depth_qualification"]
        chosen_frames = set(qualification["qualifying_component"]["frame_ids"])
        chosen_attempts = [
            attempt for attempt in qualification["attempts"]
            if attempt["valid_node"] and attempt["frame_id"] in chosen_frames
        ]
        node_frames.extend(int(item["frame_id"]) for item in chosen_attempts)
        node_rows.extend(int(item["source_row"]) for item in chosen_attempts)
        node_guide_counts.extend(int(item["guide_points"]) for item in chosen_attempts)
        node_offsets.append(len(node_frames))
        chosen_edges = qualification["qualifying_component"]["support_edges"]
        edge_source_frames.extend(int(item["source_frame_id"]) for item in chosen_edges)
        edge_target_frames.extend(int(item["target_frame_id"]) for item in chosen_edges)
        edge_vf.extend(float(item["v_f"]) for item in chosen_edges)
        edge_vb.extend(float(item["v_b"]) for item in chosen_edges)
        edge_baseline.extend(float(item["camera_baseline_m"]) for item in chosen_edges)
        edge_ray_span.extend(float(item["view_ray_span_deg"]) for item in chosen_edges)
        edge_offsets.append(len(edge_source_frames))

    candidate_arrays: dict[str, np.ndarray] = {
        "scene_ids": np.asarray(scenes, dtype="<U12"),
        "candidate_scene_index": np.asarray([row["scene_index"] for row in accepted_rows], dtype=np.int16),
        "candidate_track_id": np.asarray([row["track_id"] for row in accepted_rows], dtype=np.int32),
        "candidate_confirmation_frame_id": np.asarray([row["confirmation_frame_id"] for row in accepted_rows], dtype=np.int64),
        "candidate_qualification_frame_id": np.asarray([row["qualification_frame_id"] for row in accepted_rows], dtype=np.int64),
        "candidate_corners_world": np.asarray([row["corners_world"] for row in accepted_rows], dtype=np.float32).reshape((-1, 8, 3)),
        "candidate_raw_mean_score": np.asarray([row["raw_mean_score"] for row in accepted_rows], dtype=np.float32),
        "candidate_appended_score_diagnostic_only": np.asarray([row["appended_score_diagnostic_only"] for row in accepted_rows], dtype=np.float32),
        "candidate_median_pairwise_aabb_iou": np.asarray([row["median_pairwise_aabb_iou"] for row in accepted_rows], dtype=np.float32),
        "candidate_center_rms_m": np.asarray([row["center_rms_m"] for row in accepted_rows], dtype=np.float32),
        "candidate_max_terminal_native_aabb_iou": np.asarray([row["max_terminal_native_aabb_iou"] for row in accepted_rows], dtype=np.float32),
        "candidate_receipt_evidence_offsets": np.asarray(receipt_offsets, dtype=np.int32),
        "receipt_evidence_frame_id": np.asarray(receipt_frames, dtype=np.int64),
        "receipt_evidence_source_row": np.asarray(receipt_rows, dtype=np.int32),
        "candidate_depth_node_offsets": np.asarray(node_offsets, dtype=np.int32),
        "depth_node_frame_id": np.asarray(node_frames, dtype=np.int64),
        "depth_node_source_row": np.asarray(node_rows, dtype=np.int32),
        "depth_node_guide_count": np.asarray(node_guide_counts, dtype=np.int16),
        "candidate_support_edge_offsets": np.asarray(edge_offsets, dtype=np.int32),
        "support_edge_source_frame_id": np.asarray(edge_source_frames, dtype=np.int64),
        "support_edge_target_frame_id": np.asarray(edge_target_frames, dtype=np.int64),
        "support_edge_vf": np.asarray(edge_vf, dtype=np.float32),
        "support_edge_vb": np.asarray(edge_vb, dtype=np.float32),
        "support_edge_camera_baseline_m": np.asarray(edge_baseline, dtype=np.float32),
        "support_edge_view_ray_span_deg": np.asarray(edge_ray_span, dtype=np.float32),
    }
    for value in candidate_arrays.values():
        value.setflags(write=False)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(output_npz, candidate_arrays)
    timing = np.asarray(frame_times_ms, dtype=np.float64)
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "native_mutation_applied": False,
        "gt_access": False,
        "training_free": True,
        "online_learning": False,
        "past_only": True,
        "future_frames_used": False,
        "receipt_geometry_frozen": True,
        "receipt_provenance_frozen": True,
        "later_evidence_changes_receipt": False,
        "detector_semantics_used": False,
        "native_clip_access": False,
        "native_clip_unchanged": True,
        "score_mode_for_formal_evaluation": "constant_1.0",
        "coordinate_frame": "scannet_world",
        "scene_count": len(scenes),
        "candidate_count": len(accepted_rows),
        "npz_file": output_npz.name,
        "npz_sha256": _sha256(output_npz),
        "candidate_content_sha256": _array_content_sha256(candidate_arrays),
        "input": {
            "candidate_json": os.fspath(input_json),
            "candidate_json_sha256": _sha256(input_json),
            "candidate_npz": os.fspath(input_npz),
            "candidate_npz_sha256": _sha256(input_npz),
            "candidate_schema": INPUT_SCHEMA,
            "preregistration": os.fspath(preregistration),
            "preregistration_sha256": _sha256(preregistration),
            "receipt_source": os.fspath(receipt_source),
            "receipt_source_sha256": _sha256(receipt_source),
            "depth_source": os.fspath(depth_source),
            "depth_source_sha256": _sha256(depth_source),
            "baseline_root": os.fspath(baseline_root),
            "schedule_root": os.fspath(schedule_root),
            "schedule_sha256": schedule_hashes,
            "schedule_namespace": schedule_namespace,
            "schedule_schemas": sorted(schedule_schemas),
            "scene_rgbd_root": os.fspath(scene_rgbd_root),
        },
        "frozen_policy": {
            "receipt_schema": RECEIPT_SCHEMA,
            "max_evidence_attempts": MAX_EVIDENCE_ATTEMPTS,
            "depth_guide_min_points_per_node": MIN_GUIDE_POINTS,
            "depth_alpha": DEPTH_ALPHA,
            "edge_min_camera_baseline_m_inclusive": MIN_CAMERA_BASELINE_M,
            "edge_min_view_ray_span_deg_inclusive": MIN_VIEW_RAY_SPAN_DEG,
            "edge_min_vf_strict": MIN_FORWARD_VISIBILITY,
            "edge_min_vb_strict": MIN_BACKWARD_CONTAINMENT,
            "df_db_affinity_are_diagnostic_only": True,
            "qualification_same_weak_component": True,
            "qualification_min_distinct_frames": MIN_COMPONENT_NODES,
            "qualification_min_support_edges": MIN_COMPONENT_SUPPORT_EDGES,
            "terminal_native_novelty_iou": NATIVE_NOVELTY_IOU,
            "terminal_self_nms_iou": SELF_NMS_IOU,
            "max_outputs_per_scene": MAX_OUTPUTS_PER_SCENE,
            "depth_scale_m_per_integer_unit": DEPTH_SCALE_METERS_PER_UNIT,
            "max_cached_depth_frames": MAX_DEPTH_FRAME_CACHE,
        },
        "timing_excludes_frozen_boxer_inference": True,
        "materializer_keyframe_time_mean_ms": None if not len(timing) else float(timing.mean()),
        "materializer_keyframe_time_p95_ms": None if not len(timing) else float(np.percentile(timing, 95)),
        "materializer_keyframe_time_max_ms": None if not len(timing) else float(timing.max()),
        "native_prediction_sha256_before": native_before,
        "native_prediction_sha256_after": native_after,
        "native_before_after_identity": native_before == native_after,
        "scenes": scene_reports,
    }
    try:
        _write_json_exclusive(output_json, manifest)
    except Exception:
        output_npz.unlink(missing_ok=True)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal no-GT Boxer-Past3 S1 depth-qualified shadow receipts")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--schedule-root", required=True, type=Path)
    parser.add_argument("--scene-rgbd-root", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = materialize_boxer_past3_depth_shadow(
        input_json=args.input_json,
        input_npz=args.input_npz,
        baseline_root=args.baseline_root,
        schedule_root=args.schedule_root,
        scene_rgbd_root=args.scene_rgbd_root,
        preregistration=args.preregistration,
        output_json=args.out_json,
        output_npz=args.out_npz,
    )
    print(json.dumps({"schema": SCHEMA, "scene_count": manifest["scene_count"], "candidate_count": manifest["candidate_count"], "out_json": os.fspath(args.out_json), "out_npz": os.fspath(args.out_npz)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
