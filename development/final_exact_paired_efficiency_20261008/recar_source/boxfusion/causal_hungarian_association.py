"""Causal two-stage association for static RGB-D scenes.

The released BoxFusion association uses score-ordered 3D NMS followed by a
greedy projected-2D fallback.  This module keeps the native track rows but
replaces those two decisions with a gated one-to-one Hungarian assignment.
Only observations from the current keyframe are matched to already-existing
tracks, so the implementation is online and past-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint


_DEFAULTS = {
    "enabled": False,
    "high_score_threshold": 0.60,
    "min_confirmed_views_for_low": 2,
    "high_match_threshold": 0.33,
    "low_match_threshold": 0.39,
    "min_3d_iou": 0.035,
    "min_2d_iou": 0.12,
    "max_center_distance_m": 0.70,
    "max_normalized_center_distance": 1.10,
    "max_depth_residual_m": 0.65,
    "max_size_ratio": 2.50,
    "same_frame_3d_iou": 0.35,
    "same_frame_2d_iou": 0.85,
    "weight_3d_iou": 0.40,
    "weight_2d_iou": 0.20,
    "weight_center": 0.18,
    "weight_depth": 0.10,
    "weight_appearance": 0.12,
    "center_similarity_sigma": 0.55,
    "depth_similarity_sigma_m": 0.35,
    "appearance_floor": 0.20,
    "update_track_score": "max",
}


def resolve_causal_hungarian_config(cfg: Mapping) -> dict:
    section = cfg.get("association", {}).get("causal_hungarian", {})
    if section is None:
        section = {}
    if not hasattr(section, "get"):
        raise ValueError("association.causal_hungarian must be a mapping")
    result = dict(_DEFAULTS)
    result.update(dict(section))
    result["enabled"] = bool(result["enabled"])
    result["min_confirmed_views_for_low"] = int(
        result["min_confirmed_views_for_low"]
    )
    for key in (
        "high_score_threshold",
        "high_match_threshold",
        "low_match_threshold",
        "min_3d_iou",
        "min_2d_iou",
        "max_center_distance_m",
        "max_normalized_center_distance",
        "max_depth_residual_m",
        "max_size_ratio",
        "same_frame_3d_iou",
        "same_frame_2d_iou",
        "weight_3d_iou",
        "weight_2d_iou",
        "weight_center",
        "weight_depth",
        "weight_appearance",
        "center_similarity_sigma",
        "depth_similarity_sigma_m",
        "appearance_floor",
    ):
        result[key] = float(result[key])
    if not 0.0 <= result["high_score_threshold"] <= 1.0:
        raise ValueError("high_score_threshold must be in [0,1]")
    if result["min_confirmed_views_for_low"] < 1:
        raise ValueError("min_confirmed_views_for_low must be positive")
    if result["max_size_ratio"] <= 1.0:
        raise ValueError("max_size_ratio must exceed 1")
    weights = sum(
        result[key]
        for key in (
            "weight_3d_iou",
            "weight_2d_iou",
            "weight_center",
            "weight_depth",
            "weight_appearance",
        )
    )
    if not np.isclose(weights, 1.0, atol=1.0e-6):
        raise ValueError(f"causal Hungarian weights must sum to 1, got {weights}")
    if result["update_track_score"] not in ("keep", "max"):
        raise ValueError("update_track_score must be keep or max")
    return result


def _as_numpy(value, dtype=np.float64):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1.0e-8)
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1.0e-8)
    return np.clip(left @ right.T, -1.0, 1.0)


def _iou_2d_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    lt = np.maximum(left[:, None, :2], right[None, :, :2])
    rb = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    left_area = np.prod(np.maximum(left[:, 2:] - left[:, :2], 0.0), axis=1)
    right_area = np.prod(
        np.maximum(right[:, 2:] - right[:, :2], 0.0), axis=1
    )
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / np.maximum(union, 1.0e-8)


def _project_boxes(
    corners_world: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if corners_world.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)
    ones = np.ones((*corners_world.shape[:2], 1), dtype=np.float64)
    homogeneous = np.concatenate((corners_world, ones), axis=2)
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = homogeneous @ world_to_camera.T
    z = camera[..., 2]
    safe_z = np.where(z > 1.0e-4, z, 1.0)
    u = intrinsics[0, 0] * camera[..., 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[..., 1] / safe_z + intrinsics[1, 2]
    valid = z > 1.0e-4
    result = np.zeros((corners_world.shape[0], 4), dtype=np.float64)
    for index in range(corners_world.shape[0]):
        if not np.any(valid[index]):
            continue
        uu = np.clip(u[index, valid[index]], 0.0, float(width))
        vv = np.clip(v[index, valid[index]], 0.0, float(height))
        result[index] = (uu.min(), vv.min(), uu.max(), vv.max())
    return result


def _footprints(corners: np.ndarray):
    polygons = []
    z_min = corners[:, :, 2].min(axis=1)
    z_max = corners[:, :, 2].max(axis=1)
    volumes = np.zeros(corners.shape[0], dtype=np.float64)
    for index, box in enumerate(corners):
        polygon = MultiPoint(box[:, :2]).convex_hull
        polygons.append(polygon)
        volumes[index] = max(float(polygon.area), 0.0) * max(
            float(z_max[index] - z_min[index]), 0.0
        )
    return polygons, z_min, z_max, volumes


def _gravity_iou_3d_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return result
    left_poly, left_min, left_max, left_volume = _footprints(left)
    right_poly, right_min, right_max, right_volume = _footprints(right)
    for row in range(left.shape[0]):
        for col in range(right.shape[0]):
            height = min(left_max[row], right_max[col]) - max(
                left_min[row], right_min[col]
            )
            if height <= 0.0 or not left_poly[row].intersects(right_poly[col]):
                continue
            intersection = left_poly[row].intersection(right_poly[col])
            intersection_volume = max(float(intersection.area), 0.0) * height
            union = left_volume[row] + right_volume[col] - intersection_volume
            if union > 1.0e-8:
                result[row, col] = intersection_volume / union
    return result


def _hungarian_stage(
    detection_indices: Sequence[int],
    track_indices: Sequence[int],
    similarity: np.ndarray,
    valid: np.ndarray,
    threshold: float,
):
    detection_indices = np.asarray(detection_indices, dtype=np.int64)
    track_indices = np.asarray(track_indices, dtype=np.int64)
    if detection_indices.size == 0 or track_indices.size == 0:
        return []
    local_similarity = similarity[np.ix_(detection_indices, track_indices)]
    local_valid = valid[np.ix_(detection_indices, track_indices)]
    cost = 1.0 - local_similarity
    cost[~local_valid] = 1.0e6
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        det = int(detection_indices[row])
        track = int(track_indices[col])
        if local_valid[row, col] and local_similarity[row, col] >= threshold:
            matches.append((det, track, float(local_similarity[row, col])))
    return matches


@dataclass
class AssociationResult:
    keep_indices: np.ndarray
    correspondence_skip_indices: np.ndarray


class CausalHungarianAssociator:
    def __init__(self, cfg: Mapping):
        self.cfg = resolve_causal_hungarian_config(cfg)
        self.enabled = self.cfg["enabled"]
        self.stats = {
            "keyframes": 0,
            "pairs": 0,
            "gated_pairs": 0,
            "high_detections": 0,
            "low_detections": 0,
            "high_matches": 0,
            "low_matches": 0,
            "high_births": 0,
            "low_dropped": 0,
            "same_frame_suppressed": 0,
        }

    @property
    def needs_appearance(self) -> bool:
        return self.enabled and self.cfg["weight_appearance"] > 0.0

    def _same_frame_keep(
        self,
        detection_indices: Sequence[int],
        scores: np.ndarray,
        iou3d: np.ndarray,
        iou2d: np.ndarray,
    ) -> list[int]:
        ordered = sorted(detection_indices, key=lambda index: (-scores[index], index))
        kept = []
        for detection in ordered:
            duplicate = any(
                iou3d[detection, prior] >= self.cfg["same_frame_3d_iou"]
                or iou2d[detection, prior] >= self.cfg["same_frame_2d_iou"]
                for prior in kept
            )
            if duplicate:
                self.stats["same_frame_suppressed"] += 1
            else:
                kept.append(detection)
        return sorted(kept)

    def associate(self, instance_lists, box_manager, cam_poses) -> AssociationResult:
        if not self.enabled:
            raise RuntimeError("causal Hungarian association is disabled")
        frame_ids = _as_numpy(instance_lists.frame_id, np.int64)
        current_frame = int(frame_ids.max())
        current_absolute = np.flatnonzero(frame_ids == current_frame)
        global_absolute = np.flatnonzero(frame_ids != current_frame)
        if current_absolute.size == 0 or global_absolute.size == 0:
            keep = np.arange(len(instance_lists), dtype=np.int64)
            return AssociationResult(keep, current_absolute)

        if not np.array_equal(
            current_absolute,
            np.arange(current_absolute[0], len(instance_lists)),
        ):
            raise RuntimeError("current observations must be the final contiguous rows")

        self.stats["keyframes"] += 1
        boxes = instance_lists.pred_boxes_3d
        corners = _as_numpy(boxes.corners)
        dims = np.maximum(_as_numpy(boxes.dims), 1.0e-3)
        centers = corners.mean(axis=1)
        scores_all = _as_numpy(instance_lists.scores)
        current_scores = scores_all[current_absolute]
        current_corners = corners[current_absolute]
        global_corners = corners[global_absolute]
        current_dims = dims[current_absolute]
        global_dims = dims[global_absolute]
        current_centers = centers[current_absolute]
        global_centers = centers[global_absolute]

        iou3d = _gravity_iou_3d_matrix(current_corners, global_corners)
        camera_pose = _as_numpy(instance_lists.cam_pose[current_absolute[0]])
        cam_cfg = box_manager.cfg["cam"]
        intrinsics = np.asarray(
            [
                [cam_cfg["fx"], 0.0, cam_cfg["cx"]],
                [0.0, cam_cfg["fy"], cam_cfg["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        height, width = int(cam_cfg["H"]), int(cam_cfg["W"])
        global_2d = _project_boxes(
            global_corners, intrinsics, camera_pose, height, width
        )
        current_2d = _as_numpy(instance_lists.pred_boxes[current_absolute])
        iou2d = _iou_2d_matrix(current_2d, global_2d)

        center_distance = np.linalg.norm(
            current_centers[:, None, :] - global_centers[None, :, :], axis=2
        )
        diagonals = 0.5 * (
            np.linalg.norm(current_dims, axis=1)[:, None]
            + np.linalg.norm(global_dims, axis=1)[None, :]
        )
        normalized_center = center_distance / np.maximum(diagonals, 1.0e-3)
        world_to_camera = np.linalg.inv(camera_pose)
        current_camera = np.concatenate(
            (current_centers, np.ones((len(current_centers), 1))), axis=1
        ) @ world_to_camera.T
        global_camera = np.concatenate(
            (global_centers, np.ones((len(global_centers), 1))), axis=1
        ) @ world_to_camera.T
        depth_residual = np.abs(
            current_camera[:, None, 2] - global_camera[None, :, 2]
        )
        size_ratio = np.maximum(
            current_dims[:, None, :] / global_dims[None, :, :],
            global_dims[None, :, :] / current_dims[:, None, :],
        ).max(axis=2)

        appearance = np.full_like(iou3d, 0.5)
        if self.needs_appearance:
            if not instance_lists.has("appearance_features"):
                raise RuntimeError("Hungarian appearance evidence is missing")
            features = _as_numpy(instance_lists.appearance_features)
            appearance_cosine = _cosine_matrix(
                features[current_absolute], features[global_absolute]
            )
            appearance = np.clip(
                (appearance_cosine - self.cfg["appearance_floor"])
                / max(1.0 - self.cfg["appearance_floor"], 1.0e-6),
                0.0,
                1.0,
            )

        valid = (
            (center_distance <= self.cfg["max_center_distance_m"])
            & (
                normalized_center
                <= self.cfg["max_normalized_center_distance"]
            )
            & (depth_residual <= self.cfg["max_depth_residual_m"])
            & (size_ratio <= self.cfg["max_size_ratio"])
            & (
                (iou3d >= self.cfg["min_3d_iou"])
                | (iou2d >= self.cfg["min_2d_iou"])
            )
        )
        center_similarity = np.exp(
            -0.5
            * (normalized_center / self.cfg["center_similarity_sigma"]) ** 2
        )
        depth_similarity = np.exp(
            -0.5
            * (depth_residual / self.cfg["depth_similarity_sigma_m"]) ** 2
        )
        similarity = (
            self.cfg["weight_3d_iou"] * iou3d
            + self.cfg["weight_2d_iou"] * iou2d
            + self.cfg["weight_center"] * center_similarity
            + self.cfg["weight_depth"] * depth_similarity
            + self.cfg["weight_appearance"] * appearance
        )
        self.stats["pairs"] += int(valid.size)
        self.stats["gated_pairs"] += int(np.count_nonzero(valid))

        local_indices = np.arange(current_absolute.size, dtype=np.int64)
        high = local_indices[current_scores >= self.cfg["high_score_threshold"]]
        low = local_indices[current_scores < self.cfg["high_score_threshold"]]
        self.stats["high_detections"] += int(high.size)
        self.stats["low_detections"] += int(low.size)
        tracks = np.arange(global_absolute.size, dtype=np.int64)
        high_matches = _hungarian_stage(
            high,
            tracks,
            similarity,
            valid,
            self.cfg["high_match_threshold"],
        )
        matched_high = {item[0] for item in high_matches}
        matched_tracks = {item[1] for item in high_matches}

        eligible_low_tracks = [
            track
            for track in tracks.tolist()
            if track not in matched_tracks
            and len(box_manager.fusion_list[int(global_absolute[track])])
            >= self.cfg["min_confirmed_views_for_low"]
        ]
        low_matches = _hungarian_stage(
            low,
            eligible_low_tracks,
            similarity,
            valid,
            self.cfg["low_match_threshold"],
        )
        matched_low = {item[0] for item in low_matches}
        matches = high_matches + low_matches
        self.stats["high_matches"] += len(high_matches)
        self.stats["low_matches"] += len(low_matches)

        keep = global_absolute.tolist()
        init_id = _as_numpy(instance_lists.init_id, np.int64)
        box_size = _as_numpy(boxes.dims)
        for detection, track, match_score in matches:
            absolute_detection = int(current_absolute[detection])
            absolute_track = int(global_absolute[track])
            instance_lists.valid_num[absolute_track] += 1
            box_manager.record(
                absolute_track,
                [absolute_detection],
                init_id.tolist(),
                cam_poses,
                box_size,
                keep,
                centers,
            )
            if self.cfg["update_track_score"] == "max":
                instance_lists.scores[absolute_track] = torch.maximum(
                    instance_lists.scores[absolute_track],
                    instance_lists.scores[absolute_detection],
                )
            print(
                "hungarian",
                absolute_detection,
                "->",
                absolute_track,
                f"score={match_score:.4f}",
                "stage=low" if detection in matched_low else "stage=high",
            )

        unmatched_high = [item for item in high.tolist() if item not in matched_high]
        current_iou3d = _gravity_iou_3d_matrix(current_corners, current_corners)
        current_iou2d = _iou_2d_matrix(current_2d, current_2d)
        high_births = self._same_frame_keep(
            unmatched_high, current_scores, current_iou3d, current_iou2d
        )
        keep.extend(int(current_absolute[index]) for index in high_births)
        self.stats["high_births"] += len(high_births)
        self.stats["low_dropped"] += int(low.size - len(matched_low))
        keep = np.asarray(sorted(set(keep)), dtype=np.int64)
        # Every retained current row is already a high-confidence birth.  Mark
        # it as handled so the legacy greedy 2D correspondence stage is inert.
        skip = np.asarray(
            [int(current_absolute[index]) for index in high_births],
            dtype=np.int64,
        )
        return AssociationResult(keep, skip)

    def summary(self) -> str:
        stats = self.stats
        matches = stats["high_matches"] + stats["low_matches"]
        return (
            "Causal Hungarian association summary | "
            f"keyframes={stats['keyframes']}, pairs={stats['pairs']}, "
            f"gated_pairs={stats['gated_pairs']}, "
            f"high={stats['high_detections']}, low={stats['low_detections']}, "
            f"matches={matches} "
            f"(high={stats['high_matches']}, low={stats['low_matches']}), "
            f"births={stats['high_births']}, "
            f"low_dropped={stats['low_dropped']}, "
            f"same_frame_suppressed={stats['same_frame_suppressed']}"
        )
