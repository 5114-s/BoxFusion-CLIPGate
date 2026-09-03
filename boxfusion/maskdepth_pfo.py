"""Deterministic 7D mask/depth particle refinement with rollback.

The seven optimized variables are world center (3), log dimensions (3), and
a gravity-axis yaw delta (1).  The active candidate is accepted only when it
improves the same bounded mask/depth validation objective and passes explicit
motion, scale, yaw, and depth-support gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


DEFAULT_CONFIG = {
    "enabled": False,
    "iterations": 10,
    "particles": 192,
    "seed": 0,
    "min_valid_views": 3,
    "mask_weight": 0.65,
    "depth_weight": 0.25,
    "regularization_weight": 0.10,
    "center_sigma_start_m": 0.10,
    "center_sigma_end_m": 0.01,
    "log_size_sigma_start": 0.16,
    "log_size_sigma_end": 0.015,
    "yaw_sigma_start_deg": 12.0,
    "yaw_sigma_end_deg": 1.0,
    "depth_margin_m": 0.03,
    "min_depth_support": 0.45,
    "min_objective_improvement": 0.01,
    "max_center_shift_m": 0.20,
    "min_size_ratio": 0.70,
    "max_size_ratio": 1.40,
    "max_yaw_delta_deg": 20.0,
}


def resolve_maskdepth_pfo_config(box_fusion_cfg: Mapping) -> dict:
    raw = box_fusion_cfg.get("maskdepth_pfo", {})
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(raw or {})
    cfg["enabled"] = bool(cfg["enabled"])
    for key in ("iterations", "particles", "seed", "min_valid_views"):
        cfg[key] = int(cfg[key])
    for key in set(DEFAULT_CONFIG) - {
        "enabled",
        "iterations",
        "particles",
        "seed",
        "min_valid_views",
    }:
        cfg[key] = float(cfg[key])
    if cfg["iterations"] < 1 or cfg["particles"] < 8:
        raise ValueError("MaskDepth-PFO requires positive iterations and >=8 particles")
    if cfg["min_valid_views"] < 2:
        raise ValueError("MaskDepth-PFO requires at least two valid views")
    objective_weight = (
        cfg["mask_weight"]
        + cfg["depth_weight"]
        + cfg["regularization_weight"]
    )
    if not np.isclose(objective_weight, 1.0, atol=1.0e-6):
        raise ValueError("MaskDepth-PFO objective weights must sum to one")
    if not 0.0 <= cfg["min_depth_support"] <= 1.0:
        raise ValueError("MaskDepth-PFO min_depth_support must be in [0,1]")
    if not 0.0 < cfg["min_size_ratio"] <= 1.0 <= cfg["max_size_ratio"]:
        raise ValueError("MaskDepth-PFO size-ratio gates are invalid")
    return cfg


@dataclass(frozen=True)
class MaskDepthPFOResult:
    box_xyzlwh: np.ndarray
    rotation: np.ndarray
    accepted: bool
    reason: str
    baseline_loss: float
    candidate_loss: float
    baseline_mask_iou: float
    candidate_mask_iou: float
    baseline_depth_support: float
    candidate_depth_support: float


_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)


def _yaw_rotations(yaw: np.ndarray, base_rotation: np.ndarray) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    rotations_z = np.zeros((yaw.shape[0], 3, 3), dtype=np.float64)
    rotations_z[:, 0, 0] = cosine
    rotations_z[:, 0, 1] = -sine
    rotations_z[:, 1, 0] = sine
    rotations_z[:, 1, 1] = cosine
    rotations_z[:, 2, 2] = 1.0
    return rotations_z @ base_rotation[None]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = max(float(weights.sum()), 1.0e-12)
    return (values * weights[None]).sum(axis=1) / denominator


def _evaluate(
    states: np.ndarray,
    base_box: np.ndarray,
    base_rotation: np.ndarray,
    camera_poses: np.ndarray,
    tight_boxes: np.ndarray,
    points_world: np.ndarray,
    point_valid: np.ndarray,
    view_weights: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    cfg: Mapping,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = states[:, :3]
    dimensions = np.exp(states[:, 3:6])
    rotations = _yaw_rotations(states[:, 6], base_rotation)
    local_corners = _SIGNS[None] * (dimensions[:, None] * 0.5)
    corners_world = (
        np.einsum("pkj,pij->pki", local_corners, rotations)
        + centers[:, None]
    )

    particle_count = states.shape[0]
    view_count = camera_poses.shape[0]
    mask_iou = np.zeros((particle_count, view_count), dtype=np.float64)
    depth_support = np.zeros_like(mask_iou)
    for view_index in range(view_count):
        pose = camera_poses[view_index]
        relative = corners_world - pose[:3, 3][None, None]
        camera = relative @ pose[:3, :3]
        z = camera[:, :, 2]
        projection_valid = np.all(z > 1.0e-3, axis=1)
        safe_z = np.maximum(z, 1.0e-3)
        u = camera[:, :, 0] * intrinsics[0, 0] / safe_z + intrinsics[0, 2]
        v = camera[:, :, 1] * intrinsics[1, 1] / safe_z + intrinsics[1, 2]
        projected = np.stack(
            (u.min(axis=1), v.min(axis=1), u.max(axis=1), v.max(axis=1)),
            axis=1,
        )
        projected[:, [0, 2]] = np.clip(projected[:, [0, 2]], 0, image_width - 1)
        projected[:, [1, 3]] = np.clip(projected[:, [1, 3]], 0, image_height - 1)
        target = tight_boxes[view_index]
        intersection_width = np.maximum(
            0.0, np.minimum(projected[:, 2], target[2]) - np.maximum(projected[:, 0], target[0])
        )
        intersection_height = np.maximum(
            0.0, np.minimum(projected[:, 3], target[3]) - np.maximum(projected[:, 1], target[1])
        )
        intersection = intersection_width * intersection_height
        projected_area = np.maximum(0.0, projected[:, 2] - projected[:, 0]) * np.maximum(
            0.0, projected[:, 3] - projected[:, 1]
        )
        target_area = max(float((target[2] - target[0]) * (target[3] - target[1])), 1.0)
        union = projected_area + target_area - intersection
        mask_iou[:, view_index] = np.where(
            projection_valid & (union > 0.0), intersection / np.maximum(union, 1.0e-9), 0.0
        )

        valid_points = points_world[view_index, point_valid[view_index]]
        if valid_points.shape[0]:
            delta = valid_points[None] - centers[:, None]
            local = np.einsum("pmj,pjk->pmk", delta, rotations)
            inside = np.all(
                np.abs(local)
                <= dimensions[:, None] * 0.5 + float(cfg["depth_margin_m"]),
                axis=2,
            )
            depth_support[:, view_index] = inside.mean(axis=1)

    mean_iou = _weighted_mean(mask_iou, view_weights)
    mean_support = _weighted_mean(depth_support, view_weights)
    center_regularizer = np.linalg.norm(centers - base_box[None, :3], axis=1) / max(
        float(np.linalg.norm(base_box[3:])), 0.10
    )
    size_regularizer = np.linalg.norm(
        states[:, 3:6] - np.log(base_box[None, 3:]), axis=1
    ) / np.sqrt(3.0)
    yaw_regularizer = np.abs(states[:, 6]) / np.deg2rad(
        max(float(cfg["max_yaw_delta_deg"]), 1.0)
    )
    regularizer = (center_regularizer + size_regularizer + yaw_regularizer) / 3.0
    loss = (
        float(cfg["mask_weight"]) * (1.0 - mean_iou)
        + float(cfg["depth_weight"]) * (1.0 - mean_support)
        + float(cfg["regularization_weight"]) * regularizer
    )
    return loss, mean_iou, mean_support


def optimize_maskdepth_pfo(
    initial_box_xyzlwh: np.ndarray,
    initial_rotation: np.ndarray,
    camera_poses: np.ndarray,
    tight_boxes_xyxy: np.ndarray,
    points_world: np.ndarray,
    point_valid: np.ndarray,
    view_weights: np.ndarray,
    intrinsics: np.ndarray,
    image_height: int,
    image_width: int,
    cfg: Mapping,
    object_seed: int = 0,
) -> MaskDepthPFOResult:
    """Refine one legacy fused box, or return it byte-for-byte on rollback."""

    box = np.asarray(initial_box_xyzlwh, dtype=np.float64).copy()
    rotation = np.asarray(initial_rotation, dtype=np.float64).copy()
    poses = np.asarray(camera_poses, dtype=np.float64)
    tight = np.asarray(tight_boxes_xyxy, dtype=np.float64)
    points = np.asarray(points_world, dtype=np.float64)
    valid_points = np.asarray(point_valid, dtype=bool)
    weights = np.asarray(view_weights, dtype=np.float64).reshape(-1)
    K = np.asarray(intrinsics, dtype=np.float64)[:3, :3]
    view_count = poses.shape[0]
    if (
        box.shape != (6,)
        or rotation.shape != (3, 3)
        or poses.shape != (view_count, 4, 4)
        or tight.shape != (view_count, 4)
        or points.ndim != 3
        or points.shape[:2] != valid_points.shape
        or points.shape[0] != view_count
        or weights.shape != (view_count,)
        or K.shape != (3, 3)
        or not np.isfinite(box).all()
        or np.any(box[3:] <= 0.0)
    ):
        raise ValueError("invalid MaskDepth-PFO input shapes or values")

    baseline_state = np.concatenate((box[:3], np.log(box[3:]), [0.0]))
    baseline_loss, baseline_iou, baseline_support = _evaluate(
        baseline_state[None],
        box,
        rotation,
        poses,
        tight,
        points,
        valid_points,
        weights,
        K,
        image_height,
        image_width,
        cfg,
    )
    best_state = baseline_state.copy()
    best_loss = float(baseline_loss[0])
    rng = np.random.default_rng(int(cfg["seed"]) + int(object_seed) * 1009)
    iterations = int(cfg["iterations"])
    particles = int(cfg["particles"])
    for iteration in range(iterations):
        fraction = iteration / max(iterations - 1, 1)
        center_sigma = np.geomspace(
            cfg["center_sigma_start_m"], cfg["center_sigma_end_m"], iterations
        )[iteration]
        size_sigma = np.geomspace(
            cfg["log_size_sigma_start"], cfg["log_size_sigma_end"], iterations
        )[iteration]
        yaw_sigma = np.deg2rad(
            np.geomspace(
                cfg["yaw_sigma_start_deg"], cfg["yaw_sigma_end_deg"], iterations
            )[iteration]
        )
        perturbation = rng.standard_normal((particles, 7))
        perturbation[:, :3] *= center_sigma
        perturbation[:, 3:6] *= size_sigma
        perturbation[:, 6] *= yaw_sigma
        candidates = best_state[None] + perturbation
        candidates[0] = best_state
        candidates[:, 6] = np.clip(
            candidates[:, 6],
            -np.deg2rad(cfg["max_yaw_delta_deg"]),
            np.deg2rad(cfg["max_yaw_delta_deg"]),
        )
        candidate_loss, _, _ = _evaluate(
            candidates,
            box,
            rotation,
            poses,
            tight,
            points,
            valid_points,
            weights,
            K,
            image_height,
            image_width,
            cfg,
        )
        winner = int(np.argmin(candidate_loss))
        if float(candidate_loss[winner]) < best_loss:
            best_loss = float(candidate_loss[winner])
            best_state = candidates[winner].copy()

    candidate_loss, candidate_iou, candidate_support = _evaluate(
        best_state[None],
        box,
        rotation,
        poses,
        tight,
        points,
        valid_points,
        weights,
        K,
        image_height,
        image_width,
        cfg,
    )
    candidate_box = np.concatenate((best_state[:3], np.exp(best_state[3:6])))
    candidate_rotation = _yaw_rotations(best_state[6:7], rotation)[0]
    size_ratio = candidate_box[3:] / box[3:]
    gates = [
        (
            float(candidate_loss[0])
            <= float(baseline_loss[0]) - cfg["min_objective_improvement"],
            "objective_not_improved",
        ),
        (
            float(candidate_support[0]) >= cfg["min_depth_support"],
            "insufficient_depth_support",
        ),
        (
            float(np.linalg.norm(candidate_box[:3] - box[:3]))
            <= cfg["max_center_shift_m"],
            "center_shift_gate",
        ),
        (
            bool(
                np.all(size_ratio >= cfg["min_size_ratio"])
                and np.all(size_ratio <= cfg["max_size_ratio"])
            ),
            "size_ratio_gate",
        ),
        (
            abs(float(np.rad2deg(best_state[6]))) <= cfg["max_yaw_delta_deg"],
            "yaw_gate",
        ),
    ]
    accepted = all(passed for passed, _ in gates)
    reason = "accepted" if accepted else next(reason for passed, reason in gates if not passed)
    output_box = candidate_box if accepted else box
    output_rotation = candidate_rotation if accepted else rotation
    return MaskDepthPFOResult(
        box_xyzlwh=output_box.astype(np.float32),
        rotation=output_rotation.astype(np.float32),
        accepted=accepted,
        reason=reason,
        baseline_loss=float(baseline_loss[0]),
        candidate_loss=float(candidate_loss[0]),
        baseline_mask_iou=float(baseline_iou[0]),
        candidate_mask_iou=float(candidate_iou[0]),
        baseline_depth_support=float(baseline_support[0]),
        candidate_depth_support=float(candidate_support[0]),
    )

