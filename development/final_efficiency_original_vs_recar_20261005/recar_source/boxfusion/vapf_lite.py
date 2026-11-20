"""Training-free visibility-aware probabilistic 7D box fusion.

VAPF-lite deliberately leaves proposal generation and association untouched.
For every native observation it estimates an anisotropic covariance over

    [cx, cy, cz, log(length), log(height), log(width), yaw]

from information already present in the online RGB-D stream.  Matched
observations are fused with a Huber-robust, count-aware Covariance
Intersection update.  No future frame, ground truth, learned calibration, or
segmentation model is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


_DEFAULTS = {
    "enabled": False,
    "min_views": 3,
    "max_depth_samples": 1024,
    "min_valid_depth_samples": 16,
    "min_depth_m": 0.10,
    "max_depth_m": 8.0,
    "huber_whitened_delta": 3.0,
    "covariance_eigen_floor": 1.0e-6,
    "max_center_shift_m": 0.45,
    "max_dimension_ratio": 1.80,
    "max_yaw_shift_deg": 60.0,
}


def resolve_vapf_lite_config(box_fusion_cfg: Mapping) -> dict:
    section = box_fusion_cfg.get("vapf_lite", {})
    if section is None:
        section = {}
    if not hasattr(section, "get"):
        raise ValueError("box_fusion.vapf_lite must be a mapping")
    cfg = dict(_DEFAULTS)
    cfg.update(dict(section))
    cfg["enabled"] = bool(cfg["enabled"])
    for key in ("min_views", "max_depth_samples", "min_valid_depth_samples"):
        cfg[key] = int(cfg[key])
        if cfg[key] < 1:
            raise ValueError(f"vapf_lite.{key} must be positive")
    for key in (
        "min_depth_m",
        "max_depth_m",
        "huber_whitened_delta",
        "covariance_eigen_floor",
        "max_center_shift_m",
        "max_dimension_ratio",
        "max_yaw_shift_deg",
    ):
        cfg[key] = float(cfg[key])
        if cfg[key] <= 0.0:
            raise ValueError(f"vapf_lite.{key} must be positive")
    if cfg["max_depth_m"] <= cfg["min_depth_m"]:
        raise ValueError("vapf_lite depth interval is empty")
    if cfg["max_dimension_ratio"] <= 1.0:
        raise ValueError("vapf_lite.max_dimension_ratio must exceed one")
    return cfg


def _as_numpy(value, dtype=np.float64):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _wrap_period(value, period):
    return (value + 0.5 * period) % period - 0.5 * period


def _yaw_from_rotation(rotation):
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _projected_xyxy(projected):
    result = np.empty((projected.shape[0], 4), dtype=np.float64)
    result[:, :2] = np.min(projected, axis=1)
    result[:, 2:] = np.max(projected, axis=1)
    return result


def _box_iou_2d(left, right):
    lt = np.maximum(left[:2], right[:2])
    rb = np.minimum(left[2:], right[2:])
    wh = np.maximum(rb - lt, 0.0)
    intersection = float(wh[0] * wh[1])
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    return intersection / max(left_area + right_area - intersection, 1.0e-8)


def _safe_spd(matrix, floor):
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    return (vectors * values[None, :]) @ vectors.T


def _rz(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class VAPFFusionResult:
    accepted: bool
    reason: str
    box_xyzlhw: np.ndarray
    rotation: np.ndarray
    covariance: np.ndarray
    max_whitened_residual: float
    robust_observations: int


class VAPFLite:
    """Visibility covariance extraction and robust parameter-level fusion."""

    FEATURE_NAMES = (
        "depth_valid_ratio",
        "depth_mad_m",
        "depth_edge_ratio",
        "truncation_ratio",
        "occlusion_ratio",
        "projection_iou",
        "yaw_ambiguity",
    )

    def __init__(self, box_fusion_cfg: Mapping):
        self.cfg = resolve_vapf_lite_config(box_fusion_cfg)
        self.enabled = self.cfg["enabled"]
        self.stats = {
            "observation_batches": 0,
            "observations": 0,
            "depth_fallbacks": 0,
            "fusion_attempts": 0,
            "fusion_accepted": 0,
            "fusion_rejected": 0,
            "robust_downweighted": 0,
            "reasons": {},
            "views": [],
            "max_whitened_residual": [],
            "feature_rows": [],
        }

    def attach_observations(
        self,
        instances,
        *,
        depth_m,
        image_height,
        image_width,
        camera_to_world,
    ):
        if not self.enabled or len(instances) == 0:
            return
        depth = np.squeeze(_as_numpy(depth_m, np.float64))
        if depth.ndim != 2:
            raise ValueError(f"VAPF-lite expects a 2D depth map, got {depth.shape}")
        boxes_2d = _as_numpy(instances.pred_boxes, np.float64)
        boxes_3d = instances.pred_boxes_3d
        centers = _as_numpy(boxes_3d.tensor[:, :3], np.float64)
        dimensions = np.maximum(
            _as_numpy(boxes_3d.tensor[:, 3:6], np.float64), 1.0e-3
        )
        rotations = _as_numpy(boxes_3d.R, np.float64)
        corners = _as_numpy(boxes_3d.corners, np.float64)
        projected = _projected_xyxy(
            _as_numpy(instances.projected_boxes, np.float64)
        )
        scores = np.clip(_as_numpy(instances.scores, np.float64), 0.0, 1.0)
        pose = _as_numpy(camera_to_world, np.float64)
        world_to_camera = np.linalg.inv(pose)
        camera_rotation = pose[:3, :3]
        camera_position = pose[:3, 3]
        depth_h, depth_w = depth.shape
        scale_x = float(depth_w) / max(float(image_width), 1.0)
        scale_y = float(depth_h) / max(float(image_height), 1.0)

        covariance = np.zeros((len(instances), 7, 7), dtype=np.float64)
        features = np.zeros(
            (len(instances), len(self.FEATURE_NAMES)), dtype=np.float64
        )
        max_samples = self.cfg["max_depth_samples"]
        min_samples = self.cfg["min_valid_depth_samples"]

        for index, box in enumerate(boxes_2d):
            clipped = np.asarray(
                [
                    np.clip(box[0], 0.0, float(image_width)),
                    np.clip(box[1], 0.0, float(image_height)),
                    np.clip(box[2], 0.0, float(image_width)),
                    np.clip(box[3], 0.0, float(image_height)),
                ],
                dtype=np.float64,
            )
            original_area = max(
                float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)),
                1.0,
            )
            clipped_area = float(
                max(clipped[2] - clipped[0], 0.0)
                * max(clipped[3] - clipped[1], 0.0)
            )
            border_touch = np.mean(
                [
                    clipped[0] <= 1.0,
                    clipped[1] <= 1.0,
                    clipped[2] >= image_width - 1.0,
                    clipped[3] >= image_height - 1.0,
                ]
            )
            truncation = np.clip(
                1.0 - clipped_area / original_area + 0.25 * border_touch,
                0.0,
                1.0,
            )

            x1 = int(np.floor(clipped[0] * scale_x))
            y1 = int(np.floor(clipped[1] * scale_y))
            x2 = int(np.ceil(clipped[2] * scale_x))
            y2 = int(np.ceil(clipped[3] * scale_y))
            x1, x2 = np.clip([x1, x2], 0, depth_w)
            y1, y2 = np.clip([y1, y2], 0, depth_h)
            patch = depth[y1:y2, x1:x2]
            if patch.size:
                stride = max(1, int(np.ceil(np.sqrt(patch.size / max_samples))))
                sampled = patch[::stride, ::stride]
            else:
                sampled = patch
            valid = (
                np.isfinite(sampled)
                & (sampled >= self.cfg["min_depth_m"])
                & (sampled <= self.cfg["max_depth_m"])
            )
            valid_values = sampled[valid]
            valid_ratio = float(valid_values.size / max(sampled.size, 1))

            homogeneous = np.concatenate(
                (corners[index], np.ones((8, 1), dtype=np.float64)), axis=1
            )
            camera_corners = homogeneous @ world_to_camera.T
            predicted_front = float(np.min(camera_corners[:, 2]))
            center_h = np.concatenate((centers[index], [1.0]))
            predicted_center_depth = float((world_to_camera @ center_h)[2])

            if valid_values.size >= min_samples:
                depth_median = float(np.median(valid_values))
                depth_mad = float(
                    1.4826 * np.median(np.abs(valid_values - depth_median))
                )
                depth_residual = min(
                    abs(depth_median - predicted_center_depth), 1.0
                )
                occlusion = float(
                    np.mean(valid_values < predicted_front - 0.05)
                )
                horizontal = np.abs(np.diff(sampled, axis=1))
                vertical = np.abs(np.diff(sampled, axis=0))
                edge_threshold = max(0.05, 2.0 * depth_mad)
                edge_values = np.concatenate(
                    (horizontal.reshape(-1), vertical.reshape(-1))
                )
                edge_values = edge_values[np.isfinite(edge_values)]
                edge_ratio = float(
                    np.mean(edge_values > edge_threshold)
                ) if edge_values.size else 0.0
                depth_fallback = False
            else:
                depth_mad = 0.35
                depth_residual = 0.50
                occlusion = 0.50
                edge_ratio = 0.50
                depth_fallback = True
                self.stats["depth_fallbacks"] += 1

            projection_iou = _box_iou_2d(box, projected[index])
            planar = dimensions[index, :2]
            yaw_ambiguity = float(
                1.0 - abs(planar[0] - planar[1]) / max(planar.max(), 1.0e-3)
            )
            view_world = camera_position - centers[index]
            view_world /= max(float(np.linalg.norm(view_world)), 1.0e-8)
            local_view = np.abs(rotations[index].T @ view_world)
            local_view /= max(float(np.linalg.norm(local_view)), 1.0e-8)

            score_scale = 1.0 + 0.50 * (1.0 - scores[index])
            lateral_sigma = score_scale * np.clip(
                0.025
                + 0.050 * (1.0 - valid_ratio)
                + 0.030 * truncation
                + 0.025 * (1.0 - projection_iou),
                0.020,
                0.180,
            )
            axial_sigma = score_scale * np.clip(
                0.040
                + 0.75 * depth_mad
                + 0.20 * depth_residual
                + 0.08 * occlusion,
                0.035,
                0.500,
            )
            center_camera_cov = np.diag(
                [lateral_sigma**2, lateral_sigma**2, axial_sigma**2]
            )
            center_world_cov = (
                camera_rotation @ center_camera_cov @ camera_rotation.T
            )

            log_size_base = (
                0.080
                + 0.18 * (1.0 - valid_ratio)
                + 0.15 * occlusion
                + 0.10 * truncation
                + 0.10 * (1.0 - projection_iou)
                + (0.10 if depth_fallback else 0.0)
            )
            log_size_sigma = np.clip(
                score_scale * (log_size_base + 0.18 * local_view),
                0.060,
                0.550,
            )
            yaw_sigma = np.deg2rad(
                np.clip(
                    score_scale
                    * (
                        8.0
                        + 30.0 * yaw_ambiguity
                        + 15.0 * occlusion
                        + 12.0 * truncation
                        + 10.0 * (1.0 - projection_iou)
                    ),
                    8.0,
                    75.0,
                )
            )

            covariance[index, :3, :3] = center_world_cov
            covariance[index, 3:6, 3:6] = np.diag(log_size_sigma**2)
            covariance[index, 6, 6] = yaw_sigma**2
            covariance[index] = _safe_spd(
                covariance[index], self.cfg["covariance_eigen_floor"]
            )
            features[index] = (
                valid_ratio,
                depth_mad,
                edge_ratio,
                truncation,
                occlusion,
                projection_iou,
                yaw_ambiguity,
            )

        instances.vapf_covariance_7d = torch.from_numpy(
            covariance.astype(np.float32)
        )
        instances.vapf_visibility_features = torch.from_numpy(
            features.astype(np.float32)
        )
        self.stats["observation_batches"] += 1
        self.stats["observations"] += len(instances)
        self.stats["feature_rows"].extend(features.tolist())

    def _canonical_states(self, boxes, rotations, covariance):
        yaw_variance = covariance[:, 6, 6]
        reference_index = int(np.argmin(yaw_variance))
        reference_yaw = _yaw_from_rotation(rotations[reference_index])
        states = np.zeros((boxes.shape[0], 7), dtype=np.float64)
        canonical_covariance = covariance.copy()
        for index in range(boxes.shape[0]):
            yaw = _yaw_from_rotation(rotations[index])
            dimensions = np.maximum(boxes[index, 3:6], 1.0e-3).copy()
            direct_delta = abs(_wrap_period(yaw - reference_yaw, np.pi))
            swapped_yaw = yaw + 0.5 * np.pi
            swapped_delta = abs(
                _wrap_period(swapped_yaw - reference_yaw, np.pi)
            )
            if swapped_delta + 1.0e-9 < direct_delta:
                yaw = swapped_yaw
                dimensions[[0, 1]] = dimensions[[1, 0]]
                permutation = np.arange(7)
                permutation[[3, 4]] = permutation[[4, 3]]
                canonical_covariance[index] = canonical_covariance[index][
                    np.ix_(permutation, permutation)
                ]
            yaw = reference_yaw + _wrap_period(yaw - reference_yaw, np.pi)
            states[index, :3] = boxes[index, :3]
            states[index, 3:6] = np.log(dimensions)
            states[index, 6] = yaw
        return states, canonical_covariance, reference_index, reference_yaw

    def fuse(self, boxes_xyzlhw, rotations, covariance_7d, frame_ids=None):
        boxes = _as_numpy(boxes_xyzlhw, np.float64)
        rotations = _as_numpy(rotations, np.float64)
        covariance = _as_numpy(covariance_7d, np.float64)
        count = boxes.shape[0]
        self.stats["fusion_attempts"] += 1
        self.stats["views"].append(int(count))
        if count < self.cfg["min_views"]:
            return self._reject("insufficient_views", boxes, rotations)
        if covariance.shape != (count, 7, 7):
            return self._reject("missing_covariance", boxes, rotations)
        if not (
            np.all(np.isfinite(boxes))
            and np.all(np.isfinite(rotations))
            and np.all(np.isfinite(covariance))
        ):
            return self._reject("nonfinite_input", boxes, rotations)

        covariance = np.stack(
            [
                _safe_spd(item, self.cfg["covariance_eigen_floor"])
                for item in covariance
            ]
        )
        states, covariance, reference_index, reference_yaw = (
            self._canonical_states(boxes, rotations, covariance)
        )
        if frame_ids is None:
            order = np.arange(count)
        else:
            order = np.argsort(_as_numpy(frame_ids, np.int64), kind="stable")

        mean = states[order[0]].copy()
        fused_covariance = covariance[order[0]].copy()
        robust_observations = 0
        max_whitened = 0.0
        effective_count = 1
        for observation_index in order[1:]:
            observation = states[observation_index].copy()
            observation[6] = mean[6] + _wrap_period(
                observation[6] - mean[6], np.pi
            )
            observation_covariance = covariance[observation_index].copy()
            innovation = observation - mean
            innovation[6] = _wrap_period(innovation[6], np.pi)
            innovation_covariance = _safe_spd(
                fused_covariance + observation_covariance,
                self.cfg["covariance_eigen_floor"],
            )
            whitened = float(
                np.sqrt(
                    max(
                        innovation
                        @ np.linalg.solve(innovation_covariance, innovation),
                        0.0,
                    )
                )
            )
            max_whitened = max(max_whitened, whitened)
            huber_weight = min(
                1.0,
                self.cfg["huber_whitened_delta"] / max(whitened, 1.0e-8),
            )
            if huber_weight < 1.0:
                robust_observations += 1
                observation_covariance /= max(huber_weight**2, 1.0e-4)

            omega = float(effective_count / (effective_count + 1.0))
            prior_information = np.linalg.inv(fused_covariance)
            observation_information = np.linalg.inv(observation_covariance)
            fused_information = (
                omega * prior_information
                + (1.0 - omega) * observation_information
            )
            fused_covariance = _safe_spd(
                np.linalg.inv(fused_information),
                self.cfg["covariance_eigen_floor"],
            )
            mean = fused_covariance @ (
                omega * prior_information @ mean
                + (1.0 - omega) * observation_information @ observation
            )
            mean[6] = reference_yaw + _wrap_period(
                mean[6] - reference_yaw, np.pi
            )
            effective_count += 1

        candidate_dimensions = np.exp(mean[3:6])
        median_center = np.median(states[:, :3], axis=0)
        median_dimensions = np.exp(np.median(states[:, 3:6], axis=0))
        center_shift = float(np.linalg.norm(mean[:3] - median_center))
        dimension_ratio = float(
            np.max(
                np.maximum(
                    candidate_dimensions / np.maximum(median_dimensions, 1.0e-3),
                    median_dimensions / np.maximum(candidate_dimensions, 1.0e-3),
                )
            )
        )
        yaw_shift = abs(_wrap_period(mean[6] - reference_yaw, np.pi))
        if center_shift > self.cfg["max_center_shift_m"]:
            return self._reject("center_safety", boxes, rotations)
        if dimension_ratio > self.cfg["max_dimension_ratio"]:
            return self._reject("dimension_safety", boxes, rotations)
        if yaw_shift > np.deg2rad(self.cfg["max_yaw_shift_deg"]):
            return self._reject("yaw_safety", boxes, rotations)
        if not np.all(np.isfinite(candidate_dimensions)) or np.any(
            candidate_dimensions < 0.01
        ):
            return self._reject("invalid_candidate", boxes, rotations)

        output_box = np.concatenate((mean[:3], candidate_dimensions)).astype(
            np.float32
        )
        output_rotation = (
            _rz(_wrap_period(mean[6] - reference_yaw, np.pi))
            @ rotations[reference_index]
        ).astype(np.float32)
        self.stats["fusion_accepted"] += 1
        self.stats["robust_downweighted"] += robust_observations
        self.stats["max_whitened_residual"].append(max_whitened)
        self.stats["reasons"]["accepted"] = (
            self.stats["reasons"].get("accepted", 0) + 1
        )
        return VAPFFusionResult(
            accepted=True,
            reason="accepted",
            box_xyzlhw=output_box,
            rotation=output_rotation,
            covariance=fused_covariance.astype(np.float32),
            max_whitened_residual=max_whitened,
            robust_observations=robust_observations,
        )

    def _reject(self, reason, boxes, rotations):
        self.stats["fusion_rejected"] += 1
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
        fallback_box = (
            boxes[0].astype(np.float32)
            if boxes.size
            else np.zeros(6, dtype=np.float32)
        )
        fallback_rotation = (
            rotations[0].astype(np.float32)
            if rotations.size
            else np.eye(3, dtype=np.float32)
        )
        return VAPFFusionResult(
            accepted=False,
            reason=reason,
            box_xyzlhw=fallback_box,
            rotation=fallback_rotation,
            covariance=np.eye(7, dtype=np.float32),
            max_whitened_residual=float("nan"),
            robust_observations=0,
        )

    def summary(self):
        feature_rows = np.asarray(self.stats["feature_rows"], dtype=np.float64)
        if feature_rows.size:
            medians = np.median(feature_rows, axis=0)
            feature_summary = ",".join(
                f"{name}={value:.4f}"
                for name, value in zip(self.FEATURE_NAMES, medians)
            )
        else:
            feature_summary = "no_features"
        residuals = np.asarray(
            self.stats["max_whitened_residual"], dtype=np.float64
        )
        residual_p95 = (
            float(np.quantile(residuals, 0.95)) if residuals.size else float("nan")
        )
        return (
            "VAPF-lite summary | "
            f"observations={self.stats['observations']}, "
            f"depth_fallbacks={self.stats['depth_fallbacks']}, "
            f"attempted={self.stats['fusion_attempts']}, "
            f"accepted={self.stats['fusion_accepted']}, "
            f"rejected={self.stats['fusion_rejected']}, "
            f"robust_downweighted={self.stats['robust_downweighted']}, "
            f"max_whitened_p95={residual_p95:.4f}, "
            f"reasons={self.stats['reasons']}, "
            f"feature_medians=[{feature_summary}]"
        )
