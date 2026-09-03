"""Frozen EdgeTAM box prompts and bounded current-frame RGB-D evidence.

The provider is online/causal: it consumes only the current RGB image, depth,
intrinsics, pose, and native detector boxes.  It stores a mask tight box and a
small deterministic world-point sample for later multi-view geometry fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Mapping

import cv2
import numpy as np
import torch


EDGETAM_SOURCE_COMMIT = "7711e012a30a2402c4eaab637bdb00a521302c91"
EDGETAM_CHECKPOINT_SHA256 = (
    "ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df"
)


DEFAULT_CONFIG = {
    "enabled": False,
    "source_root": "/data/ZhaoX/BoxFusion/third_party/EdgeTAM",
    "checkpoint": (
        "/data/ZhaoX/BoxFusion/third_party/EdgeTAM/checkpoints/edgetam.pt"
    ),
    "config_name": "edgetam.yaml",
    "checkpoint_sha256": EDGETAM_CHECKPOINT_SHA256,
    "device": "cuda",
    "autocast_dtype": "bfloat16",
    "min_predicted_iou": 0.60,
    "min_mask_pixels": 80,
    "min_mask_prompt_ratio": 0.08,
    "max_mask_prompt_ratio": 2.00,
    "prompt_padding_ratio": 0.08,
    "min_depth_m": 0.10,
    "max_depth_m": 6.00,
    "depth_quantile_low": 0.02,
    "depth_quantile_high": 0.98,
    "max_points": 96,
    "erode_pixels": 1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_edgetam_config(box_fusion_cfg: Mapping) -> dict:
    raw = box_fusion_cfg.get("edgetam_maskdepth", {})
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(raw or {})
    cfg["enabled"] = bool(cfg["enabled"])
    for key in (
        "min_predicted_iou",
        "min_mask_prompt_ratio",
        "max_mask_prompt_ratio",
        "prompt_padding_ratio",
        "min_depth_m",
        "max_depth_m",
        "depth_quantile_low",
        "depth_quantile_high",
    ):
        cfg[key] = float(cfg[key])
    for key in ("min_mask_pixels", "max_points", "erode_pixels"):
        cfg[key] = int(cfg[key])
    if not 0.0 <= cfg["min_predicted_iou"] <= 1.0:
        raise ValueError("edgetam_maskdepth.min_predicted_iou must be in [0,1]")
    if not 0.0 < cfg["min_mask_prompt_ratio"] <= cfg["max_mask_prompt_ratio"]:
        raise ValueError("invalid EdgeTAM mask/prompt area-ratio interval")
    if not 0.0 <= cfg["depth_quantile_low"] < cfg["depth_quantile_high"] <= 1.0:
        raise ValueError("invalid EdgeTAM depth quantiles")
    if cfg["max_points"] < 16 or cfg["min_mask_pixels"] < 1:
        raise ValueError("EdgeTAM evidence caps are too small")
    return cfg


@dataclass(frozen=True)
class EdgeTAMFrameEvidence:
    tight_boxes_xyxy: np.ndarray
    points_world: np.ndarray
    point_valid: np.ndarray
    predicted_iou: np.ndarray
    valid: np.ndarray
    elapsed_ms: float


class EdgeTAMMaskDepthProvider:
    """Lazy official EdgeTAM image predictor with deterministic RGB-D lifting."""

    def __init__(self, box_fusion_cfg: Mapping) -> None:
        self.cfg = resolve_edgetam_config(box_fusion_cfg)
        self._predictor = None
        self._torch = torch
        self.stats = {
            "frames": 0,
            "proposals": 0,
            "valid_masks": 0,
            "elapsed_ms": 0.0,
        }

    @property
    def enabled(self) -> bool:
        return bool(self.cfg["enabled"])

    def _build(self):
        source_root = Path(self.cfg["source_root"]).resolve(strict=True)
        checkpoint = Path(self.cfg["checkpoint"]).resolve(strict=True)
        if _sha256(checkpoint) != self.cfg["checkpoint_sha256"]:
            raise RuntimeError("EdgeTAM checkpoint SHA256 mismatch")
        source_text = os.fspath(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        # The official constructor asks timm for ImageNet pretraining before it
        # immediately loads the complete EdgeTAM checkpoint.  Suppress that
        # redundant network fetch; no model parameter is omitted from the
        # authenticated EdgeTAM checkpoint.
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import sam2.modeling.backbones.timm as edgetam_timm

        package_root = Path(sam2.__file__).resolve().parent.parent
        if package_root != source_root:
            raise RuntimeError(f"wrong sam2 package loaded for EdgeTAM: {package_root}")
        original_create_model = edgetam_timm.create_model

        def create_without_imagenet(name, **kwargs):
            kwargs["pretrained"] = False
            return original_create_model(name, **kwargs)

        edgetam_timm.create_model = create_without_imagenet
        try:
            model = build_sam2(
                self.cfg["config_name"],
                os.fspath(checkpoint),
                device=self.cfg["device"],
                mode="eval",
            )
        finally:
            edgetam_timm.create_model = original_create_model
        model.eval()
        model.requires_grad_(False)
        self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    def _predict(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray):
        predictor = self._predictor if self._predictor is not None else self._build()
        dtype = getattr(self._torch, self.cfg["autocast_dtype"])
        with self._torch.inference_mode(), self._torch.autocast(
            device_type="cuda", dtype=dtype
        ):
            predictor.set_image(image_rgb)
            masks, ious, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_xyxy,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
                normalize_coords=True,
            )
        predictor.reset_predictor()
        masks = np.asarray(masks)
        ious = np.asarray(ious)
        if boxes_xyxy.shape[0] == 1 and masks.ndim == 3:
            masks = masks[None]
            ious = ious[None]
        if masks.ndim != 4 or masks.shape[:2] != ious.shape:
            raise RuntimeError(
                f"unexpected EdgeTAM output: masks={masks.shape}, iou={ious.shape}"
            )
        selected = np.argmax(ious, axis=1)
        rows = np.arange(boxes_xyxy.shape[0])
        return masks[rows, selected].astype(bool), ious[rows, selected].astype(np.float32)

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
        result = np.asarray(boxes, dtype=np.float32).copy()
        result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, width - 1)
        result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, height - 1)
        return result

    def prepare_frame(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> EdgeTAMFrameEvidence:
        if not self.enabled:
            raise RuntimeError("EdgeTAM mask-depth provider is disabled")
        image = np.ascontiguousarray(image_rgb, dtype=np.uint8)
        depth = np.asarray(depth_m, dtype=np.float32).squeeze()
        height, width = image.shape[:2]
        if image.shape != (height, width, 3) or depth.shape != (height, width):
            raise ValueError("EdgeTAM RGB and depth must be aligned HxW arrays")
        K = np.asarray(intrinsics, dtype=np.float64)
        pose = np.asarray(camera_to_world, dtype=np.float64).squeeze()
        if K.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("invalid EdgeTAM intrinsics or camera pose")

        boxes = self._clip_boxes(boxes_xyxy, height, width)
        count = boxes.shape[0]
        max_points = self.cfg["max_points"]
        tight = np.zeros((count, 4), dtype=np.float32)
        points = np.zeros((count, max_points, 3), dtype=np.float32)
        point_valid = np.zeros((count, max_points), dtype=bool)
        quality = np.zeros(count, dtype=np.float32)
        valid = np.zeros(count, dtype=bool)
        structurally_valid = (
            np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        started = time.perf_counter()
        if np.any(structurally_valid):
            masks, predicted_iou = self._predict(image, boxes[structurally_valid])
            source_indices = np.flatnonzero(structurally_valid)
            quality[source_indices] = predicted_iou
            for source_index, mask, mask_iou in zip(
                source_indices.tolist(), masks, predicted_iou.tolist()
            ):
                x1, y1, x2, y2 = boxes[source_index]
                pad_x = (x2 - x1) * self.cfg["prompt_padding_ratio"]
                pad_y = (y2 - y1) * self.cfg["prompt_padding_ratio"]
                px1 = max(0, int(np.floor(x1 - pad_x)))
                py1 = max(0, int(np.floor(y1 - pad_y)))
                px2 = min(width - 1, int(np.ceil(x2 + pad_x)))
                py2 = min(height - 1, int(np.ceil(y2 + pad_y)))
                restricted = np.zeros_like(mask, dtype=bool)
                restricted[py1 : py2 + 1, px1 : px2 + 1] = mask[
                    py1 : py2 + 1, px1 : px2 + 1
                ]
                ys, xs = np.nonzero(restricted)
                prompt_area = max((x2 - x1) * (y2 - y1), 1.0)
                area_ratio = float(xs.size) / prompt_area
                if (
                    mask_iou < self.cfg["min_predicted_iou"]
                    or xs.size < self.cfg["min_mask_pixels"]
                    or not self.cfg["min_mask_prompt_ratio"]
                    <= area_ratio
                    <= self.cfg["max_mask_prompt_ratio"]
                ):
                    continue
                tight[source_index] = [xs.min(), ys.min(), xs.max(), ys.max()]

                support_mask = restricted.astype(np.uint8)
                erode_pixels = self.cfg["erode_pixels"]
                if erode_pixels:
                    kernel_size = erode_pixels * 2 + 1
                    support_mask = cv2.erode(
                        support_mask,
                        np.ones((kernel_size, kernel_size), dtype=np.uint8),
                    )
                support = support_mask.astype(bool)
                support &= np.isfinite(depth)
                support &= depth >= self.cfg["min_depth_m"]
                support &= depth <= self.cfg["max_depth_m"]
                sy, sx = np.nonzero(support)
                if sx.size < 16:
                    continue
                z = depth[sy, sx]
                low, high = np.quantile(
                    z,
                    [self.cfg["depth_quantile_low"], self.cfg["depth_quantile_high"]],
                )
                keep = (z >= low) & (z <= high)
                sx, sy, z = sx[keep], sy[keep], z[keep]
                if z.size < 16:
                    continue
                order = np.lexsort((sx, sy))
                sample = order[
                    np.linspace(0, order.size - 1, min(max_points, order.size))
                    .round()
                    .astype(np.int64)
                ]
                sx = sx[sample].astype(np.float64)
                sy = sy[sample].astype(np.float64)
                z = z[sample].astype(np.float64)
                xyz_camera = np.stack(
                    (
                        (sx - K[0, 2]) * z / K[0, 0],
                        (sy - K[1, 2]) * z / K[1, 1],
                        z,
                    ),
                    axis=1,
                )
                xyz_world = xyz_camera @ pose[:3, :3].T + pose[:3, 3]
                sample_count = xyz_world.shape[0]
                points[source_index, :sample_count] = xyz_world.astype(np.float32)
                point_valid[source_index, :sample_count] = True
                valid[source_index] = True

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.stats["frames"] += 1
        self.stats["proposals"] += count
        self.stats["valid_masks"] += int(valid.sum())
        self.stats["elapsed_ms"] += elapsed_ms
        return EdgeTAMFrameEvidence(
            tight_boxes_xyxy=tight,
            points_world=points,
            point_valid=point_valid,
            predicted_iou=quality,
            valid=valid,
            elapsed_ms=elapsed_ms,
        )

    def attach(self, instances, **inputs) -> EdgeTAMFrameEvidence:
        evidence = self.prepare_frame(
            boxes_xyxy=instances.pred_boxes.detach().cpu().numpy(), **inputs
        )
        instances.maskdepth_tight_boxes = torch.from_numpy(
            evidence.tight_boxes_xyxy
        )
        instances.maskdepth_points_world = torch.from_numpy(evidence.points_world)
        instances.maskdepth_point_valid = torch.from_numpy(evidence.point_valid)
        instances.maskdepth_quality = torch.from_numpy(evidence.predicted_iou)
        instances.maskdepth_valid = torch.from_numpy(evidence.valid)
        return evidence

    def summary(self) -> str:
        frames = max(int(self.stats["frames"]), 1)
        proposals = max(int(self.stats["proposals"]), 1)
        return (
            "EdgeTAM mask-depth summary: "
            f"frames={self.stats['frames']}, proposals={self.stats['proposals']}, "
            f"valid={self.stats['valid_masks']} "
            f"({self.stats['valid_masks'] / proposals:.3f}), "
            f"mean_ms={self.stats['elapsed_ms'] / frames:.2f}"
        )

