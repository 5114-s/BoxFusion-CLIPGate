"""Class-agnostic target assignment for the genuine TR3D head.

The upstream TR3D assigner maps every semantic class to exactly one feature
level through ``label2level``.  Collapsing ScanNet's 18 labels to a single
foreground label would consequently disable one of the two feature levels.
This module keeps the upstream backbone, neck, prediction layers, losses and
NMS unchanged, but assigns each foreground box independently at every feature
level.  The closest ``pts_center_threshold`` locations at each level are
eligible positives, matching TR3D's original center-distance rule.
"""

from typing import List, Tuple

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes
from projects.TR3D.tr3d.tr3d_head import TR3DHead


@MODELS.register_module()
class TR3DClassAgnosticHead(TR3DHead):
    """TR3D head with semantic-independent multi-level target assignment.

    Only the training target assignment differs from :class:`TR3DHead`.
    Inference remains a one-channel TR3D objectness prediction followed by the
    upstream class-agnostic 3D NMS.
    """

    def __init__(self, *args, **kwargs):
        # One foreground logit.  ``label2level`` is consumed by the parent to
        # determine the number of output channels, but is intentionally not
        # used by the overridden target assigner.
        kwargs["label2level"] = (0, )
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def get_targets(
        self,
        points: List[Tensor],
        gt_bboxes: BaseInstance3DBoxes,
        gt_labels: Tensor,
        num_classes: int,
    ) -> Tuple[Tensor, Tensor]:
        """Assign every foreground box to nearby points at every FPN level."""
        if num_classes != 1:
            raise ValueError(
                "TR3DClassAgnosticHead requires exactly one foreground class")
        if len(points) == 0:
            raise ValueError("TR3D requires at least one feature level")
        if any(label != 0 for label in gt_labels.tolist()):
            raise ValueError(
                "class-agnostic TR3D expects all gt_labels to be zero")

        level_ids = torch.cat([
            level_points.new_full(
                (len(level_points), ), level_index, dtype=torch.long)
            for level_index, level_points in enumerate(points)
        ])
        all_points = torch.cat(points)
        num_points = len(all_points)
        num_boxes = len(gt_bboxes)

        if num_boxes == 0:
            empty_boxes = all_points.new_zeros((num_points, 6))
            background = gt_labels.new_full((num_points, ), num_classes)
            return empty_boxes, background

        box_tensor = torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        box_tensor = box_tensor.to(all_points.device)
        expanded_boxes = box_tensor.unsqueeze(0).expand(
            num_points, num_boxes, box_tensor.shape[-1])
        expanded_points = all_points.unsqueeze(1).expand(
            num_points, num_boxes, 3)

        center_distances = torch.sum(
            torch.square(expanded_boxes[..., :3] - expanded_points), dim=-1)
        eligible = torch.zeros_like(center_distances, dtype=torch.bool)

        # Select the closest locations independently at each feature level.
        # This prevents the denser level from monopolising all positives while
        # keeping both small- and large-object evidence available.
        for level_index in range(len(points)):
            level_mask = level_ids == level_index
            level_distances = center_distances[level_mask]
            if len(level_distances) == 0:
                continue
            positive_count = min(self.pts_center_threshold,
                                 len(level_distances))
            topk_indices = torch.topk(
                level_distances,
                k=positive_count,
                largest=False,
                dim=0,
            ).indices
            level_eligible = torch.zeros_like(
                level_distances, dtype=torch.bool)
            level_eligible.scatter_(0, topk_indices, True)
            eligible[level_mask] = level_eligible

        float_max = center_distances.new_tensor(torch.finfo(
            center_distances.dtype).max)
        eligible_distances = torch.where(
            eligible, center_distances, float_max)
        min_values, min_box_ids = eligible_distances.min(dim=1)
        positive = min_values < float_max

        # Use a safe index for background rows; the value is ignored there.
        safe_box_ids = torch.where(
            positive, min_box_ids, torch.zeros_like(min_box_ids))
        bbox_targets = box_tensor[safe_box_ids]
        if not gt_bboxes.with_yaw:
            bbox_targets = bbox_targets[:, :-1]
        cls_targets = torch.where(
            positive,
            gt_labels[safe_box_ids],
            gt_labels.new_full((num_points, ), num_classes),
        )
        return bbox_targets, cls_targets
