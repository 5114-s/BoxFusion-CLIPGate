"""Batched CLIP features for supplemental instance masks.

This module reuses the CLIP model already loaded by BoxFusion.  It does not
construct another image encoder and therefore adds only crop preprocessing and
one batched forward pass on scheduled supplemental-proposal keyframes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from boxfusion.supplemental_proposals import SupplementalProposal


class ClipInstanceFeatureEncoder:
    """Encode proposal crops with the already-loaded CLIP image tower."""

    def __init__(
        self,
        model: Any,
        preprocess: Any,
        *,
        masked_crop: bool = True,
        background_value: int = 255,
    ) -> None:
        if model is None or not hasattr(model, "encode_image"):
            raise TypeError("model must expose encode_image")
        if not callable(preprocess):
            raise TypeError("preprocess must be callable")
        if not isinstance(masked_crop, (bool, np.bool_)):
            raise TypeError("masked_crop must be Boolean")
        if (
            isinstance(background_value, bool)
            or not isinstance(background_value, (int, np.integer))
            or not 0 <= int(background_value) <= 255
        ):
            raise ValueError("background_value must be an integer in [0,255]")
        self.model = model
        self.preprocess = preprocess
        self.masked_crop = bool(masked_crop)
        self.background_value = int(background_value)

    @staticmethod
    def _validate_image(image: Any) -> np.ndarray:
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("image must have shape [H,W,3]")
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("image must be numeric")
        if np.issubdtype(value.dtype, np.floating):
            if not np.isfinite(value).all():
                raise ValueError("image must be finite")
            if value.max(initial=0.0) <= 1.0:
                value = value * 255.0
        return np.clip(value, 0.0, 255.0).astype(np.uint8)

    def _crop(
        self,
        image: np.ndarray,
        proposal: SupplementalProposal,
    ) -> Image.Image:
        height, width = image.shape[:2]
        x1 = int(np.floor(proposal.bbox[0]))
        y1 = int(np.floor(proposal.bbox[1]))
        x2 = int(np.ceil(proposal.bbox[2]))
        y2 = int(np.ceil(proposal.bbox[3]))
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, x1 + 1), width)
        y2 = min(max(y2, y1 + 1), height)
        crop = image[y1:y2, x1:x2].copy()
        if self.masked_crop:
            if proposal.mask.shape != (height, width):
                raise ValueError(
                    "proposal mask must match the source image shape"
                )
            mask = proposal.mask[y1:y2, x1:x2]
            crop[~mask] = self.background_value
        return Image.fromarray(crop)

    @torch.no_grad()
    def __call__(
        self,
        image: Any,
        proposals: Sequence[SupplementalProposal],
    ) -> list[np.ndarray]:
        image_array = self._validate_image(image)
        proposal_list = list(proposals)
        if not proposal_list:
            return []
        if not all(
            isinstance(proposal, SupplementalProposal)
            for proposal in proposal_list
        ):
            raise TypeError(
                "every proposal must be a SupplementalProposal"
            )
        tensors = [
            self.preprocess(self._crop(image_array, proposal))
            for proposal in proposal_list
        ]
        if not all(torch.is_tensor(tensor) for tensor in tensors):
            raise TypeError("CLIP preprocess must return tensors")
        batch = torch.stack(tensors)
        try:
            device = next(self.model.parameters()).device
        except (StopIteration, AttributeError) as error:
            raise TypeError(
                "CLIP model must expose at least one parameter"
            ) from error
        features = self.model.encode_image(batch.to(device))
        if (
            not torch.is_tensor(features)
            or features.ndim != 2
            or features.shape[0] != len(proposal_list)
            or features.shape[1] < 1
        ):
            raise ValueError(
                "CLIP encode_image must return [N, feature_dim]"
            )
        if not torch.isfinite(features).all():
            raise ValueError("CLIP instance features must be finite")
        features = features.float()
        norms = features.norm(dim=1, keepdim=True)
        if torch.any(norms <= 1e-8):
            raise ValueError("CLIP instance features must have non-zero norm")
        normalized = (features / norms).cpu().numpy()
        return [
            np.asarray(feature, dtype=np.float32).copy()
            for feature in normalized
        ]


__all__ = ["ClipInstanceFeatureEncoder"]
