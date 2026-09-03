"""Foreground-occupancy Top-K selection for the BoxFusion P2 ablation.

P2 is deliberately a *selector* layered on the frozen P1 residual proposal
observer:

``P1 residual voxels -> foreground occupancy -> stable Top-K -> P1 boxes``.

The module never writes BoxFusion detections.  It does not contain P3
grouping, P4 Mask-RGBD confirmation, semantic labels, or an output score gate.
Ground truth is accepted only by :func:`assign_foreground_occupancy_targets`,
which is used by the offline train-only utility.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.residual_proposal import (
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
    P1ResidualProposalObserver,
    ResidualObservation,
    ResidualProposal,
    ResidualProposalConfig,
    ResidualVoxelBatch,
    center_size_to_minmax,
    resolve_residual_proposal_config,
    stable_nms_aabb,
)

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - dependency preflight
    torch = None
    nn = None


P2_DIAGNOSTIC_SCHEMA = "boxfusion.p2.occupancy_topk_observer.v1"
P2_HEAD_SCHEMA = "boxfusion.p2_occupancy_topk_head.v1"
P2_SOURCE = "p2_foreground_occupancy_topk"


def _finite_float(
    value: Any,
    name: str,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    strict_lower: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None:
        invalid = result <= lower if strict_lower else result < lower
        if invalid:
            relation = "greater than" if strict_lower else "at least"
            raise ValueError(f"{name} must be {relation} {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OccupancyTopKConfig:
    """Strict P2 selector configuration."""

    enabled: bool = False
    observer_only: bool = True
    mutate: bool = False
    collect_diagnostics: bool = False
    checkpoint: Optional[str] = None
    forbidden_scene_list: Optional[str] = None
    device: str = "cpu"
    hidden_dim: int = 32
    min_occupancy_score: float = 0.05
    topk_voxels_per_step: int = 512
    max_candidates_per_step: int = 32
    max_scene_candidates: int = 64
    scene_nms_iou: float = 0.25
    max_history_steps: int = 64
    input_feature_names: Tuple[str, ...] = P1_FEATURE_NAMES

    def validated(self) -> "OccupancyTopKConfig":
        for name in (
            "enabled",
            "observer_only",
            "mutate",
            "collect_diagnostics",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"occupancy_topk.{name} must be Boolean")
        if not bool(self.observer_only):
            raise ValueError("P2 must remain observer_only")
        if bool(self.mutate):
            raise ValueError("P2 cannot mutate formal detections")
        checkpoint = self.checkpoint
        if checkpoint is not None:
            if not isinstance(checkpoint, (str, Path)):
                raise TypeError(
                    "occupancy_topk.checkpoint must be a path or null"
                )
            checkpoint = str(checkpoint).strip()
            if not checkpoint:
                raise ValueError(
                    "occupancy_topk.checkpoint cannot be empty"
                )
        forbidden_scene_list = self.forbidden_scene_list
        if forbidden_scene_list is not None:
            if not isinstance(forbidden_scene_list, (str, Path)):
                raise TypeError(
                    "occupancy_topk.forbidden_scene_list must be a path "
                    "or null"
                )
            forbidden_scene_list = str(forbidden_scene_list).strip()
            if not forbidden_scene_list:
                raise ValueError(
                    "occupancy_topk.forbidden_scene_list cannot be empty"
                )
        device = str(self.device).strip()
        if not device:
            raise ValueError("occupancy_topk.device cannot be empty")
        names = tuple(str(name) for name in self.input_feature_names)
        if names != P1_FEATURE_NAMES:
            raise ValueError(
                "occupancy_topk feature schema must match P1 exactly"
            )
        result = OccupancyTopKConfig(
            enabled=bool(self.enabled),
            observer_only=True,
            mutate=False,
            collect_diagnostics=bool(self.collect_diagnostics),
            checkpoint=checkpoint,
            forbidden_scene_list=forbidden_scene_list,
            device=device,
            hidden_dim=_positive_int(
                self.hidden_dim, "occupancy_topk.hidden_dim"
            ),
            min_occupancy_score=_finite_float(
                self.min_occupancy_score,
                "occupancy_topk.min_occupancy_score",
                lower=0.0,
                upper=1.0,
            ),
            topk_voxels_per_step=_positive_int(
                self.topk_voxels_per_step,
                "occupancy_topk.topk_voxels_per_step",
            ),
            max_candidates_per_step=_positive_int(
                self.max_candidates_per_step,
                "occupancy_topk.max_candidates_per_step",
            ),
            max_scene_candidates=_positive_int(
                self.max_scene_candidates,
                "occupancy_topk.max_scene_candidates",
            ),
            scene_nms_iou=_finite_float(
                self.scene_nms_iou,
                "occupancy_topk.scene_nms_iou",
                lower=0.0,
                upper=1.0,
            ),
            max_history_steps=_positive_int(
                self.max_history_steps,
                "occupancy_topk.max_history_steps",
            ),
            input_feature_names=names,
        )
        if result.max_candidates_per_step > result.topk_voxels_per_step:
            raise ValueError(
                "P2 max_candidates_per_step cannot exceed voxel Top-K"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_feature_names"] = list(self.input_feature_names)
        return payload


def resolve_occupancy_topk_config(
    config: Optional[Mapping[str, Any] | OccupancyTopKConfig] = None,
) -> OccupancyTopKConfig:
    if config is None:
        return OccupancyTopKConfig().validated()
    if isinstance(config, OccupancyTopKConfig):
        return config.validated()
    if not isinstance(config, Mapping):
        raise TypeError("occupancy_topk config must be a mapping")
    known = set(OccupancyTopKConfig.__dataclass_fields__)
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            "Unknown occupancy_topk key(s): " + ", ".join(unknown)
        )
    payload = dict(config)
    if "input_feature_names" in payload:
        payload["input_feature_names"] = tuple(payload["input_feature_names"])
    return OccupancyTopKConfig(**payload).validated()


if nn is not None:

    class ForegroundOccupancyHead(nn.Module):
        """Small class-agnostic foreground-occupancy MLP."""

        def __init__(
            self,
            input_dim: int = P1_FEATURE_DIM,
            hidden_dim: int = 32,
        ) -> None:
            super().__init__()
            if int(input_dim) != P1_FEATURE_DIM or int(hidden_dim) < 1:
                raise ValueError(
                    "P2 input_dim must match P1 and hidden_dim be positive"
                )
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            if features.ndim != 2 or features.shape[1] != self.input_dim:
                raise ValueError(
                    f"features must have shape [N,{self.input_dim}]"
                )
            return self.network(features)

        def model_config(self) -> dict[str, int]:
            return {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
            }

else:  # pragma: no cover

    class ForegroundOccupancyHead:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for the P2 occupancy head")


def _torch_devices_for_fork(device: str) -> list[int]:
    if torch is None or not str(device).startswith("cuda"):
        return []
    if not torch.cuda.is_available():
        return []
    parsed = torch.device(device)
    index = parsed.index
    return [torch.cuda.current_device() if index is None else int(index)]


def load_occupancy_topk_head(
    checkpoint_path: str | Path,
    *,
    expected_config: OccupancyTopKConfig,
    expected_p1_checkpoint_sha256: str,
    expected_b6_checkpoint_sha256: str,
    expected_forbidden_scene_list_sha256: Optional[str] = None,
    device: str,
) -> tuple[ForegroundOccupancyHead, str, Mapping[str, Any]]:
    """Load P2 while restoring global RNG state after module construction."""

    if torch is None:
        raise ImportError("PyTorch is required to load P2")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing P2 checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("P2 checkpoint must contain a mapping")
    if payload.get("schema") != P2_HEAD_SCHEMA:
        raise ValueError("P2 checkpoint schema mismatch")
    if tuple(payload.get("feature_names", ())) != P1_FEATURE_NAMES:
        raise ValueError("P2 checkpoint feature schema mismatch")
    model_config = payload.get("model_config")
    state_dict = payload.get("state_dict")
    provenance = payload.get("provenance")
    if not isinstance(model_config, Mapping) or not isinstance(
        state_dict, Mapping
    ):
        raise ValueError("P2 checkpoint lacks model_config/state_dict")
    if not isinstance(provenance, Mapping):
        raise ValueError("P2 checkpoint lacks train-only provenance")
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    scene_pattern = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
    train_scenes = provenance.get("train_scene_ids")
    forbidden_overlap = provenance.get("forbidden_overlap")
    recorded_p1 = str(provenance.get("p1_checkpoint_sha256", "")).lower()
    recorded_b6 = str(provenance.get("b6_checkpoint_sha256", "")).lower()
    recorded_forbidden = str(
        provenance.get("forbidden_scene_list_sha256", "")
    ).lower()
    expected_p1 = str(expected_p1_checkpoint_sha256).lower()
    expected_b6 = str(expected_b6_checkpoint_sha256).lower()
    if (
        not isinstance(train_scenes, Sequence)
        or isinstance(train_scenes, (str, bytes))
        or not train_scenes
        or any(
            not isinstance(scene, str)
            or scene_pattern.fullmatch(scene) is None
            for scene in train_scenes
        )
        or len(set(train_scenes)) != len(train_scenes)
        or forbidden_overlap != []
        or sha_pattern.fullmatch(recorded_p1) is None
        or sha_pattern.fullmatch(recorded_b6) is None
        or sha_pattern.fullmatch(expected_p1) is None
        or sha_pattern.fullmatch(expected_b6) is None
    ):
        raise ValueError("P2 checkpoint train-only provenance is invalid")
    if recorded_p1 != expected_p1:
        raise ValueError("P2 checkpoint was trained against another P1 head")
    if recorded_b6 != expected_b6:
        raise ValueError("P2 checkpoint was trained against another B6 head")
    if expected_forbidden_scene_list_sha256 is not None:
        expected_forbidden = str(
            expected_forbidden_scene_list_sha256
        ).lower()
        if (
            sha_pattern.fullmatch(expected_forbidden) is None
            or recorded_forbidden != expected_forbidden
        ):
            raise ValueError(
                "P2 checkpoint forbidden validation split mismatch"
            )
    if int(model_config.get("input_dim", -1)) != P1_FEATURE_DIM:
        raise ValueError("P2 checkpoint input_dim mismatch")
    if int(model_config.get("hidden_dim", -1)) != expected_config.hidden_dim:
        raise ValueError("P2 checkpoint hidden_dim mismatch")
    devices = _torch_devices_for_fork(device)
    with torch.random.fork_rng(devices=devices, enabled=True):
        model = ForegroundOccupancyHead(
            input_dim=int(model_config["input_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
        )
        model.load_state_dict(dict(state_dict), strict=True)
        model.to(device)
    model.eval()
    return model, sha256_file(path), payload


def assign_foreground_occupancy_targets(
    voxel_centers: Any,
    residual_gt_boxes: Any,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Label residual voxels inside a residual GT box.

    This train-only target is distinct from P1 objectness: P1 assigns a small
    number of centre anchors, whereas P2 labels every observed residual voxel
    falling inside an otherwise-uncovered object box.
    """

    centers = np.asarray(voxel_centers, dtype=np.float64)
    boxes = np.asarray(residual_gt_boxes, dtype=np.float64)
    if centers.size == 0:
        centers = np.empty((0, 3), dtype=np.float64)
    if boxes.size == 0:
        boxes = np.empty((0, 6), dtype=np.float64)
    if centers.shape != (len(centers), 3):
        raise ValueError("voxel_centers must have shape [V,3]")
    if boxes.shape != (len(boxes), 6):
        raise ValueError("residual_gt_boxes must have shape [G,6]")
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(boxes).all()
        or (len(boxes) and np.any(boxes[:, 3:] <= 0.0))
    ):
        raise ValueError("P2 occupancy target inputs are invalid")
    margin_value = _finite_float(
        margin, "occupancy target margin", lower=0.0
    )
    if not len(centers) or not len(boxes):
        return np.zeros(len(centers), dtype=np.float32)
    bounds = center_size_to_minmax(boxes)
    lower = bounds[:, :3] - margin_value
    upper = bounds[:, 3:] + margin_value
    inside = np.any(
        np.all(
            (centers[:, None, :] >= lower[None])
            & (centers[:, None, :] <= upper[None]),
            axis=2,
        ),
        axis=1,
    )
    return inside.astype(np.float32)


def stable_occupancy_topk(
    scores: Any,
    coordinates: Any,
    *,
    minimum_score: float,
    topk: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    coords = np.asarray(coordinates)
    if coords.shape != (len(values), 3) or coords.dtype.kind not in {"i", "u"}:
        raise ValueError("coordinates must be integer [V,3]")
    if not np.isfinite(values).all():
        raise ValueError("occupancy scores must be finite")
    threshold = _finite_float(
        minimum_score, "minimum_score", lower=0.0, upper=1.0
    )
    limit = _positive_int(topk, "topk")
    eligible = np.flatnonzero(values >= threshold)
    ordered = sorted(
        eligible.tolist(),
        key=lambda index: (
            -values[index],
            int(coords[index, 0]),
            int(coords[index, 1]),
            int(coords[index, 2]),
        ),
    )
    return np.asarray(ordered[:limit], dtype=np.int64)


def _candidate_coordinate(candidate_id: str) -> tuple[int, int, int]:
    fields = str(candidate_id).rsplit(":", 3)
    if len(fields) != 4:
        raise ValueError(f"invalid P1 candidate id: {candidate_id!r}")
    try:
        return int(fields[1]), int(fields[2]), int(fields[3])
    except ValueError as error:
        raise ValueError(
            f"invalid P1 candidate coordinate: {candidate_id!r}"
        ) from error


@dataclass(frozen=True)
class OccupancySelectedProposal:
    base: ResidualProposal
    occupancy_score: float
    occupancy_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.base, ResidualProposal):
            raise TypeError("base must be a ResidualProposal")
        if (
            not np.isfinite(self.occupancy_score)
            or not 0.0 <= float(self.occupancy_score) <= 1.0
        ):
            raise ValueError("occupancy_score must lie in [0,1]")
        if int(self.occupancy_rank) < 0:
            raise ValueError("occupancy_rank must be non-negative")

    @property
    def candidate_id(self) -> str:
        return self.base.candidate_id

    @property
    def box(self) -> np.ndarray:
        return self.base.box

    @property
    def corners(self) -> np.ndarray:
        return self.base.corners

    @property
    def objectness(self) -> float:
        return self.base.objectness


@dataclass(frozen=True)
class OccupancyTopKObservation:
    base: ResidualObservation
    selected: Tuple[OccupancySelectedProposal, ...]
    eligible_voxels: int
    selected_voxels: int
    occupancy_seconds: float

    @property
    def total_seconds(self) -> float:
        return float(self.base.total_seconds + self.occupancy_seconds)


class P2OccupancyTopKObserver(P1ResidualProposalObserver):
    """P1 observer plus a diagnostics-only foreground occupancy selector."""

    def __init__(
        self,
        p1_config: Mapping[str, Any] | ResidualProposalConfig,
        p2_config: Mapping[str, Any] | OccupancyTopKConfig,
        *,
        p1_head: Optional[Any] = None,
        occupancy_head: Optional[Any] = None,
        p1_device: Optional[str] = None,
        expected_b6_checkpoint_sha256: Optional[str] = None,
        expected_p1_checkpoint_sha256: Optional[str] = None,
    ) -> None:
        super().__init__(
            p1_config,
            head=p1_head,
            device=p1_device,
            expected_b6_checkpoint_sha256=expected_b6_checkpoint_sha256,
        )
        self.occupancy_config = resolve_occupancy_topk_config(p2_config)
        if not self.occupancy_config.enabled:
            raise ValueError("P2 observer requires occupancy_topk.enabled")
        self.occupancy_head = None
        self.occupancy_checkpoint_sha256 = ""
        self.occupancy_checkpoint_metadata: Mapping[str, Any] = {}
        self.training_scene_ids: frozenset[str] = frozenset()
        if occupancy_head is not None:
            self.occupancy_head = occupancy_head
            self.occupancy_checkpoint_sha256 = "injected"
        elif self.occupancy_config.checkpoint is None:
            raise ValueError("P2 observer requires an occupancy checkpoint")
        else:
            if not expected_p1_checkpoint_sha256:
                raise ValueError("P2 requires expected P1 checkpoint SHA")
            if not expected_b6_checkpoint_sha256:
                raise ValueError("P2 requires expected B6 checkpoint SHA")
            expected_forbidden_sha = None
            forbidden_scenes: set[str] = set()
            if self.occupancy_config.forbidden_scene_list is not None:
                forbidden_path = Path(
                    self.occupancy_config.forbidden_scene_list
                )
                if not forbidden_path.is_file():
                    raise FileNotFoundError(
                        "missing P2 forbidden scene list: "
                        f"{forbidden_path}"
                    )
                rows = [
                    row.strip()
                    for row in forbidden_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if row.strip()
                ]
                if not rows or len(rows) != len(set(rows)):
                    raise ValueError(
                        "P2 forbidden scene list must be non-empty "
                        "and unique"
                    )
                forbidden_scenes = set(rows)
                expected_forbidden_sha = sha256_file(forbidden_path)
            (
                self.occupancy_head,
                self.occupancy_checkpoint_sha256,
                self.occupancy_checkpoint_metadata,
            ) = load_occupancy_topk_head(
                self.occupancy_config.checkpoint,
                expected_config=self.occupancy_config,
                expected_p1_checkpoint_sha256=(
                    expected_p1_checkpoint_sha256
                ),
                expected_b6_checkpoint_sha256=(
                    expected_b6_checkpoint_sha256
                ),
                expected_forbidden_scene_list_sha256=(
                    expected_forbidden_sha
                ),
                device=self.occupancy_config.device,
            )
            provenance = self.occupancy_checkpoint_metadata.get(
                "provenance", {}
            )
            if not isinstance(provenance, Mapping):
                raise ValueError("P2 checkpoint provenance is invalid")
            self.training_scene_ids = frozenset(
                str(scene)
                for scene in provenance.get("train_scene_ids", ())
            )
            if self.training_scene_ids & forbidden_scenes:
                raise ValueError(
                    "P2 checkpoint train scenes overlap forbidden split"
                )
        if hasattr(self.occupancy_head, "eval"):
            self.occupancy_head.eval()
        self.p2_observations: list[OccupancyTopKObservation] = []

    def reset(self, scene_id: str) -> None:
        if str(scene_id) in self.training_scene_ids:
            raise ValueError(
                f"P2 refuses train-scene inference: {scene_id}"
            )
        super().reset(scene_id)
        if hasattr(self, "p2_observations"):
            self.p2_observations.clear()

    def _occupancy_probabilities(
        self, batch: ResidualVoxelBatch
    ) -> np.ndarray:
        if self.occupancy_head is None:
            raise RuntimeError("P2 occupancy head is unavailable")
        if not len(batch.features):
            return np.empty((0,), dtype=np.float64)
        if torch is not None and isinstance(self.occupancy_head, nn.Module):
            tensor = torch.as_tensor(
                np.asarray(batch.features),
                dtype=torch.float32,
                device=self.occupancy_config.device,
            )
            with torch.inference_mode():
                logits = self.occupancy_head(tensor)
            values = logits.detach().cpu().numpy().reshape(-1)
        else:
            values = np.asarray(
                self.occupancy_head(batch.features), dtype=np.float64
            ).reshape(-1)
        if len(values) != len(batch.features) or not np.isfinite(values).all():
            raise RuntimeError("P2 occupancy head returned invalid logits")
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    @staticmethod
    def _subset_batch(
        batch: ResidualVoxelBatch, indices: np.ndarray
    ) -> ResidualVoxelBatch:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        return ResidualVoxelBatch(
            coordinates=np.asarray(batch.coordinates)[selected],
            centers=np.asarray(batch.centers)[selected],
            features=np.asarray(batch.features)[selected],
            point_counts=np.asarray(batch.point_counts)[selected],
            input_point_count=batch.input_point_count,
            explained_point_count=batch.explained_point_count,
            residual_point_count=batch.residual_point_count,
        )

    def _run_frozen_p1_head(
        self, batch: ResidualVoxelBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.head is None:
            raise RuntimeError("P2 requires the frozen P1 head")
        if torch is not None and isinstance(self.head, nn.Module):
            tensor = torch.as_tensor(
                np.asarray(batch.features),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.inference_mode():
                logits, regression = self.head(tensor)
            return (
                logits.detach().cpu().numpy(),
                regression.detach().cpu().numpy(),
            )
        logits, regression = self.head(batch.features)
        return np.asarray(logits), np.asarray(regression)

    def observe(self, **kwargs: Any) -> ResidualObservation:
        base = super().observe(**kwargs)
        started = time.perf_counter()
        probabilities = self._occupancy_probabilities(base.voxel_batch)
        selected_voxel_indices = stable_occupancy_topk(
            probabilities,
            base.voxel_batch.coordinates,
            minimum_score=self.occupancy_config.min_occupancy_score,
            topk=self.occupancy_config.topk_voxels_per_step,
        )
        selected_batch = self._subset_batch(
            base.voxel_batch, selected_voxel_indices
        )
        p2_proposals: tuple[ResidualProposal, ...] = ()
        if len(selected_batch.features):
            logits, regression = self._run_frozen_p1_head(selected_batch)
            global_corners = np.asarray(
                kwargs.get("global_corners"), dtype=np.float32
            )
            global_ids = np.asarray(
                kwargs.get("global_stable_ids"), dtype=np.int64
            ).reshape(-1)
            p2_proposals = self._decode(
                selected_batch,
                logits,
                regression,
                scene_id=str(kwargs["scene_id"]),
                frame_index=int(kwargs["frame_index"]),
                provider_step=int(kwargs["provider_step"]),
                global_corners=global_corners,
                global_stable_ids=global_ids,
            )
        score_by_coordinate = {
            tuple(
                int(value)
                for value in base.voxel_batch.coordinates[int(index)]
            ): (float(probabilities[int(index)]), rank)
            for rank, index in enumerate(selected_voxel_indices.tolist())
        }
        rows: list[OccupancySelectedProposal] = []
        for proposal in p2_proposals:
            coordinate = _candidate_coordinate(proposal.candidate_id)
            selected = score_by_coordinate.get(coordinate)
            if selected is None:
                raise RuntimeError(
                    "P2 decoder emitted a non-selected anchor"
                )
            occupancy_score, occupancy_rank = selected
            rows.append(
                OccupancySelectedProposal(
                    base=proposal,
                    occupancy_score=occupancy_score,
                    occupancy_rank=occupancy_rank,
                )
            )
        rows.sort(
            key=lambda row: (
                -row.occupancy_score,
                row.candidate_id,
            )
        )
        rows = rows[: self.occupancy_config.max_candidates_per_step]
        p2_observation = OccupancyTopKObservation(
            base=base,
            selected=tuple(rows),
            eligible_voxels=int(
                np.sum(
                    probabilities
                    >= self.occupancy_config.min_occupancy_score
                )
            ),
            selected_voxels=len(selected_voxel_indices),
            occupancy_seconds=float(time.perf_counter() - started),
        )
        self.p2_observations.append(p2_observation)
        del self.p2_observations[
            : -self.occupancy_config.max_history_steps
        ]
        return base

    def p2_scene_candidates(
        self,
    ) -> tuple[OccupancySelectedProposal, ...]:
        rows = [
            proposal
            for observation in self.p2_observations
            for proposal in observation.selected
        ]
        if not rows:
            return ()
        boxes = np.stack([row.box for row in rows])
        scores = np.asarray(
            [row.occupancy_score for row in rows], dtype=np.float64
        )
        ids = [row.candidate_id for row in rows]
        keep = stable_nms_aabb(
            boxes,
            scores,
            self.occupancy_config.scene_nms_iou,
            tie_breakers=ids,
            max_output=self.occupancy_config.max_scene_candidates,
        )
        return tuple(rows[int(index)] for index in keep)

    def diagnostic_payload(self) -> dict[str, np.ndarray]:
        payload = super().diagnostic_payload()
        observations = tuple(self.p2_observations)
        candidates = self.p2_scene_candidates()
        config_json = json.dumps(
            self.occupancy_config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        payload.update(
            {
                "p2_schema": np.asarray(P2_DIAGNOSTIC_SCHEMA),
                "p2_stage": np.asarray("P2"),
                "p2_profile": np.asarray(
                    "p2_occupancy_topk_observer"
                ),
                "p2_enabled": np.asarray(True, dtype=bool),
                "p2_observer_only": np.asarray(True, dtype=bool),
                "p2_uses_ground_truth": np.asarray(False, dtype=bool),
                "p2_mutation_enabled": np.asarray(False, dtype=bool),
                "p2_applied_count": np.asarray(0, dtype=np.int64),
                "p2_complete": np.asarray(
                    bool(observations), dtype=bool
                ),
                "p2_class_agnostic": np.asarray(True, dtype=bool),
                "p2_source": np.asarray(P2_SOURCE),
                "p2_checkpoint_sha256": np.asarray(
                    self.occupancy_checkpoint_sha256
                ),
                "p2_config_json": np.asarray(config_json),
                "p2_feature_names": np.asarray(
                    P1_FEATURE_NAMES, dtype=np.str_
                ),
                "p2_step_frame_ids": np.asarray(
                    [row.base.frame_index for row in observations],
                    dtype=np.int64,
                ),
                "p2_step_provider_steps": np.asarray(
                    [row.base.provider_step for row in observations],
                    dtype=np.int64,
                ),
                "p2_step_input_voxel_counts": np.asarray(
                    [
                        len(row.base.voxel_batch.features)
                        for row in observations
                    ],
                    dtype=np.int64,
                ),
                "p2_step_eligible_voxel_counts": np.asarray(
                    [row.eligible_voxels for row in observations],
                    dtype=np.int64,
                ),
                "p2_step_selected_voxel_counts": np.asarray(
                    [row.selected_voxels for row in observations],
                    dtype=np.int64,
                ),
                "p2_step_candidate_counts": np.asarray(
                    [len(row.selected) for row in observations],
                    dtype=np.int64,
                ),
                "p2_step_seconds": np.asarray(
                    [row.occupancy_seconds for row in observations],
                    dtype=np.float64,
                ),
                "p2_candidate_ids": np.asarray(
                    [row.candidate_id for row in candidates],
                    dtype=np.str_,
                ),
                "p2_candidate_boxes": (
                    np.stack([row.box for row in candidates]).astype(
                        np.float32
                    )
                    if candidates
                    else np.empty((0, 6), dtype=np.float32)
                ),
                "p2_candidate_corners": (
                    np.stack([row.corners for row in candidates]).astype(
                        np.float32
                    )
                    if candidates
                    else np.empty((0, 8, 3), dtype=np.float32)
                ),
                "p2_candidate_objectness": np.asarray(
                    [row.objectness for row in candidates],
                    dtype=np.float32,
                ),
                "p2_candidate_occupancy_scores": np.asarray(
                    [row.occupancy_score for row in candidates],
                    dtype=np.float32,
                ),
                "p2_candidate_occupancy_ranks": np.asarray(
                    [row.occupancy_rank for row in candidates],
                    dtype=np.int64,
                ),
            }
        )
        return payload


__all__ = [
    "ForegroundOccupancyHead",
    "OccupancySelectedProposal",
    "OccupancyTopKConfig",
    "OccupancyTopKObservation",
    "P2_DIAGNOSTIC_SCHEMA",
    "P2_HEAD_SCHEMA",
    "P2OccupancyTopKObserver",
    "P2_SOURCE",
    "assign_foreground_occupancy_targets",
    "load_occupancy_topk_head",
    "resolve_occupancy_topk_config",
    "stable_occupancy_topk",
]
