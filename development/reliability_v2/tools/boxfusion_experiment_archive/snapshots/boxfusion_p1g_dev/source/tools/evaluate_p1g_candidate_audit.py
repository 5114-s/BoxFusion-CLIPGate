#!/usr/bin/env python3
"""Offline, observer-only candidate audit for the frozen P1S -> P1G path.

This tool never trains a model and never writes BoxFusion predictions.  It
replays the frozen P1S proposal head from P1 collection diagnostics, performs
the original per-step and scene-level P1 NMS, and only then replaces geometry
for the exact same candidate IDs with frozen-P1S plus bounded P1G residual
correction output.

Ground truth is consumed only after inference, by the offline evaluator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.p1_geometry_refiner import (  # noqa: E402
    load_p1g_checkpoint,
)
from boxfusion.p1_geometry_loss import (  # noqa: E402
    decode_p1g_residual_aabb,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_NAMES,
    P1S_HEAD_SCHEMA,
    P1ResidualProposalObserver,
    ResidualObservation,
    ResidualProposalConfig,
    ResidualVoxelBatch,
    load_residual_proposal_head,
    resolve_residual_proposal_config,
    sha256_file,
)
from tools.report_p1_residual_recall import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    corners_to_minmax,
    evaluate_scene,
    load_axis_alignment,
    load_gt_boxes,
    load_predictions,
    pairwise_aabb_iou,
    read_scene_ids,
    score_ordered_match,
    transform_corners,
    validate_thresholds,
)


REPORT_SCHEMA = "boxfusion.p1g_candidate_audit.v2"
FROZEN_THRESHOLDS = tuple(DEFAULT_THRESHOLDS)
MAX_CANDIDATES_PER_SCENE = 256
MAX_REFINER_SECONDS_PER_SCENE = 0.15
MAX_REFINER_P95_SECONDS_PER_SCENE = 0.30
REFINABLE_MATCH_IOU = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
AUDIT_STAGE_SCENE_COUNTS = {
    "module20": 20,
    "fresh50": 50,
}

_EXPECTED_SOURCE_DECODE = {
    "score_threshold": 0.05,
    "pre_nms_topk": 512,
    "max_candidates_per_step": 64,
    "max_scene_candidates": MAX_CANDIDATES_PER_SCENE,
    "nms_iou": 0.25,
    "scene_nms_iou": 0.25,
}


@dataclass(frozen=True)
class SourceSnapshot:
    frame_id: int
    provider_step: int
    voxel_batch: ResidualVoxelBatch


@dataclass(frozen=True)
class SourceScene:
    scene_id: str
    config: ResidualProposalConfig
    snapshots: tuple[SourceSnapshot, ...]
    diagnostic_sha256: str


@dataclass(frozen=True)
class FrozenCandidates:
    candidate_ids: np.ndarray
    frame_ids: np.ndarray
    provider_steps: np.ndarray
    boxes_world: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.candidate_ids)
        frames = np.asarray(self.frame_ids)
        steps = np.asarray(self.provider_steps)
        boxes = np.asarray(self.boxes_world, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64)
        count = len(ids)
        if (
            ids.shape != (count,)
            or ids.dtype.hasobject
            or ids.dtype.kind not in {"U", "S", "i", "u"}
            or frames.shape != (count,)
            or not np.issubdtype(frames.dtype, np.integer)
            or steps.shape != (count,)
            or not np.issubdtype(steps.dtype, np.integer)
            or boxes.shape != (count, 6)
            or scores.shape != (count,)
        ):
            raise ValueError("candidate arrays do not share a strict row axis")
        if len(np.unique(ids)) != count:
            raise ValueError("candidate IDs must be unique")
        if (
            not np.isfinite(boxes).all()
            or (count and np.any(boxes[:, 3:] <= 0.0))
            or not np.isfinite(scores).all()
            or np.any((scores < 0.0) | (scores > 1.0))
        ):
            raise ValueError("candidate geometry/scores are invalid")
        object.__setattr__(self, "candidate_ids", np.array(ids, copy=True))
        object.__setattr__(
            self, "frame_ids", np.asarray(frames, dtype=np.int64)
        )
        object.__setattr__(
            self, "provider_steps", np.asarray(steps, dtype=np.int64)
        )
        object.__setattr__(
            self, "boxes_world", np.asarray(boxes, dtype=np.float64)
        )
        object.__setattr__(
            self, "scores", np.asarray(scores, dtype=np.float64)
        )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(
    archive: Mapping[str, np.ndarray],
    name: str,
    path: Path,
) -> Any:
    if name not in archive:
        raise ValueError(f"{path}: missing {name}")
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a non-object scalar")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return result


def _require_bool(
    archive: Mapping[str, np.ndarray],
    name: str,
    expected: bool,
    path: Path,
) -> None:
    value = _scalar(archive, name, path)
    if not isinstance(value, (bool, np.bool_)) or bool(value) is not expected:
        raise ValueError(f"{path}: {name} must equal {expected}")


def _load_source_scene(path: Path, scene_id: str) -> SourceScene:
    """Load one immutable P1 collection archive and validate every row axis."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive_object:
        archive = {
            name: np.array(archive_object[name], copy=True)
            for name in archive_object.files
        }

    if _scalar(archive, "scene_id", path) != scene_id:
        raise ValueError(f"{path}: scene_id disagrees with explicit scene list")
    if _scalar(archive, "p1_schema", path) != P1_DIAGNOSTIC_SCHEMA:
        raise ValueError(f"{path}: unsupported P1 diagnostic schema")
    if _scalar(archive, "p1_stage", path) != "P1":
        raise ValueError(f"{path}: source diagnostics must be frozen P1 collect")
    _require_bool(archive, "p1_enabled", True, path)
    _require_bool(archive, "p1_observer_only", True, path)
    _require_bool(archive, "p1_uses_ground_truth", False, path)
    _require_bool(archive, "p1_mutation_enabled", False, path)
    _require_bool(archive, "p1_complete", True, path)
    _require_bool(archive, "p1_class_agnostic", True, path)
    if int(_scalar(archive, "p1_applied_count", path)) != 0:
        raise ValueError(f"{path}: source P1 observer applied formal output")
    if int(_scalar(archive, "p1_regression_dim", path)) != 6:
        raise ValueError(f"{path}: source P1 regression_dim must equal six")

    feature_names = tuple(
        str(value) for value in np.asarray(archive.get("p1_feature_names"))
    )
    if feature_names != P1_FEATURE_NAMES:
        raise ValueError(f"{path}: source P1 feature schema mismatch")
    raw_config = _scalar(archive, "p1_config_json", path)
    if not isinstance(raw_config, str):
        raise TypeError(f"{path}: p1_config_json must be text")
    try:
        config_mapping = json.loads(raw_config)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed p1_config_json") from error
    if not isinstance(config_mapping, Mapping):
        raise ValueError(f"{path}: p1_config_json must contain an object")
    config = resolve_residual_proposal_config(config_mapping)
    if (
        config.mode != "collect"
        or not config.collect_diagnostics
        or not config.collect_voxel_inputs
        or not config.observer_only
        or config.mutate
    ):
        raise ValueError(f"{path}: source is not immutable P1 collect data")
    for name, expected in _EXPECTED_SOURCE_DECODE.items():
        observed = getattr(config, name)
        if isinstance(expected, float):
            matches = math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = int(observed) == expected
        if not matches:
            raise ValueError(
                f"{path}: frozen decoder field {name}={observed!r}, "
                f"expected {expected!r}"
            )

    step_frames = np.asarray(archive.get("p1_step_frame_ids"))
    step_providers = np.asarray(archive.get("p1_step_provider_steps"))
    step_voxel_counts = np.asarray(archive.get("p1_step_voxel_counts"))
    step_input_counts = np.asarray(
        archive.get("p1_step_input_point_counts")
    )
    step_explained_counts = np.asarray(
        archive.get("p1_step_explained_point_counts")
    )
    step_residual_counts = np.asarray(
        archive.get("p1_step_residual_point_counts")
    )
    if step_frames.ndim != 1 or not np.issubdtype(
        step_frames.dtype, np.integer
    ):
        raise ValueError(f"{path}: invalid p1_step_frame_ids")
    step_count = len(step_frames)
    for name, values in (
        ("p1_step_provider_steps", step_providers),
        ("p1_step_voxel_counts", step_voxel_counts),
        ("p1_step_input_point_counts", step_input_counts),
        ("p1_step_explained_point_counts", step_explained_counts),
        ("p1_step_residual_point_counts", step_residual_counts),
    ):
        if values.shape != (step_count,) or not np.issubdtype(
            values.dtype, np.integer
        ):
            raise ValueError(f"{path}: invalid {name}")
    if (
        len(np.unique(step_providers)) != step_count
        or np.any(step_voxel_counts < 0)
        or np.any(step_input_counts < 0)
        or np.any(step_explained_counts < 0)
        or np.any(step_residual_counts < 0)
    ):
        raise ValueError(f"{path}: invalid or duplicate P1 step metadata")

    offsets = np.asarray(archive.get("p1_voxel_offsets"))
    coordinates = np.asarray(archive.get("p1_voxel_coords"))
    centers = np.asarray(archive.get("p1_voxel_centers"))
    features = np.asarray(archive.get("p1_voxel_features"))
    point_counts = np.asarray(archive.get("p1_voxel_point_counts"))
    if (
        offsets.shape != (step_count + 1,)
        or not np.issubdtype(offsets.dtype, np.integer)
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"{path}: invalid p1_voxel_offsets")
    voxel_count = int(offsets[-1])
    if (
        coordinates.shape != (voxel_count, 3)
        or not np.issubdtype(coordinates.dtype, np.integer)
        or centers.shape != (voxel_count, 3)
        or features.shape != (voxel_count, len(P1_FEATURE_NAMES))
        or point_counts.shape != (voxel_count,)
        or not np.issubdtype(point_counts.dtype, np.integer)
        or not np.isfinite(centers).all()
        or not np.isfinite(features).all()
        or (voxel_count and np.any(point_counts <= 0))
    ):
        raise ValueError(f"{path}: invalid frozen voxel tensors")
    if not np.array_equal(np.diff(offsets), step_voxel_counts):
        raise ValueError(f"{path}: voxel offsets/counts disagree")

    snapshots: list[SourceSnapshot] = []
    for index in range(step_count):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        if end == start:
            batch = ResidualVoxelBatch.empty(
                input_point_count=int(step_input_counts[index]),
                explained_point_count=int(step_explained_counts[index]),
                residual_point_count=int(step_residual_counts[index]),
            )
        else:
            batch = ResidualVoxelBatch(
                coordinates=coordinates[start:end],
                centers=centers[start:end],
                features=features[start:end],
                point_counts=point_counts[start:end],
                input_point_count=int(step_input_counts[index]),
                explained_point_count=int(step_explained_counts[index]),
                residual_point_count=int(step_residual_counts[index]),
            )
        snapshots.append(
            SourceSnapshot(
                frame_id=int(step_frames[index]),
                provider_step=int(step_providers[index]),
                voxel_batch=batch,
            )
        )
    return SourceScene(
        scene_id=scene_id,
        config=config,
        snapshots=tuple(snapshots),
        diagnostic_sha256=_file_sha256(path),
    )


def _load_torch_mapping(path: Path, role: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old torch compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} checkpoint must contain a mapping")
    return payload


def _inference_config(
    source: ResidualProposalConfig,
    p1s_metadata: Mapping[str, Any],
    *,
    device: str,
) -> ResidualProposalConfig:
    if p1s_metadata.get("schema") != P1S_HEAD_SCHEMA:
        raise ValueError("P1S checkpoint schema mismatch")
    model_config = p1s_metadata.get("model_config")
    training_config = p1s_metadata.get("training_config")
    if not isinstance(model_config, Mapping) or not isinstance(
        training_config, Mapping
    ):
        raise ValueError("P1S checkpoint lacks model/training config")
    if model_config.get("architecture") != "native_sparse_context_v1":
        raise ValueError("candidate audit requires native sparse P1S")
    if (
        training_config.get("target_assignment_scope")
        != "snapshot_inside_only"
    ):
        raise ValueError("P1S target assignment scope mismatch")
    payload = source.to_dict()
    payload.update(
        {
            "mode": "infer",
            "checkpoint": None,
            "device": device,
            "head_architecture": "native_sparse_context_v1",
            "target_assignment_scope": "snapshot_inside_only",
            "hidden_dim": int(model_config.get("hidden_dim", -1)),
        }
    )
    return resolve_residual_proposal_config(payload)


def _synchronize(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.synchronize(parsed)


def _bounded_refined_box(
    center: np.ndarray,
    frozen_p1s_raw_regression: np.ndarray,
    residual_correction: np.ndarray,
    *,
    max_center_offset: float,
    min_box_extent: float,
    max_box_extent: float,
    adapter_epsilon: float,
) -> np.ndarray:
    """Decode one P1G v2 row through the shared residual contract."""

    anchor = np.asarray(center, dtype=np.float64)
    frozen = np.asarray(frozen_p1s_raw_regression, dtype=np.float64)
    correction = np.asarray(residual_correction, dtype=np.float64)
    if (
        anchor.shape != (3,)
        or frozen.shape != (6,)
        or correction.shape != (6,)
        or not np.isfinite(anchor).all()
        or not np.isfinite(frozen).all()
        or not np.isfinite(correction).all()
    ):
        raise ValueError(
            "P1G center/frozen/correction must be finite 3/6/6-vectors"
        )
    if (
        not math.isfinite(max_center_offset)
        or max_center_offset <= 0.0
        or not math.isfinite(min_box_extent)
        or min_box_extent <= 0.0
        or not math.isfinite(max_box_extent)
        or max_box_extent <= min_box_extent
        or not math.isfinite(adapter_epsilon)
        or adapter_epsilon <= 0.0
        or adapter_epsilon >= 0.5
    ):
        raise ValueError("P1G decoder bounds are invalid")
    with torch.inference_mode():
        decoded = decode_p1g_residual_aabb(
            torch.as_tensor(frozen.reshape(1, 6), dtype=torch.float64),
            torch.as_tensor(correction.reshape(1, 6), dtype=torch.float64),
            torch.as_tensor(anchor.reshape(1, 3), dtype=torch.float64),
            max_center_offset=max_center_offset,
            min_box_extent=min_box_extent,
            max_box_extent=max_box_extent,
            adapter_epsilon=adapter_epsilon,
        )
    result = decoded[0].cpu().numpy()
    if not np.isfinite(result).all() or np.any(result[3:] <= 0.0):
        raise ValueError("bounded P1G decoder produced invalid geometry")
    return result.astype(np.float64, copy=False)


def validate_candidate_identity(
    raw: FrozenCandidates, refined: FrozenCandidates
) -> dict[str, Any]:
    """Fail closed unless P1G changes geometry and geometry alone."""

    checks = {
        "count_equal": len(raw.candidate_ids) == len(
            refined.candidate_ids
        ),
        "ids_equal": np.array_equal(
            raw.candidate_ids, refined.candidate_ids
        ),
        "frame_ids_equal": np.array_equal(
            raw.frame_ids, refined.frame_ids
        ),
        "provider_steps_equal": np.array_equal(
            raw.provider_steps, refined.provider_steps
        ),
        "scores_equal": np.array_equal(raw.scores, refined.scores),
        "raw_ids_unique": len(np.unique(raw.candidate_ids))
        == len(raw.candidate_ids),
        "refined_ids_unique": len(np.unique(refined.candidate_ids))
        == len(refined.candidate_ids),
    }
    checks["passes"] = all(checks.values())
    if not checks["passes"]:
        failed = sorted(name for name, value in checks.items() if not value)
        raise ValueError(
            "P1G candidate identity contract failed: " + ", ".join(failed)
        )
    return checks


def _replay_scene(
    source: SourceScene,
    *,
    p1s_model: Any,
    p1g_model: Any,
    p1g_model_config: Mapping[str, Any],
    device: str,
) -> tuple[FrozenCandidates, FrozenCandidates, float]:
    """Regenerate raw P1S candidates, then replace only their geometry."""

    config = _inference_config(source.config, {
        "schema": P1S_HEAD_SCHEMA,
        "model_config": p1s_model.model_config(),
        "training_config": {
            "target_assignment_scope": "snapshot_inside_only"
        },
    }, device=device)
    observer = P1ResidualProposalObserver(
        config, head=p1s_model, device=device
    )
    observer.reset(source.scene_id)
    refined_per_step: dict[str, np.ndarray] = {}
    refiner_seconds = 0.0
    torch_device = torch.device(device)

    if not callable(getattr(p1s_model, "encode", None)):
        raise ValueError("P1S model lacks the frozen encode() contract")
    if int(getattr(p1s_model, "hidden_dim", -1)) != int(
        getattr(p1g_model, "hidden_dim", -2)
    ):
        raise ValueError("P1S/P1G hidden dimensions disagree")

    for snapshot in source.snapshots:
        batch = snapshot.voxel_batch
        proposals = ()
        if len(batch.centers):
            features = torch.as_tensor(
                np.array(batch.features, copy=True),
                dtype=torch.float32,
                device=torch_device,
            )
            coordinates = torch.as_tensor(
                np.array(batch.coordinates, copy=True),
                dtype=torch.int64,
                device=torch_device,
            )
            anchor_centers = torch.as_tensor(
                np.array(batch.centers, copy=True),
                dtype=torch.float32,
                device=torch_device,
            )
            with torch.inference_mode():
                encoded = p1s_model.encode(features, coordinates)
                raw_logits_tensor = p1s_model.objectness(encoded)
                raw_regression_tensor = p1s_model.regression(encoded)
                _synchronize(device)
                started = time.perf_counter()
                residual_correction_tensor = p1g_model(encoded)
                refined_boxes_tensor = decode_p1g_residual_aabb(
                    raw_regression_tensor,
                    residual_correction_tensor,
                    anchor_centers,
                    max_center_offset=float(
                        p1g_model_config["max_center_offset"]
                    ),
                    min_box_extent=float(
                        p1g_model_config["min_box_extent"]
                    ),
                    max_box_extent=float(
                        p1g_model_config["max_box_extent"]
                    ),
                    adapter_epsilon=float(
                        p1g_model_config["adapter_epsilon"]
                    ),
                )
                refined_boxes = refined_boxes_tensor.detach().cpu().numpy()
                _synchronize(device)
                refiner_seconds += time.perf_counter() - started
            raw_logits = raw_logits_tensor.detach().cpu().numpy()
            raw_regression = raw_regression_tensor.detach().cpu().numpy()
            proposals = observer.decode(
                batch,
                raw_logits,
                raw_regression,
                scene_id=source.scene_id,
                frame_index=snapshot.frame_id,
                provider_step=snapshot.provider_step,
            )
            id_to_voxel = {
                (
                    f"{source.scene_id}:{snapshot.provider_step:06d}:"
                    f"{int(coordinate[0])}:{int(coordinate[1])}:"
                    f"{int(coordinate[2])}"
                ): index
                for index, coordinate in enumerate(batch.coordinates)
            }
            decode_started = time.perf_counter()
            for proposal in proposals:
                if proposal.candidate_id not in id_to_voxel:
                    raise ValueError(
                        "raw P1S candidate cannot be mapped to frozen voxel"
                    )
                voxel_index = id_to_voxel[proposal.candidate_id]
                if proposal.candidate_id in refined_per_step:
                    raise ValueError("duplicate raw P1S candidate ID")
                refined_per_step[proposal.candidate_id] = np.asarray(
                    refined_boxes[voxel_index], dtype=np.float64
                )
            refiner_seconds += time.perf_counter() - decode_started
        observer.observations.append(
            ResidualObservation(
                frame_index=snapshot.frame_id,
                provider_step=snapshot.provider_step,
                voxel_batch=batch,
                proposals=tuple(proposals),
            )
        )
        del observer.observations[: -config.max_history_steps]

    raw_rows = observer.scene_candidates()
    if len(raw_rows) > MAX_CANDIDATES_PER_SCENE:
        raise ValueError("raw P1S scene candidate count exceeds Top-256")
    ids = np.asarray(
        [proposal.candidate_id for proposal in raw_rows], dtype=np.str_
    )
    frames = np.asarray(
        [proposal.frame_index for proposal in raw_rows], dtype=np.int64
    )
    steps = np.asarray(
        [proposal.provider_step for proposal in raw_rows], dtype=np.int64
    )
    raw_boxes = (
        np.stack([proposal.box for proposal in raw_rows])
        if raw_rows
        else np.empty((0, 6), dtype=np.float64)
    )
    scores = np.asarray(
        [proposal.objectness for proposal in raw_rows], dtype=np.float64
    )
    missing = [
        candidate_id
        for candidate_id in ids.tolist()
        if candidate_id not in refined_per_step
    ]
    if missing:
        raise ValueError(
            f"final P1S IDs lack P1G geometry: {missing[:3]}"
        )
    refined_boxes = (
        np.stack(
            [refined_per_step[candidate_id] for candidate_id in ids.tolist()]
        )
        if len(ids)
        else np.empty((0, 6), dtype=np.float64)
    )
    raw = FrozenCandidates(ids, frames, steps, raw_boxes, scores)
    refined = FrozenCandidates(
        ids.copy(), frames.copy(), steps.copy(), refined_boxes, scores.copy()
    )
    validate_candidate_identity(raw, refined)
    return raw, refined, float(refiner_seconds)


def novel_threshold_crossings(
    *,
    baseline_boxes: np.ndarray,
    baseline_scores: np.ndarray,
    raw_boxes: np.ndarray,
    refined_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_ids: np.ndarray | None = None,
    gt_boxes: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Count one-to-one AP threshold crossings on B6-missed GT."""

    baseline_iou = pairwise_aabb_iou(baseline_boxes, gt_boxes)
    raw_iou = pairwise_aabb_iou(raw_boxes, gt_boxes)
    refined_iou = pairwise_aabb_iou(refined_boxes, gt_boxes)
    baseline_match = score_ordered_match(
        baseline_iou, baseline_scores, threshold
    )
    novel_mask = np.ones(len(gt_boxes), dtype=np.bool_)
    novel_mask[baseline_match.matched_gt] = False
    raw_match = score_ordered_match(
        raw_iou,
        candidate_scores,
        threshold,
        allowed_gt=novel_mask,
        tie_break_ids=candidate_ids,
    )
    refined_match = score_ordered_match(
        refined_iou,
        candidate_scores,
        threshold,
        allowed_gt=novel_mask,
        tie_break_ids=candidate_ids,
    )
    raw_matched = set(int(index) for index in raw_match.matched_gt.tolist())
    refined_matched = set(
        int(index) for index in refined_match.matched_gt.tolist()
    )
    up = len(refined_matched - raw_matched)
    down = len(raw_matched - refined_matched)
    return {
        "up": int(up),
        "down": int(down),
        "net": int(up - down),
        "raw_score_ordered_novel_tp": raw_match.true_positive_count,
        "refined_score_ordered_novel_tp": (
            refined_match.true_positive_count
        ),
    }


def refinable_iou_quality(
    *,
    raw_boxes: np.ndarray,
    refined_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_ids: np.ndarray,
    gt_boxes: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure paired IoU change on fixed raw-candidate/GT matches.

    Matching is score-ordered and one-to-one on the raw geometry at the
    protocol's minimum refinable overlap (IoU >= 0.05).  The same candidate
    identity and GT row are then used to read the refined IoU.
    """

    raw_iou = pairwise_aabb_iou(raw_boxes, gt_boxes)
    refined_iou = pairwise_aabb_iou(refined_boxes, gt_boxes)
    match = score_ordered_match(
        raw_iou,
        candidate_scores,
        np.nextafter(REFINABLE_MATCH_IOU, -np.inf),
        tie_break_ids=candidate_ids,
    )
    candidate_indices = np.flatnonzero(match.prediction_to_gt >= 0)
    if len(candidate_indices):
        gt_indices = match.prediction_to_gt[candidate_indices]
        raw_values = raw_iou[candidate_indices, gt_indices]
        refined_values = refined_iou[candidate_indices, gt_indices]
        deltas = np.asarray(
            refined_values - raw_values, dtype=np.float64
        )
        median = float(np.median(deltas))
        harm_rate = float(np.mean(deltas <= -0.05))
    else:
        deltas = np.empty((0,), dtype=np.float64)
        median = None
        harm_rate = None
    return (
        {
            "matching": (
                "stable score-descending one-to-one raw candidate/GT at "
                "IoU >= 0.05; paired refined IoU uses the identical "
                "candidate ID and GT"
            ),
            "matched_count": int(len(deltas)),
            "median_delta_iou": median,
            "harm_count": int(np.sum(deltas <= -0.05)),
            "harm_rate": harm_rate,
        },
        deltas,
    )


def scene_bootstrap_novel_recall_delta(
    scene_rows: Sequence[Mapping[str, int]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Deterministic scene bootstrap CI for micro novel-recall delta."""

    if not scene_rows:
        raise ValueError("scene bootstrap requires at least one scene")
    if int(resamples) <= 0:
        raise ValueError("bootstrap resamples must be positive")
    gt = np.asarray(
        [int(row["ground_truth_count"]) for row in scene_rows],
        dtype=np.int64,
    )
    raw = np.asarray(
        [int(row["raw_novel_true_positives"]) for row in scene_rows],
        dtype=np.int64,
    )
    refined = np.asarray(
        [int(row["refined_novel_true_positives"]) for row in scene_rows],
        dtype=np.int64,
    )
    if (
        np.any(gt < 0)
        or np.any(raw < 0)
        or np.any(refined < 0)
        or int(np.sum(gt)) <= 0
    ):
        raise ValueError("scene bootstrap counts are invalid")
    generator = np.random.default_rng(int(seed))
    sampled = generator.integers(
        0, len(scene_rows), size=(int(resamples), len(scene_rows))
    )
    sampled_gt = np.sum(gt[sampled], axis=1)
    if np.any(sampled_gt <= 0):
        raise ValueError("scene bootstrap sampled a zero-GT audit")
    deltas = np.sum((refined - raw)[sampled], axis=1) / sampled_gt
    return {
        "method": (
            "scene bootstrap with replacement; micro novel TP delta "
            "divided by sampled GT count"
        ),
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence": 0.95,
        "point_estimate": float(
            (np.sum(refined) - np.sum(raw)) / np.sum(gt)
        ),
        "lower": float(np.quantile(deltas, 0.025)),
        "upper": float(np.quantile(deltas, 0.975)),
    }


def _candidate_corners(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.shape != (len(values), 6):
        raise ValueError("candidate boxes must have shape [N,6]")
    if not len(values):
        return np.empty((0, 8, 3), dtype=np.float64)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return values[:, None, :3] + (
        0.5 * values[:, None, 3:] * signs[None]
    )


def _checkpoint_source_summaries(
    p1s_payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    provenance = p1s_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("P1S checkpoint lacks provenance")
    summaries = provenance.get("scene_summaries")
    if not isinstance(summaries, Sequence) or isinstance(
        summaries, (str, bytes)
    ):
        raise ValueError("P1S checkpoint lacks source scene summaries")
    result: dict[str, Mapping[str, Any]] = {}
    for row in summaries:
        if not isinstance(row, Mapping):
            raise ValueError("invalid P1S source scene summary")
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or scene_id in result:
            raise ValueError("duplicate/invalid P1S source summary")
        result[scene_id] = row
    return result


def _provenance_scene_ids(provenance: Mapping[str, Any]) -> set[str]:
    """Collect every explicitly enumerated scene ID from provenance."""

    result: set[str] = set()
    for name, value in provenance.items():
        if not name.endswith("_scene_ids"):
            continue
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes)
        ):
            raise ValueError(f"provenance {name} must be a scene sequence")
        rows = tuple(str(scene_id) for scene_id in value)
        if any(not scene_id for scene_id in rows) or len(set(rows)) != len(
            rows
        ):
            raise ValueError(f"provenance {name} contains invalid scene IDs")
        result.update(rows)
    summaries = provenance.get("scene_summaries", ())
    if not isinstance(summaries, Sequence) or isinstance(
        summaries, (str, bytes)
    ):
        raise ValueError("provenance scene_summaries must be a sequence")
    for row in summaries:
        if not isinstance(row, Mapping):
            raise ValueError("provenance scene summary must be a mapping")
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("provenance scene summary has invalid scene_id")
        result.add(scene_id)
    return result


def validate_stage_scene_binding(
    *,
    stage: str,
    scenes: Sequence[str],
    scene_list_sha256: str,
    p1s_provenance: Mapping[str, Any],
    p1g_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind module20 to its checkpoint split and keep fresh50 genuinely fresh."""

    if stage not in AUDIT_STAGE_SCENE_COUNTS:
        raise ValueError(
            "audit stage must be one of: "
            + ", ".join(AUDIT_STAGE_SCENE_COUNTS)
        )
    scene_rows = tuple(str(scene_id) for scene_id in scenes)
    if not scene_rows or len(set(scene_rows)) != len(scene_rows):
        raise ValueError("audit scenes must be non-empty and unique")
    if stage == "module20":
        checkpoint_scenes = tuple(
            str(scene_id)
            for scene_id in p1g_provenance.get("audit_scene_ids", ())
        )
        if set(checkpoint_scenes) != set(scene_rows) or len(
            checkpoint_scenes
        ) != len(scene_rows):
            raise ValueError(
                "module20 scene list disagrees with P1G audit provenance"
            )
        if (
            p1g_provenance.get("audit_scene_list_sha256")
            != scene_list_sha256
        ):
            raise ValueError(
                "module20 scene-list SHA disagrees with P1G checkpoint"
            )
        return {
            "status": "p1g_checkpoint_exact_module20_binding",
            "scene_list_sha256": scene_list_sha256,
            "overlap_with_frozen_provenance": [],
        }

    # A P1G checkpoint is frozen before fresh50 and therefore binds its
    # module20 split, not the later external fresh50 list.  Freshness is
    # checked against every enumerated upstream/P1G development scene and the
    # exact external list hash is inventoried in the immutable report.
    p1s_seen = _provenance_scene_ids(p1s_provenance)
    p1g_seen: set[str] = set()
    for name in ("fit_scene_ids", "cal_scene_ids", "audit_scene_ids"):
        values = p1g_provenance.get(name, ())
        if not isinstance(values, Sequence) or isinstance(
            values, (str, bytes)
        ):
            raise ValueError(f"P1G provenance {name} must be a sequence")
        p1g_seen.update(str(scene_id) for scene_id in values)
    overlap = sorted(set(scene_rows) & (p1s_seen | p1g_seen))
    if overlap:
        raise ValueError(
            "fresh50 overlaps frozen P1S/P1G provenance: " + overlap[0]
        )
    return {
        "status": "fresh50_external_hash_and_disjoint_provenance",
        "scene_list_sha256": scene_list_sha256,
        "overlap_with_frozen_provenance": overlap,
        "p1s_provenance_scene_count": int(len(p1s_seen)),
        "p1g_development_scene_count": int(len(p1g_seen)),
    }


def _verify_source_binding(
    *,
    stage: str,
    scene_id: str,
    source: SourceScene,
    prediction_path: Path,
    gt_path: Path,
    alignment_path: Path,
    source_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    observed = {
        "diagnostic_sha256": source.diagnostic_sha256,
        "prediction_sha256": _file_sha256(prediction_path),
        "ground_truth_sha256": _file_sha256(gt_path),
        "axis_alignment_sha256": _file_sha256(alignment_path),
    }
    frozen = source_summaries.get(scene_id)
    if frozen is not None:
        for name, value in observed.items():
            if frozen.get(name) != value:
                raise ValueError(
                    f"{scene_id}: frozen P1 source {name} mismatch"
                )
        status = "p1s_checkpoint_exact_source_binding"
    else:
        if stage != "fresh50":
            raise ValueError(
                f"{scene_id}: module20 lacks frozen P1S source provenance"
            )
        # Fresh train-only audit scenes cannot appear in the P1S checkpoint.
        # Their immutable source hashes are inventoried in this report.
        status = "fresh_audit_hash_inventory"
    return {"status": status, **observed}


def build_go_no_go(
    *,
    stage: str,
    thresholds: Mapping[str, Mapping[str, Any]],
    total_gt: int,
    scene_count: int,
    refinement_quality: Mapping[str, Any],
    bootstrap_ci: Mapping[str, Any] | None,
    safety_identity: bool,
    candidates_bounded: bool,
    refiner_seconds_per_scene: float,
    refiner_p95_seconds_per_scene: float,
    maximum_refiner_seconds_per_scene: float = (
        MAX_REFINER_SECONDS_PER_SCENE
    ),
    maximum_refiner_p95_seconds_per_scene: float = (
        MAX_REFINER_P95_SECONDS_PER_SCENE
    ),
) -> dict[str, Any]:
    """Apply the stage-specific, pre-registered train-only P1G gate."""

    if stage not in AUDIT_STAGE_SCENE_COUNTS:
        raise ValueError(
            "audit stage must be one of: "
            + ", ".join(AUDIT_STAGE_SCENE_COUNTS)
        )
    if total_gt <= 0:
        raise ValueError("P1G audit requires at least one GT instance")
    for key in ("0.25", "0.50"):
        if key not in thresholds:
            raise ValueError(f"threshold report lacks {key}")
    row25 = thresholds["0.25"]
    row50 = thresholds["0.50"]
    delta_r50 = float(row50["delta_novel_recall"])
    r25_not_degraded = bool(
        float(row25["refined"]["novel_recall"])
        >= float(row25["raw"]["novel_recall"])
    )
    expected_scene_count = AUDIT_STAGE_SCENE_COUNTS[stage]
    scene_completeness_passes = int(scene_count) == expected_scene_count
    delta_novel_tp50 = int(row50["delta_novel_true_positives"])
    up_crossings = int(row50["crossings"]["up"])
    down_crossings = int(row50["crossings"]["down"])
    net_crossings = int(row50["crossings"]["net"])
    if net_crossings != up_crossings - down_crossings:
        raise ValueError("AP50 crossing counts are internally inconsistent")
    matched_count = int(refinement_quality.get("matched_count", 0))
    median_raw = refinement_quality.get("median_delta_iou")
    harm_raw = refinement_quality.get("harm_rate")
    median_delta_iou = (
        float(median_raw) if median_raw is not None else float("nan")
    )
    harm_rate = float(harm_raw) if harm_raw is not None else float("nan")
    mean_runtime_passes = bool(
        math.isfinite(refiner_seconds_per_scene)
        and 0.0 <= refiner_seconds_per_scene
        <= maximum_refiner_seconds_per_scene
    )
    p95_runtime_passes = bool(
        math.isfinite(refiner_p95_seconds_per_scene)
        and 0.0 <= refiner_p95_seconds_per_scene
        <= maximum_refiner_p95_seconds_per_scene
    )
    runtime_passes = mean_runtime_passes and p95_runtime_passes

    if stage == "module20":
        delta_novel_tp_passes = delta_novel_tp50 >= 2
        up_down_passes = up_crossings > down_crossings
        median_delta_iou_passes = bool(
            matched_count > 0
            and math.isfinite(median_delta_iou)
            and median_delta_iou >= 0.02
        )
        harm_rate_passes = bool(
            matched_count > 0
            and math.isfinite(harm_rate)
            and harm_rate <= 0.12
        )
        stage_metric_passes = bool(
            delta_novel_tp_passes
            and r25_not_degraded
            and up_down_passes
            and median_delta_iou_passes
            and harm_rate_passes
        )
        stage_checks = {
            "required_delta_novel_true_positives_at_0p50": 2,
            "observed_delta_novel_true_positives_at_0p50": (
                delta_novel_tp50
            ),
            "delta_novel_true_positives_at_0p50_passes": (
                delta_novel_tp_passes
            ),
            "ap50_up_strictly_greater_than_down": up_down_passes,
            "required_median_delta_iou": 0.02,
            "required_maximum_harm_rate": 0.12,
        }
        decision = (
            "GO_FRESH50_AUDIT"
            if (
                scene_completeness_passes
                and safety_identity
                and candidates_bounded
                and stage_metric_passes
                and runtime_passes
            )
            else "STOP_P1G1_MODULE_AUDIT"
        )
    else:
        required_crossings = max(5, int(math.ceil(0.01 * total_gt)))
        delta_r50_passes = bool(delta_r50 >= 0.01)
        crossing_passes = bool(net_crossings >= required_crossings)
        crossing_ratio_passes = bool(
            up_crossings >= 2 * down_crossings
        )
        bootstrap_lower = (
            float(bootstrap_ci["lower"])
            if bootstrap_ci is not None and "lower" in bootstrap_ci
            else float("nan")
        )
        bootstrap_passes = bool(
            math.isfinite(bootstrap_lower) and bootstrap_lower > 0.0
        )
        median_delta_iou_passes = bool(
            matched_count > 0
            and math.isfinite(median_delta_iou)
            and median_delta_iou >= 0.03
        )
        harm_rate_passes = bool(
            matched_count > 0
            and math.isfinite(harm_rate)
            and harm_rate <= 0.10
        )
        stage_metric_passes = bool(
            delta_r50_passes
            and bootstrap_passes
            and r25_not_degraded
            and crossing_passes
            and crossing_ratio_passes
            and median_delta_iou_passes
            and harm_rate_passes
        )
        stage_checks = {
            "required_delta_novel_recall_at_0p50": 0.01,
            "observed_delta_novel_recall_at_0p50": delta_r50,
            "delta_recall_at_0p50_passes": delta_r50_passes,
            "bootstrap_ci95_lower": bootstrap_lower,
            "bootstrap_ci95_lower_positive": bootstrap_passes,
            "required_net_ap50_crossings": required_crossings,
            "net_ap50_crossings_pass": crossing_passes,
            "ap50_up_at_least_twice_down": crossing_ratio_passes,
            "required_median_delta_iou": 0.03,
            "required_maximum_harm_rate": 0.10,
        }
        decision = (
            "GO_ONE_SHOT_VAL10_OBSERVER"
            if (
                scene_completeness_passes
                and safety_identity
                and candidates_bounded
                and stage_metric_passes
                and runtime_passes
            )
            else "STOP_P1G1"
        )

    passes = bool(
        scene_completeness_passes
        and safety_identity
        and candidates_bounded
        and stage_metric_passes
        and runtime_passes
    )
    return {
        "stage": stage,
        "expected_scene_count": expected_scene_count,
        "observed_scene_count": int(scene_count),
        "scene_completeness_passes": scene_completeness_passes,
        **stage_checks,
        "novel_recall_at_0p25_not_degraded": r25_not_degraded,
        "observed_ap50_up_crossings": up_crossings,
        "observed_ap50_down_crossings": down_crossings,
        "observed_net_ap50_crossings": net_crossings,
        "refinable_matched_count": matched_count,
        "observed_median_delta_iou": (
            median_delta_iou if math.isfinite(median_delta_iou) else None
        ),
        "median_delta_iou_passes": median_delta_iou_passes,
        "observed_harm_rate": (
            harm_rate if math.isfinite(harm_rate) else None
        ),
        "harm_rate_passes": harm_rate_passes,
        "safety_identity": bool(safety_identity),
        "candidates_bounded": bool(candidates_bounded),
        "runtime_scope": "correction_forward_decode_only",
        "full_live_runtime_verified": False,
        "maximum_mean_refiner_seconds_per_scene": float(
            maximum_refiner_seconds_per_scene
        ),
        "observed_mean_refiner_seconds_per_scene": float(
            refiner_seconds_per_scene
        ),
        "mean_runtime_passes": mean_runtime_passes,
        "maximum_p95_refiner_seconds_per_scene": float(
            maximum_refiner_p95_seconds_per_scene
        ),
        "observed_p95_refiner_seconds_per_scene": float(
            refiner_p95_seconds_per_scene
        ),
        "p95_runtime_passes": p95_runtime_passes,
        "runtime_passes": runtime_passes,
        "passes": passes,
        "decision": decision,
        "frozen_protocol": (
            "This audit performs no training. module20 can only authorize "
            "fresh50; fresh50 can only authorize the one-shot val10 "
            "observer. Neither stage authorizes full100 or active output."
        ),
    }


def evaluate(
    *,
    stage: str,
    scene_list: Path,
    p1s_checkpoint: Path,
    p1g_checkpoint: Path,
    source_diagnostics_root: Path,
    prediction_root: Path,
    gt_root: Path,
    scans_root: Path,
    device: str = "cpu",
    maximum_refiner_seconds_per_scene: float = (
        MAX_REFINER_SECONDS_PER_SCENE
    ),
    maximum_refiner_p95_seconds_per_scene: float = (
        MAX_REFINER_P95_SECONDS_PER_SCENE
    ),
) -> dict[str, Any]:
    """Run the complete read-only P1G candidate audit."""

    if stage not in AUDIT_STAGE_SCENE_COUNTS:
        raise ValueError(
            "audit stage must be one of: "
            + ", ".join(AUDIT_STAGE_SCENE_COUNTS)
        )
    scenes = read_scene_ids(scene_list)
    scene_list_sha = _file_sha256(scene_list)
    thresholds = validate_thresholds(FROZEN_THRESHOLDS)
    for role, root in (
        ("source diagnostics", source_diagnostics_root),
        ("source predictions", prediction_root),
        ("ground truth", gt_root),
        ("scans", scans_root),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    if (
        not math.isfinite(maximum_refiner_seconds_per_scene)
        or maximum_refiner_seconds_per_scene <= 0.0
    ):
        raise ValueError("maximum refiner runtime must be positive")
    if (
        not math.isfinite(maximum_refiner_p95_seconds_per_scene)
        or maximum_refiner_p95_seconds_per_scene <= 0.0
    ):
        raise ValueError("maximum p95 refiner runtime must be positive")
    parsed_device = torch.device(device)
    if parsed_device.type not in {"cpu", "cuda"}:
        raise ValueError("audit device must be cpu or cuda")
    if parsed_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but CUDA is unavailable")

    first_source = _load_source_scene(
        source_diagnostics_root / f"{scenes[0]}_tracks.npz",
        scenes[0],
    )
    p1s_payload = _load_torch_mapping(p1s_checkpoint, "P1S")
    p1s_config = _inference_config(
        first_source.config, p1s_payload, device=str(parsed_device)
    )
    p1s_provenance = p1s_payload.get("provenance")
    if not isinstance(p1s_provenance, Mapping):
        raise ValueError("P1S checkpoint lacks train-only provenance")
    expected_b6_sha = str(
        p1s_provenance.get("b6_checkpoint_sha256", "")
    ).lower()
    p1s_model, p1s_sha, loaded_p1s_payload = (
        load_residual_proposal_head(
            p1s_checkpoint,
            expected_config=p1s_config,
            device=str(parsed_device),
            expected_b6_checkpoint_sha256=expected_b6_sha,
        )
    )
    if p1s_sha != sha256_file(p1s_checkpoint):
        raise RuntimeError("P1S checkpoint SHA changed during load")
    p1g_model, p1g_payload, p1g_sha = load_p1g_checkpoint(
        p1g_checkpoint,
        expected_p1s_checkpoint_sha256=p1s_sha,
        device=parsed_device,
    )
    p1g_provenance = p1g_payload.get("provenance")
    if not isinstance(p1g_provenance, Mapping):
        raise ValueError("P1G checkpoint lacks provenance")
    stage_binding = validate_stage_scene_binding(
        stage=stage,
        scenes=scenes,
        scene_list_sha256=scene_list_sha,
        p1s_provenance=p1s_provenance,
        p1g_provenance=p1g_provenance,
    )
    p1g_model_config = p1g_payload.get("model_config")
    if not isinstance(p1g_model_config, Mapping):
        raise ValueError("P1G checkpoint lacks model_config")
    # Function preservation is defined relative to the exact P1S clip/exp
    # bounds.  Refuse to replay if the residual checkpoint was trained with a
    # different parameterization.
    for name, source_value in (
        ("max_center_offset", first_source.config.max_center_offset),
        ("min_box_extent", first_source.config.min_box_extent),
        ("max_box_extent", first_source.config.max_box_extent),
    ):
        try:
            checkpoint_value = float(p1g_model_config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"P1G model_config has invalid {name}"
            ) from error
        if not math.isclose(
            checkpoint_value,
            float(source_value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"P1G {name} differs from frozen P1S decoder"
            )
    source_summaries = _checkpoint_source_summaries(
        loaded_p1s_payload
    )

    totals = {
        f"{threshold:.2f}": {
            "raw": {
                "b6_true_positives": 0,
                "p1_true_positives": 0,
                "union_true_positives": 0,
                "novel_true_positives": 0,
            },
            "refined": {
                "b6_true_positives": 0,
                "p1_true_positives": 0,
                "union_true_positives": 0,
                "novel_true_positives": 0,
            },
            "crossings": {"up": 0, "down": 0, "net": 0},
        }
        for threshold in thresholds
    }
    total_gt = 0
    total_baseline = 0
    total_candidates = 0
    total_refiner_seconds = 0.0
    refiner_scene_seconds: list[float] = []
    quality_delta_rows: list[np.ndarray] = []
    bootstrap_scene_rows: list[dict[str, int]] = []
    per_scene: dict[str, Any] = {}
    all_identity_pass = True
    candidates_bounded = True
    common_config = first_source.config.to_dict()

    for scene_index, scene_id in enumerate(scenes):
        source = (
            first_source
            if scene_index == 0
            else _load_source_scene(
                source_diagnostics_root / f"{scene_id}_tracks.npz",
                scene_id,
            )
        )
        if source.config.to_dict() != common_config:
            raise ValueError(
                f"{scene_id}: source P1 decoder config differs across audit"
            )
        prediction_path = prediction_root / f"{scene_id}_boxes.pkl"
        gt_path = gt_root / f"{scene_id}_bbox.npy"
        alignment_path = scans_root / scene_id / f"{scene_id}.txt"
        binding = _verify_source_binding(
            stage=stage,
            scene_id=scene_id,
            source=source,
            prediction_path=prediction_path,
            gt_path=gt_path,
            alignment_path=alignment_path,
            source_summaries=source_summaries,
        )
        baseline = load_predictions(prediction_path)
        alignment = load_axis_alignment(scans_root, scene_id)
        baseline_boxes = corners_to_minmax(
            transform_corners(baseline.corners_world, alignment)
        )
        gt_boxes = load_gt_boxes(gt_path)
        raw, refined, refiner_seconds = _replay_scene(
            source,
            p1s_model=p1s_model,
            p1g_model=p1g_model,
            p1g_model_config=p1g_model_config,
            device=str(parsed_device),
        )
        identity = validate_candidate_identity(raw, refined)
        all_identity_pass = all_identity_pass and bool(identity["passes"])
        scene_bounded = len(raw.candidate_ids) <= (
            MAX_CANDIDATES_PER_SCENE
        )
        candidates_bounded = candidates_bounded and scene_bounded
        raw_boxes = corners_to_minmax(
            transform_corners(_candidate_corners(raw.boxes_world), alignment)
        )
        refined_boxes = corners_to_minmax(
            transform_corners(
                _candidate_corners(refined.boxes_world), alignment
            )
        )
        scene_quality, scene_quality_deltas = refinable_iou_quality(
            raw_boxes=raw_boxes,
            refined_boxes=refined_boxes,
            candidate_scores=raw.scores,
            candidate_ids=raw.candidate_ids,
            gt_boxes=gt_boxes,
        )
        quality_delta_rows.append(scene_quality_deltas)
        raw_report = evaluate_scene(
            baseline_boxes=baseline_boxes,
            baseline_scores=baseline.scores,
            candidate_boxes=raw_boxes,
            candidate_scores=raw.scores,
            gt_boxes=gt_boxes,
            thresholds=thresholds,
            candidate_ids=raw.candidate_ids,
        )
        refined_report = evaluate_scene(
            baseline_boxes=baseline_boxes,
            baseline_scores=baseline.scores,
            candidate_boxes=refined_boxes,
            candidate_scores=refined.scores,
            gt_boxes=gt_boxes,
            thresholds=thresholds,
            candidate_ids=refined.candidate_ids,
        )
        scene_thresholds: dict[str, Any] = {}
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            raw_row = raw_report["thresholds"][key]
            refined_row = refined_report["thresholds"][key]
            if (
                raw_row["b6_true_positives"]
                != refined_row["b6_true_positives"]
            ):
                raise RuntimeError("B6 matching changed across geometry audit")
            crossing = novel_threshold_crossings(
                baseline_boxes=baseline_boxes,
                baseline_scores=baseline.scores,
                raw_boxes=raw_boxes,
                refined_boxes=refined_boxes,
                candidate_scores=raw.scores,
                candidate_ids=raw.candidate_ids,
                gt_boxes=gt_boxes,
                threshold=threshold,
            )
            observed_delta = int(
                refined_row["novel_true_positives"]
                - raw_row["novel_true_positives"]
            )
            if crossing["net"] != observed_delta:
                raise RuntimeError(
                    "one-to-one AP crossing net differs from novel TP delta"
                )
            scene_thresholds[key] = {
                "raw": raw_row,
                "refined": refined_row,
                "delta_novel_true_positives": observed_delta,
                "crossings": crossing,
            }
            for branch, row in (
                ("raw", raw_row),
                ("refined", refined_row),
            ):
                for name in totals[key][branch]:
                    totals[key][branch][name] += int(row[name])
            for name in ("up", "down", "net"):
                totals[key]["crossings"][name] += int(crossing[name])

        total_gt += len(gt_boxes)
        total_baseline += len(baseline_boxes)
        total_candidates += len(raw.candidate_ids)
        total_refiner_seconds += refiner_seconds
        refiner_scene_seconds.append(float(refiner_seconds))
        row50 = scene_thresholds["0.50"]
        bootstrap_scene_rows.append(
            {
                "ground_truth_count": int(len(gt_boxes)),
                "raw_novel_true_positives": int(
                    row50["raw"]["novel_true_positives"]
                ),
                "refined_novel_true_positives": int(
                    row50["refined"]["novel_true_positives"]
                ),
            }
        )
        per_scene[scene_id] = {
            "ground_truth_count": int(len(gt_boxes)),
            "baseline_prediction_count": int(len(baseline_boxes)),
            "candidate_count": int(len(raw.candidate_ids)),
            "candidate_identity": identity,
            "candidates_bounded": scene_bounded,
            "refiner_seconds": float(refiner_seconds),
            "refinement_quality": scene_quality,
            "source_binding": binding,
            "thresholds": scene_thresholds,
        }

    if total_gt <= 0:
        raise ValueError("P1G audit scene set contains no GT instances")
    threshold_report: dict[str, Any] = {}
    for key, row in totals.items():
        branch_report: dict[str, Any] = {}
        for branch in ("raw", "refined"):
            values = row[branch]
            branch_report[branch] = {
                **values,
                "b6_recall": float(
                    values["b6_true_positives"] / total_gt
                ),
                "p1_recall": float(
                    values["p1_true_positives"] / total_gt
                ),
                "union_recall": float(
                    values["union_true_positives"] / total_gt
                ),
                "novel_recall": float(
                    values["novel_true_positives"] / total_gt
                ),
            }
        threshold_report[key] = {
            **branch_report,
            "delta_union_recall": float(
                branch_report["refined"]["union_recall"]
                - branch_report["raw"]["union_recall"]
            ),
            "delta_novel_recall": float(
                branch_report["refined"]["novel_recall"]
                - branch_report["raw"]["novel_recall"]
            ),
            "delta_novel_true_positives": int(
                branch_report["refined"]["novel_true_positives"]
                - branch_report["raw"]["novel_true_positives"]
            ),
            "crossings": row["crossings"],
        }

    runtime_per_scene = float(
        total_refiner_seconds / max(len(scenes), 1)
    )
    runtime_p95_per_scene = float(
        np.quantile(np.asarray(refiner_scene_seconds), 0.95)
    )
    quality_deltas = (
        np.concatenate(quality_delta_rows)
        if quality_delta_rows
        else np.empty((0,), dtype=np.float64)
    )
    refinement_quality = {
        "matching": (
            "stable score-descending one-to-one raw candidate/GT at "
            "IoU >= 0.05; paired refined IoU uses the identical "
            "candidate ID and GT"
        ),
        "matched_count": int(len(quality_deltas)),
        "median_delta_iou": (
            float(np.median(quality_deltas))
            if len(quality_deltas)
            else None
        ),
        "harm_count": int(np.sum(quality_deltas <= -0.05)),
        "harm_rate": (
            float(np.mean(quality_deltas <= -0.05))
            if len(quality_deltas)
            else None
        ),
    }
    bootstrap_ci = scene_bootstrap_novel_recall_delta(
        bootstrap_scene_rows
    )
    safety_identity = bool(
        all_identity_pass
        and p1g_payload.get("observer_only") is True
        and p1g_payload.get("uses_ground_truth") is False
        and p1g_payload.get("class_agnostic") is True
        and p1g_payload.get("semantic_features") is False
    )
    gate = build_go_no_go(
        stage=stage,
        thresholds=threshold_report,
        total_gt=total_gt,
        scene_count=len(scenes),
        refinement_quality=refinement_quality,
        bootstrap_ci=bootstrap_ci,
        safety_identity=safety_identity,
        candidates_bounded=candidates_bounded,
        refiner_seconds_per_scene=runtime_per_scene,
        refiner_p95_seconds_per_scene=runtime_p95_per_scene,
        maximum_refiner_seconds_per_scene=(
            maximum_refiner_seconds_per_scene
        ),
        maximum_refiner_p95_seconds_per_scene=(
            maximum_refiner_p95_seconds_per_scene
        ),
    )
    return {
        "schema": REPORT_SCHEMA,
        "stage": stage,
        "training_performed": False,
        "observer_only": True,
        "uses_ground_truth_during_inference": False,
        "runtime_scope": "correction_forward_decode_only",
        "full_live_runtime_verified": False,
        "full_online_activation_authorized": False,
        "matching_contract": (
            "class-agnostic, strict IoU > threshold, stable "
            "score-descending one-to-one; GT used only after inference"
        ),
        "candidate_contract": (
            "P1S raw logits/regression -> frozen per-step NMS -> frozen "
            "scene NMS/Top256 -> frozen P1S raw geometry plus P1G "
            "residual correction for identical IDs"
        ),
        "scene_list": str(scene_list.resolve()),
        "scene_list_sha256": scene_list_sha,
        "stage_scene_binding": stage_binding,
        "scene_count": int(len(scenes)),
        "ground_truth_count": int(total_gt),
        "baseline_prediction_count": int(total_baseline),
        "candidate_count": int(total_candidates),
        "candidates_per_scene": float(
            total_candidates / max(len(scenes), 1)
        ),
        "candidate_identity": {
            "passes": all_identity_pass,
            "same_ids_scores_order_count": all_identity_pass,
            "maximum_candidates_per_scene": MAX_CANDIDATES_PER_SCENE,
            "candidates_bounded": candidates_bounded,
        },
        "refiner_runtime": {
            "runtime_scope": "correction_forward_decode_only",
            "full_live_runtime_verified": False,
            "seconds": float(total_refiner_seconds),
            "mean_seconds_per_scene": runtime_per_scene,
            "p95_seconds_per_scene": runtime_p95_per_scene,
            "maximum_mean_seconds_per_scene": float(
                maximum_refiner_seconds_per_scene
            ),
            "maximum_p95_seconds_per_scene": float(
                maximum_refiner_p95_seconds_per_scene
            ),
        },
        "checkpoints": {
            "p1s": {
                "path": str(p1s_checkpoint.resolve()),
                "sha256": p1s_sha,
            },
            "p1g": {
                "path": str(p1g_checkpoint.resolve()),
                "sha256": p1g_sha,
                "bound_p1s_sha256": p1s_sha,
            },
        },
        "frozen_decoder": {
            **_EXPECTED_SOURCE_DECODE,
            "candidate_order": (
                "stable P1S score/id order before P1G geometry"
            ),
        },
        "refinement_quality": refinement_quality,
        "novel_recall_at_0p50_scene_bootstrap": bootstrap_ci,
        "thresholds": threshold_report,
        "go_no_go": gate,
        "per_scene": per_scene,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=tuple(AUDIT_STAGE_SCENE_COUNTS),
    )
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--p1s-checkpoint", required=True, type=Path)
    parser.add_argument("--p1g-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--source-diagnostics-root",
        "--p1-source-diagnostics-root",
        dest="source_diagnostics_root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--prediction-root",
        "--p1-source-prediction-root",
        dest="prediction_root",
        required=True,
        type=Path,
    )
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--maximum-refiner-seconds-per-scene",
        type=float,
        default=MAX_REFINER_SECONDS_PER_SCENE,
    )
    parser.add_argument(
        "--maximum-refiner-p95-seconds-per-scene",
        type=float,
        default=MAX_REFINER_P95_SECONDS_PER_SCENE,
    )
    parser.add_argument("--output", "--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        stage=args.stage,
        scene_list=args.scene_list,
        p1s_checkpoint=args.p1s_checkpoint,
        p1g_checkpoint=args.p1g_checkpoint,
        source_diagnostics_root=args.source_diagnostics_root,
        prediction_root=args.prediction_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        device=args.device,
        maximum_refiner_seconds_per_scene=(
            args.maximum_refiner_seconds_per_scene
        ),
        maximum_refiner_p95_seconds_per_scene=(
            args.maximum_refiner_p95_seconds_per_scene
        ),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(rendered)
    return 0 if report["go_no_go"]["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
