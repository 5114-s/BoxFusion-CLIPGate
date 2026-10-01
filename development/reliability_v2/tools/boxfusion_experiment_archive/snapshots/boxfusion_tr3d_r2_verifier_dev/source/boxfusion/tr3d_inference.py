"""Official MMDetection3D/TR3D inference adapter and cache export logic.

MMDetection3D's generic ``inference_detector`` injects an identity
``axis_align_matrix``.  That is incorrect for ScanNet point files stored in
the unaligned world frame.  This adapter deliberately builds the test
pipeline with each sample's real matrix, so ``GlobalAlignment`` is applied
exactly once before official TR3D inference.

All OpenMMLab imports are lazy.  The dependency-light input, validation and
export functions can therefore be unit-tested in the BoxFusion environment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .frozen_b6_manifest import read_scene_list, sha256_file
from .tr3d_residual_cache import (
    TR3DResidualCache,
    load_tr3d_residual_cache,
    make_tr3d_residual_cache_from_aligned,
    tr3d_residual_cache_path,
    write_tr3d_residual_cache,
)


TR3D_EXPORT_REPORT_SCHEMA = "boxfusion.tr3d_residual_export_report.v1"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class TR3DInferenceInput:
    scene_id: str
    prefix_id: str
    prefix_fraction: float
    point_path: Path
    axis_align_matrix: np.ndarray
    expected_point_count: int | None = None

    @property
    def sample_idx(self) -> str:
        return f"{self.scene_id}:{self.prefix_id}"


@dataclass(frozen=True)
class AlignedTR3DOutput:
    boxes_aligned: np.ndarray
    scores_3d: np.ndarray
    labels_3d: np.ndarray
    runtime_s: float


class TR3DInferenceAdapter(Protocol):
    def infer(
        self, points_unaligned: np.ndarray, axis_align_matrix: np.ndarray
    ) -> AlignedTR3DOutput:
        ...


def _validate_axis_align(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    rotation = value[:3, :3] if value.shape == (4, 4) else None
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-6)
        or not np.allclose(rotation[2], [0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError(
            "axis_align_matrix must be a homogeneous ScanNet z-rotation"
        )
    return np.ascontiguousarray(value)


def load_points_xyzrgb(path: str | Path) -> np.ndarray:
    """Load canonical ScanNet float32 XYZRGB from ``.bin`` or ``.npy``."""

    point_path = Path(path)
    if not point_path.is_file():
        raise FileNotFoundError(point_path)
    if point_path.suffix.lower() == ".bin":
        flat = np.fromfile(point_path, dtype=np.float32)
        if flat.size % 6:
            raise ValueError(
                f"{point_path}: float count is not divisible by six"
            )
        points = flat.reshape(-1, 6)
    elif point_path.suffix.lower() == ".npy":
        points = np.load(point_path, allow_pickle=False)
        if points.dtype != np.dtype(np.float32):
            raise ValueError(f"{point_path}: points must be float32")
    else:
        raise ValueError(f"{point_path}: expected .bin or .npy points")
    if points.ndim != 2 or points.shape[1] != 6 or not len(points):
        raise ValueError(f"{point_path}: points must be non-empty [N,6]")
    if not np.isfinite(points).all():
        raise ValueError(f"{point_path}: points contain non-finite values")
    if np.any(points[:, 3:] < 0.0) or np.any(points[:, 3:] > 255.0):
        raise ValueError(f"{point_path}: RGB values must be in [0,255]")
    return np.ascontiguousarray(points, dtype=np.float32)


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _manifest_rows(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: row is not a mapping")
            rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping) and isinstance(
            value.get("data_list"), list
        ):
            value = value["data_list"]
        if not isinstance(value, list) or not all(
            isinstance(row, Mapping) for row in value
        ):
            raise ValueError(f"{path}: expected a JSON list of mappings")
        return list(value)
    raise ValueError(f"{path}: inference manifest must be .jsonl or .json")


def load_inference_manifest(
    path: str | Path,
    *,
    scene_ids: Sequence[str] | None = None,
    prefix_ids: Sequence[str] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> tuple[TR3DInferenceInput, ...]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if num_shards < 1 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("invalid shard index/count")
    scene_filter = None if scene_ids is None else set(scene_ids)
    prefix_filter = None if prefix_ids is None else set(prefix_ids)
    inputs: list[TR3DInferenceInput] = []
    seen: set[tuple[str, str]] = set()
    for row in _manifest_rows(manifest_path):
        scene_id = str(row.get("scene_id", ""))
        if _SCENE_RE.fullmatch(scene_id) is None:
            raise ValueError(f"{manifest_path}: invalid scene_id {scene_id!r}")
        prefix_id = str(
            row.get("prefix_id") or row.get("tag") or "full"
        )
        prefix_fraction = float(
            row.get("prefix_fraction", row.get("fraction", 1.0))
        )
        if _PREFIX_RE.fullmatch(prefix_id) is None:
            raise ValueError(
                f"{manifest_path}: invalid prefix_id {prefix_id!r}"
            )
        if (
            not math.isfinite(prefix_fraction)
            or prefix_fraction <= 0.0
            or prefix_fraction > 1.0
        ):
            raise ValueError(
                f"{scene_id}:{prefix_id}: fraction must be in (0,1]"
            )
        coordinate_frame = str(
            row.get("coordinate_frame", "world_unaligned")
        )
        if coordinate_frame not in {
            "world_unaligned",
            "scannet_unaligned_world",
        }:
            raise ValueError(
                f"{scene_id}:{prefix_id}: points are not world_unaligned"
            )
        point_value = row.get("point_path") or row.get("source_point_path")
        if not isinstance(point_value, str) or not point_value:
            raise ValueError(f"{scene_id}:{prefix_id}: missing point path")
        axis = row.get("axis_align_matrix")
        if axis is None:
            raise ValueError(
                f"{scene_id}:{prefix_id}: missing axis_align_matrix"
            )
        item = TR3DInferenceInput(
            scene_id=scene_id,
            prefix_id=prefix_id,
            prefix_fraction=prefix_fraction,
            point_path=_resolve_manifest_path(point_value, manifest_path),
            axis_align_matrix=_validate_axis_align(axis),
            expected_point_count=(
                None
                if row.get("point_count") is None
                else int(row["point_count"])
            ),
        )
        key = (item.scene_id, item.prefix_id)
        if key in seen:
            raise ValueError(f"duplicate inference sample: {item.sample_idx}")
        seen.add(key)
        if scene_filter is not None and scene_id not in scene_filter:
            continue
        if prefix_filter is not None and prefix_id not in prefix_filter:
            continue
        inputs.append(item)
    if scene_filter is not None:
        available = {item.scene_id for item in inputs}
        missing = sorted(scene_filter - available)
        if missing:
            raise ValueError(
                "manifest is missing requested scenes: " + ", ".join(missing[:8])
            )
    inputs.sort(key=lambda item: (item.scene_id, item.prefix_id))
    selected = tuple(
        item
        for index, item in enumerate(inputs)
        if index % num_shards == shard_index
    )
    if not selected:
        raise ValueError("inference manifest selection is empty")
    return selected


def load_axis_alignment_file(path: str | Path) -> np.ndarray:
    matrix_path = Path(path)
    if matrix_path.suffix.lower() == ".npy":
        value = np.load(matrix_path, allow_pickle=False)
    else:
        try:
            value = np.loadtxt(matrix_path, dtype=np.float64)
        except ValueError:
            value = None
            for line in matrix_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("axisAlignment") and "=" in line:
                    flat = np.fromstring(line.split("=", 1)[1], sep=" ")
                    if flat.size == 16:
                        value = flat.reshape(4, 4)
                    break
            if value is None:
                raise ValueError(
                    f"{matrix_path}: no 4x4 axisAlignment matrix"
                )
    return _validate_axis_align(value)


def direct_inference_input(
    *,
    scene_id: str,
    prefix_id: str,
    prefix_fraction: float,
    point_path: str | Path,
    axis_alignment_path: str | Path,
) -> TR3DInferenceInput:
    if _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError(f"invalid ScanNet scene id: {scene_id!r}")
    if _PREFIX_RE.fullmatch(prefix_id) is None:
        raise ValueError(f"invalid prefix_id: {prefix_id!r}")
    if (
        not math.isfinite(prefix_fraction)
        or prefix_fraction <= 0.0
        or prefix_fraction > 1.0
    ):
        raise ValueError("prefix_fraction must be in (0,1]")
    return TR3DInferenceInput(
        scene_id=scene_id,
        prefix_id=prefix_id,
        prefix_fraction=float(prefix_fraction),
        point_path=Path(point_path).resolve(),
        axis_align_matrix=load_axis_alignment_file(axis_alignment_path),
    )


def _unwrap_dataset_config(dataset: Any) -> Any:
    current = dataset
    for _ in range(8):
        if "pipeline" in current:
            return current
        if "dataset" not in current:
            break
        current = current["dataset"]
    raise ValueError("TR3D test dataset config has no pipeline")


class OfficialMMDet3DTR3DAdapter:
    """Thin adapter over the pinned official MMDetection3D TR3D project."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        project_root: str | Path,
        vendor_root: str | Path,
    ) -> None:
        project = Path(project_root).resolve()
        vendor = Path(vendor_root).resolve()
        for path in (project, vendor):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            import torch
            from mmdet3d.apis import init_model
            from mmdet3d.structures import get_box_type
            from mmengine.dataset import Compose, pseudo_collate
        except Exception as error:
            raise RuntimeError(
                "official TR3D runtime is unavailable; run "
                "scripts/bootstrap_tr3d_env.sh and the environment check"
            ) from error
        self.torch = torch
        self.pseudo_collate = pseudo_collate
        self.model = init_model(
            str(Path(config_path).resolve()),
            str(Path(checkpoint_path).resolve()),
            device=device,
        )
        self.device = device
        head = self.model.bbox_head
        label2level = tuple(getattr(head, "label2level", ()))
        if label2level != (0,):
            raise ValueError(
                "cache export requires the genuine one-class TR3D head; "
                f"observed label2level={label2level!r}"
            )
        self.voxel_size = float(head.voxel_size)
        dataset_cfg = _unwrap_dataset_config(
            self.model.cfg.test_dataloader.dataset
        )
        pipeline_cfg = deepcopy(dataset_cfg.pipeline)
        if not pipeline_cfg:
            raise ValueError("TR3D test pipeline is empty")
        pipeline_cfg[0]["type"] = "LoadPointsFromDict"
        self.pipeline = Compose(pipeline_cfg)
        self.box_type_3d, self.box_mode_3d = get_box_type(
            dataset_cfg.box_type_3d
        )

    def _synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def infer(
        self, points_unaligned: np.ndarray, axis_align_matrix: np.ndarray
    ) -> AlignedTR3DOutput:
        self._synchronize()
        started = time.perf_counter()
        data = self.pipeline(
            {
                "points": np.array(points_unaligned, copy=True),
                "timestamp": 1,
                "axis_align_matrix": np.array(
                    axis_align_matrix, copy=True
                ),
                "box_type_3d": self.box_type_3d,
                "box_mode_3d": self.box_mode_3d,
            }
        )
        collated = self.pseudo_collate([data])
        with self.torch.no_grad():
            results = self.model.test_step(collated)
        self._synchronize()
        runtime_s = time.perf_counter() - started
        if not isinstance(results, Sequence) or len(results) != 1:
            raise ValueError("official TR3D returned a non-singleton batch")
        instances = results[0].pred_instances_3d
        box_structure = instances.bboxes_3d
        # BaseInstance3DBoxes.tensor may store a bottom-origin z depending on
        # its canonical box class.  The cache contract is explicitly
        # center/size/yaw, so use gravity_center as the authoritative center.
        box_tensor = box_structure.tensor
        boxes = self.torch.cat(
            (box_structure.gravity_center, box_tensor[:, 3:]), dim=1
        ).detach().cpu().numpy()
        scores = instances.scores_3d.detach().cpu().numpy()
        labels = instances.labels_3d.detach().cpu().numpy()
        return AlignedTR3DOutput(
            boxes_aligned=np.asarray(boxes, dtype=np.float32),
            scores_3d=np.asarray(scores, dtype=np.float32),
            labels_3d=np.asarray(labels, dtype=np.int64),
            runtime_s=float(runtime_s),
        )


class SyntheticTR3DAdapter:
    """Dependency-free adapter used only by ``--dry-run-synthetic``."""

    def infer(
        self, points_unaligned: np.ndarray, axis_align_matrix: np.ndarray
    ) -> AlignedTR3DOutput:
        aligned = (
            points_unaligned[:, :3] @ axis_align_matrix[:3, :3].T
            + axis_align_matrix[None, :3, 3]
        )
        lower = np.quantile(aligned, 0.1, axis=0)
        upper = np.quantile(aligned, 0.9, axis=0)
        size = np.maximum(upper - lower, 0.05)
        box = np.concatenate(((lower + upper) / 2, size))[None]
        return AlignedTR3DOutput(
            boxes_aligned=np.asarray(box, dtype=np.float32),
            scores_3d=np.asarray([0.5], dtype=np.float32),
            labels_3d=np.asarray([0], dtype=np.int64),
            runtime_s=0.0,
        )


def _validate_output(output: AlignedTR3DOutput) -> tuple[np.ndarray, ...]:
    boxes = np.asarray(output.boxes_aligned)
    scores = np.asarray(output.scores_3d)
    labels = np.asarray(output.labels_3d)
    if (
        boxes.dtype != np.dtype(np.float32)
        or boxes.ndim != 2
        or boxes.shape[1] not in {6, 7}
        or not np.isfinite(boxes).all()
        or (len(boxes) and np.any(boxes[:, 3:6] <= 0))
    ):
        raise ValueError("official TR3D boxes must be finite float32 [N,6/7]")
    if (
        scores.dtype != np.dtype(np.float32)
        or scores.shape != (len(boxes),)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
    ):
        raise ValueError("official TR3D scores must be float32 [N] in [0,1]")
    if labels.dtype != np.dtype(np.int64) or labels.shape != (len(boxes),):
        raise ValueError("official TR3D labels must be int64 [N]")
    if np.any(labels != 0):
        raise ValueError(
            "official output is not class-agnostic; labels_3d contains nonzero"
        )
    if not math.isfinite(output.runtime_s) or output.runtime_s < 0:
        raise ValueError("official TR3D runtime must be finite and nonnegative")
    return boxes, scores, labels


def _proposal_point_count(
    points_aligned: np.ndarray, boxes_aligned: np.ndarray
) -> np.ndarray:
    counts = np.zeros(len(boxes_aligned), dtype=np.int32)
    for index, row in enumerate(boxes_aligned):
        yaw = float(row[6]) if row.shape[0] == 7 else 0.0
        relative = points_aligned - row[:3]
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        local_x = relative[:, 0] * cosine + relative[:, 1] * sine
        local_y = -relative[:, 0] * sine + relative[:, 1] * cosine
        inside = (
            (np.abs(local_x) <= row[3] / 2)
            & (np.abs(local_y) <= row[4] / 2)
            & (np.abs(relative[:, 2]) <= row[5] / 2)
        )
        counts[index] = min(int(np.count_nonzero(inside)), np.iinfo(np.int32).max)
    return counts


def build_cache_from_inference(
    *,
    item: TR3DInferenceInput,
    points: np.ndarray,
    output: AlignedTR3DOutput,
    checkpoint_sha256: str,
    config_sha256: str,
    source_scene_sha256: str,
    score_threshold: float,
    max_proposals: int,
    voxel_size: float,
) -> TR3DResidualCache:
    boxes, scores, labels = _validate_output(output)
    if (
        not math.isfinite(score_threshold)
        or score_threshold < 0
        or score_threshold > 1
    ):
        raise ValueError("score_threshold must be in [0,1]")
    if max_proposals < 1:
        raise ValueError("max_proposals must be positive")
    selected = np.flatnonzero(scores >= score_threshold)
    selected = selected[
        np.argsort(-scores[selected], kind="stable")[:max_proposals]
    ]
    boxes = boxes[selected]
    scores = scores[selected]
    labels = labels[selected]
    points_aligned = (
        points[:, :3] @ item.axis_align_matrix[:3, :3].T
        + item.axis_align_matrix[None, :3, 3]
    )
    support = _proposal_point_count(points_aligned, boxes)
    return make_tr3d_residual_cache_from_aligned(
        scene_id=item.scene_id,
        prefix_id=item.prefix_id,
        prefix_fraction=item.prefix_fraction,
        boxes_aligned=boxes,
        scores_3d=scores,
        labels_3d=labels,
        proposal_ids=np.arange(len(boxes), dtype=np.int64),
        point_count=support,
        unaligned_to_aligned=item.axis_align_matrix,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        source_scene_sha256=source_scene_sha256,
        voxel_size=voxel_size,
        runtime_s=output.runtime_s,
        num_input_points=len(points),
    )


def export_inference_inputs(
    *,
    inputs: Iterable[TR3DInferenceInput],
    adapter: TR3DInferenceAdapter,
    cache_root: str | Path,
    checkpoint_sha256: str,
    config_sha256: str,
    score_threshold: float = 0.01,
    max_proposals: int = 1000,
    voxel_size: float = 0.01,
    resume: bool = False,
    write_cache: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in inputs:
        points = load_points_xyzrgb(item.point_path)
        if (
            item.expected_point_count is not None
            and item.expected_point_count != len(points)
        ):
            raise ValueError(
                f"{item.sample_idx}: point count disagrees with manifest"
            )
        source_sha = sha256_file(item.point_path)
        target = tr3d_residual_cache_path(
            cache_root, item.scene_id, item.prefix_id
        )
        if target.exists():
            if not resume:
                raise FileExistsError(f"immutable TR3D cache exists: {target}")
            cached = load_tr3d_residual_cache(
                target,
                expected_scene_id=item.scene_id,
                expected_prefix_id=item.prefix_id,
                expected_checkpoint_sha256=checkpoint_sha256,
                expected_config_sha256=config_sha256,
                expected_source_scene_sha256=source_sha,
            )
            rows.append(
                {
                    "sample_idx": item.sample_idx,
                    "status": "verified_existing",
                    "cache_path": str(target.resolve()),
                    "proposal_count": cached.proposal_count,
                    "runtime_s": cached.runtime_s,
                    "source_scene_sha256": source_sha,
                }
            )
            continue
        output = adapter.infer(points, item.axis_align_matrix)
        cache = build_cache_from_inference(
            item=item,
            points=points,
            output=output,
            checkpoint_sha256=checkpoint_sha256,
            config_sha256=config_sha256,
            source_scene_sha256=source_sha,
            score_threshold=score_threshold,
            max_proposals=max_proposals,
            voxel_size=voxel_size,
        )
        if write_cache:
            write_tr3d_residual_cache(target, cache)
            status = "created"
            cache_path: str | None = str(target.resolve())
        else:
            status = "dry_run_not_written"
            cache_path = None
        rows.append(
            {
                "sample_idx": item.sample_idx,
                "status": status,
                "cache_path": cache_path,
                "proposal_count": cache.proposal_count,
                "runtime_s": cache.runtime_s,
                "source_scene_sha256": source_sha,
            }
        )
    return {
        "schema": TR3D_EXPORT_REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "sample_count": len(rows),
        "proposal_count": sum(int(row["proposal_count"]) for row in rows),
        "runtime_s": sum(float(row["runtime_s"]) for row in rows),
        "rows": rows,
    }


def artifact_sha256(path: str | Path) -> str:
    return sha256_file(Path(path).resolve())


def select_scenes(path: str | Path | None) -> tuple[str, ...] | None:
    return None if path is None else read_scene_list(path)
