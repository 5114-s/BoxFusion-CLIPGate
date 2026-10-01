#!/usr/bin/env python3
"""Fail-closed validation for a complete observer-only P2-v3 run.

P2-v3 is a detached geometry observer.  Its diagnostic stream may describe
reliability-weighted proposals, but it must never alter BoxFusion predictions.
Accordingly, this validator establishes formal-output safety from same-run
immutable flags and zero applied counts.  It deliberately does not require
prediction pickle bytes from independent CUDA runs to be identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.p2_local_mask_geometry import (  # noqa: E402
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)
from boxfusion.p2_reliability_fusion import (  # noqa: E402
    P2V3_DIAGNOSTIC_SCHEMA,
    P2V3_PROFILE,
    P2V3_RELIABILITY_CONTRACT,
    P2V3_SOURCE,
)
from tools.train_p1_residual_head import read_scene_ids  # noqa: E402
from tools.validate_p2v2_run_artifacts import (  # noqa: E402
    validate as validate_p2v2,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class P2V3Diagnostic:
    """Validated, detached P2-v3 arrays reusable by recall reports."""

    scene_id: str
    frame_ids: np.ndarray
    provider_steps: np.ndarray
    input_candidate_counts: np.ndarray
    eligible_candidate_counts: np.ndarray
    step_candidate_counts: np.ndarray
    step_seconds: np.ndarray
    candidate_ids: np.ndarray
    parent_candidate_ids: np.ndarray
    parent_p2_candidate_ids: np.ndarray
    mask_source_ids: np.ndarray
    component_boxes: np.ndarray
    component_corners: np.ndarray
    parent_boxes: np.ndarray
    parent_corners: np.ndarray
    fused_boxes: np.ndarray
    fused_corners: np.ndarray
    scores: np.ndarray
    component_weights: np.ndarray
    center_component_weights: np.ndarray
    extent_component_weights: np.ndarray
    component_reliabilities: np.ndarray
    parent_reliabilities: np.ndarray
    mask_reliabilities: np.ndarray
    depth_reliabilities: np.ndarray
    support_reliabilities: np.ndarray
    agreement_reliabilities: np.ndarray
    runtime_seconds: float
    parent_p2_checkpoint_sha256: str
    parent_p2v2_schema: str
    parent_p2v2_source: str
    config: Mapping[str, Any]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"{path}: missing {key}")
    try:
        value = np.asarray(archive[key])
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError(f"{path}: object dtype in {key}") from error
        raise
    if value.dtype.hasobject:
        raise ValueError(f"{path}: object dtype in {key}")
    return value


def _scalar(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> np.ndarray:
    value = _array(archive, key, path)
    if value.shape != ():
        raise ValueError(f"{path}: {key} must be a scalar")
    return value


def _text(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> str:
    value = _scalar(archive, key, path).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{path}: {key} must be text")
    return value


def _boolean(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> bool:
    value = _scalar(archive, key, path)
    if value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be Boolean")
    return bool(value.item())


def _integer(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> int:
    value = _scalar(archive, key, path)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path}: {key} must be integer")
    return int(value.item())


def _config(
    archive: np.lib.npyio.NpzFile,
    path: Path,
) -> Mapping[str, Any]:
    try:
        value = json.loads(_text(archive, "p2v3_config_json", path))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid p2v3_config_json") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: P2-v3 config must be an object")
    expected = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
    }
    for key, required in expected.items():
        if value.get(key) is not required:
            raise ValueError(f"{path}: unsafe P2-v3 config {key}")
    minimum_weight = float(value.get("minimum_component_weight", -1.0))
    maximum_weight = float(value.get("maximum_component_weight", -1.0))
    if not (
        0.0 <= minimum_weight <= maximum_weight <= 1.0
        and int(value.get("max_candidates_per_step", 0)) >= 1
        and int(value.get("max_scene_candidates", 0)) >= 1
    ):
        raise ValueError(f"{path}: invalid bounded P2-v3 config")
    return value


def _integer_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
    count: int,
) -> np.ndarray:
    value = _array(archive, key, path)
    if (
        value.shape != (count,)
        or not np.issubdtype(value.dtype, np.integer)
    ):
        raise ValueError(f"{path}: {key} must be integer[{count}]")
    return value


def _bounded_float_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
    count: int,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    value = _array(archive, key, path)
    if (
        value.shape != (count,)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or np.any(value < lower)
        or np.any(value > upper)
    ):
        raise ValueError(f"{path}: invalid {key}")
    return value


def _validate_box_corners(
    boxes: np.ndarray,
    corners: np.ndarray,
    *,
    key: str,
    path: Path,
) -> None:
    count = len(boxes)
    if (
        boxes.shape != (count, 6)
        or corners.shape != (count, 8, 3)
        or not np.issubdtype(boxes.dtype, np.floating)
        or not np.issubdtype(corners.dtype, np.floating)
        or not np.isfinite(boxes).all()
        or not np.isfinite(corners).all()
        or np.any(boxes[:, 3:] <= 0.0)
    ):
        raise ValueError(f"{path}: invalid {key} geometry")
    for index in range(count):
        lower = boxes[index, :3] - 0.5 * boxes[index, 3:]
        upper = boxes[index, :3] + 0.5 * boxes[index, 3:]
        row = corners[index]
        if not (
            np.allclose(np.min(row, axis=0), lower, atol=1e-5)
            and np.allclose(np.max(row, axis=0), upper, atol=1e-5)
            and np.all(
                np.isclose(row, lower[None], atol=1e-5)
                | np.isclose(row, upper[None], atol=1e-5)
            )
        ):
            raise ValueError(f"{path}: {key} corners disagree with box")
        binary = np.isclose(row, upper[None], atol=1e-5).astype(np.int8)
        if len({tuple(values) for values in binary.tolist()}) != 8:
            raise ValueError(f"{path}: {key} corners are not unique")


def _validate_steps(
    archive: np.lib.npyio.NpzFile,
    path: Path,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent_frames = _array(archive, "p2v2_step_frame_ids", path)
    parent_provider_steps = _array(
        archive, "p2v2_step_provider_steps", path
    )
    parent_counts = _array(
        archive, "p2v2_step_candidate_counts", path
    )
    frames = _array(archive, "p2v3_step_frame_ids", path)
    provider_steps = _array(
        archive, "p2v3_step_provider_steps", path
    )
    if (
        parent_frames.ndim != 1
        or parent_provider_steps.shape != parent_frames.shape
        or parent_counts.shape != parent_frames.shape
        or frames.shape != parent_frames.shape
        or provider_steps.shape != parent_frames.shape
        or len(frames) < 1
    ):
        raise ValueError(f"{path}: invalid P2-v2/P2-v3 step arrays")
    for key, value in (
        ("p2v2_step_frame_ids", parent_frames),
        ("p2v2_step_provider_steps", parent_provider_steps),
        ("p2v2_step_candidate_counts", parent_counts),
        ("p2v3_step_frame_ids", frames),
        ("p2v3_step_provider_steps", provider_steps),
    ):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{path}: {key} must be integer")
    if not np.array_equal(parent_frames, frames) or not np.array_equal(
        parent_provider_steps, provider_steps
    ):
        raise ValueError(
            f"{path}: P2-v2/P2-v3 scheduling is not aligned"
        )

    count = len(frames)
    input_counts = _integer_vector(
        archive, "p2v3_step_input_candidate_counts", path, count
    )
    eligible_counts = _integer_vector(
        archive, "p2v3_step_eligible_candidate_counts", path, count
    )
    candidate_counts = _integer_vector(
        archive, "p2v3_step_candidate_counts", path, count
    )
    seconds = _array(archive, "p2v3_step_seconds", path)
    failed = _array(archive, "p2v3_step_failed", path)
    errors = _array(archive, "p2v3_step_errors", path)
    if (
        seconds.shape != (count,)
        or not np.issubdtype(seconds.dtype, np.floating)
        or not np.isfinite(seconds).all()
        or np.any(seconds < 0.0)
        or failed.shape != (count,)
        or failed.dtype != np.dtype(bool)
        or errors.shape != (count,)
        or errors.dtype.kind not in {"U", "S"}
    ):
        raise ValueError(f"{path}: invalid P2-v3 status/timing arrays")
    if np.any(failed) or any(str(value) for value in errors.tolist()):
        raise ValueError(f"{path}: P2-v3 contains failed steps")
    if not np.array_equal(input_counts, parent_counts):
        raise ValueError(
            f"{path}: P2-v3 inputs disagree with parent P2-v2 outputs"
        )
    if (
        np.any(input_counts < 0)
        or np.any(eligible_counts < 0)
        or np.any(candidate_counts < 0)
        or not np.array_equal(eligible_counts, input_counts)
        or np.any(candidate_counts > eligible_counts)
        or np.any(
            candidate_counts
            > int(config["max_candidates_per_step"])
        )
    ):
        raise ValueError(f"{path}: impossible P2-v3 step counts")
    return (
        frames,
        provider_steps,
        input_counts,
        eligible_counts,
        candidate_counts,
    )


def _validate_candidates(
    archive: np.lib.npyio.NpzFile,
    path: Path,
    *,
    pre_scene_nms_count: int,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    ids = _array(archive, "p2v3_candidate_ids", path)
    parent_ids = _array(
        archive, "p2v3_parent_p2v2_candidate_ids", path
    )
    parent_p2_ids = _array(
        archive, "p2v3_parent_p2_candidate_ids", path
    )
    mask_source_ids = _array(
        archive, "p2v3_mask_source_ids", path
    )
    if (
        ids.ndim != 1
        or ids.dtype.kind not in {"U", "S"}
        or parent_ids.shape != ids.shape
        or parent_ids.dtype.kind not in {"U", "S"}
        or parent_p2_ids.shape != ids.shape
        or parent_p2_ids.dtype.kind not in {"U", "S"}
        or mask_source_ids.shape != ids.shape
        or mask_source_ids.dtype.kind not in {"U", "S"}
    ):
        raise ValueError(f"{path}: invalid P2-v3 candidate IDs")
    count = len(ids)
    candidate_ids = [str(value) for value in ids.tolist()]
    parent_candidate_ids = [str(value) for value in parent_ids.tolist()]
    if (
        len(set(candidate_ids)) != count
        or any(not value.startswith("p2v3:") for value in candidate_ids)
        or any(
            not value.startswith("p2v2:")
            for value in parent_candidate_ids
        )
        or any(not str(value) for value in parent_p2_ids.tolist())
        or any(not str(value) for value in mask_source_ids.tolist())
    ):
        raise ValueError(
            f"{path}: invalid or duplicate P2-v3 candidate IDs"
        )

    geometry = {}
    for name in ("component", "parent", "fused"):
        boxes = _array(
            archive, f"p2v3_candidate_{name}_boxes", path
        )
        corners = _array(
            archive, f"p2v3_candidate_{name}_corners", path
        )
        _validate_box_corners(
            boxes, corners, key=f"P2-v3 {name}", path=path
        )
        if len(boxes) != count:
            raise ValueError(
                f"{path}: P2-v3 {name} count does not align"
            )
        geometry[f"{name}_boxes"] = boxes
        geometry[f"{name}_corners"] = corners

    vectors = {}
    for name in (
        "scores",
        "component_weights",
        "component_reliabilities",
        "parent_reliabilities",
        "mask_reliabilities",
        "depth_reliabilities",
        "support_reliabilities",
        "agreement_reliabilities",
    ):
        vectors[name] = _bounded_float_vector(
            archive, f"p2v3_candidate_{name}", path, count
        )
    minimum_weight = float(config["minimum_component_weight"])
    maximum_weight = float(config["maximum_component_weight"])
    weights = vectors["component_weights"]
    if np.any(weights < minimum_weight) or np.any(
        weights > maximum_weight
    ):
        raise ValueError(
            f"{path}: component weight violates configured bounds"
        )
    axis_weights = {}
    for name in (
        "center_component_weights",
        "extent_component_weights",
    ):
        value = _array(
            archive, f"p2v3_candidate_{name}", path
        )
        if (
            value.shape != (count, 3)
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
            or np.any(value < minimum_weight)
            or np.any(value > maximum_weight)
        ):
            raise ValueError(f"{path}: invalid p2v3_candidate_{name}")
        axis_weights[name] = value

    expected_center = (
        axis_weights["center_component_weights"]
        * geometry["component_boxes"][:, :3]
        + (
            1.0 - axis_weights["center_component_weights"]
        )
        * geometry["parent_boxes"][:, :3]
    )
    expected_extent = (
        axis_weights["extent_component_weights"]
        * geometry["component_boxes"][:, 3:]
        + (
            1.0 - axis_weights["extent_component_weights"]
        )
        * geometry["parent_boxes"][:, 3:]
    )
    expected_fused = np.concatenate(
        (expected_center, expected_extent), axis=1
    )
    if not np.allclose(
        geometry["fused_boxes"], expected_fused, atol=1e-5
    ):
        raise ValueError(
            f"{path}: fused boxes violate component-weight contract"
        )

    applied = _array(archive, "p2v3_candidate_applied", path)
    if (
        applied.shape != (count,)
        or applied.dtype != np.dtype(bool)
        or np.any(applied)
    ):
        raise ValueError(f"{path}: unsafe P2-v3 candidate flags")
    if (
        count > int(config["max_scene_candidates"])
        or count > pre_scene_nms_count
    ):
        raise ValueError(
            f"{path}: P2-v3 scene candidate count is impossible"
        )
    return {
        "candidate_ids": ids,
        "parent_candidate_ids": parent_ids,
        "parent_p2_candidate_ids": parent_p2_ids,
        "mask_source_ids": mask_source_ids,
        **geometry,
        **vectors,
        **axis_weights,
    }


def _validate_parent_candidate_linkage(
    archive: np.lib.npyio.NpzFile,
    path: Path,
    candidates: Mapping[str, np.ndarray],
) -> None:
    """Cross-check every P2-v2 parent that survived both scene NMS passes."""

    parent_ids = _array(archive, "p2v2_candidate_ids", path)
    if parent_ids.ndim != 1 or parent_ids.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{path}: invalid parent P2-v2 candidate IDs")
    parent_count = len(parent_ids)
    parent_boxes = _array(archive, "p2v2_candidate_boxes", path)
    parent_corners = _array(archive, "p2v2_candidate_corners", path)
    parent_p2_ids = _array(
        archive, "p2v2_parent_p2_candidate_ids", path
    )
    mask_source_ids = _array(
        archive, "p2v2_mask_source_ids", path
    )
    scores = _array(archive, "p2v2_candidate_scores", path)
    if (
        parent_boxes.shape != (parent_count, 6)
        or parent_corners.shape != (parent_count, 8, 3)
        or parent_p2_ids.shape != (parent_count,)
        or parent_p2_ids.dtype.kind not in {"U", "S"}
        or mask_source_ids.shape != (parent_count,)
        or mask_source_ids.dtype.kind not in {"U", "S"}
        or scores.shape != (parent_count,)
        or not np.issubdtype(scores.dtype, np.floating)
    ):
        raise ValueError(f"{path}: invalid parent P2-v2 candidate stream")
    lookup = {
        str(candidate_id): index
        for index, candidate_id in enumerate(parent_ids.tolist())
    }
    if len(lookup) != parent_count:
        raise ValueError(f"{path}: duplicate parent P2-v2 candidate IDs")
    for index, candidate_id in enumerate(
        candidates["parent_candidate_ids"].tolist()
    ):
        parent_index = lookup.get(str(candidate_id))
        if parent_index is None:
            # P2-v2 and P2-v3 use different scene-NMS geometry.  A valid
            # P2-v3 survivor may therefore have a parent pruned from the
            # final P2-v2 scene array even though the per-step linkage above
            # was exact.
            continue
        if not (
            np.allclose(
                candidates["component_boxes"][index],
                parent_boxes[parent_index],
                atol=1e-6,
            )
            and np.allclose(
                candidates["component_corners"][index],
                parent_corners[parent_index],
                atol=1e-6,
            )
            and str(candidates["parent_p2_candidate_ids"][index])
            == str(parent_p2_ids[parent_index])
            and str(candidates["mask_source_ids"][index])
            == str(mask_source_ids[parent_index])
            and np.isclose(
                candidates["scores"][index],
                scores[parent_index],
                atol=1e-6,
            )
        ):
            raise ValueError(
                f"{path}: P2-v3 candidate disagrees with P2-v2 parent"
            )


def load_p2v3_diagnostic(
    path: Path,
    *,
    expected_scene_id: str | None = None,
    expected_p2_checkpoint_sha256: str | None = None,
) -> P2V3Diagnostic:
    """Load and fully validate one P2-v3 diagnostic archive."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            try:
                value = np.asarray(archive[key])
            except ValueError as error:
                if "Object arrays cannot be loaded" in str(error):
                    raise ValueError(
                        f"{path}: object dtype in {key}"
                    ) from error
                raise
            if value.dtype.hasobject:
                raise ValueError(f"{path}: object dtype in {key}")
        scene_id = _text(archive, "scene_id", path)
        if expected_scene_id is not None and scene_id != expected_scene_id:
            raise ValueError(f"{path}: scene_id mismatch")
        expected_text = {
            "p2v3_schema": P2V3_DIAGNOSTIC_SCHEMA,
            "p2v3_stage": "P2V3",
            "p2v3_profile": P2V3_PROFILE,
            "p2v3_source": P2V3_SOURCE,
            "p2v3_parent_p2v2_schema": P2V2_DIAGNOSTIC_SCHEMA,
            "p2v3_reliability_contract": P2V3_RELIABILITY_CONTRACT,
        }
        for key, expected in expected_text.items():
            if _text(archive, key, path) != expected:
                raise ValueError(f"{path}: invalid {key}")
        expected_bools = {
            "p2v3_enabled": True,
            "p2v3_observer_only": True,
            "p2v3_uses_ground_truth": False,
            "p2v3_reads_semantic_labels": False,
            "p2v3_mutation_enabled": False,
            "p2v3_complete": True,
        }
        for key, expected in expected_bools.items():
            if _boolean(archive, key, path) is not expected:
                raise ValueError(f"{path}: unsafe {key}")
        if _integer(archive, "p2v3_applied_count", path) != 0:
            raise ValueError(f"{path}: P2-v3 mutated formal output")

        parent_sha = _text(
            archive, "p2v3_parent_p2_checkpoint_sha256", path
        )
        embedded_p2_sha = _text(archive, "p2_checkpoint_sha256", path)
        embedded_p2v2_sha = _text(
            archive, "p2v2_parent_p2_checkpoint_sha256", path
        )
        if (
            _SHA256.fullmatch(parent_sha) is None
            or parent_sha != embedded_p2_sha
            or parent_sha != embedded_p2v2_sha
            or (
                expected_p2_checkpoint_sha256 is not None
                and parent_sha != expected_p2_checkpoint_sha256
            )
        ):
            raise ValueError(f"{path}: P2-v3 parent checkpoint mismatch")

        forbidden = [
            key
            for key in archive.files
            if key.startswith("p2v3_candidate_")
            and any(
                token in key
                for token in ("label", "class", "semantic", "clip", "text")
            )
        ]
        if forbidden:
            raise ValueError(
                f"{path}: semantic P2-v3 candidate field {forbidden[0]}"
            )
        config = _config(archive, path)
        (
            frame_ids,
            provider_steps,
            input_counts,
            eligible_counts,
            step_candidate_counts,
        ) = _validate_steps(archive, path, config)
        candidates = _validate_candidates(
            archive,
            path,
            pre_scene_nms_count=int(np.sum(step_candidate_counts)),
            config=config,
        )
        _validate_parent_candidate_linkage(archive, path, candidates)
        step_seconds = _array(archive, "p2v3_step_seconds", path)

        return P2V3Diagnostic(
            scene_id=scene_id,
            frame_ids=_readonly(frame_ids),
            provider_steps=_readonly(provider_steps),
            input_candidate_counts=_readonly(input_counts),
            eligible_candidate_counts=_readonly(eligible_counts),
            step_candidate_counts=_readonly(step_candidate_counts),
            step_seconds=_readonly(step_seconds),
            candidate_ids=_readonly(candidates["candidate_ids"]),
            parent_candidate_ids=_readonly(
                candidates["parent_candidate_ids"]
            ),
            parent_p2_candidate_ids=_readonly(
                candidates["parent_p2_candidate_ids"]
            ),
            mask_source_ids=_readonly(candidates["mask_source_ids"]),
            component_boxes=_readonly(candidates["component_boxes"]),
            component_corners=_readonly(
                candidates["component_corners"]
            ),
            parent_boxes=_readonly(candidates["parent_boxes"]),
            parent_corners=_readonly(candidates["parent_corners"]),
            fused_boxes=_readonly(candidates["fused_boxes"]),
            fused_corners=_readonly(candidates["fused_corners"]),
            scores=_readonly(candidates["scores"]),
            component_weights=_readonly(
                candidates["component_weights"]
            ),
            center_component_weights=_readonly(
                candidates["center_component_weights"]
            ),
            extent_component_weights=_readonly(
                candidates["extent_component_weights"]
            ),
            component_reliabilities=_readonly(
                candidates["component_reliabilities"]
            ),
            parent_reliabilities=_readonly(
                candidates["parent_reliabilities"]
            ),
            mask_reliabilities=_readonly(
                candidates["mask_reliabilities"]
            ),
            depth_reliabilities=_readonly(
                candidates["depth_reliabilities"]
            ),
            support_reliabilities=_readonly(
                candidates["support_reliabilities"]
            ),
            agreement_reliabilities=_readonly(
                candidates["agreement_reliabilities"]
            ),
            runtime_seconds=float(np.sum(step_seconds)),
            parent_p2_checkpoint_sha256=parent_sha,
            parent_p2v2_schema=P2V2_DIAGNOSTIC_SCHEMA,
            parent_p2v2_source=P2V2_SOURCE,
            config=config,
        )


def validate(
    *,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    expected_p1_checkpoint: Path,
    expected_p2_checkpoint: Path,
) -> dict[str, Any]:
    """Validate P1/P2/P2-v2 ancestry plus detached P2-v3 artifacts."""

    parent_report = validate_p2v2(
        scene_list=scene_list,
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        expected_p1_checkpoint=expected_p1_checkpoint,
        expected_p2_checkpoint=expected_p2_checkpoint,
    )
    scenes = read_scene_ids(scene_list, role="P2-v3 evaluation")
    expected_p2_sha = _sha256(expected_p2_checkpoint)
    step_count = 0
    input_count = 0
    pre_scene_nms_count = 0
    scene_candidate_count = 0
    runtime_seconds = 0.0
    for scene in scenes:
        diagnostic = load_p2v3_diagnostic(
            diagnostics_root / f"{scene}_tracks.npz",
            expected_scene_id=scene,
            expected_p2_checkpoint_sha256=expected_p2_sha,
        )
        step_count += len(diagnostic.frame_ids)
        input_count += int(np.sum(diagnostic.input_candidate_counts))
        pre_scene_nms_count += int(
            np.sum(diagnostic.step_candidate_counts)
        )
        scene_candidate_count += len(diagnostic.candidate_ids)
        runtime_seconds += diagnostic.runtime_seconds
    return {
        "schema": "boxfusion.p2v3.run_artifact_validation.v1",
        "scene_count": len(scenes),
        "p2v3_step_count": step_count,
        "p2v3_input_candidate_count": input_count,
        "p2v3_pre_scene_nms_candidate_count": pre_scene_nms_count,
        "p2v3_scene_candidate_count": scene_candidate_count,
        "p2v3_runtime_seconds": runtime_seconds,
        "p1_checkpoint_sha256": parent_report[
            "p1_checkpoint_sha256"
        ],
        "p2_checkpoint_sha256": expected_p2_sha,
        "prediction_root": str(prediction_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        "formal_output_safety": {
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "uses_ground_truth": False,
            "reads_semantic_labels": False,
            "cross_run_pickle_byte_equality_required": False,
            "basis": (
                "same-run immutable observer contract; independent-run "
                "drift is handled by the nondeterminism audit"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument(
        "--expected-p1-checkpoint", required=True, type=Path
    )
    parser.add_argument(
        "--expected-p2-checkpoint", required=True, type=Path
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate(
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        expected_p1_checkpoint=args.expected_p1_checkpoint,
        expected_p2_checkpoint=args.expected_p2_checkpoint,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
