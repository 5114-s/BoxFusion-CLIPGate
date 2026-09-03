#!/usr/bin/env python3
"""Paired byte audit for an R3 primary-rule shadow-active replay.

The auditor does not trust materializer diagnostics.  It independently
reconstructs the fixed primary decision from the frozen G0 predictions and
immutable R3 sidecars, then checks every active prediction row.  Labels,
scores, row order, and row count must remain exact.  Selected geometry must
equal the associated TR3D candidate bytes; all other geometry must equal the
frozen G0 bytes.  Finally, class-agnostic AP is recomputed and required to
match the pre-existing all100 counterfactual report exactly.

This proves shadow-replay equivalence only.  It does not authorize a formal
active method and never changes the frozen G0 or R3 artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import struct
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import verify_frozen_anchor_manifest  # noqa: E402
from boxfusion.tr3d_r2_cache import tr3d_r2_cache_path  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    sha256_file,
)
from boxfusion.tr3d_r2b_cache import tr3d_r2b_cache_path  # noqa: E402
from boxfusion.tr3d_residual_cache import tr3d_residual_cache_path  # noqa: E402
from tools.audit_tr3d_r3_near_correction import (  # noqa: E402
    IOU_THRESHOLDS,
    PRIMARY_RULE,
    REPORT_SCHEMA as COUNTERFACTUAL_SCHEMA,
    SceneCounterfactual,
    _alignment,
    _gt_boxes,
    _load_export_report,
    _minmax,
    _transform,
    _validate_optional_export_report_paths,
    fixed_rule_scores,
    replacement_rows_for_rule,
    scored_detection_metrics,
    select_one_per_anchor,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r3_shadow_active_paired_audit.v1"


@dataclass(frozen=True)
class RawPrediction:
    labels: tuple[Any, ...]
    corners: tuple[np.ndarray, ...]
    scores: tuple[Any, ...]

    @property
    def count(self) -> int:
        return len(self.labels)


def load_raw_prediction(path: Path) -> RawPrediction:
    """Load one trusted local BoxFusion pickle without normalising bytes."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - pinned local result
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"{path}: expected one-scene outer list")
    rows = payload[0]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: prediction rows must be a list")
    labels: list[Any] = []
    corners: list[np.ndarray] = []
    scores: list[Any] = []
    for index, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"{path}: row {index} must be a 3-tuple")
        label, raw_corners, score = row
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"{path}: row {index} label must be integer")
        geometry = np.asarray(raw_corners)
        if geometry.shape != (8, 3) or geometry.dtype.hasobject:
            raise ValueError(f"{path}: row {index} geometry must be numeric [8,3]")
        if not np.isfinite(geometry).all():
            raise ValueError(f"{path}: row {index} geometry is non-finite")
        if isinstance(score, bool) or not isinstance(score, (float, np.floating)):
            raise ValueError(f"{path}: row {index} score must be floating point")
        if not math.isfinite(float(score)):
            raise ValueError(f"{path}: row {index} score is non-finite")
        labels.append(label)
        corners.append(geometry)
        scores.append(score)
    return RawPrediction(tuple(labels), tuple(corners), tuple(scores))


def _geometry_equal(left: np.ndarray, right: np.ndarray) -> bool:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    return bool(
        lhs.dtype == rhs.dtype
        and lhs.shape == rhs.shape
        and lhs.strides == rhs.strides
        and lhs.flags.c_contiguous == rhs.flags.c_contiguous
        and lhs.flags.f_contiguous == rhs.flags.f_contiguous
        and lhs.tobytes(order="A") == rhs.tobytes(order="A")
    )


def _score_bytes(value: Any) -> bytes:
    if isinstance(value, np.floating):
        return np.asarray(value).tobytes()
    return struct.pack("!d", float(value))


def _label_bytes(value: Any) -> bytes:
    return pickle.dumps(value, protocol=5)


def primary_expected_rows(
    *,
    proposal_ids: object,
    anchor_indices: object,
    tr3d_scores: object,
    anchor_scores: object,
) -> dict[int, int]:
    """Return ``anchor output row -> R3 candidate row`` for the frozen rule."""

    ids = np.asarray(proposal_ids)
    anchors = np.asarray(anchor_indices)
    scores = np.asarray(tr3d_scores)
    frozen_scores = np.asarray(anchor_scores)
    if ids.dtype != np.int64 or anchors.dtype != np.int64:
        raise ValueError("proposal_ids and anchor_indices must be int64")
    if ids.ndim != 1 or anchors.shape != ids.shape or scores.shape != ids.shape:
        raise ValueError("R3 primary arrays must be aligned vectors")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("proposal_ids must be unique")
    if not np.isfinite(scores).all() or not np.isfinite(frozen_scores).all():
        raise ValueError("primary scores must be finite")
    if len(ids) and (np.any(anchors < 0) or np.any(anchors >= len(frozen_scores))):
        raise ValueError("anchor index is out of range")
    result: dict[int, int] = {}
    for anchor in np.unique(anchors):
        rows = np.flatnonzero(anchors == anchor)
        order = np.lexsort((ids[rows], -scores[rows].astype(np.float64)))
        candidate = int(rows[int(order[0])])
        if float(scores[candidate]) > float(frozen_scores[int(anchor)]):
            result[int(anchor)] = candidate
    return result


def audit_prediction_pair(
    baseline: RawPrediction,
    active: RawPrediction,
    *,
    expected_candidates: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Check one output pair and return exact changed-row diagnostics."""

    issues: list[str] = []
    if active.count != baseline.count:
        issues.append(f"prediction count differs ({baseline.count} != {active.count})")
    expected_rows = set(int(value) for value in expected_candidates)
    if any(value < 0 or value >= baseline.count for value in expected_rows):
        issues.append("expected replacement row is out of range")
    actual_changed: set[int] = set()
    exact_selected = exact_unselected = 0
    common = min(baseline.count, active.count)
    for index in range(common):
        if type(active.labels[index]) is not type(baseline.labels[index]) or (
            _label_bytes(active.labels[index]) != _label_bytes(baseline.labels[index])
        ):
            issues.append(f"row {index}: label bytes changed")
        if type(active.scores[index]) is not type(baseline.scores[index]) or (
            _score_bytes(active.scores[index]) != _score_bytes(baseline.scores[index])
        ):
            issues.append(f"row {index}: score bytes changed")
        if not _geometry_equal(active.corners[index], baseline.corners[index]):
            actual_changed.add(index)
        expected = (
            np.asarray(expected_candidates[index])
            if index in expected_rows
            else baseline.corners[index]
        )
        if not _geometry_equal(active.corners[index], expected):
            source = "candidate" if index in expected_rows else "frozen G0"
            issues.append(f"row {index}: geometry bytes differ from expected {source}")
        elif index in expected_rows:
            exact_selected += 1
        else:
            exact_unselected += 1
    expected_changed = {
        index
        for index, candidate in expected_candidates.items()
        if not _geometry_equal(np.asarray(candidate), baseline.corners[index])
    }
    if actual_changed != expected_changed:
        issues.append(
            "actual changed rows differ from expected byte-changing rows: "
            f"actual={sorted(actual_changed)}, expected={sorted(expected_changed)}"
        )
    baseline_order = np.lexsort(
        (
            np.arange(baseline.count, dtype=np.int64),
            -np.asarray([float(value) for value in baseline.scores]),
        )
    )
    active_order = np.lexsort(
        (
            np.arange(active.count, dtype=np.int64),
            -np.asarray([float(value) for value in active.scores]),
        )
    )
    if not np.array_equal(baseline_order, active_order):
        issues.append("stable score order changed")
    return {
        "ok": not issues,
        "issues": issues,
        "rows": baseline.count,
        "eligible_replacement_rows": len(expected_rows),
        "expected_byte_changed_rows": len(expected_changed),
        "actual_byte_changed_rows": len(actual_changed),
        "selected_geometry_exact": exact_selected,
        "unselected_geometry_exact": exact_unselected,
        "labels_exact": not any("label bytes" in value for value in issues),
        "scores_exact": not any("score bytes" in value for value in issues),
        "stable_score_order_exact": not any("score order" in value for value in issues),
    }


def _snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_name": payload["anchor_name"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "scene_list_sha256": payload["scene_list_sha256"],
    }


def _prediction_paths(root: Path, scenes: Sequence[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    found = {path.name for path in root.glob("scene*_boxes.pkl") if path.is_file()}
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise ValueError(f"prediction file set mismatch; missing={missing}, extra={extra}")
    return {scene: root / f"{scene}_boxes.pkl" for scene in scenes}


def _load_counterfactual(path: Path, r3_export_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != COUNTERFACTUAL_SCHEMA:
        raise ValueError("unsupported R3 counterfactual schema")
    contract = payload.get("observer_contract", {})
    if (
        not contract.get("observer_only")
        or contract.get("mutation_enabled")
        or int(contract.get("applied_count", -1)) != 0
        or not contract.get("clip_semantics_unchanged")
    ):
        raise ValueError("counterfactual report violates observer contract")
    if payload.get("lineage", {}).get("r3_export_report_sha256") != r3_export_sha256:
        raise ValueError("counterfactual report belongs to another R3 export")
    counterfactual = payload.get("counterfactual", {})
    if counterfactual.get("mode") != "full100_with_frozen_heldout90":
        raise ValueError("shadow-active audit requires a full100 counterfactual")
    if "all100" not in counterfactual:
        raise ValueError("counterfactual report lacks all100 metrics")
    return payload


def _metric_exact(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    integer_keys = ("predictions", "ground_truth", "matched_tp")
    float_keys = ("average_precision", "final_precision", "final_recall")
    return bool(
        all(int(observed[key]) == int(expected[key]) for key in integer_keys)
        and all(
            np.asarray(float(observed[key]), dtype=np.float64).tobytes()
            == np.asarray(float(expected[key]), dtype=np.float64).tobytes()
            for key in float_keys
        )
    )


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable shadow-active audit exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def audit(args: argparse.Namespace) -> dict[str, Any]:
    # Delayed import lets this auditor land before a shadow materializer API.
    # It depends only on the stable immutable R3 cache contract.
    from boxfusion.tr3d_r3_cache import load_tr3d_r3_cache, tr3d_r3_cache_path

    before = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    before_snapshot = _snapshot(before)
    frozen_root = Path(before["reference_result_root"]).resolve()
    if args.active_root.resolve() == frozen_root:
        raise ValueError("shadow-active root must be separate from frozen G0")
    scenes = read_scene_list(args.scene_list.resolve())
    if len(scenes) != 100:
        raise ValueError("shadow-active paired audit requires exactly full100")
    export_path = args.r3_export_report.resolve()
    export = _load_export_report(export_path, scenes)
    if export.get("prefix_id") != args.prefix_id:
        raise ValueError("R3 export prefix mismatch")
    if export.get("expected_parent_checkpoint_sha256") != args.expected_parent_checkpoint_sha256:
        raise ValueError("R3 checkpoint SHA mismatch")
    if export.get("expected_parent_config_sha256") != args.expected_parent_config_sha256:
        raise ValueError("R3 parent config SHA mismatch")
    fixed_paths = {
        "frozen_manifest": args.frozen_manifest,
        "parent_cache_root": args.parent_cache_root,
        "prefix_manifest": args.prefix_manifest,
        "r3_cache_root": args.r3_cache_root,
        "scene_list": args.scene_list,
        "scans_root": args.scans_root,
    }
    for name, path in fixed_paths.items():
        if Path(str(export.get(name, ""))).resolve() != path.resolve():
            raise ValueError(f"R3 export {name} mismatch")
    config = export["r3_config"]
    r2a_enabled = bool(config.get("r2a_enabled"))
    r2b_enabled = bool(config.get("r2b_enabled"))
    optional_paths = {
        "r2a_cache_root": (args.r2a_cache_root, r2a_enabled),
        "r2b_cache_root": (args.r2b_cache_root, r2b_enabled),
        "frames_root": (args.frames_root, r2a_enabled),
    }
    for name, (path, enabled) in optional_paths.items():
        if (path is not None) != enabled:
            raise ValueError(f"optional path {name} availability mismatch")
        exported = export.get(name)
        if path is None:
            if exported is not None:
                raise ValueError(f"disabled R3 export path {name} must be null")
        elif Path(str(exported)).resolve() != path.resolve():
            raise ValueError(f"R3 export {name} mismatch")
    _validate_optional_export_report_paths(
        export,
        r2a_enabled=r2a_enabled,
        r2b_enabled=r2b_enabled,
        r2a_export_report=args.r2a_export_report,
        r2b_export_report=args.r2b_export_report,
    )
    evidence_hashes = export["parent_evidence_hashes"]
    prefix_rows = load_prefix_manifest(args.prefix_manifest.resolve(), prefix_id=args.prefix_id)
    export_rows = {str(row["scene_id"]): row for row in export["scenes"]}
    active_paths = _prediction_paths(args.active_root.resolve(), scenes)
    counterfactual = _load_counterfactual(
        args.counterfactual_report.resolve(), sha256_file(export_path)
    )

    per_scene: dict[str, Any] = {}
    active_metric_inputs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    total_rows = total_eligible = total_expected_changed = total_actual_changed = 0
    issues: list[str] = []
    for scene_id in scenes:
        baseline_path = frozen_root / f"{scene_id}_boxes.pkl"
        baseline = load_raw_prediction(baseline_path)
        active = load_raw_prediction(active_paths[scene_id])
        anchor_corners = np.stack(baseline.corners) if baseline.count else np.empty((0, 8, 3), dtype=np.float32)
        anchor_scores = np.asarray([float(value) for value in baseline.scores], dtype=np.float64)
        metadata_path = args.scans_root.resolve() / scene_id / f"{scene_id}.txt"
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        manifest_row_sha = frame_tree_sha = ""
        r2a_path = r2b_path = None
        if r2a_enabled:
            row = prefix_rows[scene_id]
            manifest_row_sha = canonical_json_sha256(row)
            bundle = discover_frame_bundle(args.frames_root.resolve(), scene_id)
            frame_tree_sha, _ = frame_artifact_tree(row, bundle)
            r2a_path = tr3d_r2_cache_path(
                args.r2a_cache_root.resolve(), scene_id, args.prefix_id
            )
        if r2b_enabled:
            r2b_path = tr3d_r2b_cache_path(
                args.r2b_cache_root.resolve(), scene_id, args.prefix_id
            )
        sidecar_path = tr3d_r3_cache_path(
            args.r3_cache_root.resolve(), scene_id, args.prefix_id
        )
        if sha256_file(sidecar_path) != export_rows[scene_id]["r3_sidecar_sha256"]:
            raise ValueError(f"{scene_id}: R3 sidecar changed after export")
        cache = load_tr3d_r3_cache(
            sidecar_path,
            parent_tr3d_cache_path=parent_path,
            frozen_anchor_manifest_path=args.frozen_manifest.resolve(),
            anchor_prediction_path=baseline_path,
            anchor_corners_world=anchor_corners,
            anchor_scores=anchor_scores,
            axis_alignment_metadata_path=metadata_path,
            expected_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
            expected_config_sha256=args.expected_parent_config_sha256,
            expected_r3_config_sha256=str(export["r3_config_sha256"]),
            expected_r3_code_sha256=str(export["r3_code_sha256"]),
            parent_r2a_cache_path=r2a_path,
            parent_r2b_cache_path=r2b_path,
            expected_prefix_manifest_row_sha256=manifest_row_sha,
            expected_frame_artifact_tree_sha256=frame_tree_sha,
            expected_r2_config_sha256=str(evidence_hashes.get("r2_config_sha256", "")),
            expected_r2_code_sha256=str(evidence_hashes.get("r2_code_sha256", "")),
            expected_feature_checkpoint_sha256=str(
                evidence_hashes.get("feature_checkpoint_sha256", "")
            ),
            expected_feature_config_sha256=str(
                evidence_hashes.get("feature_config_sha256", "")
            ),
            expected_feature_code_sha256=str(
                evidence_hashes.get("feature_code_sha256", "")
            ),
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
        )
        expected_row_map = primary_expected_rows(
            proposal_ids=cache.proposal_ids,
            anchor_indices=cache.anchor_index,
            tr3d_scores=cache.tr3d_score,
            anchor_scores=anchor_scores,
        )
        transform = _alignment(args.scans_root.resolve(), scene_id)
        selector_scene = SceneCounterfactual(
            scene_id=scene_id,
            anchor_boxes=_minmax(
                _transform(anchor_corners.astype(np.float64), transform)
            ),
            anchor_scores=anchor_scores,
            gt_boxes=np.empty((0, 6), dtype=np.float64),
            candidate_boxes=_minmax(
                _transform(
                    cache.proposal_corners_world.astype(np.float64), transform
                )
            ),
            proposal_ids=np.asarray(cache.proposal_ids, dtype=np.int64),
            anchor_indices=np.asarray(cache.anchor_index, dtype=np.int64),
            anchor_iou=np.asarray(cache.anchor_iou, dtype=np.float64),
            tr3d_score=np.asarray(cache.tr3d_score, dtype=np.float64),
            depth_available=np.asarray(
                cache.r2a_evidence_available, dtype=np.bool_
            ),
            depth_quality=np.asarray(cache.r2a_depth_quality, dtype=np.float64),
            feature_available=np.asarray(
                cache.r2b_multiview_available, dtype=np.bool_
            ),
            feature_cosine=np.asarray(
                cache.r2b_pairwise_cosine_mean, dtype=np.float64
            ),
        )
        shared_selected = select_one_per_anchor(
            selector_scene, fixed_rule_scores(selector_scene)[PRIMARY_RULE]
        )
        shared_applied = replacement_rows_for_rule(
            selector_scene, shared_selected, PRIMARY_RULE
        )
        shared_map = {
            int(cache.anchor_index[row]): int(row) for row in shared_applied
        }
        if shared_map != expected_row_map:
            raise ValueError(
                f"{scene_id}: independent primary selector differs from "
                "counterfactual shared selector"
            )
        expected_candidates = {
            anchor: cache.proposal_corners_world[candidate]
            for anchor, candidate in expected_row_map.items()
        }
        paired = audit_prediction_pair(
            baseline, active, expected_candidates=expected_candidates
        )
        if not paired["ok"]:
            issues.extend(f"{scene_id}: {value}" for value in paired["issues"])
        active_corners = (
            np.stack(active.corners).astype(np.float64, copy=False)
            if active.count
            else np.empty((0, 8, 3), dtype=np.float64)
        )
        active_boxes = _minmax(_transform(active_corners, transform))
        gt_boxes = _gt_boxes(args.gt_root.resolve() / f"{scene_id}_bbox.npy")
        active_metric_inputs.append(
            (
                scene_id,
                active_boxes,
                np.asarray([float(value) for value in active.scores], dtype=np.float64),
                gt_boxes,
            )
        )
        per_scene[scene_id] = {
            **paired,
            "baseline_prediction_sha256": sha256_file(baseline_path),
            "active_prediction_sha256": sha256_file(active_paths[scene_id]),
            "r3_sidecar_sha256": sha256_file(sidecar_path),
            "eligible_anchor_indices": sorted(expected_row_map),
            "selected_proposal_ids": [
                int(cache.proposal_ids[expected_row_map[index]])
                for index in sorted(expected_row_map)
            ],
        }
        total_rows += paired["rows"]
        total_eligible += paired["eligible_replacement_rows"]
        total_expected_changed += paired["expected_byte_changed_rows"]
        total_actual_changed += paired["actual_byte_changed_rows"]

    metrics: dict[str, Any] = {}
    all100 = counterfactual["counterfactual"]["all100"]
    for threshold in IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        observed = scored_detection_metrics(active_metric_inputs, threshold)
        expected = all100["fixed_rules"][PRIMARY_RULE]["thresholds"][key][
            "replacement"
        ]["scored"]
        exact = _metric_exact(observed, expected)
        if not exact:
            issues.append(f"AP{int(threshold * 100):02d}: active metric differs from counterfactual")
        metrics[key] = {
            "active": observed,
            "counterfactual_expected": expected,
            "exact": exact,
        }

    after = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    after_snapshot = _snapshot(after)
    if before_snapshot != after_snapshot:
        issues.append("frozen G0 changed during shadow-active audit")
    return {
        "schema": REPORT_SCHEMA,
        "ok": not issues,
        "shadow_only": True,
        "formal_active_authorized": False,
        "issues": issues,
        "contract": {
            "primary_rule": PRIMARY_RULE,
            "labels_scores_order_count_unchanged": not any(
                any(token in value for token in ("label", "score", "count"))
                for value in issues
            ),
            "selected_candidate_geometry_exact": all(
                row["selected_geometry_exact"] == row["eligible_replacement_rows"]
                for row in per_scene.values()
            ),
            "unselected_geometry_bytes_exact": all(
                row["unselected_geometry_exact"]
                == row["rows"] - row["eligible_replacement_rows"]
                for row in per_scene.values()
            ),
            "counterfactual_all100_metrics_exact": all(
                row["exact"] for row in metrics.values()
            ),
            "frozen_anchor_verified_before_and_after": before_snapshot == after_snapshot,
            "clip_access": False,
            "ground_truth_used_by_auditor_only": True,
        },
        "counts": {
            "scenes": len(scenes),
            "rows": total_rows,
            "eligible_replacement_rows": total_eligible,
            "expected_byte_changed_rows": total_expected_changed,
            "actual_byte_changed_rows": total_actual_changed,
        },
        "metrics": metrics,
        "frozen_anchor": {"before": before_snapshot, "after": after_snapshot},
        "per_scene": per_scene,
        "input_hashes": {
            "frozen_manifest_sha256": sha256_file(args.frozen_manifest.resolve()),
            "r3_export_report_sha256": sha256_file(export_path),
            "counterfactual_report_sha256": sha256_file(
                args.counterfactual_report.resolve()
            ),
            "scene_list_sha256": sha256_file(args.scene_list.resolve()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--r2a-cache-root", type=Path)
    parser.add_argument("--r2b-cache-root", type=Path)
    parser.add_argument("--r3-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-export-report", type=Path)
    parser.add_argument("--r2b-export-report", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--r3-export-report", type=Path, required=True)
    parser.add_argument("--counterfactual-report", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-config-sha256", required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report.resolve()
    manifest = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    frozen_root = Path(manifest["reference_result_root"]).resolve()
    if report_path == frozen_root or frozen_root in report_path.parents:
        raise ValueError("paired audit report must not be written inside frozen G0")
    report = audit(args)
    _write_create_only(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
