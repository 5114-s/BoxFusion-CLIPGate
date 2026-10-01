#!/usr/bin/env python3
"""Fail-closed paired audit for MV3DIS-Depth-Lite S0 shadow.

This audit is deliberately unable to authorize an active association or birth
veto.  It checks the observer contract, aggregates native-relative shadow
diagnostics, reports (but does not require) pickle byte identity, and requires
paired prediction arrays to remain within explicit numeric tolerances.

An optional, predeclared known-event file can be joined to the observer's
generic diagnostics by ``(scene_id, frame_id, proposal_id)``.  This is an
offline consistency check only; it is not a threshold-tuning or activation
path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPORT_SCHEMA = "boxfusion.mv3dis_depth_lite_s0_paired_audit.v1"
OBSERVER_SCHEMA = "boxfusion.mv3dis_depth_lite_s0_shadow.v1"
KNOWN_EVENTS_SCHEMA = "boxfusion.mv3dis_depth_lite_known_events.v1"
MV3DIS_JSON_PREFIX = "MV3DIS-Depth-lite S0 shadow JSON | "
TIMING_PATTERN = re.compile(
    r"^Cost:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s+Average FPS:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return number


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _load_json_text(text: str, *, source: Path | str) -> object:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _nonnegative_float(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _unit_float(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a non-bool integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _optional_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field=field)


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scenes(path: Path) -> tuple[str, ...]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"scene list must be a regular file: {path}")
    scenes = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not scenes:
        raise ValueError("scene list must be non-empty")
    if len(scenes) != len(set(scenes)):
        raise ValueError("scene list must contain unique scene IDs")
    if any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes):
        raise ValueError("scene list contains an invalid ScanNet scene ID")
    return scenes


def _load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ValueError(f"prediction must be a non-empty regular file: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local artifact
        if handle.read(1):
            raise ValueError(f"prediction has trailing bytes: {path}")
    if type(payload) is not list or len(payload) != 1 or type(payload[0]) is not list:
        raise ValueError(f"prediction must contain one list batch: {path}")
    rows: list[tuple[int, np.ndarray, float]] = []
    for index, raw in enumerate(payload[0]):
        if type(raw) is not tuple or len(raw) != 3:
            raise ValueError(f"invalid prediction row {index}: {path}")
        label, raw_corners, raw_score = raw
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"invalid prediction label at row {index}: {path}")
        corners = np.asarray(raw_corners)
        score = _finite_float(raw_score, field=f"{path}: prediction score {index}")
        if (
            int(label) != 0
            or corners.shape != (8, 3)
            or not np.issubdtype(corners.dtype, np.number)
            or not np.isfinite(corners).all()
        ):
            raise ValueError(f"invalid prediction row {index}: {path}")
        rows.append((int(label), np.array(corners, copy=True), score))
    return rows


def compare_predictions(
    control_path: Path,
    observer_path: Path,
    *,
    geometry_atol: float,
    score_atol: float,
) -> dict[str, object]:
    """Require numeric identity within bounds and report stronger identities."""

    control_sha = _sha256(control_path)
    observer_sha = _sha256(observer_path)
    left = _load_prediction(control_path)
    right = _load_prediction(observer_path)
    if len(left) != len(right):
        raise ValueError("control and observer prediction counts differ")
    geometry_max = 0.0
    score_max = 0.0
    exact_arrays = True
    for index, (before, after) in enumerate(zip(left, right)):
        if before[0] != after[0]:
            raise ValueError(f"prediction label differs at row {index}")
        if before[1].shape != after[1].shape:
            raise ValueError(f"prediction geometry shape differs at row {index}")
        exact_arrays = exact_arrays and before[1].dtype == after[1].dtype
        exact_arrays = exact_arrays and bool(np.array_equal(before[1], after[1]))
        exact_arrays = exact_arrays and before[2] == after[2]
        geometry_delta = float(
            np.max(
                np.abs(
                    before[1].astype(np.float64)
                    - after[1].astype(np.float64)
                ),
                initial=0.0,
            )
        )
        score_delta = abs(before[2] - after[2])
        geometry_max = max(geometry_max, geometry_delta)
        score_max = max(score_max, score_delta)
        if geometry_delta > geometry_atol:
            raise ValueError(
                f"prediction geometry drift exceeds tolerance at row {index}: "
                f"{geometry_delta} > {geometry_atol}"
            )
        if score_delta > score_atol:
            raise ValueError(
                f"prediction score drift exceeds tolerance at row {index}: "
                f"{score_delta} > {score_atol}"
            )
    return {
        "rows": len(left),
        "control_sha256": control_sha,
        "observer_sha256": observer_sha,
        "byte_identity": control_sha == observer_sha,
        "exact_array_identity": exact_arrays,
        "numeric_identity_within_tolerance": True,
        "geometry_max_abs_delta": geometry_max,
        "score_max_abs_delta": score_max,
        "geometry_atol": geometry_atol,
        "score_atol": score_atol,
    }


def parse_log(path: Path, *, require_observer: bool) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"log must be a regular file: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    timings = TIMING_PATTERN.findall(text)
    if len(timings) != 1:
        raise ValueError(f"log must contain exactly one timing line: {path}")
    duration_s = _finite_float(timings[0][0], field=f"{path}: duration_s")
    fps = _finite_float(timings[0][1], field=f"{path}: fps")
    if duration_s <= 0.0 or fps <= 0.0:
        raise ValueError(f"log timing must be positive: {path}")
    frames = int(round(duration_s * fps))
    if frames <= 0:
        raise ValueError(f"log frame equivalent must be positive: {path}")
    summaries = [
        _load_json_text(
            line[len(MV3DIS_JSON_PREFIX) :], source=f"{path}: MV3DIS summary"
        )
        for line in text.splitlines()
        if line.startswith(MV3DIS_JSON_PREFIX)
    ]
    if require_observer:
        if len(summaries) != 1:
            raise ValueError(
                f"observer log must contain exactly one MV3DIS JSON summary: {path}"
            )
    elif summaries:
        raise ValueError(f"control log unexpectedly contains MV3DIS output: {path}")
    return {
        "duration_s": duration_s,
        "fps": fps,
        "frame_equivalent": frames,
        **({"mv3dis": summaries[0]} if summaries else {}),
    }


def _validate_config(summary: Mapping[str, object], *, field: str) -> dict[str, object]:
    config = summary.get("effective_config")
    if not isinstance(config, dict):
        raise ValueError(f"{field} effective_config must be a JSON object")
    expected_keys = {
        "enabled",
        "observer_only",
        "max_guides_per_track",
        "max_depth_frames",
        "max_proposals",
        "max_qim_candidates",
        "projection_budget_points",
        "points_per_projection",
        "frame_visibility_threshold",
        "box_visibility_threshold",
        "candidate_dominance_threshold",
        "min_history_views",
        "alpha",
        "max_diagnostic_examples",
    }
    if set(config) != expected_keys:
        raise ValueError(f"{field} effective_config keys do not match S0 schema")
    boolean_fields = ("enabled", "observer_only")
    for name in boolean_fields:
        _strict_bool(config.get(name), field=f"{field} config {name}")
    integer_bounds = {
        "max_guides_per_track": (1, 5),
        "max_depth_frames": (1, 80),
        "max_proposals": (1, 256),
        "max_qim_candidates": (1, 3),
        "projection_budget_points": (1, 8192),
        "points_per_projection": (1, 64),
        "min_history_views": (1, 5),
        "max_diagnostic_examples": (0, 1024),
    }
    for name, (minimum, maximum) in integer_bounds.items():
        value = _integer(config.get(name), field=f"{field} config {name}", minimum=minimum)
        if value > maximum:
            raise ValueError(f"{field} config {name} exceeds {maximum}")
    for name in (
        "frame_visibility_threshold",
        "box_visibility_threshold",
        "candidate_dominance_threshold",
        "alpha",
    ):
        _unit_float(config.get(name), field=f"{field} config {name}")
    frozen = {
        "enabled": True,
        "observer_only": True,
        "points_per_projection": 64,
        "frame_visibility_threshold": 0.30,
        "box_visibility_threshold": 0.90,
        "candidate_dominance_threshold": 0.90,
        "min_history_views": 2,
        "alpha": 0.05,
    }
    mismatched = [name for name, value in frozen.items() if config.get(name) != value]
    if mismatched:
        raise ValueError(f"{field} does not keep frozen S0 config: {mismatched}")
    return config


def _validate_diagnostic(
    raw: object, *, field: str, expected_scene: str
) -> dict[str, object]:
    if not isinstance(raw, list) or len(raw) != 9:
        raise ValueError(f"{field} must be a nine-field JSON array")
    (
        scene_id,
        frame_id_raw,
        proposal_id_raw,
        target_ids_raw,
        would_veto_raw,
        recommended_raw,
        dominance_raw,
        reason,
        candidates_raw,
    ) = raw
    if scene_id != expected_scene:
        raise ValueError(f"{field} scene_id does not match its summary")
    frame_id = _integer(frame_id_raw, field=f"{field} frame_id")
    proposal_id = _integer(proposal_id_raw, field=f"{field} proposal_id")
    if target_ids_raw is None:
        target_ids = None
    else:
        if not isinstance(target_ids_raw, list):
            raise ValueError(f"{field} target_ids must be null or an array")
        target_ids = tuple(
            _integer(value, field=f"{field} target_ids") for value in target_ids_raw
        )
        if tuple(sorted(set(target_ids))) != target_ids:
            raise ValueError(f"{field} target_ids must be sorted and unique")
    would_veto = _strict_bool(would_veto_raw, field=f"{field} would_veto_birth")
    recommended = _optional_integer(recommended_raw, field=f"{field} recommended_track_id")
    if dominance_raw is None:
        dominance = None
    else:
        dominance = _unit_float(dominance_raw, field=f"{field} candidate_dominance")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"{field} reason must be a non-empty string")
    if not isinstance(candidates_raw, list):
        raise ValueError(f"{field} candidates must be an array")
    candidates: list[dict[str, object]] = []
    candidate_ids: set[int] = set()
    candidate_ranks: set[int] = set()
    for candidate_index, candidate_raw in enumerate(candidates_raw):
        candidate_field = f"{field} candidate[{candidate_index}]"
        if not isinstance(candidate_raw, list) or len(candidate_raw) != 8:
            raise ValueError(f"{candidate_field} must be an eight-field array")
        (
            track_raw,
            rank_raw,
            available_raw,
            evaluated_raw,
            supporting_raw,
            support_score_raw,
            complete_raw,
            views_raw,
        ) = candidate_raw
        track_id = _integer(track_raw, field=f"{candidate_field} track_id")
        rank = _integer(rank_raw, field=f"{candidate_field} qim_rank")
        available = _integer(available_raw, field=f"{candidate_field} available")
        evaluated = _integer(evaluated_raw, field=f"{candidate_field} evaluated")
        supporting = _integer(supporting_raw, field=f"{candidate_field} supporting")
        support_score = _nonnegative_float(
            support_score_raw, field=f"{candidate_field} support_score"
        )
        complete = _strict_bool(complete_raw, field=f"{candidate_field} complete")
        if track_id in candidate_ids:
            raise ValueError(f"{field} contains duplicate candidate track ids")
        candidate_ids.add(track_id)
        if rank in candidate_ranks:
            raise ValueError(f"{field} contains duplicate QIM candidate ranks")
        candidate_ranks.add(rank)
        if not supporting <= evaluated <= available:
            raise ValueError(
                f"{candidate_field} must satisfy supporting <= evaluated <= available"
            )
        if support_score > supporting + 1e-12:
            raise ValueError(f"{candidate_field} support_score exceeds supporting views")
        if not isinstance(views_raw, list) or len(views_raw) != available:
            raise ValueError(f"{candidate_field} views do not match available count")
        valid_views = 0
        supporting_views = 0
        views: list[dict[str, object]] = []
        view_frames: set[int] = set()
        for view_index, view_raw in enumerate(views_raw):
            view_field = f"{candidate_field} view[{view_index}]"
            if not isinstance(view_raw, list) or len(view_raw) != 9:
                raise ValueError(f"{view_field} must be a nine-field array")
            (
                view_frame_raw,
                valid_raw,
                frame_visibility_raw,
                box_visibility_raw,
                depth_raw,
                affinity_raw,
                supporting_view_raw,
                points_raw,
                view_reason,
            ) = view_raw
            view_frame = _integer(view_frame_raw, field=f"{view_field} frame_id")
            valid = _strict_bool(valid_raw, field=f"{view_field} valid")
            supporting_view = _strict_bool(
                supporting_view_raw, field=f"{view_field} supporting"
            )
            points = _integer(points_raw, field=f"{view_field} projected_points")
            if view_frame in view_frames:
                raise ValueError(f"{candidate_field} contains duplicate view frames")
            view_frames.add(view_frame)
            metrics = []
            for metric_name, metric_raw in (
                ("frame_visibility", frame_visibility_raw),
                ("box_visibility", box_visibility_raw),
                ("box_depth_consistency", depth_raw),
                ("affinity", affinity_raw),
            ):
                if metric_raw is None:
                    metric = None
                else:
                    metric = _unit_float(metric_raw, field=f"{view_field} {metric_name}")
                metrics.append(metric)
            if valid != all(metric is not None for metric in metrics):
                raise ValueError(f"{view_field} valid flag disagrees with metrics")
            if supporting_view and not valid:
                raise ValueError(f"{view_field} invalid view cannot support")
            if not isinstance(view_reason, str) or not view_reason:
                raise ValueError(f"{view_field} reason must be a non-empty string")
            valid_views += int(valid)
            supporting_views += int(supporting_view)
            if view_frame >= frame_id:
                raise ValueError(f"{view_field} is not causal")
            if points > 64:
                raise ValueError(f"{view_field} exceeds frozen projection size")
            if valid:
                expected_support = bool(metrics[0] > 0.30 and metrics[1] > 0.90)
                if supporting_view != expected_support:
                    raise ValueError(
                        f"{view_field} supporting flag disagrees with S0 thresholds"
                    )
            views.append(
                {
                    "frame_id": view_frame,
                    "valid": valid,
                    "frame_visibility": metrics[0],
                    "box_visibility": metrics[1],
                    "box_depth_consistency": metrics[2],
                    "affinity": metrics[3],
                    "supporting": supporting_view,
                    "projected_points": points,
                    "reason": view_reason,
                }
            )
        if valid_views != evaluated or supporting_views != supporting:
            raise ValueError(f"{candidate_field} view counts are inconsistent")
        if complete != (evaluated == available):
            raise ValueError(f"{candidate_field} complete flag is inconsistent")
        expected_support_score = sum(
            float(view["affinity"])
            for view in views
            if bool(view["supporting"])
        )
        if not math.isclose(
            support_score,
            expected_support_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{candidate_field} support score is inconsistent")
        candidates.append(
            {
                "track_id": track_id,
                "qim_rank": rank,
                "history_views_available": available,
                "history_views_evaluated": evaluated,
                "supporting_views": supporting,
                "support_score": support_score,
                "complete": complete,
                "views": views,
            }
        )
    if candidate_ranks != set(range(len(candidates))):
        raise ValueError(f"{field} QIM candidate ranks must be contiguous")
    total_support = sum(float(candidate["support_score"]) for candidate in candidates)
    best = (
        sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate["support_score"]),
                -int(candidate["supporting_views"]),
                int(candidate["qim_rank"]),
                int(candidate["track_id"]),
            ),
        )[0]
        if candidates
        else None
    )
    expected_dominance = (
        float(best["support_score"]) / total_support
        if best is not None and total_support > 0.0
        else None
    )
    if (dominance is None) != (expected_dominance is None) or (
        dominance is not None
        and not math.isclose(
            dominance,
            float(expected_dominance),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{field} candidate dominance is inconsistent")
    eligible = [
        candidate
        for candidate in candidates
        if bool(candidate["complete"])
        and int(candidate["supporting_views"]) >= 2
        and float(candidate["support_score"]) > 0.0
    ]
    expected_veto = bool(
        candidates
        and all(bool(candidate["complete"]) for candidate in candidates)
        and len(eligible) == 1
        and best is eligible[0]
        and dominance is not None
        and dominance > 0.90
    )
    if would_veto != expected_veto:
        raise ValueError(f"{field} veto decision is inconsistent with S0 evidence")
    if would_veto:
        if recommended is None or dominance is None:
            raise ValueError(f"{field} veto must include recommendation and dominance")
        if recommended not in candidate_ids:
            raise ValueError(f"{field} recommended track is absent from candidates")
    elif recommended is not None:
        raise ValueError(f"{field} non-veto must not recommend a track")
    return {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "proposal_id": proposal_id,
        "native_target_track_ids": target_ids,
        "would_veto_birth": would_veto,
        "recommended_track_id": recommended,
        "candidate_dominance": dominance,
        "reason": reason,
        "candidates": candidates,
    }


def validate_summary(raw: object, *, scene_id: str) -> dict[str, object]:
    field = f"{scene_id}: MV3DIS"
    if not isinstance(raw, dict):
        raise ValueError(f"{field} summary must be a JSON object")
    if raw.get("schema") != OBSERVER_SCHEMA:
        raise ValueError(f"{field} has unexpected schema")
    if raw.get("scene_id") != scene_id:
        raise ValueError(f"{field} scene_id mismatch")
    safety_expected = {
        "enabled": True,
        "observer_only": True,
        "active_authorized": False,
        "training_free": True,
        "unsupervised": True,
        "causal": True,
        "bounded_history": True,
        "online_parameter_update": False,
        "ground_truth_access": False,
        "semantic_access": False,
        "semantic_mutation": False,
        "detector_score_access": False,
        "puf_access": False,
        "native_outputs_mutated": False,
        "guide_quality_computed": True,
        "fusion_weights_computed": False,
        "fusion_weights_applied": False,
        "birth_veto_applied": False,
        "hardcoded_scene_event_access": False,
        "geometry_adapter_available": True,
    }
    safety = {}
    for name, expected in safety_expected.items():
        actual = _strict_bool(raw.get(name), field=f"{field} {name}")
        safety[name] = actual == expected
    if not all(safety.values()):
        raise ValueError(f"{field} observer safety contract failed: {safety}")
    config = _validate_config(raw, field=field)

    count_names = (
        "queries",
        "commits",
        "proposals",
        "proposal_cap_batches",
        "invalid_frame_batches",
        "guide_quality_rows_valid",
        "guide_quality_rows_invalid",
        "veto_recommendations",
        "veto_evaluable",
        "veto_correct",
        "veto_wrong",
        "veto_on_native_birth",
        "native_history",
        "native_birth",
        "native_unresolved",
        "native_diagnostics_skipped",
        "geometry_calls",
        "geometry_errors",
        "projection_points",
        "guide_quality_projection_points",
        "birth_veto_projection_points",
        "guide_quality_budget_exhaustions",
        "birth_veto_budget_exhaustions",
        "guides_committed",
        "guides_replaced_same_frame",
        "guides_evicted_track_cap",
        "guides_evicted_frame_cap",
        "committed_frames_evicted",
        "max_committed_frames_observed",
        "max_tracks_observed",
        "max_guides_observed",
        "pipeline_query_calls",
        "pipeline_commit_calls",
        "committed_frames_retained",
        "tracks_retained",
        "guides_retained",
    )
    counts = {
        name: _integer(raw.get(name), field=f"{field} {name}")
        for name in count_names
    }
    if counts["queries"] != counts["commits"]:
        raise ValueError(f"{field} must close every query with one commit")
    if (
        counts["guide_quality_rows_valid"]
        + counts["guide_quality_rows_invalid"]
        != counts["proposals"]
    ):
        raise ValueError(f"{field} guide-quality rows do not cover proposals")
    if counts["veto_correct"] + counts["veto_wrong"] != counts["veto_evaluable"]:
        raise ValueError(f"{field} veto precision counts are inconsistent")
    if counts["veto_evaluable"] > counts["veto_recommendations"]:
        raise ValueError(f"{field} evaluable vetoes exceed recommendations")
    if counts["veto_recommendations"] > counts["proposals"]:
        raise ValueError(f"{field} recommendations exceed proposals")
    if counts["veto_on_native_birth"] > counts["veto_wrong"]:
        raise ValueError(f"{field} native-birth vetoes exceed wrong vetoes")
    native_total = (
        counts["native_history"]
        + counts["native_birth"]
        + counts["native_unresolved"]
        + counts["native_diagnostics_skipped"]
    )
    if native_total != counts["proposals"]:
        raise ValueError(f"{field} native diagnostic rows do not cover proposals")
    if counts["pipeline_query_calls"] != counts["queries"]:
        raise ValueError(f"{field} pipeline query timing coverage is incomplete")
    if counts["pipeline_commit_calls"] != counts["commits"]:
        raise ValueError(f"{field} pipeline commit timing coverage is incomplete")
    if counts["committed_frames_retained"] > int(config["max_depth_frames"]):
        raise ValueError(f"{field} retained frame bound is violated")
    if counts["max_committed_frames_observed"] > int(config["max_depth_frames"]):
        raise ValueError(f"{field} observed frame bound is violated")
    if counts["guides_retained"] > counts["tracks_retained"] * int(config["max_guides_per_track"]):
        raise ValueError(f"{field} retained guide bound is violated")
    if counts["projection_points"] != (
        counts["guide_quality_projection_points"]
        + counts["birth_veto_projection_points"]
    ):
        raise ValueError(f"{field} projection point counts are inconsistent")
    if counts["proposal_cap_batches"] > counts["queries"]:
        raise ValueError(f"{field} proposal-cap batches exceed queries")
    if counts["invalid_frame_batches"] > counts["queries"]:
        raise ValueError(f"{field} invalid-frame batches exceed queries")
    if counts["committed_frames_retained"] > counts["max_committed_frames_observed"]:
        raise ValueError(f"{field} retained frames exceed observed maximum")
    if counts["tracks_retained"] > counts["max_tracks_observed"]:
        raise ValueError(f"{field} retained tracks exceed observed maximum")
    if counts["guides_retained"] > counts["max_guides_observed"]:
        raise ValueError(f"{field} retained guides exceed observed maximum")

    timing_names = (
        "query_ms_total",
        "query_ms_max",
        "commit_ms_total",
        "commit_ms_max",
        "pipeline_query_ms_total",
        "pipeline_query_ms_max",
        "pipeline_commit_ms_total",
        "pipeline_commit_ms_max",
        "query_ms_mean",
        "query_ms_p95",
        "commit_ms_mean",
        "commit_ms_p95",
        "pipeline_query_ms_mean",
        "pipeline_commit_ms_mean",
    )
    timings = {
        name: _nonnegative_float(raw.get(name), field=f"{field} {name}")
        for name in timing_names
    }
    for stage in ("query", "commit"):
        calls = counts[f"pipeline_{stage}_calls"]
        expected_mean = timings[f"pipeline_{stage}_ms_total"] / calls if calls else 0.0
        if not math.isclose(
            timings[f"pipeline_{stage}_ms_mean"], expected_mean, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{field} pipeline {stage} mean is inconsistent")
        internal_count_name = "queries" if stage == "query" else "commits"
        internal_count = counts[internal_count_name]
        internal_mean = (
            timings[f"{stage}_ms_total"] / internal_count
            if internal_count
            else 0.0
        )
        if not math.isclose(
            timings[f"{stage}_ms_mean"], internal_mean, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{field} {stage} mean is inconsistent")
        if timings[f"{stage}_ms_p95"] > timings[f"{stage}_ms_max"] + 1e-12:
            raise ValueError(f"{field} {stage} p95 exceeds max")
        if timings[f"pipeline_{stage}_ms_max"] > timings[f"pipeline_{stage}_ms_total"] + 1e-12:
            raise ValueError(f"{field} pipeline {stage} max exceeds total")

    precision_raw = raw.get("veto_precision")
    if counts["veto_evaluable"] == 0:
        if precision_raw is not None:
            raise ValueError(f"{field} veto_precision must be null with no samples")
        precision = None
    else:
        precision = _unit_float(precision_raw, field=f"{field} veto_precision")
        expected = counts["veto_correct"] / counts["veto_evaluable"]
        if not math.isclose(precision, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{field} veto_precision is inconsistent")

    diagnostics_raw = raw.get("diagnostic_examples")
    if not isinstance(diagnostics_raw, list):
        raise ValueError(f"{field} diagnostic_examples must be an array")
    if len(diagnostics_raw) > int(config["max_diagnostic_examples"]):
        raise ValueError(f"{field} diagnostic examples exceed configured cap")
    diagnostics = [
        _validate_diagnostic(item, field=f"{field} diagnostic[{index}]", expected_scene=scene_id)
        for index, item in enumerate(diagnostics_raw)
    ]
    keys = [(row["scene_id"], row["frame_id"], row["proposal_id"]) for row in diagnostics]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field} contains duplicate diagnostic keys")

    invalid_reasons = raw.get("invalid_frame_reasons")
    if not isinstance(invalid_reasons, list):
        raise ValueError(f"{field} invalid_frame_reasons must be an array")
    seen_reasons: set[str] = set()
    invalid_reason_total = 0
    for index, item in enumerate(invalid_reasons):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{field} invalid_frame_reasons[{index}] is malformed")
        reason, count = item
        if not isinstance(reason, str) or not reason or reason in seen_reasons:
            raise ValueError(f"{field} invalid frame reason key is invalid")
        seen_reasons.add(reason)
        invalid_reason_total += _integer(
            count, field=f"{field} invalid frame reason count", minimum=1
        )
    if invalid_reason_total != counts["invalid_frame_batches"]:
        raise ValueError(f"{field} invalid frame reasons are inconsistent")

    return {
        "summary": raw,
        "counts": counts,
        "timings": timings,
        "veto_precision": precision,
        "diagnostics": diagnostics,
        "safety": safety,
    }


def _load_known_events(path: Path, *, allowed_scenes: set[str]) -> list[dict[str, object]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"known-events JSON must be a regular file: {path}")
    raw = _load_json_text(path.read_text(encoding="utf-8"), source=path)
    if not isinstance(raw, dict) or set(raw) != {"schema", "events"}:
        raise ValueError("known-events JSON must contain exactly schema and events")
    if raw.get("schema") != KNOWN_EVENTS_SCHEMA:
        raise ValueError("known-events JSON has unexpected schema")
    events_raw = raw.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        raise ValueError("known-events JSON events must be a non-empty array")
    allowed_keys = {
        "scene_id",
        "frame_id",
        "proposal_id",
        "expected_native_target_track_ids",
        "expected_would_veto_birth",
        "expected_recommended_track_id",
    }
    required_keys = {"scene_id", "frame_id", "proposal_id"}
    events: list[dict[str, object]] = []
    for index, event_raw in enumerate(events_raw):
        field = f"{path}: event[{index}]"
        if not isinstance(event_raw, dict):
            raise ValueError(f"{field} must be a JSON object")
        if not required_keys <= set(event_raw) <= allowed_keys:
            raise ValueError(f"{field} has missing or unknown keys")
        scene_id = event_raw.get("scene_id")
        if not isinstance(scene_id, str) or scene_id not in allowed_scenes:
            raise ValueError(f"{field} scene_id is not in the paired scene list")
        event: dict[str, object] = {
            "scene_id": scene_id,
            "frame_id": _integer(event_raw.get("frame_id"), field=f"{field} frame_id"),
            "proposal_id": _integer(
                event_raw.get("proposal_id"), field=f"{field} proposal_id"
            ),
        }
        if "expected_native_target_track_ids" in event_raw:
            values = event_raw["expected_native_target_track_ids"]
            if values is None:
                event["expected_native_target_track_ids"] = None
            else:
                if not isinstance(values, list):
                    raise ValueError(f"{field} expected targets must be null or an array")
                targets = tuple(
                    _integer(value, field=f"{field} expected target")
                    for value in values
                )
                if tuple(sorted(set(targets))) != targets:
                    raise ValueError(f"{field} expected targets must be sorted and unique")
                event["expected_native_target_track_ids"] = targets
        if "expected_would_veto_birth" in event_raw:
            event["expected_would_veto_birth"] = _strict_bool(
                event_raw["expected_would_veto_birth"],
                field=f"{field} expected_would_veto_birth",
            )
        if "expected_recommended_track_id" in event_raw:
            event["expected_recommended_track_id"] = _optional_integer(
                event_raw["expected_recommended_track_id"],
                field=f"{field} expected_recommended_track_id",
            )
        events.append(event)
    keys = [(event["scene_id"], event["frame_id"], event["proposal_id"]) for event in events]
    if len(keys) != len(set(keys)):
        raise ValueError("known-events JSON contains duplicate event keys")
    return events


def _join_known_events(
    events: Sequence[Mapping[str, object]],
    diagnostics: Mapping[tuple[str, int, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    joined: list[dict[str, object]] = []
    for event in events:
        key = (
            str(event["scene_id"]),
            int(event["frame_id"]),
            int(event["proposal_id"]),
        )
        diagnostic = diagnostics.get(key)
        if diagnostic is None:
            raise ValueError(f"known event is absent from capped diagnostics: {key}")
        checks: dict[str, bool] = {}
        for expected_name, observed_name in (
            ("expected_native_target_track_ids", "native_target_track_ids"),
            ("expected_would_veto_birth", "would_veto_birth"),
            ("expected_recommended_track_id", "recommended_track_id"),
        ):
            if expected_name in event:
                checks[expected_name] = event[expected_name] == diagnostic[observed_name]
        if not all(checks.values()):
            raise ValueError(f"known event expectation mismatch for {key}: {checks}")
        joined.append(
            {
                "scene_id": key[0],
                "frame_id": key[1],
                "proposal_id": key[2],
                "expectation_checks": checks,
                "diagnostic": diagnostic,
            }
        )
    return joined


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit(args: argparse.Namespace) -> dict[str, object]:
    for name in (
        "geometry_atol",
        "score_atol",
        "min_fps_ratio",
        "max_pipeline_ms_per_input_frame",
        "max_query_p95_ms",
        "max_commit_p95_ms",
        "min_veto_precision",
        "min_veto_coverage",
    ):
        _finite_float(getattr(args, name), field=f"--{name.replace('_', '-')}")
    if not 0.0 <= args.geometry_atol <= 1e-3:
        raise ValueError("--geometry-atol must lie in [0, 1e-3]")
    if not 0.0 <= args.score_atol <= 1e-5:
        raise ValueError("--score-atol must lie in [0, 1e-5]")
    if not 0.0 < args.min_fps_ratio <= 1.0:
        raise ValueError("--min-fps-ratio must lie in (0, 1]")
    if args.max_pipeline_ms_per_input_frame <= 0.0:
        raise ValueError("--max-pipeline-ms-per-input-frame must be positive")
    if args.max_query_p95_ms <= 0.0 or args.max_commit_p95_ms <= 0.0:
        raise ValueError("p95 latency bounds must be positive")
    if not 0.0 <= args.min_veto_precision <= 1.0:
        raise ValueError("--min-veto-precision must lie in [0, 1]")
    if not 0.0 <= args.min_veto_coverage <= 1.0:
        raise ValueError("--min-veto-coverage must lie in [0, 1]")
    if isinstance(args.min_veto_evaluable, bool) or args.min_veto_evaluable < 0:
        raise ValueError("--min-veto-evaluable must be non-negative")

    scenes = read_scenes(args.scene_list.resolve())
    per_scene: dict[str, object] = {}
    diagnostics: dict[tuple[str, int, int], Mapping[str, object]] = {}
    totals = {
        "frames": 0,
        "pipeline_query_ms": 0.0,
        "pipeline_commit_ms": 0.0,
        "veto_recommendations": 0,
        "veto_evaluable": 0,
        "veto_correct": 0,
        "veto_wrong": 0,
        "native_history": 0,
        "native_birth": 0,
        "native_unresolved": 0,
        "proposals": 0,
        "byte_identity_scenes": 0,
        "exact_array_identity_scenes": 0,
        "rows": 0,
    }
    min_fps_ratio = float("inf")
    max_query_p95 = 0.0
    max_commit_p95 = 0.0
    for scene_id in scenes:
        identity = compare_predictions(
            args.control_root.resolve() / f"{scene_id}_boxes.pkl",
            args.observer_root.resolve() / f"{scene_id}_boxes.pkl",
            geometry_atol=args.geometry_atol,
            score_atol=args.score_atol,
        )
        control = parse_log(
            args.control_log_root.resolve() / f"{scene_id}.log",
            require_observer=False,
        )
        observer = parse_log(
            args.observer_log_root.resolve() / f"{scene_id}.log",
            require_observer=True,
        )
        if observer["frame_equivalent"] != control["frame_equivalent"]:
            raise ValueError(
                f"{scene_id}: control/observer frame equivalents differ"
            )
        validated = validate_summary(observer["mv3dis"], scene_id=scene_id)
        counts = validated["counts"]
        timings = validated["timings"]
        fps_ratio = _finite_float(
            float(observer["fps"]) / float(control["fps"]),
            field=f"{scene_id}: observer/control FPS ratio",
        )
        if fps_ratio < args.min_fps_ratio:
            raise ValueError(f"{scene_id}: observer/control FPS ratio is too low")
        if timings["query_ms_p95"] > args.max_query_p95_ms:
            raise ValueError(f"{scene_id}: MV3DIS query p95 exceeds threshold")
        if timings["commit_ms_p95"] > args.max_commit_p95_ms:
            raise ValueError(f"{scene_id}: MV3DIS commit p95 exceeds threshold")
        frames = int(observer["frame_equivalent"])
        pipeline_ms = timings["pipeline_query_ms_total"] + timings["pipeline_commit_ms_total"]
        scene_pipeline_per_frame = pipeline_ms / frames
        if scene_pipeline_per_frame > args.max_pipeline_ms_per_input_frame:
            raise ValueError(
                f"{scene_id}: MV3DIS pipeline overhead exceeds per-input-frame threshold"
            )
        for row in validated["diagnostics"]:
            key = (str(row["scene_id"]), int(row["frame_id"]), int(row["proposal_id"]))
            if key in diagnostics:
                raise ValueError(f"duplicate cross-scene diagnostic key: {key}")
            diagnostics[key] = row
        totals["frames"] += frames
        totals["pipeline_query_ms"] += timings["pipeline_query_ms_total"]
        totals["pipeline_commit_ms"] += timings["pipeline_commit_ms_total"]
        for name in (
            "veto_recommendations",
            "veto_evaluable",
            "veto_correct",
            "veto_wrong",
            "native_history",
            "native_birth",
            "native_unresolved",
            "proposals",
        ):
            totals[name] += counts[name]
        totals["byte_identity_scenes"] += int(identity["byte_identity"])
        totals["exact_array_identity_scenes"] += int(identity["exact_array_identity"])
        totals["rows"] += int(identity["rows"])
        min_fps_ratio = min(min_fps_ratio, fps_ratio)
        max_query_p95 = max(max_query_p95, timings["query_ms_p95"])
        max_commit_p95 = max(max_commit_p95, timings["commit_ms_p95"])
        per_scene[scene_id] = {
            "identity": identity,
            "control_timing": control,
            "observer_timing": {
                "duration_s": observer["duration_s"],
                "fps": observer["fps"],
                "frame_equivalent": observer["frame_equivalent"],
            },
            "fps_ratio": fps_ratio,
            "pipeline_ms": pipeline_ms,
            "pipeline_ms_per_input_frame": scene_pipeline_per_frame,
            "veto_precision": validated["veto_precision"],
            "veto_coverage": (
                counts["veto_correct"] / counts["native_history"]
                if counts["native_history"]
                else None
            ),
            "summary": validated["summary"],
        }

    veto_precision = (
        totals["veto_correct"] / totals["veto_evaluable"]
        if totals["veto_evaluable"]
        else None
    )
    veto_coverage = (
        totals["veto_correct"] / totals["native_history"]
        if totals["native_history"]
        else None
    )
    if totals["veto_evaluable"] < args.min_veto_evaluable:
        raise ValueError("aggregate evaluable veto samples are below threshold")
    if args.min_veto_precision > 0.0 and (
        veto_precision is None or veto_precision < args.min_veto_precision
    ):
        raise ValueError("aggregate veto precision is below threshold")
    if args.min_veto_coverage > 0.0 and (
        veto_coverage is None or veto_coverage < args.min_veto_coverage
    ):
        raise ValueError("aggregate veto coverage is below threshold")
    pipeline_total_ms = totals["pipeline_query_ms"] + totals["pipeline_commit_ms"]
    pipeline_per_frame = pipeline_total_ms / totals["frames"]
    if pipeline_per_frame > args.max_pipeline_ms_per_input_frame:
        raise ValueError("aggregate MV3DIS pipeline overhead exceeds threshold")

    known_event_rows: list[dict[str, object]] = []
    known_events_sha256 = None
    if args.known_events_json is not None:
        known_path = args.known_events_json.resolve()
        events = _load_known_events(known_path, allowed_scenes=set(scenes))
        known_event_rows = _join_known_events(events, diagnostics)
        known_events_sha256 = _sha256(known_path)

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "observer_only": True,
        "shadow_only": True,
        "active_authorized": False,
        "activation_decision": "not_authorized_by_design",
        "accuracy_gain_measured": False,
        "ground_truth_access": False,
        "native_relative_evaluation": True,
        "training_free": True,
        "unsupervised": True,
        "causal": True,
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": _sha256(args.scene_list.resolve()),
        "scene_count": len(scenes),
        "prediction_rows": totals["rows"],
        "prediction_numeric_identity_within_tolerance": True,
        "prediction_byte_identity_scene_count": totals["byte_identity_scenes"],
        "same_prediction_bytes": totals["byte_identity_scenes"] == len(scenes),
        "prediction_exact_array_identity_scene_count": totals["exact_array_identity_scenes"],
        "same_prediction_arrays_exact": totals["exact_array_identity_scenes"] == len(scenes),
        "geometry_atol": args.geometry_atol,
        "score_atol": args.score_atol,
        "frames": totals["frames"],
        "proposals": totals["proposals"],
        "veto_recommendations": totals["veto_recommendations"],
        "veto_evaluable": totals["veto_evaluable"],
        "veto_correct": totals["veto_correct"],
        "veto_wrong": totals["veto_wrong"],
        "native_history": totals["native_history"],
        "native_birth": totals["native_birth"],
        "native_unresolved": totals["native_unresolved"],
        "veto_precision": veto_precision,
        "veto_coverage": veto_coverage,
        "veto_precision_definition": "veto_correct / veto_evaluable",
        "veto_coverage_definition": "veto_correct / native_history",
        "min_veto_evaluable": args.min_veto_evaluable,
        "min_veto_precision": args.min_veto_precision,
        "min_veto_coverage": args.min_veto_coverage,
        "pipeline_query_ms_total": totals["pipeline_query_ms"],
        "pipeline_commit_ms_total": totals["pipeline_commit_ms"],
        "pipeline_ms_total": pipeline_total_ms,
        "pipeline_ms_per_input_frame": pipeline_per_frame,
        "max_pipeline_ms_per_input_frame": args.max_pipeline_ms_per_input_frame,
        "minimum_observer_control_fps_ratio": min_fps_ratio,
        "required_minimum_fps_ratio": args.min_fps_ratio,
        "maximum_scene_query_p95_ms": max_query_p95,
        "maximum_scene_commit_p95_ms": max_commit_p95,
        "required_max_query_p95_ms": args.max_query_p95_ms,
        "required_max_commit_p95_ms": args.max_commit_p95_ms,
        "known_event_join_enabled": args.known_events_json is not None,
        "known_events_sha256": known_events_sha256,
        "known_event_count": len(known_event_rows),
        "known_events_all_joined": True if args.known_events_json is not None else None,
        "known_event_rows": known_event_rows,
        "per_scene": per_scene,
    }
    _write_json(args.output.resolve(), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--control-root", type=Path, required=True)
    value.add_argument("--observer-root", type=Path, required=True)
    value.add_argument("--control-log-root", type=Path, required=True)
    value.add_argument("--observer-log-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--known-events-json", type=Path)
    value.add_argument("--geometry-atol", type=float, default=5e-5)
    value.add_argument("--score-atol", type=float, default=1e-8)
    value.add_argument("--min-fps-ratio", type=float, default=0.95)
    value.add_argument("--max-pipeline-ms-per-input-frame", type=float, default=1.0)
    value.add_argument("--max-query-p95-ms", type=float, default=25.0)
    value.add_argument("--max-commit-p95-ms", type=float, default=5.0)
    value.add_argument("--min-veto-evaluable", type=int, default=0)
    value.add_argument("--min-veto-precision", type=float, default=0.0)
    value.add_argument("--min-veto-coverage", type=float, default=0.0)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    report = audit(parser().parse_args(argv))
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
