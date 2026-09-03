#!/usr/bin/env python3
"""Aggregate strict-live Stream3Dv2 diagnostics and per-scene runtime logs.

The scene diagnostics intentionally contain distribution summaries instead of
raw frame latency samples.  Consequently, an official-set mean can be
reconstructed by sample-count weighting, while cross-scene p50/p95 values are
only sample-count-weighted approximations.  The JSON output names those
quantities accordingly and never presents them as pooled raw percentiles.

The reported FPS is internal pipeline throughput.  Each scene log prints
``Cost`` and ``Average FPS`` rounded to two decimals; the aggregate frame count
is reconstructed as ``Cost * FPS`` and divided by total cost.  Evaluation and
process startup are outside that clock.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.stream3dv2_live.official100_summary.v1"
DIAGNOSTIC_SCHEMA = "boxfusion.stream3dv2_live.v1"
DEFAULT_SCENE_LIST = (
    REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
)
DEFAULT_DIAGNOSTICS_ROOT = (
    REPOSITORY_ROOT / "diagnostics/cbest_f4_stream3dv2_live/route"
)
DEFAULT_SCENE_LOG_ROOT = (
    REPOSITORY_ROOT / "logs/scannet_cbest_f4_stream3dv2_live_score05/scenes"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "logs/scannet_cbest_f4_stream3dv2_live_score05/OFFICIAL100_LIVE_SUMMARY.json"
)

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COST_FPS_RE = re.compile(
    rf"Cost:\s*({_NUMBER})\s*s\s+Average\s+FPS:\s*({_NUMBER})",
    re.IGNORECASE,
)
STRICT_SUMMARY_MARKER = "Strict live summary |"


class SummaryInputError(RuntimeError):
    """A top-level input cannot be interpreted safely."""


def read_scene_list(path: Path) -> list[str]:
    if not path.is_file():
        raise SummaryInputError(f"scene list does not exist: {path}")
    scenes = [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes:
        raise SummaryInputError(f"scene list is empty: {path}")
    duplicates = sorted(scene for scene, count in Counter(scenes).items() if count > 1)
    if duplicates:
        raise SummaryInputError(f"duplicate scene IDs in {path}: {duplicates}")
    return scenes


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _nonnegative_int(
    value: object,
    *,
    field: str,
    issues: list[dict[str, Any]],
    scene: str,
) -> int | None:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < 0
    ):
        issues.append(
            {"kind": "invalid_nonnegative_integer", "scene": scene, "field": field, "actual": value}
        )
        return None
    return int(value)


def _nonnegative_number(
    value: object,
    *,
    field: str,
    issues: list[dict[str, Any]],
    scene: str,
) -> float | None:
    if not _is_number(value) or float(value) < 0.0:
        issues.append(
            {"kind": "invalid_nonnegative_number", "scene": scene, "field": field, "actual": value}
        )
        return None
    return float(value)


def _mapping(
    value: object,
    *,
    field: str,
    issues: list[dict[str, Any]],
    scene: str,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.append(
            {"kind": "invalid_mapping", "scene": scene, "field": field, "actual_type": type(value).__name__}
        )
        return None
    return value


def _parse_distribution(
    value: object,
    *,
    scene: str,
    stage: str,
    issues: list[dict[str, Any]],
) -> dict[str, float | int] | None:
    row = _mapping(value, field=f"timing_ms.{stage}", issues=issues, scene=scene)
    if row is None:
        return None
    count = _nonnegative_int(
        row.get("count"), field=f"timing_ms.{stage}.count", issues=issues, scene=scene
    )
    if count is None:
        return None
    result: dict[str, float | int] = {"count": count}
    if count == 0:
        for metric in ("mean", "p50", "p95", "max"):
            if row.get(metric) is not None:
                issues.append(
                    {
                        "kind": "nonempty_zero_count_distribution",
                        "scene": scene,
                        "field": f"timing_ms.{stage}.{metric}",
                        "actual": row.get(metric),
                    }
                )
                return None
        return result
    for metric in ("mean", "p50", "p95", "max"):
        parsed = _nonnegative_number(
            row.get(metric),
            field=f"timing_ms.{stage}.{metric}",
            issues=issues,
            scene=scene,
        )
        if parsed is None:
            return None
        result[metric] = parsed
    if not (
        float(result["max"]) >= float(result["p95"]) >= float(result["p50"])
    ):
        issues.append(
            {
                "kind": "invalid_distribution_order",
                "scene": scene,
                "field": f"timing_ms.{stage}",
                "actual": result,
            }
        )
        return None
    return result


def _read_log(
    path: Path, *, scene: str, issues: list[dict[str, Any]]
) -> dict[str, float] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        issues.append({"kind": "unreadable_scene_log", "scene": scene, "error": str(error)})
        return None
    matches = COST_FPS_RE.findall(text)
    if len(matches) != 1:
        issues.append(
            {"kind": "cost_fps_line_count", "scene": scene, "expected": 1, "actual": len(matches)}
        )
        return None
    if text.count(STRICT_SUMMARY_MARKER) != 1:
        issues.append(
            {
                "kind": "strict_live_summary_line_count",
                "scene": scene,
                "expected": 1,
                "actual": text.count(STRICT_SUMMARY_MARKER),
            }
        )
        return None
    cost = float(matches[0][0])
    fps = float(matches[0][1])
    if not math.isfinite(cost) or not math.isfinite(fps) or cost <= 0.0 or fps <= 0.0:
        issues.append(
            {"kind": "invalid_cost_or_fps", "scene": scene, "cost_seconds": cost, "fps": fps}
        )
        return None
    return {
        "cost_seconds": cost,
        "reported_fps": fps,
        "reconstructed_frames_approx": cost * fps,
    }


def _load_scene(
    *, scene: str, diagnostic_path: Path, log_path: Path, issues: list[dict[str, Any]]
) -> dict[str, Any] | None:
    try:
        payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.append({"kind": "unreadable_diagnostic", "scene": scene, "error": str(error)})
        return None
    if not isinstance(payload, Mapping):
        issues.append({"kind": "diagnostic_not_object", "scene": scene})
        return None
    fatal_identity_mismatch = False
    for field, expected in (
        ("schema", DIAGNOSTIC_SCHEMA),
        ("scene_id", scene),
        ("complete", True),
        ("training_free", True),
        ("gt_access", False),
        ("annotation_access", False),
        ("evaluator_access", False),
        ("proposal_cache_access", False),
        ("teacher_cache_access", False),
        ("terminal_cache_access", False),
        ("past_only", True),
        ("query_before_commit", True),
    ):
        if payload.get(field) != expected:
            issues.append(
                {"kind": "diagnostic_contract_mismatch", "scene": scene, "field": field, "expected": expected, "actual": payload.get(field)}
            )
            if field in {"schema", "scene_id", "complete"}:
                fatal_identity_mismatch = True
    if fatal_identity_mismatch:
        return None

    counts = _mapping(payload.get("counts"), field="counts", issues=issues, scene=scene)
    timing = _mapping(payload.get("timing_ms"), field="timing_ms", issues=issues, scene=scene)
    sam3 = _mapping(payload.get("sam3"), field="sam3", issues=issues, scene=scene)
    state = _mapping(payload.get("state"), field="state", issues=issues, scene=scene)
    bounded = _mapping(payload.get("bounded"), field="bounded", issues=issues, scene=scene)
    if None in (counts, timing, sam3, state, bounded):
        return None

    parsed_counts: dict[str, int] = {}
    for key, value in counts.items():
        parsed = _nonnegative_int(value, field=f"counts.{key}", issues=issues, scene=scene)
        if parsed is not None:
            parsed_counts[str(key)] = parsed
    for key in ("keyframes", "native", "candidates", "births", "overlays", "output"):
        if key not in parsed_counts:
            issues.append({"kind": "missing_required_count", "scene": scene, "field": f"counts.{key}"})
    if all(key in parsed_counts for key in ("native", "births", "output")) and (
        parsed_counts["native"] + parsed_counts["births"] != parsed_counts["output"]
    ):
        issues.append(
            {
                "kind": "output_count_equation_failed",
                "scene": scene,
                "native": parsed_counts["native"],
                "births": parsed_counts["births"],
                "output": parsed_counts["output"],
            }
        )
    if parsed_counts.get("births", 0) > 6:
        issues.append({"kind": "birth_cap_exceeded", "scene": scene, "actual": parsed_counts["births"], "cap": 6})
    if parsed_counts.get("overlays", 0) > 1:
        issues.append({"kind": "overlay_cap_exceeded", "scene": scene, "actual": parsed_counts["overlays"], "cap": 1})

    distributions: dict[str, dict[str, float | int]] = {}
    for stage, value in timing.items():
        parsed = _parse_distribution(value, scene=scene, stage=str(stage), issues=issues)
        if parsed is not None:
            distributions[str(stage)] = parsed
    if "keyframe_total" not in distributions:
        issues.append({"kind": "missing_keyframe_total_timing", "scene": scene})
    elif parsed_counts.get("keyframes") != distributions["keyframe_total"]["count"]:
        issues.append(
            {
                "kind": "keyframe_timing_count_mismatch",
                "scene": scene,
                "counts_keyframes": parsed_counts.get("keyframes"),
                "timing_count": distributions["keyframe_total"]["count"],
            }
        )

    future = _nonnegative_int(payload.get("future_access_count"), field="future_access_count", issues=issues, scene=scene)
    state_future = _nonnegative_int(state.get("future_access_count"), field="state.future_access_count", issues=issues, scene=scene)
    late = _nonnegative_int(payload.get("late_result_count"), field="late_result_count", issues=issues, scene=scene)
    sam3_late = _nonnegative_int(sam3.get("late_count"), field="sam3.late_count", issues=issues, scene=scene)
    queue_capacity = _nonnegative_int(bounded.get("sam3_queue_capacity"), field="bounded.sam3_queue_capacity", issues=issues, scene=scene)
    queue_max = _nonnegative_int(sam3.get("max_queue_depth"), field="sam3.max_queue_depth", issues=issues, scene=scene)
    queue_depth = _nonnegative_int(sam3.get("queue_depth"), field="sam3.queue_depth", issues=issues, scene=scene)
    client_drops = _nonnegative_int(sam3.get("drop_count"), field="sam3.drop_count", issues=issues, scene=scene)
    dropped_late = _nonnegative_int(sam3.get("dropped_late"), field="sam3.dropped_late", issues=issues, scene=scene)
    worker_errors = _nonnegative_int(sam3.get("worker_error_count"), field="sam3.worker_error_count", issues=issues, scene=scene)
    for key in ("submitted", "completed", "delivered"):
        _nonnegative_int(sam3.get(key), field=f"sam3.{key}", issues=issues, scene=scene)

    if future is not None and state_future is not None and future != state_future:
        issues.append({"kind": "future_counter_mismatch", "scene": scene, "top_level": future, "state": state_future})
    if late is not None and sam3_late is not None and late != sam3_late:
        issues.append({"kind": "late_counter_mismatch", "scene": scene, "top_level": late, "sam3": sam3_late})
    if queue_capacity != 1:
        issues.append({"kind": "queue_capacity_contract_mismatch", "scene": scene, "expected": 1, "actual": queue_capacity})
    if queue_max is not None and queue_capacity is not None and queue_max > queue_capacity:
        issues.append({"kind": "queue_capacity_exceeded", "scene": scene, "capacity": queue_capacity, "observed": queue_max})
    if queue_depth not in (None, 0):
        issues.append({"kind": "queue_not_empty_at_terminal", "scene": scene, "actual": queue_depth})
    # LiveSAM3 is configured fail-closed: every late result must be dropped.
    if late is not None and dropped_late is not None and late != dropped_late:
        issues.append({"kind": "late_result_not_fail_closed", "scene": scene, "late": late, "dropped_late": dropped_late})
    if state.get("query_before_commit") is not True:
        issues.append({"kind": "state_query_before_commit_false", "scene": scene, "actual": state.get("query_before_commit")})
    state_frames = _nonnegative_int(state.get("committed_frame_count"), field="state.committed_frame_count", issues=issues, scene=scene)
    if state_frames is not None and parsed_counts.get("keyframes") != state_frames:
        issues.append({"kind": "state_keyframe_count_mismatch", "scene": scene, "counts_keyframes": parsed_counts.get("keyframes"), "state_committed_frames": state_frames})

    peak_allocated = _nonnegative_int(payload.get("peak_cuda_allocated_bytes"), field="peak_cuda_allocated_bytes", issues=issues, scene=scene)
    peak_reserved = _nonnegative_int(payload.get("peak_cuda_reserved_bytes"), field="peak_cuda_reserved_bytes", issues=issues, scene=scene)
    deadline = _nonnegative_number(payload.get("deadline_ms"), field="deadline_ms", issues=issues, scene=scene)
    runtime = _read_log(log_path, scene=scene, issues=issues)
    required_numbers = (
        future,
        state_future,
        late,
        sam3_late,
        queue_capacity,
        queue_max,
        queue_depth,
        client_drops,
        dropped_late,
        worker_errors,
        peak_allocated,
        peak_reserved,
        deadline,
    )
    required_counts_present = all(
        key in parsed_counts
        for key in ("keyframes", "native", "candidates", "births", "overlays", "output")
    )
    if (
        runtime is None
        or any(value is None for value in required_numbers)
        or not required_counts_present
        or "keyframe_total" not in distributions
    ):
        return None

    return {
        "scene_id": scene,
        "counts": parsed_counts,
        "timing_ms": distributions,
        "future_access_count": future,
        "late_result_count": late,
        "queue_capacity": queue_capacity,
        "queue_max_depth": queue_max,
        "queue_depth_terminal": queue_depth,
        "client_drop_count": client_drops,
        "route_result_drop_count": parsed_counts.get("sam3_result_drops", 0),
        "submit_drop_count": parsed_counts.get("sam3_submit_drops", 0),
        "drain_timeout_count": parsed_counts.get("sam3_drain_timeouts", 0),
        "dropped_late_count": dropped_late,
        "worker_error_count": worker_errors,
        "deadline_ms": deadline,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "runtime": runtime,
    }


def _aggregate_stages(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_names = sorted({stage for row in rows for stage in row["timing_ms"]})
    result: dict[str, Any] = {}
    for stage in stage_names:
        distributions = [row["timing_ms"][stage] for row in rows if stage in row["timing_ms"]]
        nonempty = [row for row in distributions if int(row["count"]) > 0]
        count = sum(int(row["count"]) for row in nonempty)
        if count == 0:
            result[stage] = {
                "scene_count": len(distributions),
                "sample_count": 0,
                "mean": None,
                "p50_weighted_scene_quantile_approx": None,
                "p95_weighted_scene_quantile_approx": None,
                "max": None,
                "raw_samples_available": False,
            }
            continue
        weights = np.asarray([int(row["count"]) for row in nonempty], dtype=np.float64)
        result[stage] = {
            "scene_count": len(distributions),
            "sample_count": count,
            "mean": float(np.average([float(row["mean"]) for row in nonempty], weights=weights)),
            "p50_weighted_scene_quantile_approx": float(
                np.average([float(row["p50"]) for row in nonempty], weights=weights)
            ),
            "p95_weighted_scene_quantile_approx": float(
                np.average([float(row["p95"]) for row in nonempty], weights=weights)
            ),
            "max": max(float(row["max"]) for row in nonempty),
            "raw_samples_available": False,
        }
    return result


def summarize(
    *, scene_list: Path, diagnostics_root: Path, scene_log_root: Path
) -> dict[str, Any]:
    scenes = read_scene_list(scene_list)
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    missing_diagnostics: list[str] = []
    missing_logs: list[str] = []
    invalid_scenes: list[str] = []

    for scene in scenes:
        diagnostic_path = diagnostics_root / f"{scene}.json"
        log_path = scene_log_root / f"{scene}.log"
        if not diagnostic_path.is_file():
            missing_diagnostics.append(scene)
        if not log_path.is_file():
            missing_logs.append(scene)
        if not diagnostic_path.is_file() or not log_path.is_file():
            continue
        issue_start = len(issues)
        row = _load_scene(
            scene=scene,
            diagnostic_path=diagnostic_path,
            log_path=log_path,
            issues=issues,
        )
        if row is None:
            invalid_scenes.append(scene)
            if len(issues) == issue_start:
                issues.append({"kind": "invalid_scene_without_detail", "scene": scene})
        else:
            rows.append(row)

    expected_set = set(scenes)
    extra_diagnostics = sorted(
        path.stem for path in diagnostics_root.glob("*.json") if path.stem not in expected_set
    ) if diagnostics_root.is_dir() else []
    extra_logs = sorted(
        path.stem for path in scene_log_root.glob("*.log") if path.stem not in expected_set
    ) if scene_log_root.is_dir() else []
    for scene in extra_diagnostics:
        issues.append({"kind": "extra_diagnostic", "scene": scene})
    for scene in extra_logs:
        issues.append({"kind": "extra_scene_log", "scene": scene})

    valid_count = len(rows)
    artifacts_complete = (
        valid_count == len(scenes)
        and not missing_diagnostics
        and not missing_logs
        and not invalid_scenes
        and not extra_diagnostics
        and not extra_logs
    )
    official100_requested = len(scenes) == 100
    official100_complete = official100_requested and artifacts_complete

    def total_count(name: str) -> int:
        return sum(int(row["counts"].get(name, 0)) for row in rows)

    future_total = sum(int(row["future_access_count"]) for row in rows)
    late_total = sum(int(row["late_result_count"]) for row in rows)
    client_drop_total = sum(int(row["client_drop_count"]) for row in rows)
    route_drop_total = sum(int(row["route_result_drop_count"]) for row in rows)
    submit_drop_total = sum(int(row["submit_drop_count"]) for row in rows)
    drain_timeout_total = sum(int(row["drain_timeout_count"]) for row in rows)
    worker_error_total = sum(int(row["worker_error_count"]) for row in rows)
    max_queue = max((int(row["queue_max_depth"]) for row in rows), default=None)
    deadline_misses = total_count("deadline_misses")
    late_fail_closed = all(
        int(row["late_result_count"]) == int(row["dropped_late_count"])
        for row in rows
    )
    causal_contract_pass = (
        valid_count > 0
        and future_total == 0
        and late_fail_closed
        and max_queue is not None
        and max_queue <= 1
        and all(int(row["queue_depth_terminal"]) == 0 for row in rows)
    )
    deadline_contract_pass = valid_count > 0 and deadline_misses == 0
    worker_health_pass = valid_count > 0 and worker_error_total == 0
    diagnostic_contract_pass = valid_count > 0 and not issues

    total_cost = sum(float(row["runtime"]["cost_seconds"]) for row in rows)
    reconstructed_frames = sum(
        float(row["runtime"]["reconstructed_frames_approx"]) for row in rows
    )
    internal_fps = reconstructed_frames / total_cost if total_cost > 0.0 else None
    scene_fps = [float(row["runtime"]["reported_fps"]) for row in rows]
    allocated_peak = max((int(row["peak_cuda_allocated_bytes"]) for row in rows), default=None)
    reserved_peak = max((int(row["peak_cuda_reserved_bytes"]) for row in rows), default=None)

    scene_rows = []
    for row in rows:
        scene_rows.append(
            {
                "scene_id": row["scene_id"],
                "cost_seconds": row["runtime"]["cost_seconds"],
                "reported_fps": row["runtime"]["reported_fps"],
                "reconstructed_frames_approx": row["runtime"]["reconstructed_frames_approx"],
                "keyframes": row["counts"].get("keyframes", 0),
                "deadline_misses": row["counts"].get("deadline_misses", 0),
                "future_access_count": row["future_access_count"],
                "late_result_count": row["late_result_count"],
                "queue_max_depth": row["queue_max_depth"],
                "client_drop_count": row["client_drop_count"],
                "route_result_drop_count": row["route_result_drop_count"],
                "submit_drop_count": row["submit_drop_count"],
                "births": row["counts"].get("births", 0),
                "overlays": row["counts"].get("overlays", 0),
                "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": row["peak_cuda_reserved_bytes"],
            }
        )

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scene_list": str(scene_list.resolve()),
            "diagnostics_root": str(diagnostics_root.resolve()),
            "scene_log_root": str(scene_log_root.resolve()),
            "diagnostic_schema": DIAGNOSTIC_SCHEMA,
        },
        "status": {
            "coverage": "complete" if artifacts_complete else "partial",
            "partial": not artifacts_complete,
            "artifacts_complete": artifacts_complete,
            "official100_requested": official100_requested,
            "official100_complete": official100_complete,
            "causal_contract_pass": causal_contract_pass,
            "deadline_contract_pass": deadline_contract_pass,
            "worker_health_pass": worker_health_pass,
            "diagnostic_contract_pass": diagnostic_contract_pass,
            "strict_realtime_online_pass": (
                official100_complete
                and causal_contract_pass
                and deadline_contract_pass
                and worker_health_pass
                and diagnostic_contract_pass
            ),
        },
        "coverage": {
            "expected_scene_count": len(scenes),
            "valid_scene_count": valid_count,
            "missing_diagnostics": missing_diagnostics,
            "missing_scene_logs": missing_logs,
            "invalid_scenes": invalid_scenes,
            "extra_diagnostics": extra_diagnostics,
            "extra_scene_logs": extra_logs,
        },
        "internal_runtime": {
            "scene_count": valid_count,
            "total_cost_seconds": total_cost,
            "reconstructed_frame_count_approx": reconstructed_frames,
            "aggregate_internal_fps_approx": internal_fps,
            "scene_fps_mean": float(np.mean(scene_fps)) if scene_fps else None,
            "scene_fps_min": min(scene_fps, default=None),
            "scene_fps_max": max(scene_fps, default=None),
            "method": "sum(scene Cost * rounded scene FPS) / sum(scene Cost)",
            "precision_note": "Cost and FPS are parsed from two-decimal scene-log values; aggregate FPS is approximate.",
            "clock_scope": "internal scene processing clock including terminal live drain; excludes startup and official evaluation",
        },
        "stage_latency_ms": {
            "method": {
                "mean": "sample-count-weighted from per-scene count and mean",
                "p50_p95": "sample-count-weighted averages of per-scene quantiles; approximate, not pooled percentiles",
                "max": "maximum of per-scene maxima",
                "raw_samples_available": False,
            },
            "stages": _aggregate_stages(rows),
        },
        "causality_and_queue": {
            "future_access_count": future_total,
            "late_result_count": late_total,
            "dropped_late_count": sum(int(row["dropped_late_count"]) for row in rows),
            "late_results_fail_closed": late_fail_closed,
            "queue_capacity": 1,
            "queue_max_depth": max_queue,
            "terminal_nonempty_queue_scene_count": sum(
                int(int(row["queue_depth_terminal"]) != 0) for row in rows
            ),
            "client_drop_count": client_drop_total,
            "route_result_drop_count": route_drop_total,
            "submit_drop_count": submit_drop_total,
            "drain_timeout_count": drain_timeout_total,
            "drop_total_not_reported": client_drop_total + route_drop_total + submit_drop_total,
            "drop_accounting_note": "Drop counters can refer to the same request at different layers and are not summed as a deduplicated total.",
            "worker_error_count": worker_error_total,
        },
        "counts": {
            "keyframes": total_count("keyframes"),
            "deadline_misses": deadline_misses,
            "native": total_count("native"),
            "candidates": total_count("candidates"),
            "births": total_count("births"),
            "overlays": total_count("overlays"),
            "output": total_count("output"),
            "sam3_submitted": total_count("sam3_submitted"),
            "sam3_results_accepted": total_count("sam3_results_accepted"),
        },
        "cuda_main_process_peak": {
            "allocated_bytes_max": allocated_peak,
            "reserved_bytes_max": reserved_peak,
            "allocated_gib_max": (allocated_peak / 1024**3) if allocated_peak is not None else None,
            "reserved_gib_max": (reserved_peak / 1024**3) if reserved_peak is not None else None,
            "scope": "maximum per-scene PyTorch allocator peak in the main BoxFusion process",
            "includes_sam3_subprocess": False,
            "includes_non_torch_cuda_allocations": False,
            "end_to_end_device_vram_peak_available": False,
        },
        "scenes": scene_rows,
        "issues": issues,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _print_summary(summary: Mapping[str, Any]) -> None:
    coverage = summary["coverage"]
    status = summary["status"]
    runtime = summary["internal_runtime"]
    causal = summary["causality_and_queue"]
    counts = summary["counts"]
    memory = summary["cuda_main_process_peak"]
    fps = runtime["aggregate_internal_fps_approx"]
    fps_text = "n/a" if fps is None else f"{fps:.3f}"
    peak = memory["allocated_gib_max"]
    peak_text = "n/a" if peak is None else f"{peak:.3f} GiB"
    print(
        f"Coverage: {coverage['valid_scene_count']}/{coverage['expected_scene_count']} "
        f"({status['coverage']}); official100_complete={status['official100_complete']}"
    )
    print(
        f"Internal FPS: {fps_text} (Cost*rounded-FPS weighted; "
        f"cost={runtime['total_cost_seconds']:.2f}s)"
    )
    print(
        "Causal/queue: "
        f"future={causal['future_access_count']} late={causal['late_result_count']} "
        f"queue_max={causal['queue_max_depth']}/1 client_drops={causal['client_drop_count']} "
        f"submit_drops={causal['submit_drop_count']} deadline_misses={counts['deadline_misses']}"
    )
    print(
        f"Outputs: births={counts['births']} overlays={counts['overlays']}; "
        f"main-process CUDA allocated peak={peak_text}"
    )
    print(
        "Stage p50/p95 are sample-count-weighted per-scene quantile approximations; "
        "raw pooled samples are unavailable."
    )
    print(f"Strict realtime-online pass: {status['strict_realtime_online_pass']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument("--scene-log-root", type=Path, default=DEFAULT_SCENE_LOG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit nonzero unless every listed scene has valid diagnostics and log",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize(
            scene_list=args.scene_list,
            diagnostics_root=args.diagnostics_root,
            scene_log_root=args.scene_log_root,
        )
        _atomic_write_json(args.output, summary)
    except (OSError, ValueError, SummaryInputError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    _print_summary(summary)
    print(f"JSON: {args.output}")
    if args.require_complete and not summary["status"]["artifacts_complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
