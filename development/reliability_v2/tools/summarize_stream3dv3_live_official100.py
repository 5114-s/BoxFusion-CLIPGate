#!/usr/bin/env python3
"""Validate and aggregate strict-live Stream3Dv3 official100 diagnostics.

Unlike the legacy Stream3Dv2 summary, this tool does not reconstruct frame
counts from rounded log messages.  Every V3 scene diagnostic must provide the
exact top-level fields ``raw_frame_count`` and ``pipeline_seconds``.  The
reported throughput is therefore exactly::

    sum(raw_frame_count) / sum(pipeline_seconds)

The summary also audits the causal, held-out, no-cache, native-score,
fingerprint, and bounded-birth contracts before declaring a strict
realtime-online pass.  The per-keyframe add-on deadline is reported as an
advisory diagnostic: the authoritative realtime criterion is the exact
end-to-end aggregate FPS, so an isolated model cold-start does not invalidate
an otherwise realtime stream.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.stream3dv3_live.official100_summary.v1"
DIAGNOSTIC_SCHEMA = "boxfusion.stream3dv3_live.v1"
DEFAULT_SCENE_LIST = (
    REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
)
DEFAULT_DIAGNOSTICS_ROOT = (
    REPOSITORY_ROOT / "diagnostics/cbest_f4_stream3dv3_live/route"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "logs/scannet_cbest_f4_stream3dv3_live_score05/OFFICIAL100_LIVE_SUMMARY.json"
)
DEFAULT_MINIMUM_FPS = 20.0
MAX_BIRTHS_PER_SCENE = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SummaryInputError(RuntimeError):
    """A top-level summary input cannot be interpreted safely."""


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
    duplicates = sorted(
        scene for scene, count in Counter(scenes).items() if count > 1
    )
    if duplicates:
        raise SummaryInputError(f"duplicate scene IDs in {path}: {duplicates}")
    return scenes


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _issue(
    issues: list[dict[str, Any]],
    kind: str,
    scene: str,
    *,
    field: str | None = None,
    expected: object = None,
    actual: object = None,
) -> None:
    row: dict[str, Any] = {"kind": kind, "scene": scene}
    if field is not None:
        row["field"] = field
    if expected is not None:
        row["expected"] = expected
    if actual is not None:
        row["actual"] = actual
    issues.append(row)


def _mapping(
    value: object,
    *,
    scene: str,
    field: str,
    issues: list[dict[str, Any]],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        _issue(
            issues,
            "invalid_mapping",
            scene,
            field=field,
            expected="object",
            actual=type(value).__name__,
        )
        return None
    return value


def _nonnegative_integer(
    value: object,
    *,
    scene: str,
    field: str,
    issues: list[dict[str, Any]],
) -> int | None:
    if not _is_integer(value) or int(value) < 0:
        _issue(
            issues,
            "invalid_nonnegative_integer",
            scene,
            field=field,
            actual=value,
        )
        return None
    return int(value)


def _positive_integer(
    value: object,
    *,
    scene: str,
    field: str,
    issues: list[dict[str, Any]],
) -> int | None:
    result = _nonnegative_integer(
        value, scene=scene, field=field, issues=issues
    )
    if result is not None and result == 0:
        _issue(issues, "invalid_positive_integer", scene, field=field, actual=value)
        return None
    return result


def _positive_number(
    value: object,
    *,
    scene: str,
    field: str,
    issues: list[dict[str, Any]],
) -> float | None:
    if not _is_number(value) or float(value) <= 0.0:
        _issue(
            issues,
            "invalid_positive_number",
            scene,
            field=field,
            actual=value,
        )
        return None
    return float(value)


def _contract_field(
    payload: Mapping[str, Any],
    *,
    scene: str,
    field: str,
    expected: object,
    kind: str,
    issues: list[dict[str, Any]],
) -> bool:
    actual = payload.get(field)
    passed = actual == expected
    if not passed:
        _issue(
            issues,
            kind,
            scene,
            field=field,
            expected=expected,
            actual=actual,
        )
    return passed


def _parse_keyframe_timing(
    value: object,
    *,
    scene: str,
    issues: list[dict[str, Any]],
) -> dict[str, float | int] | None:
    row = _mapping(
        value, scene=scene, field="timing_ms.keyframe_total", issues=issues
    )
    if row is None:
        return None
    count = _nonnegative_integer(
        row.get("count"),
        scene=scene,
        field="timing_ms.keyframe_total.count",
        issues=issues,
    )
    if count is None:
        return None
    parsed: dict[str, float | int] = {"count": count}
    if count == 0:
        for metric in ("mean", "p50", "p95", "max"):
            if row.get(metric) is not None:
                _issue(
                    issues,
                    "nonempty_zero_count_timing",
                    scene,
                    field=f"timing_ms.keyframe_total.{metric}",
                    actual=row.get(metric),
                )
                return None
        return parsed
    for metric in ("mean", "p50", "p95", "max"):
        value_metric = row.get(metric)
        if not _is_number(value_metric) or float(value_metric) < 0.0:
            _issue(
                issues,
                "invalid_nonnegative_timing",
                scene,
                field=f"timing_ms.keyframe_total.{metric}",
                actual=value_metric,
            )
            return None
        parsed[metric] = float(value_metric)
    if not (
        float(parsed["max"])
        >= float(parsed["p95"])
        >= float(parsed["p50"])
        >= 0.0
    ):
        _issue(
            issues,
            "invalid_timing_order",
            scene,
            field="timing_ms.keyframe_total",
            actual=parsed,
        )
        return None
    return parsed


def _load_scene(
    *,
    scene: str,
    path: Path,
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        _issue(issues, "unreadable_diagnostic", scene, actual=str(error))
        return None
    if not isinstance(payload, Mapping):
        _issue(issues, "diagnostic_not_object", scene)
        return None

    identity_ok = True
    for field, expected in (
        ("schema", DIAGNOSTIC_SCHEMA),
        ("complete", True),
        ("scene_id", scene),
    ):
        identity_ok &= _contract_field(
            payload,
            scene=scene,
            field=field,
            expected=expected,
            kind="diagnostic_identity_mismatch",
            issues=issues,
        )
    if not identity_ok:
        return None

    raw_frames = _positive_integer(
        payload.get("raw_frame_count"),
        scene=scene,
        field="raw_frame_count",
        issues=issues,
    )
    pipeline_seconds = _positive_number(
        payload.get("pipeline_seconds"),
        scene=scene,
        field="pipeline_seconds",
        issues=issues,
    )
    target_fps = _positive_number(
        payload.get("target_end_to_end_fps"),
        scene=scene,
        field="target_end_to_end_fps",
        issues=issues,
    )
    deadline_ms = _positive_number(
        payload.get("addon_deadline_ms"),
        scene=scene,
        field="addon_deadline_ms",
        issues=issues,
    )
    fingerprint = payload.get("run_fingerprint")
    if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
        _issue(
            issues,
            "invalid_run_fingerprint",
            scene,
            field="run_fingerprint",
            expected="64 lowercase hexadecimal characters",
            actual=fingerprint,
        )
        fingerprint = None

    counts = _mapping(payload.get("counts"), scene=scene, field="counts", issues=issues)
    bounded = _mapping(
        payload.get("bounded"), scene=scene, field="bounded", issues=issues
    )
    f3 = _mapping(payload.get("f3"), scene=scene, field="f3", issues=issues)
    timing_root = _mapping(
        payload.get("timing_ms"), scene=scene, field="timing_ms", issues=issues
    )
    if None in (counts, bounded, f3, timing_root):
        return None
    assert counts is not None and bounded is not None and f3 is not None
    assert timing_root is not None

    parsed_counts: dict[str, int] = {}
    for name in (
        "keyframes",
        "native",
        "births",
        "overlays",
        "output",
        "addon_deadline_misses",
    ):
        parsed = _nonnegative_integer(
            counts.get(name), scene=scene, field=f"counts.{name}", issues=issues
        )
        if parsed is not None:
            parsed_counts[name] = parsed

    bounded_births = _nonnegative_integer(
        bounded.get("max_births_per_scene"),
        scene=scene,
        field="bounded.max_births_per_scene",
        issues=issues,
    )
    f3_keyframes = _nonnegative_integer(
        f3.get("keyframes"),
        scene=scene,
        field="f3.keyframes",
        issues=issues,
    )
    f3_max_ordinal = _nonnegative_integer(
        f3.get("max_logical_accessed_ordinal"),
        scene=scene,
        field="f3.max_logical_accessed_ordinal",
        issues=issues,
    )
    keyframe_timing = _parse_keyframe_timing(
        timing_root.get("keyframe_total"), scene=scene, issues=issues
    )

    essential = (
        raw_frames,
        pipeline_seconds,
        target_fps,
        deadline_ms,
        fingerprint,
        bounded_births,
        f3_keyframes,
        f3_max_ordinal,
        keyframe_timing,
    )
    if any(value is None for value in essential) or len(parsed_counts) != 6:
        return None
    assert raw_frames is not None and pipeline_seconds is not None
    assert target_fps is not None and deadline_ms is not None
    assert isinstance(fingerprint, str)
    assert bounded_births is not None and f3_keyframes is not None
    assert f3_max_ordinal is not None and keyframe_timing is not None

    keyframes = parsed_counts["keyframes"]
    births = parsed_counts["births"]
    native = parsed_counts["native"]
    overlays = parsed_counts["overlays"]
    output = parsed_counts["output"]
    misses = parsed_counts["addon_deadline_misses"]

    causal_checks = [
        _contract_field(
            payload,
            scene=scene,
            field="past_only",
            expected=True,
            kind="causal_contract_mismatch",
            issues=issues,
        ),
        _contract_field(
            payload,
            scene=scene,
            field="query_before_commit",
            expected=True,
            kind="causal_contract_mismatch",
            issues=issues,
        ),
        _contract_field(
            payload,
            scene=scene,
            field="future_access_count",
            expected=0,
            kind="causal_contract_mismatch",
            issues=issues,
        ),
        _contract_field(
            f3,
            scene=scene,
            field="audit_complete",
            expected=True,
            kind="causal_contract_mismatch",
            issues=issues,
        ),
    ]
    if f3_keyframes != keyframes:
        _issue(
            issues,
            "causal_keyframe_count_mismatch",
            scene,
            field="f3.keyframes",
            expected=keyframes,
            actual=f3_keyframes,
        )
        causal_checks.append(False)
    if keyframes == 0 or f3_max_ordinal > keyframes - 1:
        _issue(
            issues,
            "causal_future_ordinal",
            scene,
            field="f3.max_logical_accessed_ordinal",
            expected=f"<= {keyframes - 1}",
            actual=f3_max_ordinal,
        )
        causal_checks.append(False)
    if raw_frames < keyframes:
        _issue(
            issues,
            "raw_frame_count_below_keyframes",
            scene,
            expected=f">= {keyframes}",
            actual=raw_frames,
        )
        causal_checks.append(False)

    held_out_pass = _contract_field(
        payload,
        scene=scene,
        field="selection_and_acceptance_held_out",
        expected=True,
        kind="held_out_contract_mismatch",
        issues=issues,
    )

    cache_checks = []
    for field in (
        "proposal_cache_access",
        "teacher_cache_access",
        "terminal_cache_access",
    ):
        cache_checks.append(
            _contract_field(
                payload,
                scene=scene,
                field=field,
                expected=False,
                kind="cache_contract_mismatch",
                issues=issues,
            )
        )

    access_checks = []
    for field, expected in (
        ("training_free", True),
        ("ground_truth_access", False),
        ("annotation_access", False),
        ("evaluator_access", False),
    ):
        access_checks.append(
            _contract_field(
                payload,
                scene=scene,
                field=field,
                expected=expected,
                kind="access_contract_mismatch",
                issues=issues,
            )
        )

    native_score_pass = _contract_field(
        payload,
        scene=scene,
        field="native_scores_preserved",
        expected=True,
        kind="native_score_contract_mismatch",
        issues=issues,
    )
    if overlays != 0 or output != native + births:
        _issue(
            issues,
            "native_output_equation_mismatch",
            scene,
            expected={"overlays": 0, "output": native + births},
            actual={"overlays": overlays, "output": output},
        )
        native_score_pass = False

    birth_pass = True
    if bounded_births > MAX_BIRTHS_PER_SCENE:
        _issue(
            issues,
            "configured_birth_cap_exceeded",
            scene,
            field="bounded.max_births_per_scene",
            expected=f"<= {MAX_BIRTHS_PER_SCENE}",
            actual=bounded_births,
        )
        birth_pass = False
    if births > min(MAX_BIRTHS_PER_SCENE, bounded_births):
        _issue(
            issues,
            "birth_cap_exceeded",
            scene,
            field="counts.births",
            expected=f"<= {min(MAX_BIRTHS_PER_SCENE, bounded_births)}",
            actual=births,
        )
        birth_pass = False

    deadline_pass = misses == 0
    maximum_ms = float(keyframe_timing.get("max", 0.0))
    if int(keyframe_timing["count"]) != keyframes:
        _issue(
            issues,
            "deadline_timing_count_mismatch",
            scene,
            expected=keyframes,
            actual=keyframe_timing["count"],
        )
        deadline_pass = False
    if misses == 0 and maximum_ms > deadline_ms + 1.0e-9:
        _issue(
            issues,
            "deadline_miss_underreported",
            scene,
            expected=f"max <= {deadline_ms}",
            actual=maximum_ms,
        )
        deadline_pass = False
    if misses > 0 and maximum_ms <= deadline_ms + 1.0e-9:
        _issue(
            issues,
            "deadline_miss_overreported",
            scene,
            expected=f"max > {deadline_ms}",
            actual=maximum_ms,
        )
        deadline_pass = False

    return {
        "scene_id": scene,
        "raw_frame_count": raw_frames,
        "pipeline_seconds": pipeline_seconds,
        "scene_fps": raw_frames / pipeline_seconds,
        "run_fingerprint": fingerprint,
        "target_end_to_end_fps": target_fps,
        "addon_deadline_ms": deadline_ms,
        "counts": parsed_counts,
        "contracts": {
            "causal": all(causal_checks),
            "held_out": held_out_pass,
            "cache": all(cache_checks),
            "access": all(access_checks),
            "native_score": native_score_pass,
            "birth": birth_pass,
            "deadline": deadline_pass,
        },
        "keyframe_timing_ms": keyframe_timing,
    }


def summarize(
    *,
    scene_list: Path,
    diagnostics_root: Path,
    minimum_fps: float = DEFAULT_MINIMUM_FPS,
) -> dict[str, Any]:
    if not math.isfinite(float(minimum_fps)) or float(minimum_fps) <= 0.0:
        raise SummaryInputError("minimum_fps must be finite and positive")
    minimum = float(minimum_fps)
    scenes = read_scene_list(scene_list)
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []

    for scene in scenes:
        path = diagnostics_root / f"{scene}.json"
        if not path.is_file():
            missing.append(scene)
            continue
        issue_start = len(issues)
        row = _load_scene(scene=scene, path=path, issues=issues)
        if row is None:
            invalid.append(scene)
            if len(issues) == issue_start:
                _issue(issues, "invalid_scene_without_detail", scene)
        else:
            rows.append(row)

    expected = set(scenes)
    extra = (
        sorted(
            path.stem
            for path in diagnostics_root.glob("scene*.json")
            if path.stem not in expected
        )
        if diagnostics_root.is_dir()
        else []
    )
    for scene in extra:
        _issue(issues, "extra_diagnostic", scene)

    artifacts_complete = (
        len(rows) == len(scenes) and not missing and not invalid and not extra
    )
    official100_requested = len(scenes) == 100
    official100_complete = official100_requested and artifacts_complete

    fingerprints = sorted({str(row["run_fingerprint"]) for row in rows})
    fingerprint_pass = bool(rows) and len(fingerprints) == 1
    if rows and not fingerprint_pass:
        issues.append(
            {
                "kind": "cross_scene_fingerprint_mismatch",
                "scene": "*",
                "actual": fingerprints,
            }
        )

    target_fps_values = sorted(
        {float(row["target_end_to_end_fps"]) for row in rows}
    )
    if len(target_fps_values) > 1:
        issues.append(
            {
                "kind": "cross_scene_target_fps_mismatch",
                "scene": "*",
                "actual": target_fps_values,
            }
        )

    deadline_values = sorted({float(row["addon_deadline_ms"]) for row in rows})
    if len(deadline_values) > 1:
        issues.append(
            {
                "kind": "cross_scene_deadline_mismatch",
                "scene": "*",
                "actual": deadline_values,
            }
        )

    total_frames = sum(int(row["raw_frame_count"]) for row in rows)
    total_seconds = sum(float(row["pipeline_seconds"]) for row in rows)
    aggregate_fps = total_frames / total_seconds if total_seconds > 0.0 else None
    throughput_pass = (
        aggregate_fps is not None and aggregate_fps + 1.0e-12 >= minimum
    )

    def all_contract(name: str) -> bool:
        return bool(rows) and all(bool(row["contracts"][name]) for row in rows)

    causal_pass = all_contract("causal")
    held_out_pass = all_contract("held_out")
    cache_pass = all_contract("cache")
    access_pass = all_contract("access")
    native_score_pass = all_contract("native_score")
    birth_pass = all_contract("birth")
    deadline_pass = all_contract("deadline")
    diagnostic_contract_pass = bool(rows) and not issues
    strict_pass = (
        official100_complete
        and causal_pass
        and held_out_pass
        and cache_pass
        and access_pass
        and native_score_pass
        and fingerprint_pass
        and birth_pass
        and throughput_pass
        and diagnostic_contract_pass
    )

    def total_count(name: str) -> int:
        return sum(int(row["counts"].get(name, 0)) for row in rows)

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scene_list": str(scene_list.resolve()),
            "diagnostics_root": str(diagnostics_root.resolve()),
            "diagnostic_schema": DIAGNOSTIC_SCHEMA,
            "minimum_fps": minimum,
        },
        "status": {
            "coverage": "complete" if artifacts_complete else "partial",
            "artifacts_complete": artifacts_complete,
            "official100_requested": official100_requested,
            "official100_complete": official100_complete,
            "causal_contract_pass": causal_pass,
            "held_out_contract_pass": held_out_pass,
            "cache_contract_pass": cache_pass,
            "access_contract_pass": access_pass,
            "native_score_contract_pass": native_score_pass,
            "fingerprint_contract_pass": fingerprint_pass,
            "birth_contract_pass": birth_pass,
            "deadline_contract_pass": deadline_pass,
            "throughput_contract_pass": throughput_pass,
            "diagnostic_contract_pass": diagnostic_contract_pass,
            "strict_realtime_online_pass": strict_pass,
        },
        "coverage": {
            "expected_scene_count": len(scenes),
            "valid_scene_count": len(rows),
            "missing_diagnostics": missing,
            "invalid_scenes": invalid,
            "extra_diagnostics": extra,
        },
        "runtime": {
            "raw_frame_count": total_frames,
            "pipeline_seconds": total_seconds,
            "aggregate_fps": aggregate_fps,
            "minimum_fps": minimum,
            "exact": True,
            "formula": "sum(raw_frame_count) / sum(pipeline_seconds)",
            "source": "per-scene V3 diagnostics; no rounded log reconstruction",
            "scene_fps_min": min(
                (float(row["scene_fps"]) for row in rows), default=None
            ),
            "scene_fps_max": max(
                (float(row["scene_fps"]) for row in rows), default=None
            ),
        },
        "run_fingerprint": fingerprints[0] if fingerprint_pass else None,
        "configured": {
            "target_end_to_end_fps": (
                target_fps_values[0] if len(target_fps_values) == 1 else None
            ),
            "addon_deadline_ms": (
                deadline_values[0] if len(deadline_values) == 1 else None
            ),
        },
        "counts": {
            "keyframes": total_count("keyframes"),
            "addon_deadline_misses": total_count("addon_deadline_misses"),
            "native": total_count("native"),
            "births": total_count("births"),
            "overlays": total_count("overlays"),
            "output": total_count("output"),
        },
        "scenes": rows,
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
    runtime = summary["runtime"]
    fps = runtime["aggregate_fps"]
    fps_text = "n/a" if fps is None else f"{float(fps):.6f}"
    print(
        f"Coverage: {coverage['valid_scene_count']}/{coverage['expected_scene_count']} "
        f"({status['coverage']}); official100_complete={status['official100_complete']}"
    )
    print(
        f"Exact pipeline FPS: {fps_text} "
        f"({runtime['raw_frame_count']} frames / {runtime['pipeline_seconds']:.6f} s; "
        f"minimum={runtime['minimum_fps']:.6f})"
    )
    print(
        "Contracts: "
        f"causal={status['causal_contract_pass']} "
        f"held_out={status['held_out_contract_pass']} "
        f"cache={status['cache_contract_pass']} "
        f"native_score={status['native_score_contract_pass']} "
        f"fingerprint={status['fingerprint_contract_pass']} "
        f"birth={status['birth_contract_pass']} "
        f"deadline={status['deadline_contract_pass']}"
    )
    print(f"Strict realtime-online pass: {status['strict_realtime_online_pass']}")


def _minimum_fps(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("minimum FPS must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("minimum FPS must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument(
        "--diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--minimum-fps", type=_minimum_fps, default=DEFAULT_MINIMUM_FPS
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-realtime-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize(
            scene_list=args.scene_list,
            diagnostics_root=args.diagnostics_root,
            minimum_fps=args.minimum_fps,
        )
    except SummaryInputError as error:
        print(f"Input error: {error}")
        return 2
    _atomic_write_json(args.output, summary)
    _print_summary(summary)
    if args.require_complete and not summary["status"]["artifacts_complete"]:
        return 1
    if (
        args.require_realtime_pass
        and not summary["status"]["strict_realtime_online_pass"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
