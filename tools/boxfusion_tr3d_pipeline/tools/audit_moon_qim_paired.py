#!/usr/bin/env python3
"""Fail-closed paired audit for Moon-QIM-lite and optional PUF-lite shadow."""

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

import numpy as np


TIMING_PATTERN = re.compile(
    r"^Cost:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s+Average FPS:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)
QIM_JSON_PREFIX = "Moon-QIM-lite observer JSON | "
PUF_JSON_PREFIX = "PUF-lite shadow JSON | "
ARBITRATION_JSON_PREFIX = "PUF-arbitration-lite shadow JSON | "


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return number


def _load_json_summary(
    payload: str, *, path: Path, summary_name: str
) -> object:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"invalid {summary_name} JSON summary in {path}: {error}"
        ) from error


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _nonnegative_float(value: object, *, field: str) -> float:
    number = _finite_float(value, field=field)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _unit_interval_float(value: object, *, field: str) -> float:
    number = _finite_float(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return number


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a non-bool integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _positive_int(value: object, *, field: str) -> int:
    number = _nonnegative_int(value, field=field)
    if number == 0:
        raise ValueError(f"{field} must be positive")
    return number


def _validate_rate_fields(summary: object, *, field: str) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"{field} summary must be a JSON object")
    for name, value in summary.items():
        if name.endswith(("_rate", "_precision", "_recall")):
            if value is not None:
                _unit_interval_float(value, field=f"{field} {name}")


def _validate_precision_counts(
    *,
    correct_value: object,
    evaluable_value: object,
    precision_value: object,
    field: str,
) -> tuple[int, int, float | None]:
    correct = _nonnegative_int(correct_value, field=f"{field} correct")
    evaluable = _nonnegative_int(
        evaluable_value, field=f"{field} evaluable"
    )
    if correct > evaluable:
        raise ValueError(f"{field} must satisfy correct <= evaluable")
    if evaluable == 0:
        if precision_value is not None:
            raise ValueError(
                f"{field} precision must be null iff evaluable is zero"
            )
        return correct, evaluable, None
    if precision_value is None:
        raise ValueError(
            f"{field} precision must be non-null when evaluable is positive"
        )
    precision = _unit_interval_float(
        precision_value, field=f"{field} precision"
    )
    expected = correct / evaluable
    if precision != expected:
        raise ValueError(
            f"{field} precision is inconsistent with correct/evaluable"
        )
    return correct, evaluable, precision


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scenes(path: Path) -> tuple[str, ...]:
    scenes = tuple(
        line.strip() for line in path.read_text().splitlines() if line.strip()
    )
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and unique")
    if any(not re.fullmatch(r"scene\d{4}_\d{2}", scene) for scene in scenes):
        raise ValueError("scene list contains an invalid ScanNet scene ID")
    return scenes


def load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ValueError(f"prediction must be a non-empty regular file: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local artifact
        if handle.read(1):
            raise ValueError(f"prediction has trailing bytes: {path}")
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], list)
    ):
        raise ValueError(f"prediction must contain one list batch: {path}")
    rows = []
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"invalid prediction row {index}: {path}")
        label, corners, score = row
        corners = np.asarray(corners)
        if (
            int(label) != 0
            or corners.shape != (8, 3)
            or not np.isfinite(corners).all()
            or not np.isfinite(float(score))
        ):
            raise ValueError(f"invalid prediction row {index}: {path}")
        rows.append((int(label), np.array(corners, copy=True), float(score)))
    return rows


def compare_predictions(control: Path, observer: Path) -> dict[str, object]:
    control_sha, observer_sha = sha256(control), sha256(observer)
    if control_sha != observer_sha:
        raise ValueError("control and observer prediction bytes differ")
    left, right = load_prediction(control), load_prediction(observer)
    if len(left) != len(right):
        raise ValueError("control and observer prediction counts differ")
    for index, (control_row, observer_row) in enumerate(zip(left, right)):
        if (
            control_row[0] != observer_row[0]
            or control_row[2] != observer_row[2]
            or not np.array_equal(control_row[1], observer_row[1])
        ):
            raise ValueError(f"prediction row {index} is not identical")
    return {
        "rows": len(left),
        "byte_identity": True,
        "array_identity": True,
        "sha256": control_sha,
    }


def parse_log(
    path: Path,
    *,
    require_qim: bool,
    require_puf: bool = False,
    require_arbitration: bool = False,
) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"log must be a regular file: {path}")
    text = path.read_text(errors="replace")
    timings = TIMING_PATTERN.findall(text)
    if len(timings) != 1:
        raise ValueError(f"log must contain exactly one timing line: {path}")
    duration_s, fps = (
        _finite_float(value, field=f"{path}: log timing")
        for value in timings[0]
    )
    if duration_s <= 0.0 or fps <= 0.0:
        raise ValueError(f"log timing must be positive: {path}")
    frame_product = _finite_float(
        duration_s * fps, field=f"{path}: frame equivalent"
    )
    frame_equivalent = int(round(frame_product))
    if frame_equivalent <= 0:
        raise ValueError(f"log frame equivalent must be positive: {path}")
    result: dict[str, object] = {
        "duration_s": duration_s,
        "fps": fps,
        "frame_equivalent": frame_equivalent,
    }
    summaries = [
        _load_json_summary(
            line[len(QIM_JSON_PREFIX) :],
            path=path,
            summary_name="QIM",
        )
        for line in text.splitlines()
        if line.startswith(QIM_JSON_PREFIX)
    ]
    if require_qim:
        if len(summaries) != 1:
            raise ValueError(
                f"observer log must contain one QIM JSON summary: {path}"
            )
        result["qim"] = summaries[0]
    elif summaries:
        raise ValueError(f"control log unexpectedly contains QIM output: {path}")
    puf_summaries = [
        _load_json_summary(
            line[len(PUF_JSON_PREFIX) :],
            path=path,
            summary_name="PUF",
        )
        for line in text.splitlines()
        if line.startswith(PUF_JSON_PREFIX)
    ]
    if require_puf:
        if len(puf_summaries) != 1:
            raise ValueError(
                f"observer log must contain one PUF JSON summary: {path}"
            )
        result["puf"] = puf_summaries[0]
    elif puf_summaries:
        raise ValueError(f"log unexpectedly contains PUF output: {path}")
    arbitration_summaries = [
        _load_json_summary(
            line[len(ARBITRATION_JSON_PREFIX) :],
            path=path,
            summary_name="arbitration",
        )
        for line in text.splitlines()
        if line.startswith(ARBITRATION_JSON_PREFIX)
    ]
    if require_arbitration:
        if len(arbitration_summaries) != 1:
            raise ValueError(
                "observer log must contain one arbitration JSON summary: "
                f"{path}"
            )
        result["arbitration"] = arbitration_summaries[0]
    elif arbitration_summaries:
        raise ValueError(
            f"log unexpectedly contains arbitration output: {path}"
        )
    return result


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--control-log-root", type=Path, required=True)
    parser.add_argument("--observer-log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-recall-at-k", type=float, default=0.995)
    parser.add_argument("--min-fps-ratio", type=float, default=0.95)
    parser.add_argument("--require-puf", action="store_true")
    parser.add_argument("--require-arbitration", action="store_true")
    parser.add_argument(
        "--min-post-fallback-coverage", type=float, default=0.995
    )
    parser.add_argument(
        "--max-combined-ms-per-input-frame", type=float, default=0.10
    )
    parser.add_argument("--min-puf-top1-agreement", type=float, default=0.95)
    parser.add_argument("--max-puf-query-p95-ms", type=float, default=2.0)
    parser.add_argument(
        "--min-arbitration-selective-precision", type=float, default=0.99
    )
    parser.add_argument(
        "--min-conflict-owner-precision", type=float, default=1.0
    )
    parser.add_argument("--min-conflict-owner-samples", type=int, default=0)
    parser.add_argument(
        "--max-arbitration-query-p95-ms", type=float, default=0.10
    )
    args = parser.parse_args()
    for name in (
        "min_recall_at_k",
        "min_fps_ratio",
        "min_post_fallback_coverage",
        "max_combined_ms_per_input_frame",
        "min_puf_top1_agreement",
        "max_puf_query_p95_ms",
        "min_arbitration_selective_precision",
        "min_conflict_owner_precision",
        "max_arbitration_query_p95_ms",
    ):
        if not math.isfinite(getattr(args, name)):
            parser.error(
                f"--{name.replace('_', '-')} must be a finite number"
            )
    if not 0.0 <= args.min_recall_at_k <= 1.0:
        parser.error("--min-recall-at-k must lie in [0, 1]")
    if not 0.0 < args.min_fps_ratio <= 1.0:
        parser.error("--min-fps-ratio must lie in (0, 1]")
    if not 0.0 <= args.min_post_fallback_coverage <= 1.0:
        parser.error("--min-post-fallback-coverage must lie in [0, 1]")
    if args.max_combined_ms_per_input_frame <= 0.0:
        parser.error("--max-combined-ms-per-input-frame must be positive")
    if args.require_arbitration and not args.require_puf:
        parser.error("--require-arbitration requires --require-puf")
    for name in (
        "min_puf_top1_agreement",
        "min_arbitration_selective_precision",
        "min_conflict_owner_precision",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if args.max_puf_query_p95_ms <= 0.0:
        parser.error("--max-puf-query-p95-ms must be positive")
    if args.max_arbitration_query_p95_ms <= 0.0:
        parser.error("--max-arbitration-query-p95-ms must be positive")
    if args.min_conflict_owner_samples < 0:
        parser.error("--min-conflict-owner-samples must be non-negative")

    scenes = read_scenes(args.scene_list.resolve())
    per_scene = {}
    total_pipeline_ms = 0.0
    total_pipeline_puf_ms = 0.0
    total_pipeline_arbitration_ms = 0.0
    total_conflict_owner_correct = 0
    total_conflict_owner_evaluable = 0
    total_selected_correct = 0
    total_selected_evaluable = 0
    total_frames = 0
    for scene in scenes:
        identity = compare_predictions(
            args.control_root.resolve() / f"{scene}_boxes.pkl",
            args.observer_root.resolve() / f"{scene}_boxes.pkl",
        )
        control = parse_log(
            args.control_log_root.resolve() / f"{scene}.log",
            require_qim=False,
            require_puf=False,
            require_arbitration=False,
        )
        observer = parse_log(
            args.observer_log_root.resolve() / f"{scene}.log",
            require_qim=True,
            require_puf=args.require_puf,
            require_arbitration=args.require_arbitration,
        )
        qim = observer["qim"]
        _validate_rate_fields(qim, field=f"{scene}: QIM")
        native_unresolved = _nonnegative_int(
            qim.get("native_unresolved"),
            field=f"{scene}: QIM native_unresolved",
        )
        safety = {
            "observer_only": qim.get("observer_only") is True,
            "training_free": qim.get("training_free") is True,
            "causal": qim.get("causal") is True,
            "semantic_access_false": qim.get("semantic_access") is False,
            "semantic_mutation_false": (
                qim.get("semantic_mutation") is False
            ),
            "native_unresolved_zero": native_unresolved == 0,
        }
        if not all(safety.values()):
            raise ValueError(f"{scene}: QIM safety contract failed: {safety}")
        recall_at_k = qim.get("recall_at_k_rate")
        if recall_at_k is None or _unit_interval_float(
            recall_at_k, field=f"{scene}: QIM Recall@K"
        ) < args.min_recall_at_k:
            raise ValueError(f"{scene}: QIM Recall@K is below threshold")
        fps_ratio = _finite_float(
            float(observer["fps"]) / float(control["fps"]),
            field=f"{scene}: observer/control FPS ratio",
        )
        if fps_ratio < args.min_fps_ratio:
            raise ValueError(f"{scene}: observer/control FPS ratio is too low")
        qim_query_ms = _nonnegative_float(
            qim["pipeline_query_ms_total"],
            field=f"{scene}: QIM pipeline_query_ms_total",
        )
        qim_update_ms = _nonnegative_float(
            qim["pipeline_update_ms_total"],
            field=f"{scene}: QIM pipeline_update_ms_total",
        )
        pipeline_ms = _nonnegative_float(
            qim_query_ms + qim_update_ms,
            field=f"{scene}: combined QIM pipeline time",
        )
        pipeline_puf_ms = 0.0
        puf_safety = None
        if args.require_puf:
            puf = observer["puf"]
            _validate_rate_fields(puf, field=f"{scene}: PUF")
            proposals = _nonnegative_int(
                puf.get("proposals"), field=f"{scene}: PUF proposals"
            )
            probability_rows = _nonnegative_int(
                puf.get("probability_rows"),
                field=f"{scene}: PUF probability_rows",
            )
            invalid_rows = _nonnegative_int(
                puf.get("invalid_rows"),
                field=f"{scene}: PUF invalid_rows",
            )
            nonfinite_probability_rows = _nonnegative_int(
                puf.get("nonfinite_probability_rows"),
                field=f"{scene}: PUF nonfinite_probability_rows",
            )
            effective_config = puf.get("effective_config")
            if not isinstance(effective_config, dict):
                raise ValueError(
                    f"{scene}: PUF effective_config must be a JSON object"
                )
            max_tracks = _positive_int(
                effective_config.get("max_tracks"),
                field=f"{scene}: PUF max_tracks",
            )
            normalization_error = _nonnegative_float(
                puf.get("max_normalization_error", float("inf")),
                field=f"{scene}: PUF maximum normalization error",
            )
            puf_safety = {
                "observer_only": puf.get("observer_only") is True,
                "training_free": puf.get("training_free") is True,
                "causal": puf.get("causal") is True,
                "online_update_false": puf.get("online_update") is False,
                "semantic_access_false": puf.get("semantic_access") is False,
                "semantic_mutation_false": (
                    puf.get("semantic_mutation") is False
                ),
                "ground_truth_access_false": (
                    puf.get("ground_truth_access") is False
                ),
                "detector_score_access_false": (
                    puf.get("detector_score_access") is False
                ),
                "all_rows_have_probabilities": (
                    probability_rows == proposals
                ),
                "invalid_rows_zero": invalid_rows == 0,
                "nonfinite_probability_rows_zero": (
                    nonfinite_probability_rows == 0
                ),
                "normalization_error_bounded": normalization_error <= 1e-12,
                "frozen_birth_likelihood": (
                    effective_config.get("birth_likelihood")
                    == 0.4
                ),
                "bounded_track_pool": max_tracks <= 1024,
            }
            if not all(puf_safety.values()):
                raise ValueError(
                    f"{scene}: PUF-lite safety contract failed: {puf_safety}"
                )
            coverage = puf.get("post_fallback_target_coverage_rate")
            if coverage is None or _unit_interval_float(
                coverage, field=f"{scene}: PUF post-fallback coverage"
            ) < args.min_post_fallback_coverage:
                raise ValueError(
                    f"{scene}: PUF post-fallback coverage is below threshold"
                )
            top1 = puf.get("top1_native_agreement_rate")
            if top1 is None or _unit_interval_float(
                top1, field=f"{scene}: PUF Top-1 native agreement"
            ) < args.min_puf_top1_agreement:
                raise ValueError(
                    f"{scene}: PUF Top-1 native agreement is below threshold"
                )
            puf_query_p95_ms = _nonnegative_float(
                puf.get("query_ms_p95", float("inf")),
                field=f"{scene}: PUF query_ms_p95",
            )
            if puf_query_p95_ms > args.max_puf_query_p95_ms:
                raise ValueError(f"{scene}: PUF query p95 is above threshold")
            puf_query_ms = _nonnegative_float(
                puf["pipeline_query_ms_total"],
                field=f"{scene}: PUF pipeline_query_ms_total",
            )
            puf_observe_ms = _nonnegative_float(
                puf["pipeline_observe_ms_total"],
                field=f"{scene}: PUF pipeline_observe_ms_total",
            )
            pipeline_puf_ms = _nonnegative_float(
                puf_query_ms + puf_observe_ms,
                field=f"{scene}: combined PUF pipeline time",
            )
        pipeline_arbitration_ms = 0.0
        arbitration_safety = None
        if args.require_arbitration:
            arbitration = observer["arbitration"]
            _validate_rate_fields(
                arbitration, field=f"{scene}: arbitration"
            )
            effective = arbitration.get("effective_config")
            if not isinstance(effective, dict):
                raise ValueError(
                    f"{scene}: arbitration effective_config must be a JSON object"
                )
            source_invalid_rows = _nonnegative_int(
                arbitration.get("source_invalid_rows"),
                field=f"{scene}: arbitration source_invalid_rows",
            )
            proposal_cap_batches = _nonnegative_int(
                arbitration.get("proposal_cap_batches"),
                field=f"{scene}: arbitration proposal_cap_batches",
            )
            duplicate_selected_tracks = _nonnegative_int(
                arbitration.get("duplicate_selected_tracks"),
                field=f"{scene}: arbitration duplicate_selected_tracks",
            )
            selected_wrong = _nonnegative_int(
                arbitration.get("selected_wrong"),
                field=f"{scene}: arbitration selected_wrong",
            )
            false_track_overrides = _nonnegative_int(
                arbitration.get("false_track_overrides"),
                field=f"{scene}: arbitration false_track_overrides",
            )
            false_birth_overrides = _nonnegative_int(
                arbitration.get("false_birth_overrides"),
                field=f"{scene}: arbitration false_birth_overrides",
            )
            (
                conflict_owner_correct,
                conflict_owner_evaluable,
                owner_precision,
            ) = _validate_precision_counts(
                correct_value=arbitration.get(
                    "conflict_owner_group_correct"
                ),
                evaluable_value=arbitration.get(
                    "conflict_owner_group_evaluable"
                ),
                precision_value=arbitration.get(
                    "conflict_owner_group_precision"
                ),
                field=f"{scene}: conflict owner group",
            )
            (
                selected_correct,
                selected_evaluable,
                selective_precision,
            ) = _validate_precision_counts(
                correct_value=arbitration.get("selected_correct"),
                evaluable_value=arbitration.get("selected_evaluable"),
                precision_value=arbitration.get("selective_precision"),
                field=f"{scene}: arbitration selected",
            )
            if selected_wrong != selected_evaluable - selected_correct:
                raise ValueError(
                    f"{scene}: arbitration selected_wrong is inconsistent "
                    "with selected counts"
                )
            arbitration_safety = {
                "observer_only": arbitration.get("observer_only") is True,
                "active_authorized_false": (
                    arbitration.get("active_authorized") is False
                ),
                "training_free": arbitration.get("training_free") is True,
                "causal": arbitration.get("causal") is True,
                "online_update_false": (
                    arbitration.get("online_update") is False
                ),
                "semantic_access_false": (
                    arbitration.get("semantic_access") is False
                ),
                "semantic_mutation_false": (
                    arbitration.get("semantic_mutation") is False
                ),
                "ground_truth_access_false": (
                    arbitration.get("ground_truth_access") is False
                ),
                "detector_score_access_false": (
                    arbitration.get("detector_score_access") is False
                ),
                "losers_not_reassigned": (
                    arbitration.get("reassigns_losers") is False
                ),
                "proposals_not_suppressed": (
                    arbitration.get("suppresses_proposals") is False
                ),
                "source_invalid_zero": (
                    source_invalid_rows == 0
                ),
                "proposal_cap_zero": (
                    proposal_cap_batches == 0
                ),
                "duplicate_directives_zero": (
                    duplicate_selected_tracks == 0
                ),
                "selected_wrong_zero": (
                    selected_wrong == 0
                ),
                "false_track_zero": (
                    false_track_overrides == 0
                ),
                "false_birth_zero": (
                    false_birth_overrides == 0
                ),
                "frozen_track_probability": (
                    effective.get("track_min_probability") == 0.70
                ),
                "frozen_track_margin": (
                    effective.get("track_min_margin") == 0.20
                ),
                "frozen_birth_probability": (
                    effective.get("birth_min_probability") == 0.70
                ),
                "frozen_birth_margin": (
                    effective.get("birth_min_margin") == 0.20
                ),
                "frozen_owner_gap": (
                    effective.get("conflict_min_owner_gap") == 0.10
                ),
            }
            if not all(arbitration_safety.values()):
                raise ValueError(
                    f"{scene}: arbitration safety contract failed: "
                    f"{arbitration_safety}"
                )
            arbitration_query_p95_ms = _nonnegative_float(
                arbitration.get("query_ms_p95", float("inf")),
                field=f"{scene}: arbitration query_ms_p95",
            )
            if arbitration_query_p95_ms > args.max_arbitration_query_p95_ms:
                raise ValueError(
                    f"{scene}: arbitration query p95 is above threshold"
                )
            if (
                selective_precision is not None
                and selective_precision < args.min_arbitration_selective_precision
            ):
                raise ValueError(
                    f"{scene}: arbitration selective precision is below threshold"
                )
            if (
                owner_precision is not None
                and owner_precision < args.min_conflict_owner_precision
            ):
                raise ValueError(
                    f"{scene}: conflict owner precision is below threshold"
                )
            arbitration_query_ms = _nonnegative_float(
                arbitration["pipeline_query_ms_total"],
                field=f"{scene}: arbitration pipeline_query_ms_total",
            )
            arbitration_observe_ms = _nonnegative_float(
                arbitration["pipeline_observe_ms_total"],
                field=f"{scene}: arbitration pipeline_observe_ms_total",
            )
            pipeline_arbitration_ms = _nonnegative_float(
                arbitration_query_ms + arbitration_observe_ms,
                field=f"{scene}: combined arbitration pipeline time",
            )
            total_conflict_owner_correct += conflict_owner_correct
            total_conflict_owner_evaluable += conflict_owner_evaluable
            total_selected_correct += selected_correct
            total_selected_evaluable += selected_evaluable
        frames = int(observer["frame_equivalent"])
        scene_combined_pipeline_ms = _nonnegative_float(
            pipeline_ms + pipeline_puf_ms + pipeline_arbitration_ms,
            field=f"{scene}: combined observer pipeline time",
        )
        total_pipeline_ms = _nonnegative_float(
            total_pipeline_ms + pipeline_ms,
            field="aggregate QIM pipeline time",
        )
        total_pipeline_puf_ms = _nonnegative_float(
            total_pipeline_puf_ms + pipeline_puf_ms,
            field="aggregate PUF pipeline time",
        )
        total_pipeline_arbitration_ms = _nonnegative_float(
            total_pipeline_arbitration_ms + pipeline_arbitration_ms,
            field="aggregate arbitration pipeline time",
        )
        total_frames += frames
        per_scene[scene] = {
            "identity": identity,
            "control": control,
            "observer": observer,
            "fps_ratio": fps_ratio,
            "pipeline_qim_ms": pipeline_ms,
            "pipeline_qim_ms_per_input_frame": pipeline_ms / frames,
            "pipeline_puf_ms": pipeline_puf_ms,
            "pipeline_puf_ms_per_input_frame": pipeline_puf_ms / frames,
            "pipeline_arbitration_ms": pipeline_arbitration_ms,
            "pipeline_arbitration_ms_per_input_frame": (
                pipeline_arbitration_ms / frames
            ),
            "pipeline_combined_ms_per_input_frame": (
                scene_combined_pipeline_ms / frames
            ),
            "safety": safety,
            "puf_safety": puf_safety,
            "arbitration_safety": arbitration_safety,
        }

    total_combined_pipeline_ms = _nonnegative_float(
        total_pipeline_ms
        + total_pipeline_puf_ms
        + total_pipeline_arbitration_ms,
        field="aggregate combined observer pipeline time",
    )
    report = {
        "schema": (
            "boxfusion.moon_qim_puf_arbitration_lite_paired_audit.v1"
            if args.require_arbitration
            else (
                "boxfusion.moon_qim_puf_lite_paired_audit.v1"
                if args.require_puf
                else "boxfusion.moon_qim_lite_paired_audit.v1"
            )
        ),
        "ok": True,
        "scenes": len(scenes),
        "scene_list_sha256": sha256(args.scene_list.resolve()),
        "thresholds": {
            "min_recall_at_k": args.min_recall_at_k,
            "min_fps_ratio": args.min_fps_ratio,
            "min_post_fallback_coverage": (
                args.min_post_fallback_coverage
                if args.require_puf
                else None
            ),
            "max_combined_ms_per_input_frame": (
                args.max_combined_ms_per_input_frame
                if args.require_puf
                else None
            ),
            "min_puf_top1_agreement": (
                args.min_puf_top1_agreement if args.require_puf else None
            ),
            "max_puf_query_p95_ms": (
                args.max_puf_query_p95_ms if args.require_puf else None
            ),
            "min_arbitration_selective_precision": (
                args.min_arbitration_selective_precision
                if args.require_arbitration
                else None
            ),
            "min_conflict_owner_precision": (
                args.min_conflict_owner_precision
                if args.require_arbitration
                else None
            ),
            "min_conflict_owner_samples": (
                args.min_conflict_owner_samples
                if args.require_arbitration
                else None
            ),
            "max_arbitration_query_p95_ms": (
                args.max_arbitration_query_p95_ms
                if args.require_arbitration
                else None
            ),
        },
        "same_prediction_bytes": True,
        "training_free": True,
        "causal": True,
        "semantic_access": False,
        "semantic_mutation": False,
        "total_pipeline_qim_ms": total_pipeline_ms,
        "total_pipeline_puf_ms": total_pipeline_puf_ms,
        "total_pipeline_arbitration_ms": total_pipeline_arbitration_ms,
        "total_frame_equivalent": total_frames,
        "pipeline_qim_ms_per_input_frame": total_pipeline_ms / total_frames,
        "pipeline_puf_ms_per_input_frame": (
            total_pipeline_puf_ms / total_frames
        ),
        "pipeline_arbitration_ms_per_input_frame": (
            total_pipeline_arbitration_ms / total_frames
        ),
        "pipeline_combined_ms_per_input_frame": (
            total_combined_pipeline_ms
        )
        / total_frames,
        "conflict_owner_group_correct": total_conflict_owner_correct,
        "conflict_owner_group_evaluable": total_conflict_owner_evaluable,
        "conflict_owner_group_precision": (
            total_conflict_owner_correct / total_conflict_owner_evaluable
            if total_conflict_owner_evaluable
            else None
        ),
        "selected_correct": total_selected_correct,
        "selected_evaluable": total_selected_evaluable,
        "selective_precision": (
            total_selected_correct / total_selected_evaluable
            if total_selected_evaluable
            else None
        ),
        "per_scene": per_scene,
    }
    if (
        args.require_puf
        and report["pipeline_combined_ms_per_input_frame"]
        > args.max_combined_ms_per_input_frame
    ):
        raise ValueError(
            "combined observer pipeline overhead per input frame is too high"
        )
    if args.require_arbitration:
        if total_conflict_owner_evaluable < args.min_conflict_owner_samples:
            raise ValueError(
                "conflict owner sample count is below the activation threshold"
            )
        if (
            total_conflict_owner_evaluable
            and report["conflict_owner_group_precision"]
            < args.min_conflict_owner_precision
        ):
            raise ValueError(
                "aggregate conflict owner precision is below threshold"
            )
        if (
            total_selected_evaluable
            and report["selective_precision"]
            < args.min_arbitration_selective_precision
        ):
            raise ValueError(
                "aggregate arbitration selective precision is below threshold"
            )
    atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "scenes": report["scenes"],
                "same_prediction_bytes": report["same_prediction_bytes"],
                "pipeline_qim_ms_per_input_frame": report[
                    "pipeline_qim_ms_per_input_frame"
                ],
                "pipeline_puf_ms_per_input_frame": report[
                    "pipeline_puf_ms_per_input_frame"
                ],
                "pipeline_arbitration_ms_per_input_frame": report[
                    "pipeline_arbitration_ms_per_input_frame"
                ],
                "pipeline_combined_ms_per_input_frame": report[
                    "pipeline_combined_ms_per_input_frame"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
