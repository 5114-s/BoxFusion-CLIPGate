from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from boxfusion.moon_qim_lite import QIMCandidate, QIMQueryBatch
from boxfusion.mv3dis_depth_lite import (
    DepthGuideProjectionMetrics,
    MV3DISDepthLiteObserver,
)
from tools import audit_mv3dis_depth_lite as audit


def _config() -> dict[str, object]:
    return {
        "enabled": True,
        "observer_only": True,
        "max_guides_per_track": 5,
        "max_depth_frames": 80,
        "max_proposals": 256,
        "max_qim_candidates": 3,
        "projection_budget_points": 8192,
        "points_per_projection": 64,
        "frame_visibility_threshold": 0.30,
        "box_visibility_threshold": 0.90,
        "candidate_dominance_threshold": 0.90,
        "min_history_views": 2,
        "alpha": 0.05,
        "max_diagnostic_examples": 256,
    }


def _diagnostic(scene: str = "scene0000_00") -> list[object]:
    return [
        scene,
        175,
        18,
        [15],
        True,
        15,
        1.0,
        "two_view_unique_candidate_shadow_veto",
        [
            [
                15,
                0,
                2,
                2,
                2,
                1.8,
                True,
                [
                    [150, True, 0.9, 0.95, 0.95, 0.90, True, 64, "support"],
                    [160, True, 0.9, 0.96, 0.96, 0.90, True, 64, "support"],
                ],
            ]
        ],
    ]


def _summary(scene: str = "scene0000_00") -> dict[str, object]:
    return {
        "schema": audit.OBSERVER_SCHEMA,
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
        "scene_id": scene,
        "effective_config": _config(),
        "queries": 1,
        "commits": 1,
        "proposals": 1,
        "proposal_cap_batches": 0,
        "invalid_frame_batches": 0,
        "guide_quality_rows_valid": 1,
        "guide_quality_rows_invalid": 0,
        "veto_recommendations": 1,
        "veto_evaluable": 1,
        "veto_correct": 1,
        "veto_wrong": 0,
        "veto_on_native_birth": 0,
        "native_history": 1,
        "native_birth": 0,
        "native_unresolved": 0,
        "native_diagnostics_skipped": 0,
        "geometry_calls": 3,
        "geometry_errors": 0,
        "projection_points": 192,
        "guide_quality_projection_points": 64,
        "birth_veto_projection_points": 128,
        "guide_quality_budget_exhaustions": 0,
        "birth_veto_budget_exhaustions": 0,
        "guides_committed": 1,
        "guides_replaced_same_frame": 0,
        "guides_evicted_track_cap": 0,
        "guides_evicted_frame_cap": 0,
        "committed_frames_evicted": 0,
        "max_committed_frames_observed": 1,
        "max_tracks_observed": 1,
        "max_guides_observed": 1,
        "committed_frames_retained": 1,
        "tracks_retained": 1,
        "guides_retained": 1,
        "query_ms_total": 4.0,
        "query_ms_max": 4.0,
        "query_ms_mean": 4.0,
        "query_ms_p95": 4.0,
        "commit_ms_total": 0.5,
        "commit_ms_max": 0.5,
        "commit_ms_mean": 0.5,
        "commit_ms_p95": 0.5,
        "pipeline_query_calls": 1,
        "pipeline_query_ms_total": 5.0,
        "pipeline_query_ms_max": 5.0,
        "pipeline_query_ms_mean": 5.0,
        "pipeline_commit_calls": 1,
        "pipeline_commit_ms_total": 1.0,
        "pipeline_commit_ms_max": 1.0,
        "pipeline_commit_ms_mean": 1.0,
        "veto_precision": 1.0,
        "invalid_frame_reasons": [],
        "diagnostic_examples": [_diagnostic(scene)],
    }


def _write_fixture(
    tmp_path,
    *,
    summary: dict[str, object] | None = None,
    observer_geometry_delta: float = 0.0,
    raw_observer_summary: str | None = None,
    control_has_summary: bool = False,
):
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    roots = {}
    for name in ("control", "observer", "control_logs", "observer_logs"):
        roots[name] = tmp_path / name
        roots[name].mkdir()
    control_corners = np.arange(24, dtype=np.float32).reshape(8, 3)
    observer_corners = control_corners.copy()
    observer_corners[0, 0] += np.float32(observer_geometry_delta)
    with (roots["control"] / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[(0, control_corners, 0.75)]], handle)
    with (roots["observer"] / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[(0, observer_corners, 0.75)]], handle)
    control_prefix = ""
    if control_has_summary:
        control_prefix = audit.MV3DIS_JSON_PREFIX + json.dumps(_summary()) + "\n"
    (roots["control_logs"] / f"{scene}.log").write_text(
        control_prefix + "Cost: 10.00 s Average FPS: 20.00\n", encoding="utf-8"
    )
    payload = raw_observer_summary
    if payload is None:
        payload = json.dumps(summary if summary is not None else _summary())
    (roots["observer_logs"] / f"{scene}.log").write_text(
        audit.MV3DIS_JSON_PREFIX
        + payload
        + "\nCost: 10.20 s Average FPS: 19.61\n",
        encoding="utf-8",
    )
    return scene, scene_list, roots


def _run(tmp_path, *, fixture_kwargs=None, extra_args=()):
    scene, scene_list, roots = _write_fixture(
        tmp_path, **(fixture_kwargs or {})
    )
    output = tmp_path / "report.json"
    argv = [
        "--scene-list",
        str(scene_list),
        "--control-root",
        str(roots["control"]),
        "--observer-root",
        str(roots["observer"]),
        "--control-log-root",
        str(roots["control_logs"]),
        "--observer-log-root",
        str(roots["observer_logs"]),
        "--output",
        str(output),
        *extra_args,
    ]
    result = audit.main(argv)
    return result, json.loads(output.read_text(encoding="utf-8")), scene


def test_paired_shadow_audit_reports_identity_precision_and_runtime(tmp_path):
    result, report, _ = _run(tmp_path)
    assert result == 0
    assert report["schema"] == audit.REPORT_SCHEMA
    assert report["ok"] is True
    assert report["active_authorized"] is False
    assert report["activation_decision"] == "not_authorized_by_design"
    assert report["same_prediction_bytes"] is True
    assert report["same_prediction_arrays_exact"] is True
    assert report["prediction_numeric_identity_within_tolerance"] is True
    assert report["veto_precision"] == 1.0
    assert report["veto_coverage"] == 1.0
    assert report["pipeline_ms_per_input_frame"] == pytest.approx(6.0 / 200.0)


def test_validator_accepts_json_roundtrip_of_real_observer_summary():
    def geometry(*args, **kwargs):
        return DepthGuideProjectionMetrics(
            visibility=0.8,
            depth_consistency=0.9,
            quality=0.85,
            frame_visibility=0.8,
            box_visibility=0.95,
            box_depth_consistency=0.9,
            affinity=0.9,
        )

    scene = "scene0000_00"
    observer = MV3DISDepthLiteObserver(
        {"enabled": True, "observer_only": True},
        projection_adapter=geometry,
    )
    candidate = QIMCandidate(
        track_id=15,
        shared_key_count=2,
        shared_key_fraction=1.0,
        center_distance_m=0.0,
        aabb_iou=1.0,
        age_keyframes=0,
        active_at_last_commit=True,
    )
    for frame_id in range(3):
        qim = QIMQueryBatch(
            scene_id=scene,
            frame_id=frame_id,
            proposal_ids=(18,),
            candidates=((candidate,),) if frame_id else ((),),
            history_max_frame_id=None if frame_id == 0 else frame_id - 1,
            query_ms=0.0,
        )
        batch = observer.query(
            qim_batch=qim,
            proposal_points_world=(np.asarray([[0.0, 0.0, 1.0]]),),
            depth_m=np.ones((10, 10), dtype=np.float32),
            K=np.eye(3),
            T_wc=np.eye(4),
            proposal_boxes_xyxy=((0.0, 0.0, 2.0, 2.0),),
        )
        observer.record_pipeline_timing(query_ms=1.0)
        observer.commit(
            batch,
            committed_track_ids=(15,),
            native_target_track_ids=((),) if frame_id == 0 else ((15,),),
        )
        observer.record_pipeline_timing(commit_ms=0.5)
    strict_roundtrip = json.loads(json.dumps(observer.summary(), allow_nan=False))
    validated = audit.validate_summary(strict_roundtrip, scene_id=scene)
    assert validated["counts"]["native_birth"] == 1
    assert validated["counts"]["guide_quality_rows_valid"] == 3
    assert validated["counts"]["veto_correct"] == 1


def test_numeric_tolerance_does_not_misreport_byte_or_exact_identity(tmp_path):
    _, report, scene = _run(
        tmp_path, fixture_kwargs={"observer_geometry_delta": 2e-5}
    )
    assert report["same_prediction_bytes"] is False
    assert report["same_prediction_arrays_exact"] is False
    identity = report["per_scene"][scene]["identity"]
    assert identity["numeric_identity_within_tolerance"] is True
    assert identity["geometry_max_abs_delta"] == pytest.approx(2e-5, abs=2e-6)


def test_prediction_drift_beyond_bound_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="geometry drift exceeds tolerance"):
        _run(tmp_path, fixture_kwargs={"observer_geometry_delta": 2e-4})


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity", "1e9999"))
def test_observer_json_rejects_nonfinite_tokens(tmp_path, token):
    payload = json.dumps(_summary()).replace('"query_ms_total": 4.0', f'"query_ms_total": {token}')
    with pytest.raises(ValueError, match="non-finite JSON number is forbidden"):
        _run(tmp_path, fixture_kwargs={"raw_observer_summary": payload})


def test_numeric_string_nonfinite_is_rejected_by_schema_validator(tmp_path):
    summary = _summary()
    summary["pipeline_query_ms_total"] = "Infinity"
    with pytest.raises(ValueError, match="pipeline_query_ms_total must be a finite"):
        _run(tmp_path, fixture_kwargs={"summary": summary})


def test_wrong_observer_schema_fails_closed(tmp_path):
    summary = _summary()
    summary["schema"] = "wrong"
    with pytest.raises(ValueError, match="unexpected schema"):
        _run(tmp_path, fixture_kwargs={"summary": summary})


def test_duplicate_json_object_keys_are_rejected(tmp_path):
    payload = json.dumps(_summary()).replace(
        '{"schema":', '{"schema": "duplicate", "schema":', 1
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _run(tmp_path, fixture_kwargs={"raw_observer_summary": payload})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("active_authorized", True),
        ("observer_only", False),
        ("native_outputs_mutated", True),
        ("birth_veto_applied", True),
        ("fusion_weights_computed", True),
        ("puf_access", True),
        ("hardcoded_scene_event_access", True),
    ),
)
def test_shadow_safety_flags_cannot_authorize_or_mutate(tmp_path, field, value):
    summary = _summary()
    summary[field] = value
    with pytest.raises(ValueError, match="observer safety contract failed"):
        _run(tmp_path, fixture_kwargs={"summary": summary})


def test_veto_precision_counts_must_be_consistent(tmp_path):
    summary = _summary()
    summary["veto_correct"] = 0
    with pytest.raises(ValueError, match="veto precision counts are inconsistent"):
        _run(tmp_path, fixture_kwargs={"summary": summary})


def test_pipeline_timing_call_coverage_is_required(tmp_path):
    summary = _summary()
    summary["pipeline_query_calls"] = 0
    summary["pipeline_query_ms_mean"] = 0.0
    with pytest.raises(ValueError, match="pipeline query timing coverage is incomplete"):
        _run(tmp_path, fixture_kwargs={"summary": summary})


def test_control_log_must_not_contain_observer_output(tmp_path):
    with pytest.raises(ValueError, match="control log unexpectedly contains MV3DIS"):
        _run(tmp_path, fixture_kwargs={"control_has_summary": True})


def test_optional_known_event_join_checks_generic_diagnostic(tmp_path):
    scene, scene_list, roots = _write_fixture(tmp_path)
    known = tmp_path / "known.json"
    known.write_text(
        json.dumps(
            {
                "schema": audit.KNOWN_EVENTS_SCHEMA,
                "events": [
                    {
                        "scene_id": scene,
                        "frame_id": 175,
                        "proposal_id": 18,
                        "expected_native_target_track_ids": [15],
                        "expected_would_veto_birth": True,
                        "expected_recommended_track_id": 15,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    result = audit.main(
        [
            "--scene-list",
            str(scene_list),
            "--control-root",
            str(roots["control"]),
            "--observer-root",
            str(roots["observer"]),
            "--control-log-root",
            str(roots["control_logs"]),
            "--observer-log-root",
            str(roots["observer_logs"]),
            "--known-events-json",
            str(known),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert report["known_event_count"] == 1
    assert report["known_events_all_joined"] is True
    assert report["known_event_rows"][0]["diagnostic"]["recommended_track_id"] == 15


def test_known_event_missing_from_capped_diagnostics_fails(tmp_path):
    scene, scene_list, roots = _write_fixture(tmp_path)
    known = tmp_path / "known.json"
    known.write_text(
        json.dumps(
            {
                "schema": audit.KNOWN_EVENTS_SCHEMA,
                "events": [
                    {"scene_id": scene, "frame_id": 999, "proposal_id": 18}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from capped diagnostics"):
        audit.main(
            [
                "--scene-list",
                str(scene_list),
                "--control-root",
                str(roots["control"]),
                "--observer-root",
                str(roots["observer"]),
                "--control-log-root",
                str(roots["control_logs"]),
                "--observer-log-root",
                str(roots["observer_logs"]),
                "--known-events-json",
                str(known),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_optional_evidence_and_realtime_gates_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="evaluable veto samples"):
        _run(tmp_path, extra_args=("--min-veto-evaluable", "2"))
    # A separate temporary directory avoids colliding fixture subdirectories.
    second = tmp_path / "latency"
    second.mkdir()
    with pytest.raises(ValueError, match="pipeline overhead exceeds"):
        _run(
            second,
            extra_args=("--max-pipeline-ms-per-input-frame", "0.01"),
        )


def test_diagnostic_nested_metrics_and_counts_are_strict(tmp_path):
    summary = _summary()
    diagnostic = _diagnostic()
    diagnostic[8][0][3] = 1  # evaluated, but two valid view rows remain
    diagnostic[8][0][4] = 1
    diagnostic[8][0][5] = 0.9
    summary["diagnostic_examples"] = [diagnostic]
    with pytest.raises(ValueError, match="view counts are inconsistent"):
        _run(tmp_path, fixture_kwargs={"summary": summary})
