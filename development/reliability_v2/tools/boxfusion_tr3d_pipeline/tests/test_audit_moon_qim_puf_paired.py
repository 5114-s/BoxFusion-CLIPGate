import json
import pickle
import sys

import numpy as np
import pytest

from tools import audit_moon_qim_paired as audit


def prediction_payload():
    corners = np.arange(24, dtype=np.float32).reshape(8, 3)
    return [[(0, corners, 0.75)]]


def write_fixture(
    tmp_path,
    *,
    invalid_puf=False,
    arbitration=False,
    invalid_arbitration=False,
    qim_overrides=None,
    puf_overrides=None,
    arbitration_overrides=None,
):
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n")
    roots = {}
    for name in ("control", "observer", "control_logs", "observer_logs"):
        roots[name] = tmp_path / name
        roots[name].mkdir()
    for name in ("control", "observer"):
        with (roots[name] / f"{scene}_boxes.pkl").open("wb") as handle:
            pickle.dump(prediction_payload(), handle)

    (roots["control_logs"] / f"{scene}.log").write_text(
        "Cost: 10.00 s Average FPS: 20.00\n"
    )
    qim = {
        "observer_only": True,
        "training_free": True,
        "causal": True,
        "semantic_access": False,
        "semantic_mutation": False,
        "native_unresolved": 0,
        "recall_at_k_rate": 1.0,
        "pipeline_query_ms_total": 2.0,
        "pipeline_update_ms_total": 3.0,
    }
    qim.update(qim_overrides or {})
    puf = {
        "observer_only": True,
        "training_free": True,
        "causal": True,
        "online_update": False,
        "semantic_access": False,
        "semantic_mutation": False,
        "ground_truth_access": False,
        "detector_score_access": False,
        "proposals": 10,
        "probability_rows": 10,
        "invalid_rows": int(invalid_puf),
        "nonfinite_probability_rows": 0,
        "max_normalization_error": 1e-16,
        "post_fallback_target_coverage_rate": 1.0,
        "top1_native_agreement_rate": 1.0,
        "query_ms_p95": 0.5,
        "pipeline_query_ms_total": 2.0,
        "pipeline_observe_ms_total": 1.0,
        "effective_config": {
            "birth_likelihood": 0.4,
            "max_tracks": 1024,
        },
    }
    puf.update(puf_overrides or {})
    observer_log = (
        "Moon-QIM-lite observer JSON | "
        + json.dumps(qim)
        + "\nPUF-lite shadow JSON | "
        + json.dumps(puf)
        + "\n"
    )
    if arbitration:
        arb = {
            "observer_only": True,
            "active_authorized": False,
            "training_free": True,
            "causal": True,
            "online_update": False,
            "semantic_access": False,
            "semantic_mutation": False,
            "ground_truth_access": False,
            "detector_score_access": False,
            "reassigns_losers": False,
            "suppresses_proposals": False,
            "source_invalid_rows": 0,
            "proposal_cap_batches": 0,
            "duplicate_selected_tracks": int(invalid_arbitration),
            "selected_wrong": 0,
            "false_track_overrides": 0,
            "false_birth_overrides": 0,
            "query_ms_p95": 0.05,
            "pipeline_query_ms_total": 1.5,
            "pipeline_observe_ms_total": 0.5,
            "selective_precision": 1.0,
            "conflict_owner_group_precision": 1.0,
            "conflict_owner_group_correct": 1,
            "conflict_owner_group_evaluable": 1,
            "selected_correct": 4,
            "selected_evaluable": 4,
            "effective_config": {
                "track_min_probability": 0.70,
                "track_min_margin": 0.20,
                "birth_min_probability": 0.70,
                "birth_min_margin": 0.20,
                "conflict_min_owner_gap": 0.10,
            },
        }
        arb.update(arbitration_overrides or {})
        observer_log += (
            "PUF-arbitration-lite shadow JSON | "
            + json.dumps(arb)
            + "\n"
        )
    observer_log += "Cost: 10.00 s Average FPS: 20.00\n"
    (roots["observer_logs"] / f"{scene}.log").write_text(observer_log)
    return scene_list, roots


def run_audit(
    monkeypatch,
    tmp_path,
    *,
    invalid_puf=False,
    arbitration=False,
    invalid_arbitration=False,
    qim_overrides=None,
    puf_overrides=None,
    arbitration_overrides=None,
    extra_args=(),
):
    scene_list, roots = write_fixture(
        tmp_path,
        invalid_puf=invalid_puf,
        arbitration=arbitration,
        invalid_arbitration=invalid_arbitration,
        qim_overrides=qim_overrides,
        puf_overrides=puf_overrides,
        arbitration_overrides=arbitration_overrides,
    )
    output = tmp_path / "report.json"
    argv = [
        "audit_moon_qim_paired.py",
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
        "--require-puf",
        "--max-combined-ms-per-input-frame",
        "0.1",
    ]
    if arbitration:
        argv.extend(("--require-arbitration", "--min-conflict-owner-samples", "1"))
    argv.extend(extra_args)
    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )
    result = audit.main()
    return result, json.loads(output.read_text()) if output.exists() else None


def test_combined_audit_requires_identity_safety_and_bounded_overhead(
    monkeypatch, tmp_path
):
    result, report = run_audit(monkeypatch, tmp_path)
    assert result == 0
    assert report["ok"] is True
    assert report["same_prediction_bytes"] is True
    assert report["schema"] == "boxfusion.moon_qim_puf_lite_paired_audit.v1"
    assert report["pipeline_combined_ms_per_input_frame"] == pytest.approx(
        8.0 / 200.0
    )


def test_combined_audit_rejects_any_invalid_probability_row(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="PUF-lite safety contract failed"):
        run_audit(monkeypatch, tmp_path, invalid_puf=True)


def test_parser_rejects_puf_output_in_control_log(tmp_path):
    path = tmp_path / "control.log"
    path.write_text(
        "PUF-lite shadow JSON | {}\nCost: 1.00 s Average FPS: 1.00\n"
    )
    with pytest.raises(ValueError, match="unexpectedly contains PUF"):
        audit.parse_log(path, require_qim=False, require_puf=False)


def test_arbitration_audit_checks_frozen_safety_and_aggregate_samples(
    monkeypatch, tmp_path
):
    result, report = run_audit(monkeypatch, tmp_path, arbitration=True)
    assert result == 0
    assert report["schema"] == (
        "boxfusion.moon_qim_puf_arbitration_lite_paired_audit.v1"
    )
    assert report["conflict_owner_group_evaluable"] == 1
    assert report["conflict_owner_group_precision"] == 1.0
    assert report["pipeline_arbitration_ms_per_input_frame"] == pytest.approx(
        2.0 / 200.0
    )
    assert report["pipeline_combined_ms_per_input_frame"] == pytest.approx(
        10.0 / 200.0
    )


def test_arbitration_audit_rejects_duplicate_selected_track_directive(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="arbitration safety contract failed"):
        run_audit(
            monkeypatch,
            tmp_path,
            arbitration=True,
            invalid_arbitration=True,
        )


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity", "1e9999"))
def test_log_parser_strictly_rejects_nonfinite_json_numbers(tmp_path, token):
    path = tmp_path / "observer.log"
    path.write_text(
        f'Moon-QIM-lite observer JSON | {{"metric": {token}}}\n'
        "Cost: 1.00 s Average FPS: 1.00\n"
    )
    with pytest.raises(ValueError, match="non-finite JSON number is forbidden"):
        audit.parse_log(path, require_qim=True)


def test_audit_rejects_nonfinite_gate_metric_encoded_as_string(
    monkeypatch, tmp_path
):
    with pytest.raises(
        ValueError, match="QIM recall_at_k_rate must be a finite number"
    ):
        run_audit(
            monkeypatch,
            tmp_path,
            qim_overrides={"recall_at_k_rate": "NaN"},
        )


def test_audit_rejects_nonfinite_upper_bound_metric_encoded_as_string(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="PUF query_ms_p95 must be a finite"):
        run_audit(
            monkeypatch,
            tmp_path,
            puf_overrides={"query_ms_p95": "Infinity"},
        )


def test_audit_rejects_negative_pipeline_timing(monkeypatch, tmp_path):
    with pytest.raises(
        ValueError, match="QIM pipeline_query_ms_total must be non-negative"
    ):
        run_audit(
            monkeypatch,
            tmp_path,
            qim_overrides={"pipeline_query_ms_total": -1.0},
        )


@pytest.mark.parametrize("threshold", ("nan", "inf", "-inf"))
def test_audit_rejects_nonfinite_cli_threshold(
    monkeypatch, tmp_path, threshold
):
    with pytest.raises(SystemExit):
        run_audit(
            monkeypatch,
            tmp_path,
            extra_args=("--max-combined-ms-per-input-frame", threshold),
        )


@pytest.mark.parametrize("invalid_count", ("NaN", -1, False, 1.0))
def test_audit_rejects_non_integer_or_negative_puf_equality_counts(
    monkeypatch, tmp_path, invalid_count
):
    with pytest.raises(ValueError, match="PUF proposals"):
        run_audit(
            monkeypatch,
            tmp_path,
            puf_overrides={
                "proposals": invalid_count,
                "probability_rows": invalid_count,
            },
        )


@pytest.mark.parametrize(
    "effective_config",
    (
        {"birth_likelihood": 0.4, "max_tracks": 0},
        {"birth_likelihood": 0.4, "max_tracks": False},
        {"birth_likelihood": 0.4, "max_tracks": -1},
        {"birth_likelihood": 0.4, "max_tracks": 1.0},
        {"birth_likelihood": 0.4},
    ),
)
def test_audit_requires_strict_positive_integer_puf_max_tracks(
    monkeypatch, tmp_path, effective_config
):
    with pytest.raises(ValueError, match="PUF max_tracks"):
        run_audit(
            monkeypatch,
            tmp_path,
            puf_overrides={"effective_config": effective_config},
        )


def test_arbitration_audit_rejects_truncated_float_counts(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="correct must be a non-bool integer"):
        run_audit(
            monkeypatch,
            tmp_path,
            arbitration=True,
            arbitration_overrides={
                "conflict_owner_group_correct": 1.9,
                "conflict_owner_group_evaluable": 1.9,
            },
        )


def test_arbitration_audit_rejects_correct_above_evaluable(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="correct <= evaluable"):
        run_audit(
            monkeypatch,
            tmp_path,
            arbitration=True,
            arbitration_overrides={
                "conflict_owner_group_correct": 2,
                "conflict_owner_group_evaluable": 1,
            },
        )


def test_arbitration_audit_requires_precision_when_denominator_is_positive(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="precision must be non-null"):
        run_audit(
            monkeypatch,
            tmp_path,
            arbitration=True,
            arbitration_overrides={"selective_precision": None},
        )


def test_arbitration_audit_requires_null_precision_for_zero_denominator(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="precision must be null iff"):
        run_audit(
            monkeypatch,
            tmp_path,
            arbitration=True,
            arbitration_overrides={
                "selected_correct": 0,
                "selected_evaluable": 0,
                "selective_precision": 0.0,
            },
        )


def test_arbitration_audit_rejects_inconsistent_reported_precision(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="precision is inconsistent"):
        run_audit(
            monkeypatch,
            tmp_path,
            arbitration=True,
            arbitration_overrides={
                "selected_correct": 1,
                "selected_evaluable": 2,
                "selected_wrong": 1,
                "selective_precision": 1.0,
            },
        )


@pytest.mark.parametrize("invalid_rate", (1.01, -0.01, True))
def test_audit_rejects_rate_outside_unit_interval_or_bool(
    monkeypatch, tmp_path, invalid_rate
):
    with pytest.raises(ValueError, match="QIM recall_at_k_rate"):
        run_audit(
            monkeypatch,
            tmp_path,
            qim_overrides={"recall_at_k_rate": invalid_rate},
        )


def test_audit_checks_all_reported_puf_rates_not_only_gate_result(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="PUF diagnostic_rate"):
        run_audit(
            monkeypatch,
            tmp_path,
            puf_overrides={"diagnostic_rate": 1.01},
        )
