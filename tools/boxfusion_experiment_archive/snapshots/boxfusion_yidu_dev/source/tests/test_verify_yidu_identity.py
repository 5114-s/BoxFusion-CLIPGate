from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.verify_yidu_identity import (
    _compare_values,
    audit_identity,
    main,
)


SCENE = "scene0001_00"


def _prediction_payload() -> list[list[tuple[str, np.ndarray, float]]]:
    corners = np.arange(24, dtype=np.float32).reshape(8, 3)
    return [[("chair", corners, 0.75)]]


def _write_pickle(root: Path, payload=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{SCENE}_boxes.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            _prediction_payload() if payload is None else payload,
            handle,
        )
    return path


def _write_diagnostic(root: Path, **overrides) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "yidu_mutation_enabled": np.asarray(False, dtype=np.bool_),
        "yidu_applied_count": np.asarray(0, dtype=np.int64),
        "yidu_applied": np.zeros(3, dtype=np.bool_),
        "yidu_stage": np.asarray("A1"),
        "yidu_profile": np.asarray(
            "yidu_a1_adaptive_erosion_observer"
        ),
        "yidu_modules_json": np.asarray(
            json.dumps(
                {
                    "adaptive_erosion": True,
                    "dfu_filter": False,
                    "voxel_components": False,
                    "occupancy_msr": False,
                    "raw_fused_query": False,
                    "quality_gate": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }
    payload.update(overrides)
    path = root / f"{SCENE}_tracks.npz"
    np.savez(path, **payload)
    return path


def _valid_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    baseline = tmp_path / "b0"
    observer = tmp_path / "a1"
    diagnostics = tmp_path / "diag"
    _write_pickle(baseline)
    _write_pickle(observer)
    _write_diagnostic(diagnostics)
    return baseline, observer, diagnostics


def test_recursive_compare_requires_array_dtype_shape_and_bytes() -> None:
    value = {"row": [np.asarray([-0.0, np.nan], dtype=np.float32)]}
    identical = {"row": [np.asarray([-0.0, np.nan], dtype=np.float32)]}
    assert _compare_values(value, identical) is None

    dtype_changed = {
        "row": [np.asarray([-0.0, np.nan], dtype=np.float64)]
    }
    assert "dtype mismatch" in _compare_values(value, dtype_changed)

    sign_changed = {"row": [np.asarray([0.0, np.nan], dtype=np.float32)]}
    assert "value bytes differ" in _compare_values(value, sign_changed)

    shape_changed = {
        "row": [np.asarray([[-0.0, np.nan]], dtype=np.float32)]
    }
    assert "shape mismatch" in _compare_values(value, shape_changed)


def test_identical_pickle_npy_and_npz_artifacts_pass(tmp_path: Path) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    array = np.arange(8, dtype=np.int16).reshape(2, 4)
    np.save(baseline / f"{SCENE}_scores.npy", array)
    np.save(observer / f"{SCENE}_scores.npy", array.copy())
    np.savez(
        baseline / f"{SCENE}_extra.npz",
        names=np.asarray(["chair"]),
        scores=np.asarray([0.25], dtype=np.float32),
    )
    np.savez(
        observer / f"{SCENE}_extra.npz",
        scores=np.asarray([0.25], dtype=np.float32),
        names=np.asarray(["chair"]),
    )

    report = audit_identity(baseline, observer, diagnostics)

    assert report.ok
    assert report.prediction_files == 3
    assert report.prediction_scenes == (SCENE,)


def test_prediction_value_change_reports_structural_path(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    changed = _prediction_payload()
    changed[0][0] = ("chair", changed[0][0][1].copy(), 0.5)
    _write_pickle(observer, changed)

    report = audit_identity(baseline, observer, diagnostics)

    assert not report.ok
    issue = next(
        item for item in report.issues
        if item.kind == "prediction_mismatch"
    )
    assert issue.relative_path == f"{SCENE}_boxes.pkl"
    assert issue.object_path.endswith("[0][0][2]")
    assert "float value bytes differ" in issue.message


def test_numeric_envelope_is_explicit_and_reports_drift(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    changed = _prediction_payload()
    shifted = changed[0][0][1].copy()
    shifted += np.float32(0.005)
    changed[0][0] = ("chair", shifted, 0.751)
    _write_pickle(observer, changed)

    strict = audit_identity(baseline, observer, diagnostics)
    assert not strict.ok
    assert strict.comparison_mode == "strict_bitwise"
    assert strict.numeric_envelope is None

    tolerant = audit_identity(
        baseline,
        observer,
        diagnostics,
        max_corner_abs=0.01,
        max_score_abs=0.01,
        max_matched_iou_loss=0.01,
    )
    assert tolerant.ok
    assert tolerant.comparison_mode == "explicit_numeric_envelope"
    assert tolerant.numeric_envelope is not None
    assert tolerant.numeric_envelope.max_corner_abs == 0.01
    assert tolerant.numeric_summary["prediction_rows"] == 1
    assert tolerant.numeric_summary["changed_prediction_rows"] == 1
    assert 0.0049 < tolerant.numeric_summary["corner_abs"]["max"] < 0.0051
    assert 0.0009 < tolerant.numeric_summary["score_abs"]["max"] < 0.0011
    assert (
        tolerant.numeric_summary["ordering"]
        == "same_index_no_rematching_and_exact_score_rank"
    )


def test_numeric_envelope_rejects_limit_excess_and_partial_configuration(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    changed = _prediction_payload()
    shifted = changed[0][0][1].copy()
    shifted += np.float32(0.05)
    changed[0][0] = ("chair", shifted, 0.8)
    _write_pickle(observer, changed)

    report = audit_identity(
        baseline,
        observer,
        diagnostics,
        max_corner_abs=0.01,
        max_score_abs=0.01,
        max_matched_iou_loss=0.001,
    )
    assert not report.ok
    envelope_issues = [
        issue
        for issue in report.issues
        if issue.kind == "prediction_numeric_envelope_exceeded"
    ]
    assert len(envelope_issues) >= 2
    messages = "\n".join(issue.message for issue in envelope_issues)
    assert "corner absolute difference" in messages
    assert "score absolute difference" in messages

    with pytest.raises(ValueError, match="requires"):
        audit_identity(
            baseline,
            observer,
            diagnostics,
            max_corner_abs=0.01,
        )
    with pytest.raises(ValueError, match="must not exceed 1"):
        audit_identity(
            baseline,
            observer,
            diagnostics,
            max_corner_abs=0.01,
            max_score_abs=0.01,
            max_matched_iou_loss=1.1,
        )


def test_numeric_envelope_keeps_labels_row_order_and_safety_strict(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    first = np.arange(24, dtype=np.float32).reshape(8, 3)
    second = first + np.float32(100.0)
    baseline_payload = [
        [
            ("chair", first, 0.8),
            ("chair", second, 0.4),
        ]
    ]
    observer_payload = [
        [
            ("chair", second.copy(), 0.4),
            ("chair", first.copy(), 0.8),
        ]
    ]
    _write_pickle(baseline, baseline_payload)
    _write_pickle(observer, observer_payload)

    reordered = audit_identity(
        baseline,
        observer,
        diagnostics,
        max_corner_abs=200.0,
        max_score_abs=1.0,
        max_matched_iou_loss=1.0,
    )
    assert not reordered.ok
    assert any(
        issue.kind == "prediction_label_or_order_mismatch"
        for issue in reordered.issues
    )

    observer_payload[0][0] = ("table", first.copy(), 0.8)
    observer_payload[0][1] = ("chair", second.copy(), 0.4)
    _write_pickle(observer, observer_payload)
    label_changed = audit_identity(
        baseline,
        observer,
        diagnostics,
        max_corner_abs=200.0,
        max_score_abs=1.0,
        max_matched_iou_loss=1.0,
    )
    assert not label_changed.ok
    assert any(
        issue.kind == "prediction_label_or_order_mismatch"
        for issue in label_changed.issues
    )

    _write_pickle(observer, baseline_payload)
    _write_diagnostic(
        diagnostics,
        yidu_applied=np.asarray([True], dtype=np.bool_),
    )
    unsafe = audit_identity(
        baseline,
        observer,
        diagnostics,
        max_corner_abs=200.0,
        max_score_abs=1.0,
        max_matched_iou_loss=1.0,
    )
    assert not unsafe.ok
    assert any(issue.kind == "observer_applied" for issue in unsafe.issues)


def test_missing_and_extra_prediction_files_are_failures(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    (observer / f"{SCENE}_boxes.pkl").unlink()
    np.save(observer / "scene0002_00_boxes.npy", np.zeros(1))

    report = audit_identity(baseline, observer, diagnostics)
    kinds = {issue.kind for issue in report.issues}

    assert "missing_observer_file" in kinds
    assert "extra_observer_file" in kinds


def test_missing_diagnostic_keys_are_explicit_failures(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    _write_diagnostic(diagnostics)
    path = diagnostics / f"{SCENE}_tracks.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {
            key: np.array(archive[key], copy=True)
            for key in archive.files
            if key != "yidu_mutation_enabled"
        }
    np.savez(path, **payload)

    report = audit_identity(baseline, observer, diagnostics)
    missing = [
        issue for issue in report.issues
        if issue.kind == "missing_or_invalid_safety_key"
    ]

    assert not report.ok
    assert any(
        "missing required diagnostic key 'yidu_mutation_enabled'"
        in issue.message
        for issue in missing
    )


def test_every_true_or_nonzero_safety_indicator_fails(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    _write_diagnostic(
        diagnostics,
        yidu_mutation_enabled=np.asarray(True, dtype=np.bool_),
        yidu_applied_count=np.asarray(2, dtype=np.int64),
        yidu_applied=np.asarray([False, True], dtype=np.bool_),
    )

    report = audit_identity(baseline, observer, diagnostics)
    messages = "\n".join(issue.message for issue in report.issues)

    assert not report.ok
    assert "mutation_enabled=true" in messages
    assert "applied_count=2" in messages
    assert "1 applied row" in messages


def test_required_in_process_zero_write_contract(tmp_path: Path) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    digest = "a" * 64
    _write_diagnostic(
        diagnostics,
        yidu_zero_write_check_enabled=np.asarray(True, dtype=np.bool_),
        yidu_zero_write_verified=np.asarray(True, dtype=np.bool_),
        yidu_zero_write_pre_sha256=np.asarray(digest),
        yidu_zero_write_post_sha256=np.asarray(digest),
        yidu_zero_write_array_names=np.asarray(
            ["corners", "boxes", "scores"], dtype=np.str_
        ),
        yidu_zero_write_changed_fields=np.asarray([], dtype=np.str_),
    )

    passing = audit_identity(
        baseline,
        observer,
        diagnostics,
        expected_stage="A1",
        require_zero_write=True,
    )
    assert passing.ok
    assert passing.require_zero_write is True

    _write_diagnostic(
        diagnostics,
        yidu_zero_write_check_enabled=np.asarray(True, dtype=np.bool_),
        yidu_zero_write_verified=np.asarray(True, dtype=np.bool_),
        yidu_zero_write_pre_sha256=np.asarray(digest),
        yidu_zero_write_post_sha256=np.asarray("b" * 64),
        yidu_zero_write_array_names=np.asarray(["scores"], dtype=np.str_),
        yidu_zero_write_changed_fields=np.asarray(
            ["scores"], dtype=np.str_
        ),
    )
    failing = audit_identity(
        baseline,
        observer,
        diagnostics,
        expected_stage="A1",
        require_zero_write=True,
    )
    kinds = {issue.kind for issue in failing.issues}
    assert "observer_zero_write_hash_mismatch" in kinds
    assert "observer_zero_write_changed" in kinds


def test_required_zero_write_rejects_legacy_diagnostics(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)

    report = audit_identity(
        baseline,
        observer,
        diagnostics,
        expected_stage="A1",
        require_zero_write=True,
    )

    assert not report.ok
    assert any(
        issue.kind == "missing_or_invalid_zero_write_key"
        for issue in report.issues
    )


def test_expected_stage_requires_exact_profile_and_module_matrix(
    tmp_path: Path,
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)

    passing = audit_identity(
        baseline,
        observer,
        diagnostics,
        expected_stage="A1",
    )
    assert passing.ok

    failing = audit_identity(
        baseline,
        observer,
        diagnostics,
        expected_stage="A2",
    )
    assert not failing.ok
    assert any(
        issue.kind == "stage_contract_mismatch"
        for issue in failing.issues
    )


def test_missing_scene_diagnostic_and_empty_directory_fail(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "b0"
    observer = tmp_path / "a1"
    diagnostics = tmp_path / "diag"
    _write_pickle(baseline)
    _write_pickle(observer)
    diagnostics.mkdir()

    report = audit_identity(baseline, observer, diagnostics)
    kinds = {issue.kind for issue in report.issues}

    assert "missing_diagnostics" in kinds
    assert "missing_scene_diagnostic" in kinds


def test_empty_baseline_is_never_reported_as_identity(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "b0"
    observer = tmp_path / "a1"
    diagnostics = tmp_path / "diag"
    baseline.mkdir()
    observer.mkdir()
    diagnostics.mkdir()

    report = audit_identity(baseline, observer, diagnostics)

    assert not report.ok
    assert any(
        issue.kind == "missing_baseline_predictions"
        for issue in report.issues
    )


def test_cli_text_and_json_exit_codes(tmp_path: Path, capsys) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    common = [
        "--baseline-root",
        str(baseline),
        "--observer-root",
        str(observer),
        "--diagnostics-root",
        str(diagnostics),
        "--expected-stage",
        "A1",
    ]

    assert main(common) == 0
    text = capsys.readouterr().out
    assert "identity audit: PASS" in text
    assert "comparison mode: strict_bitwise" in text

    _write_diagnostic(
        diagnostics,
        yidu_applied=np.asarray([True], dtype=np.bool_),
    )
    assert main([*common, "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["comparison_mode"] == "strict_bitwise"
    assert payload["numeric_summary"]["prediction_rows"] == 1
    assert payload["issues"][0]["kind"] == "observer_applied"


def test_cli_explicit_numeric_mode_and_incomplete_limits(
    tmp_path: Path, capsys
) -> None:
    baseline, observer, diagnostics = _valid_roots(tmp_path)
    changed = _prediction_payload()
    shifted = changed[0][0][1].copy()
    shifted += np.float32(0.001)
    changed[0][0] = ("chair", shifted, 0.7501)
    _write_pickle(observer, changed)
    common = [
        "--baseline-root",
        str(baseline),
        "--observer-root",
        str(observer),
        "--diagnostics-root",
        str(diagnostics),
        "--expected-stage",
        "A1",
    ]

    assert (
        main(
            [
                *common,
                "--max-corner-abs",
                "0.01",
                "--max-score-abs",
                "0.01",
                "--max-matched-iou-loss",
                "0.01",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparison_mode"] == "explicit_numeric_envelope"
    assert payload["numeric_envelope"] == {
        "max_corner_abs": 0.01,
        "max_matched_iou_loss": 0.01,
        "max_score_abs": 0.01,
    }
    assert payload["numeric_summary"]["changed_prediction_rows"] == 1

    assert main([*common, "--max-corner-abs", "0.01"]) == 2
    assert "requires" in capsys.readouterr().err
