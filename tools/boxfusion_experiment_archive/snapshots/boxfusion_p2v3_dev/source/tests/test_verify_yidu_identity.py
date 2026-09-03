from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

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
    assert "identity audit: PASS" in capsys.readouterr().out

    _write_diagnostic(
        diagnostics,
        yidu_applied=np.asarray([True], dtype=np.bool_),
    )
    assert main([*common, "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["issues"][0]["kind"] == "observer_applied"
