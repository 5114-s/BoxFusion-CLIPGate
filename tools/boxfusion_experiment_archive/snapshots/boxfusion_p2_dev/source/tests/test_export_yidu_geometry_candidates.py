import json
import pickle

import numpy as np
import pytest

from boxfusion.yidu_ablation import (
    YIDU_SCHEMA,
    YIDU_STAGE_MODULE_MATRIX,
    YIDU_STAGE_TO_PROFILE,
)
from boxfusion.yidu_local_observer import (
    YIDU_GATE_FEATURE_DIM,
    YIDU_GATE_FEATURE_NAMES,
    YIDU_LOCAL_OBSERVER_SCHEMA,
)
from tools.export_yidu_geometry_candidates import (
    OUTPUT_SUFFIX,
    export_directory,
    export_scene,
    main,
)
from tools.report_trifusion_oracles import (
    GEOMETRY_CANDIDATE_SCHEMA,
    load_geometry_candidates,
)


_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def _corners(center, extent):
    return (
        np.asarray(center, dtype=np.float32)[None, :]
        + _SIGNS * (0.5 * np.asarray(extent, dtype=np.float32))[None, :]
    )


def _write_predictions(path, corners):
    detections = [
        (0, np.asarray(value, dtype=np.float32), 0.9 - 0.01 * index)
        for index, value in enumerate(corners)
    ]
    with path.open("wb") as handle:
        pickle.dump([detections], handle)


def _diagnostic_payload(
    scene_id,
    stage,
    original,
    selected,
    *,
    attempted=None,
    valid=None,
    sources=None,
    gate_evaluated=None,
    gate_accepted=None,
):
    rows = len(original)

    def booleans(value, default):
        if value is None:
            value = [default] * rows
        return np.asarray(value, dtype=np.bool_)

    if sources is None:
        sources = ["raw_mask"] * rows
    gate_features = np.arange(
        rows * YIDU_GATE_FEATURE_DIM, dtype=np.float32
    ).reshape(rows, YIDU_GATE_FEATURE_DIM)
    return {
        "scene_id": np.asarray(scene_id),
        "yidu_diagnostics_schema": np.asarray(
            YIDU_LOCAL_OBSERVER_SCHEMA
        ),
        "yidu_ablation_schema": np.asarray(YIDU_SCHEMA),
        "yidu_stage": np.asarray(stage),
        "yidu_profile": np.asarray(YIDU_STAGE_TO_PROFILE[stage]),
        "yidu_enabled": np.asarray(True, dtype=np.bool_),
        "yidu_mutation_enabled": np.asarray(False, dtype=np.bool_),
        "yidu_applied_count": np.asarray(0, dtype=np.int64),
        "yidu_modules_json": np.asarray(
            json.dumps(
                dict(YIDU_STAGE_MODULE_MATRIX[stage]),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "yidu_result_indices": np.arange(rows, dtype=np.int64),
        "yidu_stable_ids": np.arange(100, 100 + rows, dtype=np.int64),
        "yidu_attempted": booleans(attempted, True),
        "yidu_valid": booleans(valid, True),
        "yidu_applied": np.zeros(rows, dtype=np.bool_),
        "yidu_selected_source": np.asarray(sources, dtype="<U32"),
        "yidu_original_corners": np.asarray(original, dtype=np.float32),
        "yidu_selected_candidate_corners": np.asarray(
            selected, dtype=np.float32
        ),
        "yidu_gate_feature_names": np.asarray(
            YIDU_GATE_FEATURE_NAMES, dtype="<U96"
        ),
        "yidu_gate_features": gate_features,
        "yidu_gate_evaluated": booleans(gate_evaluated, False),
        "yidu_gate_accepted": booleans(gate_accepted, False),
    }


def _write_diagnostics(path, payload):
    np.savez_compressed(path, **payload)


def _scene_artifacts(tmp_path, *, stage="A5", rows=1, **updates):
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene_id = "scene0000_00"
    original = np.stack(
        [
            _corners([3.0 * index, 0.0, 0.0], [1.0, 1.2, 1.4])
            for index in range(rows)
        ]
    )
    selected = original.copy()
    selected[:, :, 0] += 0.1
    prediction_path = tmp_path / f"{scene_id}_boxes.pkl"
    diagnostic_path = tmp_path / f"{scene_id}_tracks.npz"
    _write_predictions(prediction_path, original)
    payload = _diagnostic_payload(
        scene_id, stage, original, selected, **updates
    )
    _write_diagnostics(diagnostic_path, payload)
    return scene_id, prediction_path, diagnostic_path, payload


@pytest.mark.parametrize("stage", ["A1", "A2", "A3", "A4", "A5"])
def test_a1_to_a5_export_one_verified_selected_candidate(tmp_path, stage):
    scene, prediction_path, diagnostic_path, payload = _scene_artifacts(
        tmp_path, stage=stage
    )
    result = export_scene(
        scene_id=scene,
        diagnostic_path=diagnostic_path,
        prediction_path=prediction_path,
        expected_stage=stage,
    )
    assert str(result["schema"].item()) == GEOMETRY_CANDIDATE_SCHEMA
    assert result["prediction_indices"].tolist() == [0]
    assert result["candidate_offsets"].tolist() == [0, 1]
    assert result["candidate_sources"].tolist() == ["raw_mask"]
    assert result["candidate_valid"].tolist() == [True]
    assert result["candidate_verified"].tolist() == [True]
    assert result["candidate_gate_evaluated"].tolist() == [False]
    assert result["candidate_gate_accepted"].tolist() == [False]
    assert result["candidate_features"].shape == (
        1,
        YIDU_GATE_FEATURE_DIM,
    )
    np.testing.assert_array_equal(
        result["candidate_features"][0],
        payload["yidu_gate_features"][0],
    )
    assert tuple(result["candidate_feature_names"].tolist()) == (
        YIDU_GATE_FEATURE_NAMES
    )


def test_single_scene_can_validate_and_derive_its_canonical_stage(tmp_path):
    scene, prediction_path, diagnostic_path, _ = _scene_artifacts(
        tmp_path, stage="A3"
    )
    result = export_scene(
        scene_id=scene,
        diagnostic_path=diagnostic_path,
        prediction_path=prediction_path,
    )
    assert str(result["yidu_stage"].item()) == "A3"


def test_each_diagnostic_row_has_zero_or_one_candidate(tmp_path):
    scene_id = "scene0000_00"
    original = np.stack(
        [
            _corners([3.0 * index, 0.0, 0.0], [1.0, 1.0, 1.0])
            for index in range(5)
        ]
    )
    selected = original.copy()
    selected[:, :, 0] += 0.2
    selected[1] = original[1]  # identity
    selected[4, :, 2] = selected[4, 0, 2]  # degenerate
    prediction_path = tmp_path / f"{scene_id}_boxes.pkl"
    diagnostic_path = tmp_path / f"{scene_id}_tracks.npz"
    _write_predictions(prediction_path, original)
    payload = _diagnostic_payload(
        scene_id,
        "A5",
        original,
        selected,
        attempted=[True, True, True, False, True],
        valid=[True, True, True, True, True],
        sources=[
            "occupancy",
            "raw_mask",
            "original",
            "superpoint",
            "superpoint",
        ],
    )
    _write_diagnostics(diagnostic_path, payload)

    result = export_scene(
        scene_id=scene_id,
        diagnostic_path=diagnostic_path,
        prediction_path=prediction_path,
        expected_stage="A5",
    )
    # Keep every diagnostic-to-prediction row in the ragged contract, but
    # flatten only the one row which satisfies every export condition.
    assert result["prediction_indices"].tolist() == [0, 1, 2, 3, 4]
    assert result["candidate_offsets"].tolist() == [0, 1, 1, 1, 1, 1]
    assert len(result["candidate_corners"]) == 1
    assert result["candidate_sources"].tolist() == ["occupancy"]
    assert result["candidate_stable_ids"].tolist() == [100]


def test_a6_keeps_all_valid_candidates_and_marks_only_gate_accept_verified(
    tmp_path,
):
    scene, prediction_path, diagnostic_path, _ = _scene_artifacts(
        tmp_path,
        stage="A6",
        rows=3,
        gate_evaluated=[True, True, False],
        gate_accepted=[True, False, False],
        sources=["occupancy", "superpoint", "raw_mask"],
    )
    result = export_scene(
        scene_id=scene,
        diagnostic_path=diagnostic_path,
        prediction_path=prediction_path,
        expected_stage="A6",
    )
    assert result["candidate_offsets"].tolist() == [0, 1, 2, 3]
    assert result["candidate_sources"].tolist() == [
        "occupancy",
        "superpoint",
        "raw_mask",
    ]
    assert result["candidate_gate_evaluated"].tolist() == [
        True,
        True,
        False,
    ]
    assert result["candidate_gate_accepted"].tolist() == [
        True,
        False,
        False,
    ]
    assert result["candidate_verified"].tolist() == [True, False, False]


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "yidu_mutation_enabled",
            np.asarray(True, dtype=np.bool_),
            "not observer-only",
        ),
        (
            "yidu_applied_count",
            np.asarray(1, dtype=np.int64),
            "applied count must be zero",
        ),
        (
            "yidu_applied",
            np.asarray([True], dtype=np.bool_),
            "applied rows must be empty",
        ),
    ],
)
def test_export_rejects_any_mutation_or_application(
    tmp_path, field, value, match
):
    scene, prediction_path, diagnostic_path, payload = _scene_artifacts(
        tmp_path
    )
    payload[field] = value
    _write_diagnostics(diagnostic_path, payload)
    with pytest.raises(ValueError, match=match):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A5",
        )


def test_export_rejects_wrong_stage_profile_or_module_matrix(tmp_path):
    scene, prediction_path, diagnostic_path, payload = _scene_artifacts(
        tmp_path
    )
    with pytest.raises(ValueError, match="does not match expected A4"):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A4",
        )

    payload["yidu_profile"] = np.asarray(
        YIDU_STAGE_TO_PROFILE["A4"]
    )
    _write_diagnostics(diagnostic_path, payload)
    with pytest.raises(ValueError, match="not canonical"):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A5",
        )

    payload["yidu_profile"] = np.asarray(
        YIDU_STAGE_TO_PROFILE["A5"]
    )
    modules = dict(YIDU_STAGE_MODULE_MATRIX["A5"])
    modules["raw_fused_query"] = False
    payload["yidu_modules_json"] = np.asarray(json.dumps(modules))
    _write_diagnostics(diagnostic_path, payload)
    with pytest.raises(ValueError, match="module matrix"):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A5",
        )


def test_export_rejects_gate_execution_before_a6(tmp_path):
    scene, prediction_path, diagnostic_path, payload = _scene_artifacts(
        tmp_path, stage="A5"
    )
    payload["yidu_gate_evaluated"][:] = True
    payload["yidu_gate_accepted"][:] = True
    _write_diagnostics(diagnostic_path, payload)
    with pytest.raises(ValueError, match="invalid before A6"):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A5",
        )


def test_export_rejects_original_corners_disagreeing_with_pickle(tmp_path):
    scene, prediction_path, diagnostic_path, payload = _scene_artifacts(
        tmp_path
    )
    payload["yidu_original_corners"][0, :, 0] += 0.01
    _write_diagnostics(diagnostic_path, payload)
    with pytest.raises(ValueError, match="disagree"):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A5",
        )


def test_export_rejects_nonfinite_or_non_91d_features(tmp_path):
    scene, prediction_path, diagnostic_path, payload = _scene_artifacts(
        tmp_path
    )
    payload["yidu_gate_features"][0, 0] = np.nan
    _write_diagnostics(diagnostic_path, payload)
    with pytest.raises(ValueError, match="finite shape"):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A5",
        )

    payload["yidu_gate_features"] = np.zeros(
        (1, YIDU_GATE_FEATURE_DIM), dtype=np.float32
    )
    payload["yidu_gate_feature_names"] = np.asarray(
        YIDU_GATE_FEATURE_NAMES[:-1]
    )
    _write_diagnostics(diagnostic_path, payload)
    with pytest.raises(ValueError, match="feature names"):
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic_path,
            prediction_path=prediction_path,
            expected_stage="A5",
        )


def test_directory_cli_writes_valid_npz_atomically_and_refuses_overwrite(
    tmp_path, capsys
):
    diagnostics_root = tmp_path / "diagnostics"
    prediction_root = tmp_path / "predictions"
    output_root = tmp_path / "geometry"
    diagnostics_root.mkdir()
    prediction_root.mkdir()
    scene, prediction_path, diagnostic_path, _ = _scene_artifacts(
        tmp_path / "source"
    )
    # Copy the trusted synthetic artifacts to the roots used by the
    # directory API.
    diagnostics_root.joinpath(diagnostic_path.name).write_bytes(
        diagnostic_path.read_bytes()
    )
    prediction_root.joinpath(prediction_path.name).write_bytes(
        prediction_path.read_bytes()
    )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")

    assert (
        main(
            [
                "--diagnostics-root",
                str(diagnostics_root),
                "--prediction-root",
                str(prediction_root),
                "--scene-list",
                str(scene_list),
                "--output-root",
                str(output_root),
                "--stage",
                "A5",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["stage"] == "A5"
    assert summary["scenes"] == 1
    assert summary["candidates"] == 1
    destination = output_root / f"{scene}{OUTPUT_SUFFIX}"
    checked = load_geometry_candidates(
        destination, expected_scene_id=scene
    )
    assert checked.candidate_sources == ("raw_mask",)
    assert not list(output_root.glob(f".{destination.name}.*"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_directory(
            diagnostics_root=diagnostics_root,
            prediction_root=prediction_root,
            scene_list=scene_list,
            output_root=output_root,
            expected_stage="A5",
        )
