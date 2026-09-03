import json
import pickle

import numpy as np
import pytest

from tools.export_yidu_geometry_candidates import (
    OUTPUT_FORMAT_VERSION,
    OUTPUT_SUFFIX,
)
from tools.report_trifusion_oracles import (
    CORNER_FRAME,
    GEOMETRY_CANDIDATE_SCHEMA,
)
from tools.report_yidu_candidate_deltas import (
    DELTA_EPSILON,
    REPORT_SCHEMA,
    main,
    report_yidu_candidate_deltas,
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
        + 0.5
        * _SIGNS
        * np.asarray(extent, dtype=np.float32)[None, :]
    )


def _write_predictions(path, corners):
    detections = [
        (0, np.asarray(value, dtype=np.float32), 0.9 - 0.1 * index)
        for index, value in enumerate(corners)
    ]
    with path.open("wb") as handle:
        pickle.dump([detections], handle)


def _write_scan_metadata(scan_root, scene):
    scene_root = scan_root / scene
    scene_root.mkdir(parents=True)
    identity = " ".join(
        str(float(value)) for value in np.eye(4).reshape(-1)
    )
    (scene_root / f"{scene}.txt").write_text(
        f"axisAlignment = {identity}\n", encoding="utf-8"
    )


def _write_geometry(
    path,
    *,
    scene,
    stage,
    originals,
    candidates,
    offsets,
    sources,
    valid,
    verified,
    gate_evaluated=None,
    gate_accepted=None,
):
    count = len(candidates)
    if gate_evaluated is None:
        gate_evaluated = [False] * count
    if gate_accepted is None:
        gate_accepted = [False] * count
    np.savez_compressed(
        path,
        schema=np.asarray(GEOMETRY_CANDIDATE_SCHEMA),
        format_version=np.asarray(
            OUTPUT_FORMAT_VERSION, dtype=np.int64
        ),
        scene_id=np.asarray(scene),
        corner_frame=np.asarray(CORNER_FRAME),
        prediction_indices=np.arange(
            len(originals), dtype=np.int64
        ),
        original_corners=np.asarray(originals, dtype=np.float32),
        candidate_offsets=np.asarray(offsets, dtype=np.int64),
        candidate_corners=np.asarray(candidates, dtype=np.float32).reshape(
            count, 8, 3
        ),
        candidate_ids=np.asarray(
            [f"{scene}:candidate:{index}" for index in range(count)]
        ),
        candidate_sources=np.asarray(sources),
        candidate_valid=np.asarray(valid, dtype=np.bool_),
        candidate_verified=np.asarray(verified, dtype=np.bool_),
        candidate_gate_evaluated=np.asarray(
            gate_evaluated, dtype=np.bool_
        ),
        candidate_gate_accepted=np.asarray(
            gate_accepted, dtype=np.bool_
        ),
        yidu_stage=np.asarray(stage),
        observer_only=np.asarray(True, dtype=np.bool_),
        uses_ground_truth=np.asarray(False, dtype=np.bool_),
    )


def _fixture(tmp_path, *, stage="A6"):
    geometry_root = tmp_path / "geometry"
    prediction_root = tmp_path / "predictions"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    for root in (
        geometry_root,
        prediction_root,
        gt_root,
        scan_root,
    ):
        root.mkdir()
    scenes = ("scene0000_00", "scene0001_00")
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")

    scene0_original = np.stack(
        (
            _corners([0, 0, 0], [4, 4, 4]),
            _corners([5, 0, 0], [2, 2, 2]),
        )
    )
    scene0_candidates = np.stack(
        (
            _corners([0, 0, 0], [2, 2, 2]),
            _corners([5, 0, 0], [3, 3, 3]),
        )
    )
    _write_predictions(
        prediction_root / f"{scenes[0]}_boxes.pkl", scene0_original
    )
    _write_geometry(
        geometry_root / f"{scenes[0]}{OUTPUT_SUFFIX}",
        scene=scenes[0],
        stage=stage,
        originals=scene0_original,
        candidates=scene0_candidates,
        offsets=[0, 1, 2],
        sources=["occupancy", "superpoint"],
        valid=[True, True],
        verified=[True, False] if stage == "A6" else [True, True],
        gate_evaluated=(
            [True, True] if stage == "A6" else [False, False]
        ),
        gate_accepted=(
            [True, False] if stage == "A6" else [False, False]
        ),
    )
    np.save(
        gt_root / f"{scenes[0]}_bbox.npy",
        np.asarray(
            [
                [0, 0, 0, 2, 2, 2, 1],
                [5, 0, 0, 2, 2, 2, 1],
            ],
            dtype=np.float32,
        ),
    )
    _write_scan_metadata(scan_root, scenes[0])

    scene1_original = np.stack(
        (_corners([0, 0, 0], [2, 2, 2]),)
    )
    scene1_candidates = np.stack(
        (_corners([0.1, 0, 0], [2, 2, 2]),)
    )
    _write_predictions(
        prediction_root / f"{scenes[1]}_boxes.pkl", scene1_original
    )
    _write_geometry(
        geometry_root / f"{scenes[1]}{OUTPUT_SUFFIX}",
        scene=scenes[1],
        stage=stage,
        originals=scene1_original,
        candidates=scene1_candidates,
        offsets=[0, 1],
        sources=["raw_mask"],
        valid=[True],
        verified=[True],
        gate_evaluated=[stage == "A6"],
        gate_accepted=[stage == "A6"],
    )
    np.save(
        gt_root / f"{scenes[1]}_bbox.npy",
        np.empty((0, 7), dtype=np.float32),
    )
    _write_scan_metadata(scan_root, scenes[1])
    return {
        "geometry_root": geometry_root,
        "prediction_root": prediction_root,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "scene_list": scene_list,
        "scenes": scenes,
    }


def _run(paths, output=None):
    return report_yidu_candidate_deltas(
        geometry_root=paths["geometry_root"],
        prediction_root=paths["prediction_root"],
        scene_list=paths["scene_list"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
        output=output,
    )


def test_reports_all_valid_verified_source_scene_and_empty_gt(tmp_path):
    paths = _fixture(tmp_path)
    report = _run(paths)
    assert report["schema"] == REPORT_SCHEMA
    assert report["stage"] == "A6"
    assert report["verified_only_semantics"] == "a6_gate_accepted"
    assert report["runtime_artifacts_mutated"] is False
    assert report["delta_epsilon"] == DELTA_EPSILON

    all_valid = report["all_valid"]
    assert all_valid["candidates"] == 3
    assert all_valid["improved"] == 1
    assert all_valid["harmed"] == 1
    assert all_valid["identity"] == 1
    assert all_valid["cross25_up"] == 1
    assert all_valid["cross50_up"] == 1
    assert all_valid["covered_prediction_rows"] == 3
    assert all_valid["geometry_prediction_row_coverage"] == 1.0

    verified = report["verified_only"]
    assert verified["candidates"] == 2
    assert verified["improved"] == 1
    assert verified["harmed"] == 0
    assert verified["identity"] == 1
    assert verified["covered_prediction_rows"] == 2
    assert verified["all_prediction_coverage"] == pytest.approx(2 / 3)

    assert set(report["by_source"]) == {
        "occupancy",
        "raw_mask",
        "superpoint",
    }
    assert (
        report["by_source"]["superpoint"]["verified_only"]["candidates"]
        == 0
    )
    no_gt = report["by_scene"][paths["scenes"][1]]
    assert no_gt["ground_truth"] == 0
    assert no_gt["all_valid"]["identity"] == 1
    assert no_gt["all_valid"]["original_iou"]["mean"] == 0.0
    assert no_gt["all_valid"]["candidate_iou"]["mean"] == 0.0

    expected = np.asarray([0.875, (8 / 27) - 1.0, 0.0])
    assert all_valid["delta"]["q10"] == pytest.approx(
        np.quantile(expected, 0.10)
    )
    assert all_valid["delta"]["q50"] == pytest.approx(
        np.quantile(expected, 0.50)
    )
    assert all_valid["delta"]["q90"] == pytest.approx(
        np.quantile(expected, 0.90)
    )


def test_candidate_cannot_jump_to_neighbour_ground_truth(tmp_path):
    paths = _fixture(tmp_path)
    scene = paths["scenes"][0]
    prediction = np.stack(
        (_corners([0, 0, 0], [2, 2, 2]),)
    )
    jumped = np.stack(
        (_corners([5, 0, 0], [2, 2, 2]),)
    )
    _write_predictions(
        paths["prediction_root"] / f"{scene}_boxes.pkl", prediction
    )
    _write_geometry(
        paths["geometry_root"] / f"{scene}{OUTPUT_SUFFIX}",
        scene=scene,
        stage="A6",
        originals=prediction,
        candidates=jumped,
        offsets=[0, 1],
        sources=["raw_mask"],
        valid=[True],
        verified=[True],
        gate_evaluated=[True],
        gate_accepted=[True],
    )
    report = _run(paths)
    scene_report = report["by_scene"][scene]["all_valid"]
    assert scene_report["original_iou"]["mean"] == 1.0
    assert scene_report["candidate_iou"]["mean"] == 0.0
    assert scene_report["delta"]["mean"] == -1.0
    assert scene_report["harmed"] == 1


def test_invalid_candidates_are_excluded(tmp_path):
    paths = _fixture(tmp_path)
    scene = paths["scenes"][1]
    geometry_path = paths["geometry_root"] / f"{scene}{OUTPUT_SUFFIX}"
    with np.load(geometry_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["candidate_valid"] = np.asarray([False], dtype=np.bool_)
    arrays["candidate_verified"] = np.asarray([False], dtype=np.bool_)
    arrays["candidate_gate_accepted"] = np.asarray(
        [False], dtype=np.bool_
    )
    np.savez_compressed(geometry_path, **arrays)
    report = _run(paths)
    assert report["all_valid"]["candidates"] == 2
    assert report["by_scene"][scene]["all_valid"]["candidates"] == 0
    assert (
        report["by_scene"][scene]["all_valid"]["delta"]["q50"] is None
    )


def test_original_corners_must_match_predictions_exactly(tmp_path):
    paths = _fixture(tmp_path)
    scene = paths["scenes"][0]
    geometry_path = paths["geometry_root"] / f"{scene}{OUTPUT_SUFFIX}"
    with np.load(geometry_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["original_corners"] = arrays["original_corners"].copy()
    arrays["original_corners"][0, 0, 0] += 1.0e-5
    np.savez_compressed(geometry_path, **arrays)
    with pytest.raises(ValueError, match="disagree exactly"):
        _run(paths)


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "uses_ground_truth",
            np.asarray(True, dtype=np.bool_),
            "must not use ground truth",
        ),
        (
            "observer_only",
            np.asarray(False, dtype=np.bool_),
            "not observer-only",
        ),
        (
            "candidate_gate_accepted",
            np.asarray([False, False], dtype=np.bool_),
            "must exactly equal",
        ),
    ],
)
def test_export_provenance_fails_closed(tmp_path, field, value, match):
    paths = _fixture(tmp_path)
    scene = paths["scenes"][0]
    geometry_path = paths["geometry_root"] / f"{scene}{OUTPUT_SUFFIX}"
    with np.load(geometry_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[field] = value
    np.savez_compressed(geometry_path, **arrays)
    with pytest.raises(ValueError, match=match):
        _run(paths)


def test_mixed_stage_root_is_rejected(tmp_path):
    paths = _fixture(tmp_path)
    scene = paths["scenes"][1]
    geometry_path = paths["geometry_root"] / f"{scene}{OUTPUT_SUFFIX}"
    with np.load(geometry_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["yidu_stage"] = np.asarray("A5")
    arrays["candidate_verified"] = np.asarray([True], dtype=np.bool_)
    arrays["candidate_gate_evaluated"] = np.asarray(
        [False], dtype=np.bool_
    )
    arrays["candidate_gate_accepted"] = np.asarray(
        [False], dtype=np.bool_
    )
    np.savez_compressed(geometry_path, **arrays)
    with pytest.raises(ValueError, match="mixes YiDu stages"):
        _run(paths)


def test_output_is_atomic_external_and_inputs_remain_unchanged(
    tmp_path, capsys
):
    paths = _fixture(tmp_path)
    protected = [
        paths["scene_list"],
        *paths["geometry_root"].glob("*"),
        *paths["prediction_root"].glob("*"),
        *paths["gt_root"].glob("*"),
    ]
    before = {path: path.read_bytes() for path in protected}
    output = tmp_path / "reports" / "deltas.json"
    report = _run(paths, output=output)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert {path: path.read_bytes() for path in protected} == before
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(paths, output=output)
    with pytest.raises(ValueError, match="outside every input"):
        _run(paths, output=paths["geometry_root"] / "report.json")

    cli_output = tmp_path / "reports" / "cli.json"
    assert main(
        [
            "--geometry-root",
            str(paths["geometry_root"]),
            "--prediction-root",
            str(paths["prediction_root"]),
            "--scene-list",
            str(paths["scene_list"]),
            "--gt-root",
            str(paths["gt_root"]),
            "--scan-root",
            str(paths["scan_root"]),
            "--output",
            str(cli_output),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == REPORT_SCHEMA
    assert json.loads(cli_output.read_text(encoding="utf-8")) == printed
