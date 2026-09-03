from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.compare_prediction_roots import (
    PredictionComparisonError,
    compare_prediction_roots,
    main,
)


def _write(root: Path, scene: str, rows) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([rows], handle, protocol=pickle.HIGHEST_PROTOCOL)


def _row(class_id: int, x: float, score: float = 1.0):
    corners = np.zeros((8, 3), dtype=np.float32)
    corners[:, 0] = x
    return (class_id, corners, score)


def test_three_root_comparison_reports_pairwise_meter_and_mm_statistics(tmp_path):
    scenes = ("scene0001_00", "scene0002_00")
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    roots = [tmp_path / f"root{index}" for index in range(3)]

    offsets = ((0.0, 0.0), (0.001, 0.002), (0.003, 0.004))
    for root, root_offsets in zip(roots, offsets):
        for scene, offset in zip(scenes, root_offsets):
            _write(root, scene, [_row(0, offset, 0.75)])

    result = compare_prediction_roots(roots, scene_list)

    assert result["validation"]["passed"] is True
    assert result["validation"]["total_box_count"] == 2
    assert [(row["root_a"], row["root_b"]) for row in result["pairwise"]] == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    first = result["pairwise"][0]
    meters = first["per_box_max_corner_euclidean_error"]["meters"]
    millimeters = first["per_box_max_corner_euclidean_error"]["millimeters"]
    assert meters["p50"] == pytest.approx(0.0015)
    assert meters["p95"] == pytest.approx(0.00195)
    assert meters["max"] == pytest.approx(0.002)
    assert millimeters == pytest.approx({"p50": 1.5, "p95": 1.95, "max": 2.0})
    assert first["worst_corner"]["scene"] == scenes[1]
    assert first["worst_corner"]["box_row"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("row_count", "row count mismatch"),
        ("class", "class order mismatch"),
        ("score", "score order/value mismatch"),
    ],
)
def test_strict_contract_rejects_row_class_and_score_mismatch(
    tmp_path, mutation, message
):
    scene = "scene0003_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    root0, root1 = tmp_path / "root0", tmp_path / "root1"
    _write(root0, scene, [_row(2, 0.0, 0.5)])
    candidate = [_row(2, 0.0, 0.5)]
    if mutation == "row_count":
        candidate.append(_row(2, 0.0, 0.5))
    elif mutation == "class":
        candidate[0] = _row(3, 0.0, 0.5)
    else:
        candidate[0] = _row(2, 0.0, 0.5000001)
    _write(root1, scene, candidate)

    with pytest.raises(PredictionComparisonError, match=message):
        compare_prediction_roots([root0, root1], scene_list)


def test_cli_outputs_json_and_exact_scene_set_rejects_extra_prediction(
    tmp_path, capsys
):
    scene = "scene0004_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    root0, root1 = tmp_path / "root0", tmp_path / "root1"
    _write(root0, scene, [])
    _write(root1, scene, [])

    assert main(
        [
            "--root",
            str(root0),
            "--root",
            str(root1),
            "--scene-list",
            str(scene_list),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pairwise"][0]["box_count"] == 0
    assert payload["pairwise"][0]["worst_corner"] is None

    _write(root1, "scene9999_00", [])
    with pytest.raises(PredictionComparisonError, match="extra=scene9999_00"):
        compare_prediction_roots([root0, root1], scene_list)
