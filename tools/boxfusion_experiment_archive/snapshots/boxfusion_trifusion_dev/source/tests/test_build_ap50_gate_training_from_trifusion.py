import pickle

import numpy as np
import pytest

from tools.build_ap50_gate_training_from_trifusion import (
    build_gate_training_archive,
)
from tools.report_trifusion_oracles import (
    CORNER_FRAME,
    GEOMETRY_CANDIDATE_SCHEMA,
)
from tools.train_ap50_safety_gate import _load_archive


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


def _corners(extent):
    return _SIGNS * (0.5 * np.asarray(extent, dtype=np.float32))


def _synthetic(tmp_path):
    scene = "scene0000_00"
    geometry_root = tmp_path / "geometry"
    prediction_root = tmp_path / "predictions"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    for root in (geometry_root, prediction_root, gt_root, scan_root):
        root.mkdir()
    (scan_root / scene).mkdir()
    identity = " ".join(
        str(float(value)) for value in np.eye(4).reshape(-1)
    )
    (scan_root / scene / f"{scene}.txt").write_text(
        f"axisAlignment = {identity}\n", encoding="utf-8"
    )
    scene_list = tmp_path / "train.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    forbidden = tmp_path / "val.txt"
    forbidden.write_text("scene9999_00\n", encoding="utf-8")

    original = _corners([2.8, 2.8, 2.8])
    candidate = _corners([2.0, 2.0, 2.0])
    with (prediction_root / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[(0, original, 0.8)]], handle)
    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.asarray([[0, 0, 0, 2, 2, 2, 3]], dtype=np.float32),
    )
    np.savez_compressed(
        geometry_root / f"{scene}_geometry_candidates.npz",
        schema=np.asarray(GEOMETRY_CANDIDATE_SCHEMA),
        scene_id=np.asarray(scene),
        corner_frame=np.asarray(CORNER_FRAME),
        prediction_indices=np.asarray([0], dtype=np.int64),
        original_corners=original[None],
        candidate_offsets=np.asarray([0, 1], dtype=np.int64),
        candidate_corners=candidate[None],
        candidate_ids=np.asarray([f"{scene}:candidate:0"]),
        candidate_sources=np.asarray(["occupancy_msr"]),
        candidate_valid=np.asarray([True], dtype=np.bool_),
        candidate_verified=np.asarray([True], dtype=np.bool_),
        candidate_feature_names=np.asarray(["x", "y"]),
        candidate_features=np.asarray([[0.25, 0.75]], dtype=np.float32),
    )
    return {
        "scene": scene,
        "geometry_root": geometry_root,
        "prediction_root": prediction_root,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "scene_list": scene_list,
        "forbidden": forbidden,
    }


def test_builder_targets_same_original_gt_and_emits_training_schema(tmp_path):
    paths = _synthetic(tmp_path)
    output = tmp_path / "training.npz"
    summary = build_gate_training_archive(
        geometry_root=paths["geometry_root"],
        prediction_root=paths["prediction_root"],
        scene_list=paths["scene_list"],
        forbidden_scene_list=paths["forbidden"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
        output=output,
    )
    data = _load_archive(output)
    assert data.feature_names == ("x", "y")
    assert data.features.tolist() == [[0.25, 0.75]]
    assert data.scene_ids.tolist() == [paths["scene"]]
    assert data.original_iou[0] == pytest.approx(8.0 / (2.8**3))
    assert data.candidate_iou[0] == pytest.approx(1.0)
    assert summary["improved"] == 1
    assert summary["cross_iou50_up"] == 1


def test_builder_aborts_on_forbidden_scene_before_writing(tmp_path):
    paths = _synthetic(tmp_path)
    paths["forbidden"].write_text(
        paths["scene"] + "\n", encoding="utf-8"
    )
    output = tmp_path / "must_not_exist.npz"
    with pytest.raises(ValueError, match="overlaps forbidden"):
        build_gate_training_archive(
            geometry_root=paths["geometry_root"],
            prediction_root=paths["prediction_root"],
            scene_list=paths["scene_list"],
            forbidden_scene_list=paths["forbidden"],
            gt_root=paths["gt_root"],
            scan_root=paths["scan_root"],
            output=output,
        )
    assert not output.exists()
