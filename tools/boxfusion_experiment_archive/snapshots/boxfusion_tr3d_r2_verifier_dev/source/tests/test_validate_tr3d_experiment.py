from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "validate_tr3d_experiment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_tr3d_experiment", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(scene: str, *, prefix: bool = False):
    row = {
        "lidar_points": {
            "num_pts_feats": 6,
            "lidar_path": f"full/{scene}.bin",
        },
        "instances": [{"bbox_label_3d": 0}],
        "axis_align_matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "coordinate_frame": "world_unaligned",
        "box_coordinate_frame": "scannet_axis_aligned",
    }
    if prefix:
        row["lidar_points"]["lidar_path"] = (
            f"prefixes/{scene}/{scene}__p025.bin"
        )
        row["trajectory_prefix"] = {"tag": "p025", "fraction": 0.25}
    return row


def _assets(tmp_path: Path, annotation_scenes, *, prefix=False):
    data_root = tmp_path / "data"
    split_root = data_root / "splits"
    annotation_root = data_root / "annotations"
    split_root.mkdir(parents=True)
    annotation_root.mkdir()
    train = ("scene0000_00", "scene0001_00")
    val = ("scene0700_00",)
    (split_root / "train.txt").write_text(
        "".join(f"{scene}\n" for scene in train), encoding="utf-8"
    )
    (split_root / "official_val.txt").write_text(
        "".join(f"{scene}\n" for scene in val), encoding="utf-8"
    )
    contract = {
        "schema": MODULE.CONTRACT_SCHEMA,
        "scene_list_sha256": {"train": MODULE.sha256_lines(train)},
    }
    contract_path = data_root / "DATASET_CONTRACT.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    annotation_path = annotation_root / "info.pkl"
    with annotation_path.open("wb") as handle:
        pickle.dump(
            {
                "metainfo": {
                    "classes": ("foreground",),
                    "categories": {"foreground": 0},
                },
                "data_list": [
                    _row(scene, prefix=prefix)
                    for scene in annotation_scenes
                ],
            },
            handle,
        )
    return contract_path, annotation_path, split_root / "train.txt"


def test_full_training_contract_passes(tmp_path: Path):
    contract, annotation, split = _assets(
        tmp_path, ("scene0000_00", "scene0001_00")
    )
    report = MODULE.validate_training(
        contract_path=contract,
        annotation_path=annotation,
        expected_split_path=split,
        prefix=False,
    )
    assert report["ok"]
    assert report["scene_count"] == 2
    assert report["official_val_overlap"] == 0


def test_prefix_must_be_train_subset_and_foreground(tmp_path: Path):
    contract, annotation, split = _assets(
        tmp_path, ("scene0000_00",), prefix=True
    )
    report = MODULE.validate_training(
        contract_path=contract,
        annotation_path=annotation,
        expected_split_path=split,
        prefix=True,
    )
    assert report["mode"] == "prefix_train"
    assert report["sample_count"] == 1


def test_official_val_leak_is_rejected(tmp_path: Path):
    contract, annotation, split = _assets(
        tmp_path, ("scene0700_00",), prefix=True
    )
    with pytest.raises(ValueError, match="official-val"):
        MODULE.validate_training(
            contract_path=contract,
            annotation_path=annotation,
            expected_split_path=split,
            prefix=True,
        )
