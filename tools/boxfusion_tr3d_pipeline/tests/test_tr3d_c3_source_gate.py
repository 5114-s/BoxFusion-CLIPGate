from argparse import Namespace
from pathlib import Path

import numpy as np

from boxfusion.tr3d_c3_online_active import C3SourceGatePolicy, FEATURE_NAMES
from boxfusion.tr3d_c3_online_identity import ROUTE
from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file
from tools.build_tr3d_c3_source_gate_dataset import DATASET_SCHEMA
from tools.train_tr3d_c3_source_gate import train


def _write_list(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_scene_group_oof_policy_is_loadable_when_gate_passes(tmp_path: Path) -> None:
    train_scenes = [f"scene{index:04d}_00" for index in range(25)]
    validation_scenes = [f"scene{index:04d}_01" for index in range(100, 200)]
    train_list = tmp_path / "train.txt"
    validation_list = tmp_path / "val.txt"
    _write_list(train_list, train_scenes)
    _write_list(validation_list, validation_scenes)

    scene_ids = np.repeat(np.asarray(train_scenes), 4)
    targets = np.tile(np.asarray([0.72, 0.58, 0.08, 0.04], dtype=np.float32), 25)
    features = np.zeros((len(targets), len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = np.where(targets >= 0.25, 1.0, 5.0)
    features[:, 1] = targets
    features[:, 2:] = np.where(targets[:, None] >= 0.25, 8.0, 1.0)
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        features=features,
        target_iou=targets,
        scene_ids=scene_ids,
        proposal_ids=np.arange(len(targets), dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES),
        schema=np.asarray(DATASET_SCHEMA),
        route=np.asarray(ROUTE),
        train_scene_list_sha256=np.asarray(sha256_file(train_list)),
        forbidden_validation_scene_list_sha256=np.asarray(sha256_file(validation_list)),
    )
    dataset.chmod(0o444)
    policy_path = tmp_path / "policy.json"
    result = train(
        Namespace(
            dataset=dataset,
            train_scene_list=train_list,
            forbidden_validation_scene_list=validation_list,
            output=policy_path,
            folds=5,
            iterations=600,
            learning_rate=0.08,
            l2=0.002,
            min_selected=20,
            min_precision_iou25=0.60,
            min_precision_iou50=0.25,
            min_positive_folds=4,
            output_score_min=0.02,
            output_score_max=0.39,
            max_candidates_per_scene=8,
            max_anchor_iou=0.15,
        )
    )
    assert result["activation_authorized"] is True
    policy = C3SourceGatePolicy.load(policy_path)
    assert policy.training_data_sha256 == sha256_file(dataset)
    assert policy.max_anchor_iou == 0.15
