"""Train-only target, loss, split, and provenance contracts for P2."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxfusion.occupancy_topk import (  # noqa: E402
    OccupancyTopKConfig,
    P2_HEAD_SCHEMA,
    load_occupancy_topk_head,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
    P1_HEAD_SCHEMA,
    ResidualVoxelProposalHead,
)
from tools.train_p2_occupancy_topk import (  # noqa: E402
    TRAINING_SCHEMA,
    _parser,
    build_training_data,
    file_sha256,
    main,
    validate_parent_artifact_chain,
    validate_parent_provenance,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_scene_inputs(
    root: Path, scene: str, *, feature_offset: float
) -> None:
    diagnostics = root / "diagnostics"
    predictions = root / "predictions"
    ground_truth = root / "gt"
    scans = root / "scans" / scene
    for directory in (diagnostics, predictions, ground_truth, scans):
        directory.mkdir(parents=True, exist_ok=True)
    features = np.zeros((2, P1_FEATURE_DIM), dtype=np.float32)
    features[:, 0] = [feature_offset, feature_offset + 0.5]
    np.savez_compressed(
        diagnostics / f"{scene}_tracks.npz",
        scene_id=np.asarray(scene),
        p1_schema=np.asarray(P1_DIAGNOSTIC_SCHEMA),
        p1_stage=np.asarray("P1"),
        p1_profile=np.asarray("p1_residual_proposal_observer"),
        p1_enabled=np.asarray(True, dtype=bool),
        p1_observer_only=np.asarray(True, dtype=bool),
        p1_uses_ground_truth=np.asarray(False, dtype=bool),
        p1_mutation_enabled=np.asarray(False, dtype=bool),
        p1_applied_count=np.asarray(0, dtype=np.int64),
        p1_complete=np.asarray(True, dtype=bool),
        p1_class_agnostic=np.asarray(True, dtype=bool),
        p1_regression_dim=np.asarray(6, dtype=np.int64),
        p1_feature_names=np.asarray(P1_FEATURE_NAMES, dtype=np.str_),
        p1_voxel_features=features,
        p1_voxel_centers=np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        ),
        p1_voxel_offsets=np.asarray([0, 2], dtype=np.int64),
    )
    with (predictions / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[]], handle)
    np.save(
        ground_truth / f"{scene}_bbox.npy",
        np.asarray([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float32),
    )
    scans.joinpath(f"{scene}.txt").write_text(
        "axisAlignment = "
        + " ".join(str(value) for value in np.eye(4).reshape(-1))
        + "\n",
        encoding="utf-8",
    )


def _parent_checkpoints(
    root: Path,
    *,
    train_list: Path,
    forbidden_list: Path,
    scenes: list[str],
) -> tuple[Path, Path]:
    b6 = root / "b6.npz"
    b6.write_bytes(b"frozen-b6")
    head = ResidualVoxelProposalHead(
        input_dim=P1_FEATURE_DIM, hidden_dim=8
    )
    p1 = root / "p1.pt"
    torch.save(
        {
            "schema": P1_HEAD_SCHEMA,
            "feature_names": list(P1_FEATURE_NAMES),
            "model_config": {
                "input_dim": P1_FEATURE_DIM,
                "hidden_dim": 8,
                "regression_dim": 6,
            },
            "state_dict": head.state_dict(),
            "provenance": {
                "train_scene_ids": scenes,
                "forbidden_overlap": [],
                "train_scene_list_sha256": _sha(train_list),
                "forbidden_scene_list_sha256": _sha(forbidden_list),
                "b6_checkpoint_sha256": _sha(b6),
                "scene_summaries": [
                    {
                        "scene_id": scene,
                        "diagnostic_sha256": _sha(
                            root
                            / "diagnostics"
                            / f"{scene}_tracks.npz"
                        ),
                        "prediction_sha256": _sha(
                            root
                            / "predictions"
                            / f"{scene}_boxes.pkl"
                        ),
                        "ground_truth_sha256": _sha(
                            root / "gt" / f"{scene}_bbox.npy"
                        ),
                    }
                    for scene in scenes
                ],
            },
        },
        p1,
    )
    return p1, b6


def _fixture(tmp_path: Path):
    scenes = ["scene0001_00", "scene0002_00"]
    for index, scene in enumerate(scenes):
        _write_scene_inputs(
            tmp_path, scene, feature_offset=float(index)
        )
    train_list = tmp_path / "train.txt"
    train_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    forbidden_list = tmp_path / "val.txt"
    forbidden_list.write_text("scene0700_00\n", encoding="utf-8")
    p1, b6 = _parent_checkpoints(
        tmp_path,
        train_list=train_list,
        forbidden_list=forbidden_list,
        scenes=scenes,
    )
    return scenes, train_list, forbidden_list, p1, b6


def test_training_cli_parser_has_one_required_output_option():
    parser = _parser()
    output_actions = [
        action
        for action in parser._actions
        if "--output" in action.option_strings
    ]
    assert len(output_actions) == 1
    assert output_actions[0].required is True
    assert "--output OUTPUT" in parser.format_help()


def test_training_script_defaults_to_immutable_p1_artifact_root():
    repository = Path(__file__).resolve().parents[1]
    script = (
        repository / "scripts" / "train_scannet_p2.sh"
    ).read_text(encoding="utf-8")
    assert (
        "P1_ARTIFACT_ROOT="
        '"${BOXFUSION_P1_ARTIFACT_ROOT:-'
        "/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev}"
        '"'
    ) in script
    assert (
        "DIAGNOSTICS="
        '"${BOXFUSION_P2_TRAIN_DIAGNOSTICS:-'
        "$P1_ARTIFACT_ROOT/diagnostics/p1_training/$P1_TAG}"
        '"'
    ) in script
    assert (
        "PREDICTIONS="
        '"${BOXFUSION_P2_TRAIN_PREDICTIONS:-'
        "$P1_ARTIFACT_ROOT/results/p1_training/$P1_TAG}"
        '"'
    ) in script


def test_training_rows_are_inside_residual_gt_targets_only(tmp_path):
    scenes, _, _, _, _ = _fixture(tmp_path)
    data = build_training_data(
        scenes=scenes,
        diagnostics_root=tmp_path / "diagnostics",
        prediction_root=tmp_path / "predictions",
        gt_root=tmp_path / "gt",
        scans_root=tmp_path / "scans",
        covered_iou=0.15,
        occupancy_margin=0.0,
        maximum_voxels_per_scene=0,
        negative_ratio=1.0,
        seed=7,
    )
    assert data.feature_names == P1_FEATURE_NAMES
    assert data.features.shape == (4, P1_FEATURE_DIM)
    assert data.occupancy.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert set(data.scene_ids.tolist()) == set(scenes)
    assert all(row["positive_voxels"] == 1 for row in data.scene_summaries)
    assert all(row["negative_voxels"] == 1 for row in data.scene_summaries)


def test_parent_provenance_is_bound_to_exact_p1_and_b6(tmp_path):
    scenes, train_list, forbidden, p1, b6 = _fixture(tmp_path)
    provenance = validate_parent_provenance(
        p1_checkpoint=p1,
        b6_checkpoint=b6,
        train_scene_list=train_list,
        forbidden_scene_list=forbidden,
        train_scenes=scenes,
    )
    assert provenance["p1_checkpoint_sha256"] == file_sha256(p1)
    assert provenance["b6_checkpoint_sha256"] == file_sha256(b6)
    data = build_training_data(
        scenes=scenes,
        diagnostics_root=tmp_path / "diagnostics",
        prediction_root=tmp_path / "predictions",
        gt_root=tmp_path / "gt",
        scans_root=tmp_path / "scans",
        maximum_voxels_per_scene=0,
        negative_ratio=1.0,
    )
    validate_parent_artifact_chain(
        data.scene_summaries,
        provenance["scene_artifact_hashes"],
    )
    changed = dict(data.scene_summaries[0])
    changed["prediction_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="exact P1 artifact"):
        validate_parent_artifact_chain(
            (changed, *data.scene_summaries[1:]),
            provenance["scene_artifact_hashes"],
        )

    b6.write_bytes(b"different-b6")
    with pytest.raises(ValueError, match="provenance disagrees"):
        validate_parent_provenance(
            p1_checkpoint=p1,
            b6_checkpoint=b6,
            train_scene_list=train_list,
            forbidden_scene_list=forbidden,
            train_scenes=scenes,
        )


def test_cli_trains_bce_only_scene_disjoint_checkpoint(tmp_path):
    scenes, train_list, forbidden, p1, b6 = _fixture(tmp_path)
    output = tmp_path / "p2.pt"
    summary = tmp_path / "p2_summary.json"
    assert main(
        [
            "--diagnostics-root",
            str(tmp_path / "diagnostics"),
            "--prediction-root",
            str(tmp_path / "predictions"),
            "--gt-root",
            str(tmp_path / "gt"),
            "--scans-root",
            str(tmp_path / "scans"),
            "--train-scene-list",
            str(train_list),
            "--forbidden-scene-list",
            str(forbidden),
            "--p1-checkpoint",
            str(p1),
            "--b6-checkpoint",
            str(b6),
            "--output",
            str(output),
            "--hidden-dim",
            "4",
            "--validation-fraction",
            "0.5",
            "--epochs",
            "3",
            "--batch-size",
            "2",
            "--negative-ratio",
            "1",
            "--max-voxels-per-scene",
            "0",
            "--seed",
            "11",
            "--device",
            "cpu",
            "--summary-json",
            str(summary),
        ]
    ) == 0
    try:
        payload = torch.load(
            output, map_location="cpu", weights_only=False
        )
    except TypeError:  # pragma: no cover
        payload = torch.load(output, map_location="cpu")
    assert payload["schema"] == P2_HEAD_SCHEMA
    assert payload["training_config"]["schema"] == TRAINING_SCHEMA
    assert payload["training_config"]["objective"] == "occupancy_bce_only"
    assert payload["training_config"]["loss_terms"] == {
        "occupancy_bce": 1.0
    }
    assert "regression" not in json.dumps(
        payload["training_config"], sort_keys=True
    ).lower()
    train_scenes = set(payload["metrics"]["training_scenes"])
    validation_scenes = set(payload["metrics"]["validation_scenes"])
    assert train_scenes | validation_scenes == set(scenes)
    assert not train_scenes & validation_scenes
    assert payload["provenance"]["p1_checkpoint_sha256"] == _sha(p1)
    assert payload["provenance"]["b6_checkpoint_sha256"] == _sha(b6)

    config = OccupancyTopKConfig(
        enabled=True,
        checkpoint=str(output),
        hidden_dim=4,
        device="cpu",
    ).validated()
    model, checkpoint_sha, _ = load_occupancy_topk_head(
        output,
        expected_config=config,
        expected_p1_checkpoint_sha256=_sha(p1),
        expected_b6_checkpoint_sha256=_sha(b6),
        device="cpu",
    )
    assert checkpoint_sha == _sha(output)
    assert tuple(model(torch.zeros(2, P1_FEATURE_DIM)).shape) == (2, 1)
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["objective"] == "occupancy_bce_only"
