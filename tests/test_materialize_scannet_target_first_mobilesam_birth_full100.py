import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.materialize_scannet_target_first_mobilesam_birth_full100 import (
    APPENDED_CLASS_ID,
    APPENDED_SCORE,
    MANIFEST_NAME,
    SCHEMA,
    BirthMaterializationError,
    _hash_array,
    _parser,
    materialize_scannet_target_first_mobilesam_birth_full100,
)


SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float32,
)


def _corners(center, extent=1.0):
    extent_xyz = np.broadcast_to(np.asarray(extent, dtype=np.float32), (3,))
    return SIGNS * (extent_xyz / 2.0) + np.asarray(center, dtype=np.float32)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(track_id=1, center=(0.0, 0.0, 0.0), **updates):
    frame0 = track_id * 100
    row0 = track_id * 10
    record = {
        "track_id": track_id,
        "confirmation_frame_id": frame0 + 50,
        "evidence_frame_ids": [frame0, frame0 + 25, frame0 + 50],
        "evidence_source_rows": [row0, row0 + 1, row0 + 2],
        "target_group": "chair",
        "evidence_scores": [0.80, 0.80, 0.80],
        "raw_mean_score": 0.80,
        "median_pairwise_mask_aabb_iou": 0.60,
        "max_pairwise_mask_center_distance_m": 0.10,
        "first_last_frame_span": 50,
        "max_camera_baseline_m": 0.20,
        "max_view_ray_span_deg": 10.0,
        "supported_voxel_count": 40,
        "view_supported_voxel_counts": [12, 13, 14],
        "fused_obb": {
            "extent_xyz": [1.0, 1.0, 1.0],
            "corners_world": _corners(center).tolist(),
        },
        "fused_center_to_raw_medoid_m": 0.10,
    }
    record.update(updates)
    return record


def _fixture(tmp_path: Path, receipts, native_center=(20.0, 20.0, 20.0), native_extent=1.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    native_row = (7, _corners(native_center, native_extent), np.float32(1.0))
    native_path = baseline / f"{scene}_boxes.pkl"
    with native_path.open("wb") as handle:
        pickle.dump([[native_row]], handle, protocol=4)

    sidecar = tmp_path / "masklift.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "boxfusion.scannet_target_first_mobilesam_masklift_full100.v1",
                "contracts": {
                    "gt_access": False,
                    "evaluator_access": False,
                    "target_dataset_training": False,
                    "online_learning": False,
                },
                "past_only_confirmation": True,
                "receipt_count": len(receipts),
                "scenes": {scene: {"receipts": receipts}},
            }
        ),
        encoding="utf-8",
    )
    return {
        "scene": scene,
        "scene_list": scene_list,
        "baseline": baseline,
        "native_path": native_path,
        "native_row": native_row,
        "sidecar": sidecar,
    }


def _run(inputs, output, plan_only=False):
    return materialize_scannet_target_first_mobilesam_birth_full100(
        scene_list=inputs["scene_list"],
        baseline_root=inputs["baseline"],
        masklift_sidecar=inputs["sidecar"],
        output_root=output,
        expected_scene_count=1,
        plan_only=plan_only,
    )


def test_active_appends_constant_score_and_preserves_native_prefix(tmp_path):
    inputs = _fixture(tmp_path, [_receipt()])
    output = tmp_path / "active"
    baseline_digest = _sha256(inputs["native_path"])
    manifest = _run(inputs, output)

    assert manifest["schema"] == SCHEMA
    assert manifest["birth_count"] == 1
    assert manifest["native_count"] == 1
    assert manifest["native_prediction_sha256"][inputs["scene"]] == baseline_digest
    assert manifest["inputs"]["masklift_sidecar_sha256"] == _sha256(inputs["sidecar"])
    assert (output / MANIFEST_NAME).is_file()

    with (output / f"{inputs['scene']}_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    before = inputs["native_row"]
    after = rows[0]
    assert type(after) is type(before)
    assert type(after[0]) is type(before[0]) and after[0] == before[0]
    assert after[1].dtype == before[1].dtype
    assert after[1].tobytes() == before[1].tobytes()
    assert type(after[2]) is type(before[2])
    assert np.asarray(after[2]).tobytes() == np.asarray(before[2]).tobytes()
    assert rows[1][0] == APPENDED_CLASS_ID
    assert rows[1][2] == APPENDED_SCORE
    np.testing.assert_allclose(rows[1][1], _corners((0.0, 0.0, 0.0)))


def test_plan_only_fully_selects_but_creates_nothing(tmp_path):
    inputs = _fixture(tmp_path, [_receipt()])
    output = tmp_path / "planned"
    manifest = _run(inputs, output, plan_only=True)
    assert manifest["mode"] == "plan_only"
    assert manifest["birth_count"] == 1
    assert manifest["output_prediction_sha256"] == {}
    assert not output.exists()
    assert manifest["scenes"][inputs["scene"]]["native_prefix_row_identity_verified"] is False


def test_frozen_gates_self_nms_and_cap_four(tmp_path):
    receipts = [
        _receipt(1, (0.0, 0.0, 0.0)),
        _receipt(
            2,
            (0.05, 0.0, 0.0),
            evidence_scores=[0.70, 0.70, 0.70],
            raw_mean_score=0.70,
        ),  # self-NMS
        _receipt(3, (3.0, 0.0, 0.0)),
        _receipt(4, (6.0, 0.0, 0.0)),
        _receipt(5, (9.0, 0.0, 0.0)),
        _receipt(6, (12.0, 0.0, 0.0)),  # cap four
        _receipt(
            7,
            (15.0, 0.0, 0.0),
            evidence_scores=[0.49, 0.49, 0.49],
            raw_mean_score=0.49,
        ),
        _receipt(8, (18.0, 0.0, 0.0), median_pairwise_mask_aabb_iou=0.14),
    ]
    inputs = _fixture(tmp_path, receipts, native_center=(100.0, 100.0, 100.0))
    manifest = _run(inputs, tmp_path / "active")
    counts = manifest["scenes"][inputs["scene"]]["decision_counts"]
    assert manifest["birth_count"] == 4
    assert counts["self_nms"] == 1
    assert counts["scene_cap"] == 1
    assert counts["score"] == 1
    assert counts["r15"] == 1


def test_native_iou_or_containment_rejects_candidate(tmp_path):
    overlap = _receipt(1, (0.0, 0.0, 0.0))
    contained = _receipt(
        2,
        (4.0, 0.0, 0.0),
        fused_obb={
            "extent_xyz": [0.2, 0.2, 0.2],
            "corners_world": _corners((4.0, 0.0, 0.0), 0.2).tolist(),
        },
    )
    # Test each native relationship independently because there is one native box.
    first = _fixture(tmp_path / "iou", [overlap], native_center=(0.0, 0.0, 0.0))
    manifest = _run(first, tmp_path / "iou-out", plan_only=True)
    assert manifest["scenes"][first["scene"]]["decision_counts"]["native_overlap"] == 1

    second = _fixture(
        tmp_path / "contain", [contained], native_center=(4.0, 0.0, 0.0), native_extent=2.0
    )
    manifest = _run(second, tmp_path / "contain-out", plan_only=True)
    assert manifest["scenes"][second["scene"]]["decision_counts"]["native_containment"] == 1


def test_empty_fusion_sentinel_is_audited_not_materialized(tmp_path):
    empty = _receipt(
        fused_obb={
            "extent_xyz": [0.0, 0.0, 0.0],
            "corners_world": np.zeros((8, 3), dtype=np.float32).tolist(),
        },
        supported_voxel_count=0,
        view_supported_voxel_counts=[0, 0, 0],
    )
    inputs = _fixture(tmp_path, [empty])
    manifest = _run(inputs, tmp_path / "empty-out", plan_only=True)
    scene = manifest["scenes"][inputs["scene"]]
    assert manifest["birth_count"] == 0
    assert scene["decision_counts"]["voxel_support"] == 1


def test_rejects_forbidden_contract_and_existing_output(tmp_path):
    inputs = _fixture(tmp_path, [_receipt()])
    payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
    payload["contracts"]["gt_access"] = True
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="gt_access"):
        _run(inputs, tmp_path / "out", plan_only=True)

    inputs = _fixture(tmp_path / "fresh", [_receipt()])
    output = tmp_path / "already"
    output.mkdir()
    with pytest.raises(BirthMaterializationError, match="overwrite"):
        _run(inputs, output)


def test_runner_canonical_scene_list_and_npz_are_cross_checked(tmp_path):
    inputs = _fixture(tmp_path, [_receipt()])
    payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
    record = payload["scenes"][inputs["scene"]]["receipts"][0]
    record["fused_min_obb_extent_m"] = 1.0
    record["fused_corners_world"] = record["fused_obb"]["corners_world"]
    record["accepted"] = True
    record["decision"] = "accepted_shadow"
    payload["scenes"] = [
        {"scene_id": inputs["scene"], "receipts": [record]}
    ]
    payload["scene_order"] = [inputs["scene"]]
    payload["gt_access"] = False
    payload["evaluator_access"] = False

    arrays = {
        "track_scene_index": np.asarray([0], dtype=np.int16),
        "track_group_track_id": np.asarray([1], dtype=np.int32),
        "track_fused_obb_corners": np.asarray(
            [record["fused_corners_world"]], dtype=np.float32
        ),
        "track_accepted_shadow": np.asarray([True], dtype=bool),
    }
    npz_path = tmp_path / "TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.npz"
    np.savez(npz_path, **arrays)
    payload["npz_file"] = npz_path.name
    payload["npz_arrays"] = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _hash_array(value),
        }
        for name, value in arrays.items()
    }
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")

    manifest = _run(inputs, tmp_path / "canonical-out", plan_only=True)
    assert manifest["birth_count"] == 1
    assert manifest["inputs"]["masklift_npz_sha256"] == _sha256(npz_path)
    assert manifest["inputs"]["masklift_npz_array_sha256"] == {
        name: _hash_array(value) for name, value in arrays.items()
    }

    payload["npz_arrays"]["track_accepted_shadow"]["sha256"] = "0" * 64
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="metadata/hash"):
        _run(inputs, tmp_path / "tampered-out", plan_only=True)


def test_cli_has_no_gt_or_evaluator_surface():
    destinations = {action.dest for action in _parser()._actions}
    assert "plan_only" in destinations
    assert not any("gt" in name or "eval" in name or "annot" in name for name in destinations)
