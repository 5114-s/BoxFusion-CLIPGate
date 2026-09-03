import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.materialize_scannet_udc_mobilesam_birth_full100 import (
    APPENDED_CLASS_ID,
    APPENDED_SCORE,
    MANIFEST_NAME,
    SCHEMA,
    SIDECAR_NAME,
    SIDECAR_SCHEMA,
    BirthMaterializationError,
    _parser,
    materialize_scannet_udc_mobilesam_birth_full100,
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


def _receipt(
    track_id=1,
    center=(0.0, 0.0, 0.0),
    extent=1.0,
    mean_predicted_iou=0.80,
    supported_voxel_count=40,
    **updates,
):
    frame0 = track_id * 100
    record = {
        "track_id": track_id,
        "confirmation_frame_id": frame0 + 50,
        "evidence_frame_ids": [frame0, frame0 + 25, frame0 + 50],
        "mean_predicted_iou": mean_predicted_iou,
        "supported_voxel_count": supported_voxel_count,
        "pre_novelty_pass": True,
        "fused_obb": {
            "extent_xyz": np.broadcast_to(
                np.asarray(extent, dtype=np.float64), (3,)
            ).tolist(),
            "corners_world": _corners(center, extent).tolist(),
        },
    }
    record.update(updates)
    return record


def _fixture(
    tmp_path: Path,
    receipts,
    native_center=(20.0, 20.0, 20.0),
    native_extent=1.0,
):
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

    sidecar_dir = tmp_path / "udc"
    sidecar_dir.mkdir()
    sidecar = sidecar_dir / SIDECAR_NAME
    sidecar.write_text(
        json.dumps(
            {
                "schema": SIDECAR_SCHEMA,
                "contracts": {
                    "causal_shadow_generation": True,
                    "native_prediction_access": False,
                    "output_inert": True,
                    "current_frame_cutr_boxes_only": True,
                    "past_only_tracking_and_confirmation": True,
                    "gt_access": False,
                    "evaluator_access": False,
                    "terminal_prediction_access": False,
                    "target_dataset_training": False,
                    "online_learning": False,
                },
                "receipt_count": len(receipts),
                "scenes": [{"scene_id": scene, "receipts": receipts}],
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
        "sidecar_dir": sidecar_dir,
    }


def _run(inputs, output, plan_only=False, sidecar_as_directory=False):
    return materialize_scannet_udc_mobilesam_birth_full100(
        scene_list=inputs["scene_list"],
        baseline_root=inputs["baseline"],
        udc_sidecar=(
            inputs["sidecar_dir"] if sidecar_as_directory else inputs["sidecar"]
        ),
        output_root=output,
        expected_scene_count=1,
        plan_only=plan_only,
    )


def test_active_appends_constant_score_and_preserves_native_prefix(tmp_path):
    inputs = _fixture(tmp_path, [_receipt()])
    output = tmp_path / "active"
    baseline_digest = _sha256(inputs["native_path"])
    manifest = _run(inputs, output, sidecar_as_directory=True)

    assert manifest["schema"] == SCHEMA
    assert manifest["birth_count"] == 1
    assert manifest["native_count"] == 1
    assert manifest["native_prediction_sha256"][inputs["scene"]] == baseline_digest
    assert manifest["inputs"]["udc_sidecar_sha256"] == _sha256(inputs["sidecar"])
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [MANIFEST_NAME, f"{inputs['scene']}_boxes.pkl"]
    )

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


def test_plan_only_fully_selects_and_creates_nothing(tmp_path):
    inputs = _fixture(tmp_path, [_receipt()])
    output = tmp_path / "planned"
    manifest = _run(inputs, output, plan_only=True)
    assert manifest["mode"] == "plan_only"
    assert manifest["birth_count"] == 1
    assert manifest["output_prediction_sha256"] == {}
    assert not output.exists()
    assert (
        manifest["scenes"][inputs["scene"]][
            "native_prefix_row_identity_verified"
        ]
        is False
    )


def test_fixed_ranking_pre_novelty_and_cap_four(tmp_path):
    common_frames = {
        "confirmation_frame_id": 150,
        "evidence_frame_ids": [100, 125, 150],
    }
    receipts = [
        _receipt(10, (0.0, 0.0, 0.0), mean_predicted_iou=0.90, **common_frames),
        _receipt(2, (3.0, 0.0, 0.0), mean_predicted_iou=0.90, **common_frames),
        _receipt(
            3,
            (6.0, 0.0, 0.0),
            mean_predicted_iou=0.90,
            supported_voxel_count=50,
        ),
        _receipt(4, (9.0, 0.0, 0.0), mean_predicted_iou=0.80),
        _receipt(5, (12.0, 0.0, 0.0), mean_predicted_iou=0.70),
        _receipt(
            6,
            (15.0, 0.0, 0.0),
            mean_predicted_iou=0.99,
            pre_novelty_pass=False,
        ),
    ]
    inputs = _fixture(
        tmp_path, receipts, native_center=(100.0, 100.0, 100.0)
    )
    manifest = _run(inputs, tmp_path / "planned", plan_only=True)
    scene = manifest["scenes"][inputs["scene"]]

    # False pre-novelty receipt ranks first but cannot consume the cap.  Then
    # the exact frozen key is mean IoU, support, confirmation, and track.
    assert [row["track_id"] for row in scene["suffix"]] == [3, 2, 10, 4]
    assert scene["decision_counts"]["pre_novelty"] == 1
    assert scene["decision_counts"]["scene_cap"] == 1


def test_native_gate_has_no_iou_cutoff_but_rejects_containment(tmp_path):
    # Equal cubes offset 0.4 have IoU ~= .43 and both containments .60.  This
    # must survive because native IoU is diagnostic only.
    partial = _fixture(
        tmp_path / "partial",
        [_receipt(1, (0.4, 0.0, 0.0))],
        native_center=(0.0, 0.0, 0.0),
    )
    manifest = _run(partial, tmp_path / "partial-out", plan_only=True)
    scene = manifest["scenes"][partial["scene"]]
    assert scene["birth_count"] == 1
    diagnostic = scene["receipt_decisions"][0][
        "max_native_aabb_iou_diagnostic_only"
    ]
    assert diagnostic > 0.15
    assert manifest["frozen_policy"]["native_iou_hard_gate"] is False

    # A small candidate fully inside native is rejected by directed
    # containment even though there is still no IoU policy.
    contained = _fixture(
        tmp_path / "contained",
        [_receipt(1, (0.0, 0.0, 0.0), extent=0.4)],
        native_center=(0.0, 0.0, 0.0),
        native_extent=2.0,
    )
    manifest = _run(contained, tmp_path / "contained-out", plan_only=True)
    scene = manifest["scenes"][contained["scene"]]
    assert scene["birth_count"] == 0
    assert scene["decision_counts"]["native_containment"] == 1


def test_self_nms_uses_iou_or_either_containment(tmp_path):
    receipts = [
        _receipt(1, (0.0, 0.0, 0.0), mean_predicted_iou=0.95),
        # IoU > .15 with the first candidate.
        _receipt(2, (0.6, 0.0, 0.0), mean_predicted_iou=0.90),
        # IoU < .15 but this small candidate is fully contained by the first.
        _receipt(3, (0.0, 0.0, 0.0), extent=0.4, mean_predicted_iou=0.85),
        _receipt(4, (4.0, 0.0, 0.0), mean_predicted_iou=0.80),
    ]
    inputs = _fixture(
        tmp_path, receipts, native_center=(100.0, 100.0, 100.0)
    )
    manifest = _run(inputs, tmp_path / "planned", plan_only=True)
    scene = manifest["scenes"][inputs["scene"]]
    assert [row["track_id"] for row in scene["suffix"]] == [1, 4]
    assert scene["decision_counts"]["self_nms"] == 2


def test_rejects_nonlist_scenes_bad_causal_receipt_and_duplicate_track(tmp_path):
    inputs = _fixture(tmp_path / "layout", [_receipt()])
    payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
    payload["scenes"] = {inputs["scene"]: {"receipts": [_receipt()]}}
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="scenes must be a list"):
        _run(inputs, tmp_path / "layout-out", plan_only=True)

    inputs = _fixture(tmp_path / "causal", [_receipt()])
    payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
    payload["scenes"][0]["receipts"][0]["confirmation_frame_id"] += 1
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="third causal"):
        _run(inputs, tmp_path / "causal-out", plan_only=True)

    duplicate = _fixture(tmp_path / "duplicate", [_receipt(), _receipt()])
    with pytest.raises(BirthMaterializationError, match="duplicate receipt track_id"):
        _run(duplicate, tmp_path / "duplicate-out", plan_only=True)


def test_rejects_forbidden_contract_schema_and_existing_output(tmp_path):
    inputs = _fixture(tmp_path / "contract", [_receipt()])
    payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
    payload["contracts"]["gt_access"] = True
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="gt_access"):
        _run(inputs, tmp_path / "contract-out", plan_only=True)

    for key, bad_value in (
        ("causal_shadow_generation", False),
        ("native_prediction_access", True),
        ("output_inert", False),
        ("current_frame_cutr_boxes_only", False),
        ("past_only_tracking_and_confirmation", False),
    ):
        inputs = _fixture(tmp_path / f"contract-{key}", [_receipt()])
        payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
        payload["contracts"][key] = bad_value
        inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(BirthMaterializationError, match=key):
            _run(inputs, tmp_path / f"contract-{key}-out", plan_only=True)

    inputs = _fixture(tmp_path / "contract-missing", [_receipt()])
    payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
    del payload["contracts"]["output_inert"]
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="output_inert"):
        _run(inputs, tmp_path / "contract-missing-out", plan_only=True)

    inputs = _fixture(tmp_path / "schema", [_receipt()])
    payload = json.loads(inputs["sidecar"].read_text(encoding="utf-8"))
    payload["schema"] = "boxfusion.wrong.v1"
    inputs["sidecar"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="unsupported"):
        _run(inputs, tmp_path / "schema-out", plan_only=True)

    inputs = _fixture(tmp_path / "overwrite", [_receipt()])
    output = tmp_path / "already"
    output.mkdir()
    with pytest.raises(BirthMaterializationError, match="overwrite"):
        _run(inputs, output)


def test_full100_publishes_exactly_100_predictions_and_one_manifest(tmp_path):
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    native_row = (7, _corners((20.0, 20.0, 20.0)), np.float32(1.0))
    for scene in scenes:
        with (baseline / f"{scene}_boxes.pkl").open("wb") as handle:
            pickle.dump([[native_row]], handle, protocol=4)
    sidecar = tmp_path / SIDECAR_NAME
    sidecar.write_text(
        json.dumps(
            {
                "schema": SIDECAR_SCHEMA,
                "contracts": {
                    "causal_shadow_generation": True,
                    "native_prediction_access": False,
                    "output_inert": True,
                    "current_frame_cutr_boxes_only": True,
                    "past_only_tracking_and_confirmation": True,
                    "gt_access": False,
                    "evaluator_access": False,
                },
                "receipt_count": 0,
                "scenes": [
                    {"scene_id": scene, "receipts": []} for scene in scenes
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "full100"
    manifest = materialize_scannet_udc_mobilesam_birth_full100(
        scene_list=scene_list,
        baseline_root=baseline,
        udc_sidecar=sidecar,
        output_root=output,
    )
    assert manifest["scene_count"] == 100
    assert manifest["native_count"] == 100
    assert manifest["birth_count"] == 0
    assert len(list(output.glob("*_boxes.pkl"))) == 100
    assert {path.name for path in output.iterdir()} == {
        *(f"{scene}_boxes.pkl" for scene in scenes),
        MANIFEST_NAME,
    }


def test_cli_has_plan_only_and_no_gt_or_evaluator_surface():
    destinations = {action.dest for action in _parser()._actions}
    assert "plan_only" in destinations
    assert "udc_sidecar" in destinations
    assert not any(
        "gt" in name or "eval" in name or "annot" in name
        for name in destinations
    )
