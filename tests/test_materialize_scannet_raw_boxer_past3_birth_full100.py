import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (
    APPENDED_CLASS_ID,
    APPENDED_SCORE,
    BirthMaterializationError,
    RAW_COLUMNS,
    SCHEMA,
    V2_SCHEMA,
    V3_SCHEMA,
    materialize_scannet_raw_boxer_past3_birth_full100,
)


SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float32,
)


def _corners(center, extent=1.0):
    return SIGNS * (extent / 2.0) + np.asarray(center, dtype=np.float32)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, native_center=(10.0, 0.0, 0.0), native_extent=1.0):
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(f"{scene}\n", encoding="utf-8")

    raw_root = tmp_path / "raw"
    raw_scene = raw_root / "boxer_raw" / scene
    raw_scene.mkdir(parents=True)
    raw_csv = raw_scene / "boxer_3dbbs.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for row_index, (frame_id, x, score) in enumerate(
            ((0, 0.00, 0.9), (25, 0.02, 0.8), (50, -0.01, 0.7))
        ):
            writer.writerow(
                {
                    "time_ns": frame_id,
                    "tx_world_object": x,
                    "ty_world_object": 0.0,
                    "tz_world_object": 2.0,
                    "qw_world_object": 1.0,
                    "qx_world_object": 0.0,
                    "qy_world_object": 0.0,
                    "qz_world_object": 0.0,
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "scale_z": 1.0,
                    # These semantic fields are deliberately inconsistent;
                    # the materializer must not use them.
                    "name": f"unused_label_{row_index}",
                    "instance": row_index,
                    "sem_id": 9999 - row_index,
                    "prob": score,
                }
            )
    ledger = raw_root / "schedule_audit_worker0_of_1.tsv"
    ledger.write_text(
        "scene\tmanifest_keyframes\tvalid_keyframes\tinvalid_pose_keyframes\t"
        "raw_candidate_frames\traw_candidates\n"
        f"{scene}\t3\t3\t0\t3\t3\n",
        encoding="utf-8",
    )

    schedule_root = tmp_path / "schedule"
    schedule_scene = schedule_root / scene
    schedule_scene.mkdir(parents=True)
    (schedule_scene / "manifest.json").write_text(
        json.dumps(
            {
                "namespace": "fixture-score05-gap25",
                "record_count": 3,
                "recorded_frame_ids": [0, 25, 50],
            }
        ),
        encoding="utf-8",
    )

    rgbd_root = tmp_path / "rgbd"
    pose_root = rgbd_root / scene / "frames" / "pose"
    pose_root.mkdir(parents=True)
    pose_translations = {
        0: (3.0, 4.0, 5.0),
        25: (3.3, 4.0, 5.0),
        50: (3.0, 4.3, 5.0),
    }
    for frame_id, translation in pose_translations.items():
        pose = np.eye(4)
        pose[:3, 3] = translation
        np.savetxt(pose_root / f"{frame_id}.txt", pose)

    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    native_path = baseline_root / f"{scene}_boxes.pkl"
    native_row = (7, _corners(native_center, native_extent), 0.4321)
    with native_path.open("wb") as handle:
        pickle.dump([[native_row]], handle, protocol=4)
    return {
        "scene": scene,
        "scene_list": scene_list,
        "raw_root": raw_root,
        "schedule_root": schedule_root,
        "rgbd_root": rgbd_root,
        "baseline_root": baseline_root,
        "native_path": native_path,
        "native_row": native_row,
    }


def _run(
    inputs,
    output_root,
    *,
    selection_policy="v1",
    clip_gate_sidecar=None,
):
    return materialize_scannet_raw_boxer_past3_birth_full100(
        scene_list=inputs["scene_list"],
        raw_log_root=inputs["raw_root"],
        baseline_root=inputs["baseline_root"],
        schedule_root=inputs["schedule_root"],
        scene_rgbd_root=inputs["rgbd_root"],
        output_root=output_root,
        expected_scene_count=1,
        wait_timeout_seconds=0.0,
        poll_seconds=0.01,
        selection_policy=selection_policy,
        clip_gate_sidecar=clip_gate_sidecar,
    )


def _set_semantic_ids(inputs, semantic_ids):
    raw_csv = (
        inputs["raw_root"]
        / "boxer_raw"
        / inputs["scene"]
        / "boxer_3dbbs.csv"
    )
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(semantic_ids)
    with raw_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for row, semantic_id in zip(rows, semantic_ids):
            row["sem_id"] = semantic_id
            writer.writerow(row)


def _v2_receipt_identity(inputs, tmp_path):
    manifest = _run(
        inputs, tmp_path / "v2_identity", selection_policy="v2_m50"
    )
    decision = manifest["scenes"][inputs["scene"]]["receipt_decisions"][0]
    return decision["track_id"], decision["evidence_source_rows"]


def _write_clip_sidecar(path, scene, track_id, evidence_rows, gate_pass, layout):
    record = {
        "track_id": track_id,
        "evidence_source_rows": evidence_rows,
        "gate_pass": gate_pass,
        "supporting_view_count": 2,
        "frozen_target_label": "chair",
    }
    if layout == "tracks_list":
        scenes = {scene: {"tracks": [record]}}
        contract_fields = {
            "gt_access": False,
            "evaluator_access": False,
            "contracts": {
                "gt_access": False,
                "evaluator_access": False,
            },
        }
    elif layout == "tracks_mapping":
        record = dict(record)
        record.pop("track_id")
        record["clip_summary"] = {"gate_pass": record.pop("gate_pass")}
        scenes = {scene: {"tracks": {str(track_id): record}}}
        contract_fields = {
            "contracts": {
                "gt_access": False,
                "evaluator_access": False,
            }
        }
    else:
        raise AssertionError(layout)
    path.write_text(
        json.dumps(
            {
                "schema": (
                    "boxfusion.scannet_raw_boxer_clip_vocab_shadow_full100.v1"
                ),
                **contract_fields,
                "scenes": scenes,
            }
        ),
        encoding="utf-8",
    )


def test_materializer_appends_one_constant_score_birth_and_preserves_native_prefix(tmp_path):
    inputs = _fixture(tmp_path)
    output_root = tmp_path / "output"
    native_hash = _sha256(inputs["native_path"])

    manifest = _run(inputs, output_root)

    assert manifest["schema"] == SCHEMA
    assert manifest["training_free"] is True
    assert manifest["gt_access"] is False
    assert manifest["evaluator_access"] is False
    assert manifest["detector_semantics_used"] is False
    assert manifest["birth_count"] == 1
    assert _sha256(inputs["native_path"]) == native_hash

    with (output_root / f"{inputs['scene']}_boxes.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert len(payload) == 1
    assert len(payload[0]) == 2
    before = inputs["native_row"]
    after = payload[0][0]
    assert type(before) is type(after)
    assert type(before[0]) is type(after[0]) and before[0] == after[0]
    assert before[1].dtype == after[1].dtype
    assert before[1].tobytes() == after[1].tobytes()
    assert type(before[2]) is type(after[2]) and before[2] == after[2]
    suffix = payload[0][1]
    assert suffix[0] == APPENDED_CLASS_ID
    assert suffix[2] == APPENDED_SCORE
    assert suffix[1].shape == (8, 3)
    assert suffix[1].dtype == np.float32
    # CSV center (0,0,2) is recentered; first valid pose translation (3,4,5)
    # must be restored before geometry and novelty are evaluated.
    np.testing.assert_allclose(suffix[1].mean(axis=0), [3.0, 4.0, 7.0], atol=1e-6)
    assert manifest["scenes"][inputs["scene"]]["world_offset_xyz"] == [3.0, 4.0, 5.0]
    assert (output_root / "RAW_BOXER_PAST3_BIRTH_FULL100.json").is_file()


def test_frozen_novelty_rule_uses_iou_only_without_containment(tmp_path):
    # The 1 m candidate occupies only 1/64 of this native AABB.  The frozen
    # active protocol deliberately has no containment branch, so IoU < 0.10
    # remains novel.
    inputs = _fixture(
        tmp_path, native_center=(3.0, 4.0, 7.0), native_extent=4.0
    )
    output_root = tmp_path / "output"
    manifest = _run(inputs, output_root)
    scene_report = manifest["scenes"][inputs["scene"]]
    assert manifest["past3_receipt_count"] == 1
    assert manifest["birth_count"] == 1
    assert scene_report["decision_counts"]["accepted"] == 1
    decision = scene_report["receipt_decisions"][0]
    assert decision["max_native_aabb_iou"] < 0.10


def test_missing_completion_ledger_fails_before_partial_csv_is_parsed(tmp_path):
    inputs = _fixture(tmp_path)
    next(inputs["raw_root"].glob("schedule_audit_worker*.tsv")).unlink()
    raw_csv = inputs["raw_root"] / "boxer_raw" / inputs["scene"] / "boxer_3dbbs.csv"
    raw_csv.write_text("this,is,a,partial,csv\n", encoding="utf-8")
    output_root = tmp_path / "output"
    with pytest.raises(BirthMaterializationError, match="incomplete"):
        _run(inputs, output_root)
    assert not output_root.exists()


def test_existing_output_root_is_never_overwritten(tmp_path):
    inputs = _fixture(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    sentinel = output_root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(BirthMaterializationError, match="overwrite"):
        _run(inputs, output_root)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_v2_requires_exact_three_view_semantic_identity(tmp_path):
    inputs = _fixture(tmp_path)
    output_root = tmp_path / "output"
    manifest = _run(inputs, output_root, selection_policy="v2_m50")
    assert manifest["schema"] == V2_SCHEMA
    assert manifest["detector_semantics_used"] is True
    assert manifest["birth_count"] == 0
    counts = manifest["scenes"][inputs["scene"]]["decision_counts"]
    assert counts["semantic_inconsistent"] == 1


def test_v2_accepts_consistent_semantics_and_preserves_native_prefix(tmp_path):
    inputs = _fixture(tmp_path)
    _set_semantic_ids(inputs, [42, 42, 42])
    output_root = tmp_path / "output"
    manifest = _run(inputs, output_root, selection_policy="v2_m50")
    assert manifest["birth_count"] == 1
    assert manifest["target_dataset_training"] is False
    assert manifest["native_clip_unchanged"] is True
    decision = manifest["scenes"][inputs["scene"]]["receipt_decisions"][0]
    assert decision["semantic_consistent"] is True
    assert decision["evidence_semantic_ids"] == [42, 42, 42]


def test_v2_rejects_candidate_contained_by_native_box(tmp_path):
    inputs = _fixture(
        tmp_path, native_center=(3.0, 4.0, 7.0), native_extent=4.0
    )
    _set_semantic_ids(inputs, [42, 42, 42])
    output_root = tmp_path / "output"
    manifest = _run(inputs, output_root, selection_policy="v2_m50")
    assert manifest["birth_count"] == 0
    decision = manifest["scenes"][inputs["scene"]]["receipt_decisions"][0]
    assert decision["decision"] == "native_containment"
    assert decision["max_candidate_in_native_containment"] == pytest.approx(1.0)


@pytest.mark.parametrize("layout", ["tracks_list", "tracks_mapping"])
def test_v3_clip_gate_pass_preserves_prefix_score_and_full_provenance(
    tmp_path, layout
):
    inputs = _fixture(tmp_path)
    _set_semantic_ids(inputs, [42, 42, 42])
    track_id, evidence_rows = _v2_receipt_identity(inputs, tmp_path)
    sidecar = tmp_path / "clip_gate.json"
    _write_clip_sidecar(
        sidecar,
        inputs["scene"],
        track_id,
        evidence_rows,
        True,
        layout,
    )
    native_hash = _sha256(inputs["native_path"])
    output_root = tmp_path / f"v3_{layout}"
    manifest = _run(
        inputs,
        output_root,
        selection_policy="v3_clip_vocab",
        clip_gate_sidecar=sidecar,
    )

    assert manifest["schema"] == V3_SCHEMA
    assert manifest["gt_access"] is False
    assert manifest["clip_access"] is True
    assert manifest["birth_count"] == 1
    assert manifest["clip_gate_evaluated_count"] == 1
    assert _sha256(inputs["native_path"]) == native_hash
    with (output_root / f"{inputs['scene']}_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    assert rows[0][1].tobytes() == inputs["native_row"][1].tobytes()
    assert rows[0][2] == inputs["native_row"][2]
    assert rows[1][2] == APPENDED_SCORE
    decision = manifest["scenes"][inputs["scene"]]["receipt_decisions"][0]
    assert decision["decision"] == "accepted"
    assert decision["clip_gate_pass"] is True
    assert decision["clip_gate_sidecar_record"]["frozen_target_label"] == "chair"
    assert manifest["inputs"]["clip_gate_sidecar_sha256"] == _sha256(sidecar)


def test_v3_clip_gate_rejects_before_nms_and_emits_no_suffix(tmp_path):
    inputs = _fixture(tmp_path)
    _set_semantic_ids(inputs, [42, 42, 42])
    track_id, evidence_rows = _v2_receipt_identity(inputs, tmp_path)
    sidecar = tmp_path / "clip_gate.json"
    _write_clip_sidecar(
        sidecar,
        inputs["scene"],
        track_id,
        evidence_rows,
        False,
        "tracks_list",
    )
    output_root = tmp_path / "v3_reject"
    manifest = _run(
        inputs,
        output_root,
        selection_policy="v3_clip_vocab",
        clip_gate_sidecar=sidecar,
    )
    assert manifest["birth_count"] == 0
    decision = manifest["scenes"][inputs["scene"]]["receipt_decisions"][0]
    assert decision["decision"] == "clip_gate"
    assert decision["clip_gate_evaluated"] is True
    assert decision["clip_gate_pass"] is False
    with (output_root / f"{inputs['scene']}_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    assert len(rows) == 1


def test_v3_missing_sidecar_record_for_v2_admissible_receipt_is_fatal(tmp_path):
    inputs = _fixture(tmp_path)
    _set_semantic_ids(inputs, [42, 42, 42])
    sidecar = tmp_path / "clip_gate.json"
    sidecar.write_text(
        json.dumps(
            {
                "gt_access": False,
                "evaluator_access": False,
                "scenes": {inputs["scene"]: {"tracks": []}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BirthMaterializationError, match="missing a v2-admissible"):
        _run(
            inputs,
            tmp_path / "v3_missing",
            selection_policy="v3_clip_vocab",
            clip_gate_sidecar=sidecar,
        )
