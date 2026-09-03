from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tools.run_scannet_s3a_boxer_mobilesam_masklift_shadow as s3a


class _FakeEngine:
    def __init__(self, _device: str = "cuda") -> None:
        pass

    def predict(self, image_rgb, boxes_xyxy):
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        masks = np.zeros(
            (len(boxes), s3a.IMAGE_HEIGHT, s3a.IMAGE_WIDTH), dtype=bool
        )
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            masks[
                index,
                max(0, int(np.floor(y1))) : min(s3a.IMAGE_HEIGHT, int(np.ceil(y2))),
                max(0, int(np.floor(x1))) : min(s3a.IMAGE_WIDTH, int(np.ceil(x2))),
            ] = True
        return (
            masks,
            np.linspace(0.8, 0.9, len(boxes), dtype=np.float32),
            np.arange(len(boxes), dtype=np.int8) % 3,
            {
                "encoder_ms": 2.0,
                "decoder_and_host_mask_ms": 1.0,
                "provider_ms": 3.0,
            },
        )

    def runtime_metadata(self):
        return {"device": "fake", "parameter_count": 10_130_092}


def _record(npz_row: int, box: list[float]) -> dict[str, object]:
    return {
        "scene": "scene0377_02",
        "scene_index_full": 2,
        "frame_id": 0,
        "schedule_ordinal": 0,
        "manifest_schedule_ordinal": 0,
        "sealed_npz_row": npz_row,
        "boxer_source_row": npz_row,
        "boxer_csv_line_number": npz_row + 2,
        "source_instance_id": npz_row,
        "owl_csv_source_row": npz_row,
        "owl_csv_line_number": npz_row + 2,
        "source_score": 0.9 - npz_row * 0.01,
        "owl_box_xyxy_960": np.asarray(box, dtype=np.float32)
        / np.asarray([2 / 3, 1 / 2, 2 / 3, 1 / 2], dtype=np.float32),
        "prompt_box_xyxy_640x480": np.asarray(box, dtype=np.float32),
        "raw_boxer_center_world": np.ones(3, dtype=np.float32),
        "raw_boxer_quaternion_wxyz": np.asarray([1, 0, 0, 0], dtype=np.float32),
        "raw_boxer_extent_xyz": np.ones(3, dtype=np.float32),
    }


def test_real_frozen_top4_membership_and_numeric_pairing_are_exact():
    manifest, arrays = s3a._load_sealed_candidates()
    selections, digest = s3a._select_top4(arrays)
    assert digest == s3a.EXPECTED_SELECTION_SHA256
    assert [len(value) for value in selections] == [262, 436, 116]
    schedules = s3a._load_schedule_contract(manifest)
    records, sources = s3a._validate_pair_and_build_records(
        manifest=manifest,
        arrays=arrays,
        selections=selections,
        schedules=schedules,
        requested_scenes=("scene0377_02",),
    )
    assert len(records) == 116
    assert records[0]["source_instance_id"] == records[0]["owl_csv_source_row"]
    assert records[0]["schedule_ordinal"] == 0
    assert records[-1]["schedule_ordinal"] == 29
    assert sources["scene0377_02"]["semantic_columns_decoded"] is False
    assert sources["scene0377_02"]["semantic_columns_consumed"] is False


def test_owl_mapping_is_exact_clip_and_mask_pack_roundtrips():
    mapped = s3a._map_owl_box_to_depth(
        np.asarray([-3.0, 20.0, 963.0, 1000.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(mapped, [0.0, 10.0, 640.0, 480.0])
    mask = np.zeros((s3a.IMAGE_HEIGHT, s3a.IMAGE_WIDTH), dtype=bool)
    mask[5:9, 7:11] = True
    packed = s3a._pack_mask(mask)
    restored = np.unpackbits(packed, bitorder="little")[: mask.size].reshape(mask.shape)
    np.testing.assert_array_equal(restored.astype(bool), mask)


def test_numeric_csv_parser_never_decodes_or_returns_semantics(tmp_path):
    owl = tmp_path / "owl.csv"
    owl.write_bytes(
        s3a.OWL_HEADER
        + b"\n0,0,scannet,ScanNet,960,960,3,4,30,40,Dixie_cup,-1,99,0.8\n"
    )
    boxer = tmp_path / "boxer.csv"
    boxer.write_bytes(
        s3a.BOXER_HEADER
        + b"\n0,1,2,3,1,0,0,0,0.2,0.3,0.4,dixie_cup,0,99,0.7\n"
    )
    owl_rows, groups = s3a._read_owl_numeric_rows(owl)
    boxer_rows = s3a._read_boxer_numeric_rows(boxer)
    assert groups == {0: [0]}
    assert set(owl_rows[0]) == {
        "data_row",
        "line_number",
        "time_ns",
        "frame_ordinal",
        "box_xyxy_960",
    }
    assert "name" not in boxer_rows[0] and "sem_id" not in boxer_rows[0]
    assert boxer_rows[0]["instance_id"] == 0


def test_fixed_lifting_emits_primary_and_diagnostic_boxes_and_abstains():
    memory = s3a._load_object_memory_module()
    intrinsic = np.asarray(
        [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    depth = np.full((s3a.IMAGE_HEIGHT, s3a.IMAGE_WIDTH), 1000, dtype=np.uint16)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[200:220, 300:320] = True
    emitted = s3a._lift_mask_row(
        mask=mask,
        predicted_iou=0.9,
        hypothesis_index=1,
        depth=depth,
        intrinsic=intrinsic,
        pose=pose,
        object_memory=memory,
    )
    assert emitted["accepted"] is True
    assert emitted["abstention_code"] == 0
    assert emitted["unique_voxel_count"] >= s3a.MIN_CLEAN_VOXELS
    assert emitted["diagnostic_box_valid"] is True
    assert np.all(emitted["reported_q02_q98_extent_xyz"] > 0)

    no_depth = s3a._lift_mask_row(
        mask=mask,
        predicted_iou=0.9,
        hypothesis_index=1,
        depth=np.zeros_like(depth),
        intrinsic=intrinsic,
        pose=pose,
        object_memory=memory,
    )
    assert no_depth["accepted"] is False
    assert no_depth["abstention_code"] == 3

    empty = s3a._lift_mask_row(
        mask=np.zeros_like(mask),
        predicted_iou=0.8,
        hypothesis_index=0,
        depth=depth,
        intrinsic=intrinsic,
        pose=pose,
        object_memory=memory,
    )
    assert empty["abstention_code"] == 2


def test_process_records_exports_every_row_and_no_semantic_or_track_arrays(monkeypatch):
    rgb = np.zeros((s3a.IMAGE_HEIGHT, s3a.IMAGE_WIDTH, 3), dtype=np.uint8)
    depth = np.full((s3a.IMAGE_HEIGHT, s3a.IMAGE_WIDTH), 1000, dtype=np.uint16)
    pose = np.eye(4, dtype=np.float64)
    monkeypatch.setattr(s3a, "_decode_frame", lambda scene, frame: (rgb, depth, pose))
    intrinsic = np.asarray(
        [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    records = [_record(0, [280, 190, 330, 240]), _record(1, [330, 240, 380, 290])]
    arrays, summary = s3a._process_records(
        records=records,
        requested_scenes=("scene0377_02",),
        schedules={"scene0377_02": {"intrinsic": intrinsic}},
        engine=_FakeEngine(),
        object_memory=s3a._load_object_memory_module(),
    )
    assert summary["row_count"] == 2
    assert arrays["sam_mask_packed"].shape == (2, s3a.MASK_PACKED_BYTES)
    assert arrays["cleaned_depth_mask_packed"].shape == (
        2,
        s3a.MASK_PACKED_BYTES,
    )
    assert arrays["point_offsets"].shape == (3,)
    assert np.array_equal(
        np.diff(arrays["point_offsets"]), arrays["retained_point_count"]
    )
    assert not any(
        token in name.lower()
        for name in arrays
        for token in ("label", "class", "clip", "track")
    )


def test_publish_is_create_only_and_deterministic(tmp_path):
    arrays = {
        "value": np.asarray([1, 2, 3], dtype=np.int32),
        "scene_ids": np.asarray(["scene0377_02"], dtype="<U12"),
    }
    manifest = {"schema": s3a.SCHEMA, "npz_file": s3a.OUTPUT_NPZ_NAME}
    output = tmp_path / "shadow"
    s3a._publish_create_only(output_root=output, arrays=arrays, manifest=manifest)
    assert (output / s3a.OUTPUT_NPZ_NAME).is_file()
    with pytest.raises(s3a.S3aShadowError, match="refusing to overwrite"):
        s3a._publish_create_only(output_root=output, arrays=arrays, manifest=manifest)
    with np.load(output / s3a.OUTPUT_NPZ_NAME, allow_pickle=False) as source:
        np.testing.assert_array_equal(source["value"], arrays["value"])


def test_cli_and_source_have_no_annotation_or_oracle_input_surface():
    options = {
        option
        for action in s3a._build_parser()._actions
        for option in action.option_strings
    }
    assert not any("gt" in option.lower() or "oracle" in option.lower() for option in options)
    source = Path(s3a.__file__).read_text(encoding="utf-8")
    assert "from evaluation" not in source
    assert "from tools.audit_scannet" not in source
    assert "import tools.audit_scannet" not in source
    assert "CandidateTrackManager" not in source


def test_native_identity_is_bound_directly_to_formal_t05_files():
    native = s3a._hash_formal_t05_predictions()
    assert tuple(native) == s3a.DEV3_SCENES
    assert {
        scene: native[scene]["sha256"] for scene in s3a.DEV3_SCENES
    } == dict(s3a.EXPECTED_FORMAL_T05_SHA256)
    assert {
        Path(native[scene]["path"]).parent.resolve() for scene in s3a.DEV3_SCENES
    } == {s3a.FORMAL_T05_ROOT.resolve()}


def test_graw_replay_root_is_rejected_as_native_identity(monkeypatch):
    graw_root = (
        s3a.REPOSITORY_ROOT / "results" / "scannet_graw_e2_replay1_score05"
    )
    assert graw_root.is_dir()
    monkeypatch.setattr(s3a, "FORMAL_T05_ROOT", graw_root)
    with pytest.raises(s3a.S3aShadowError, match="formal T05 root mismatch"):
        s3a._hash_formal_t05_predictions()


def test_old_graw_hashes_are_rejected_at_formal_root(monkeypatch):
    old_graw_hashes = {
        "scene0568_00": "ed9b3f7a40c67d4467fb71869dc0a9d2f035dc18c0a148e3cad55a2616829908",
        "scene0606_01": "10e64a931b051c4e6e8fb4115777db1d0a85b18d6b57ecbc9ada24f59953683d",
        "scene0377_02": "104b27281b6d0ba1d6e7dd8f86deb91a066c12c6c30451a95d191c0874df63f1",
    }
    assert set(old_graw_hashes.values()).isdisjoint(
        s3a.EXPECTED_FORMAL_T05_SHA256.values()
    )
    monkeypatch.setattr(s3a, "EXPECTED_FORMAL_T05_SHA256", old_graw_hashes)
    with pytest.raises(s3a.S3aShadowError, match="formal T05 SHA-256 mismatch"):
        s3a._hash_formal_t05_predictions()


def test_runner_source_does_not_trust_boxer_native_ledger_or_name_graw_root():
    source = Path(s3a.__file__).read_text(encoding="utf-8")
    assert "native_identity_ledger_sha256" not in source
    assert "native_before_sha256.txt" not in source
    assert "native_after_sha256.txt" not in source
    assert "scannet_graw_e2_replay1_score05" not in source
