import json

import numpy as np
import pytest

from tools.merge_scannet_target_first_sam2_masklift_paper100 import (
    EXPECTED_ARRAYS,
    OUTPUT_JSON,
    OUTPUT_NPZ,
    SCHEMA,
    SAM2MaskLiftMergeError,
    _deterministic_npz,
    _hash_array,
    _parser,
    merge_target_first_sam2_masklift_paper100,
)


def _arrays(point_count, source_start):
    proposal_count = 3
    track_count = 1
    values = {
        "target_group_names": np.asarray(["chair"], dtype="<U32"),
        "proposal_scene_index": np.zeros(proposal_count, dtype=np.int16),
        "proposal_frame_id": np.asarray([0, 25, 50], dtype=np.int64),
        "proposal_source_row": np.arange(source_start, source_start + 3, dtype=np.int32),
        "proposal_source_instance_id": np.arange(3, dtype=np.int32),
        "proposal_semantic_id": np.ones(3, dtype=np.int32),
        "proposal_target_group_index": np.zeros(3, dtype=np.int8),
        "proposal_score": np.full(3, 0.8, dtype=np.float32),
        "proposal_prompt_box_xyxy": np.ones((3, 4), dtype=np.float32),
        "proposal_raw_center_world": np.ones((3, 3), dtype=np.float32),
        "proposal_raw_quaternion_wxyz": np.ones((3, 4), dtype=np.float32),
        "proposal_raw_extent_xyz": np.ones((3, 3), dtype=np.float32),
        "proposal_lift_accepted": np.ones(3, dtype=bool),
        "proposal_abstention_code": np.zeros(3, dtype=np.int8),
        "proposal_predicted_iou": np.full(3, 0.9, dtype=np.float32),
        "proposal_retained_point_count": np.full(3, 20, dtype=np.int32),
        "proposal_points_sha256": np.asarray(["a" * 64] * 3, dtype="<U64"),
        "proposal_lift_center_world": np.ones((3, 3), dtype=np.float32),
        "proposal_lift_extent_xyz": np.ones((3, 3), dtype=np.float32),
        "track_scene_index": np.zeros(track_count, dtype=np.int16),
        "track_target_group_index": np.zeros(track_count, dtype=np.int8),
        "track_semantic_id": np.ones(track_count, dtype=np.int32),
        "track_group_track_id": np.asarray([7], dtype=np.int32),
        "track_confirmation_frame_id": np.asarray([50], dtype=np.int64),
        "track_evidence_global_rows": np.asarray([[0, 1, 2]], dtype=np.int64),
        "track_fused_point_offsets": np.asarray([0, point_count], dtype=np.int64),
        "track_fused_points_world": np.arange(point_count * 3, dtype=np.float32).reshape(-1, 3),
        "track_fused_aabb_corners": np.ones((1, 8, 3), dtype=np.float32),
        "track_fused_obb_corners": np.ones((1, 8, 3), dtype=np.float32),
        "track_pre_novelty_pass": np.ones(1, dtype=bool),
        "track_native_novelty_pass": np.ones(1, dtype=bool),
        "track_accepted_shadow": np.ones(1, dtype=bool),
    }
    assert set(values) == EXPECTED_ARRAYS
    return values


def _write_shard(tmp_path, name, scene, point_count, source_start, device):
    root = tmp_path / name
    root.mkdir()
    arrays = _arrays(point_count, source_start)
    _deterministic_npz(root / OUTPUT_NPZ, arrays)
    source_rows = arrays["proposal_source_row"].tolist()
    receipt = {
        "track_id": 7,
        "group_track_id": 7,
        "evidence_source_rows": source_rows,
        "evidence_global_rows": [0, 1, 2],
        "accepted": True,
        "decision": "accepted_shadow",
    }
    track = {
        "scene_index": 0,
        "scene_id": scene,
        "group_track_id": 7,
        "evidence_source_rows": source_rows,
        "evidence_global_rows": [0, 1, 2],
        "decision": "accepted_shadow",
    }
    manifest = {
        "schema": SCHEMA,
        "mode": "shadow",
        "scene_count": 1,
        "scene_order": [scene],
        "target_prompt_count": 3,
        "target_prompt_frame_count": 3,
        "top_k_per_frame": 4,
        "raw_min_score": 0.5,
        "selection_source": "all_raw_rows_not_old_receipt_membership",
        "output_inert": True,
        "birth": False,
        "native_mutation_applied": False,
        "gt_access": False,
        "evaluator_access": False,
        "annotation_input_surface": False,
        "annotation_path_argument": False,
        "training": False,
        "target_dataset_training": False,
        "online_learning": False,
        "external_pretraining_frozen": True,
        "past_only_tracking": True,
        "past_only_confirmation": True,
        "native_clip_unchanged": True,
        "old_receipt_membership_consumed": False,
        "old_receipt_decisions_consumed": False,
        "exact_raw_to_owl_key": "(time_ns,instance)",
        "target_alias_matching": "normalized_exact_lookup_only",
        "routing_policy": {"frozen": True},
        "coordinate_frame": "scannet_world",
        "checkpoint": {"sha256": "c" * 64},
        "runner_source": {"sha256": "d" * 64},
        "receipt_scene_ledger": {"sha256": "e" * 64},
        "scene_list": {"sha256": "f" * 64},
        "inputs": {scene: {"native_prediction_sha256": "1" * 64}},
        "scenes": [{"scene_id": scene, "receipts": [receipt]}],
        "scene_summaries": {scene: {"prompt_count": 3, "receipt_count": 1}},
        "lifted_row_count": 3,
        "accepted_lifted_row_count": 3,
        "receipt_count": 1,
        "pre_novelty_pass_count": 1,
        "accepted_shadow_count": 1,
        "decision_counts": {"accepted_shadow": 1},
        "tracks": [track],
        "runtime": {
            "mask_engine": "FrozenSAM2BoxPromptProvider",
            "sam2_checkpoint_sha256": "c" * 64,
            "sam2_device": device,
            "measured_frame_count": 3,
            "provider_sum_seconds": 0.3,
            "provider_mean_ms": 100.0,
            "provider_p50_ms": 90.0,
            "provider_p95_ms": 120.0,
            "incremental_total_mean_ms": 110.0,
            "incremental_total_p50_ms": 100.0,
            "incremental_total_p95_ms": 130.0,
            "incremental_runtime_gate_ms": 200.0,
            "incremental_runtime_gate_pass": True,
        },
        "npz_file": OUTPUT_NPZ,
        "npz_arrays": {
            key: {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": _hash_array(value)}
            for key, value in arrays.items()
        },
    }
    (root / OUTPUT_JSON).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return root


def test_merge_rebases_official_scene_and_global_rows(tmp_path):
    scenes = ["scene0000_00", "scene0001_00"]
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    first = _write_shard(tmp_path, "first", scenes[0], 2, 10, "cuda:0")
    second = _write_shard(tmp_path, "second", scenes[1], 3, 20, "cuda:1")
    output = tmp_path / "merged"
    manifest = merge_target_first_sam2_masklift_paper100(
        shard_roots=[second, first],
        scene_list=scene_list,
        output_root=output,
        expected_scene_count=2,
    )
    assert manifest["scene_order"] == scenes
    assert manifest["receipt_count"] == 2
    assert manifest["scenes"][0]["receipts"][0]["evidence_global_rows"] == [0, 1, 2]
    assert manifest["scenes"][1]["receipts"][0]["evidence_global_rows"] == [3, 4, 5]
    assert manifest["runtime"]["sam2_devices"] == ["cuda:1", "cuda:0"]
    assert (output / OUTPUT_JSON).is_file() and (output / OUTPUT_NPZ).is_file()
    with np.load(output / OUTPUT_NPZ, allow_pickle=False) as merged:
        np.testing.assert_array_equal(merged["proposal_scene_index"], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(merged["track_scene_index"], [0, 1])
        np.testing.assert_array_equal(merged["track_evidence_global_rows"], [[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(merged["track_fused_point_offsets"], [0, 2, 5])
    with pytest.raises(SAM2MaskLiftMergeError, match="overwrite"):
        merge_target_first_sam2_masklift_paper100(
            shard_roots=[first, second],
            scene_list=scene_list,
            output_root=output,
            expected_scene_count=2,
        )


def test_merge_rejects_overlap_existing_output_and_gt_cli(tmp_path):
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    shard = _write_shard(tmp_path, "shard", scene, 1, 0, "cuda:0")
    duplicate = _write_shard(tmp_path, "duplicate", scene, 1, 0, "cuda:1")
    with pytest.raises(SAM2MaskLiftMergeError, match="overlap"):
        merge_target_first_sam2_masklift_paper100(
            shard_roots=[shard, duplicate],
            scene_list=scene_list,
            output_root=tmp_path / "out",
            expected_scene_count=1,
        )
    destinations = {action.dest for action in _parser()._actions}
    assert not any("gt" in name or "eval" in name or "annot" in name for name in destinations)
