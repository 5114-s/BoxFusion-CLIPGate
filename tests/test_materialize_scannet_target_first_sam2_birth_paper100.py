import json
import pickle

import numpy as np
import pytest

from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (
    BirthMaterializationError,
)
from tools.materialize_scannet_target_first_sam2_birth_paper100 import (
    MANIFEST_NAME,
    SAM2_MASKLIFT_MANIFEST_NAME,
    SAM2_MASKLIFT_SCHEMA,
    SCHEMA,
    _parser,
    materialize_scannet_target_first_sam2_birth_paper100,
)


def _corners(center):
    signs = np.asarray(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float32,
    )
    return signs * 0.5 + np.asarray(center, dtype=np.float32)


def _fixture(tmp_path, *, schema=SAM2_MASKLIFT_SCHEMA):
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    native = (7, _corners((20, 20, 20)), np.float32(1.0))
    with (baseline / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[native]], handle, protocol=4)
    receipt = {
        "track_id": 1,
        "confirmation_frame_id": 50,
        "evidence_frame_ids": [0, 25, 50],
        "evidence_source_rows": [0, 1, 2],
        "evidence_scores": [0.8, 0.8, 0.8],
        "raw_mean_score": 0.8,
        "median_pairwise_mask_aabb_iou": 0.6,
        "max_pairwise_mask_center_distance_m": 0.1,
        "first_last_frame_span": 50,
        "max_camera_baseline_m": 0.2,
        "max_view_ray_span_deg": 10.0,
        "supported_voxel_count": 40,
        "view_supported_voxel_counts": [12, 13, 14],
        "fused_obb": {
            "extent_xyz": [1.0, 1.0, 1.0],
            "corners_world": _corners((0, 0, 0)).tolist(),
        },
        "fused_center_to_raw_medoid_m": 0.1,
    }
    sidecar_dir = tmp_path / "shadow"
    sidecar_dir.mkdir()
    sidecar = sidecar_dir / SAM2_MASKLIFT_MANIFEST_NAME
    sidecar.write_text(
        json.dumps(
            {
                "schema": schema,
                "contracts": {"gt_access": False, "evaluator_access": False},
                "past_only_confirmation": True,
                "receipt_count": 1,
                "scenes": {scene: {"receipts": [receipt]}},
            }
        ),
        encoding="utf-8",
    )
    return scene, scene_list, baseline, sidecar_dir


def test_sam2_adapter_materializes_shared_r15_policy(tmp_path):
    scene, scene_list, baseline, sidecar_dir = _fixture(tmp_path)
    output = tmp_path / "active"
    manifest = materialize_scannet_target_first_sam2_birth_paper100(
        scene_list=scene_list,
        baseline_root=baseline,
        masklift_sidecar=sidecar_dir,
        output_root=output,
        expected_scene_count=1,
    )
    assert manifest["schema"] == SCHEMA
    assert manifest["birth_count"] == 1
    assert manifest["frozen_policy"]["receipt_admission_name"] == "R15"
    assert manifest["inputs"]["materializer_adapter_source"].endswith(
        "materialize_scannet_target_first_sam2_birth_paper100.py"
    )
    assert len(manifest["inputs"]["materializer_adapter_source_sha256"]) == 64
    assert (output / MANIFEST_NAME).is_file()
    with (output / f"{scene}_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    assert len(rows) == 2
    assert rows[1][0] == 0 and rows[1][2] == 1.0


def test_sam2_adapter_rejects_nonexact_schema_and_has_no_gt_cli(tmp_path):
    _scene, scene_list, baseline, sidecar_dir = _fixture(
        tmp_path, schema="boxfusion.scannet_target_first_sam2_masklift_paper100.v2"
    )
    with pytest.raises(BirthMaterializationError, match="unsupported"):
        materialize_scannet_target_first_sam2_birth_paper100(
            scene_list=scene_list,
            baseline_root=baseline,
            masklift_sidecar=sidecar_dir,
            output_root=tmp_path / "out",
            expected_scene_count=1,
            plan_only=True,
        )
    destinations = {action.dest for action in _parser()._actions}
    assert not any("gt" in name or "eval" in name or "annot" in name for name in destinations)
