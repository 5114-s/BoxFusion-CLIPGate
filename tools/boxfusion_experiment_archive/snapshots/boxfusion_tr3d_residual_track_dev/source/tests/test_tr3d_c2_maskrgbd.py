from __future__ import annotations

import numpy as np
import pytest

from boxfusion.supplemental_proposals import SupplementalProposal
from boxfusion.tr3d_c2_maskrgbd_cache import (
    TR3DC2MaskRGBDCache,
    canonical_json,
    load_sidecar,
    sha256_bytes,
    validate_payload,
    write_sidecar,
)
from boxfusion.tr3d_c2_maskrgbd_observer import (
    C2Frame,
    C2MaskRGBDConfig,
    GATE_NAMES,
    observe_scene,
)


def _frame(frame_id: int, label: str = "chair") -> C2Frame:
    shape = (100, 100)
    depth = np.full(shape, 2.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[30:70, 30:70] = True
    proposal = SupplementalProposal(
        bbox=np.asarray([30, 30, 70, 70], dtype=np.float32),
        score=0.9,
        mask=mask,
        label=label,
    )
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 50.0, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return C2Frame(
        frame_id=frame_id,
        depth_meters=depth,
        intrinsics=intrinsic,
        depth_camera_to_world=np.eye(4, dtype=np.float64),
        proposals=(proposal,),
        cache_sha256=(str(frame_id % 10) * 64),
    )


def _observation():
    box = np.asarray([[0.0, 0.0, 2.0, 0.8, 0.8, 0.8, 0.0]], dtype=np.float32)
    return observe_scene(box, (_frame(0, "chair"), _frame(1, "sofa")), C2MaskRGBDConfig())


def test_two_view_mask_rgbd_confirmation_is_geometry_only():
    observation = _observation()
    assert observation.projected_view_count.tolist() == [2]
    assert observation.matched_view_count.tolist() == [2]
    assert observation.strong_view_count.tolist() == [2]
    gates = dict(zip(GATE_NAMES, observation.gate_mask[0].tolist()))
    assert gates["mask_any"]
    assert gates["mask1"]
    assert gates["mask2"]
    assert gates["mask2_depth"]
    assert not gates["mask3_strict"]
    assert observation.best_mask_label.tolist() == [["chair", "sofa"]]

    # A diagnostic label change cannot alter a geometry gate.
    box = np.asarray([[0.0, 0.0, 2.0, 0.8, 0.8, 0.8, 0.0]], dtype=np.float32)
    renamed = observe_scene(
        box, (_frame(0, "unknown-a"), _frame(1, "unknown-b")), C2MaskRGBDConfig()
    )
    np.testing.assert_array_equal(renamed.gate_mask, observation.gate_mask)
    np.testing.assert_allclose(renamed.evidence_score, observation.evidence_score)


def test_no_mask_is_fail_closed():
    frame = _frame(0)
    empty = C2Frame(
        frame_id=frame.frame_id,
        depth_meters=frame.depth_meters,
        intrinsics=frame.intrinsics,
        depth_camera_to_world=frame.depth_camera_to_world,
        proposals=(),
        cache_sha256=frame.cache_sha256,
    )
    box = np.asarray([[0.0, 0.0, 2.0, 0.8, 0.8, 0.8, 0.0]], dtype=np.float32)
    observation = observe_scene(box, (empty,), C2MaskRGBDConfig())
    assert observation.projected_view_count.tolist() == [1]
    assert observation.matched_view_count.tolist() == [0]
    assert not observation.gate_mask.any()


def test_c2_sidecar_round_trip_and_strong_gate_validation(tmp_path):
    config = C2MaskRGBDConfig()
    config_json = canonical_json(config.as_dict())
    observation = _observation()
    cache = TR3DC2MaskRGBDCache(
        scene_id="scene0000_00",
        prefix_id="p100",
        c1_sidecar_sha256="a" * 64,
        parent_cache_sha256="b" * 64,
        anchor_prediction_sha256="c" * 64,
        teacher_manifest_set_sha256="d" * 64,
        runtime_manifest_set_sha256="e" * 64,
        scene_frame_input_sha256="f" * 64,
        config_sha256=sha256_bytes(config_json.encode("utf-8")),
        code_sha256="0" * 64,
        config_json=config_json,
        source_c1_rows=np.asarray([0], dtype=np.int64),
        source_ranks=np.asarray([1], dtype=np.int32),
        proposal_ids=np.asarray([5], dtype=np.int64),
        parent_rows=np.asarray([2], dtype=np.int64),
        c1_track_scores=np.asarray([0.8], dtype=np.float32),
        frame_cache_sha256=np.asarray(["1" * 64, "2" * 64]),
        observation=observation,
        runtime_s=0.1,
    )
    target = tmp_path / "scene0000_00" / "p100.c2-maskrgbd.npz"
    digest = write_sidecar(target, cache)
    assert len(digest) == 64
    loaded = load_sidecar(target)
    np.testing.assert_array_equal(loaded.observation.gate_mask, observation.gate_mask)
    assert loaded.candidate_count == 1

    corrupted = cache.as_payload()
    corrupted["view_strong"] = np.zeros_like(corrupted["view_strong"])
    with pytest.raises(ValueError, match="strong-view decision mismatch"):
        validate_payload(corrupted)
