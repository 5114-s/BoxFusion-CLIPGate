from types import MappingProxyType

import numpy as np

from boxfusion.graw_fragments import (
    PreparedRawKeyframe,
    RawFragmentCoverage,
    RawProposalDiagnostic,
    RawViewFragment,
)
from boxfusion.graw_shadow import GrawShadow, graw_result_to_dict
from boxfusion.observer_track_registry import IdentityResolution


def _fragment(proposal_id, frame_id, start=0, score=0.9):
    voxels = np.asarray([[start + index, 0, 0] for index in range(20)], dtype=np.int64)
    coverage = RawFragmentCoverage(
        sampled_rays=20,
        usable_rays=20,
        unique_voxels=20,
        output_voxels=20,
        valid_depth_ratio=1.0,
    )
    return RawViewFragment(
        proposal_id=proposal_id,
        frame_id=frame_id,
        score=score,
        crop_xyxy_depth=np.asarray([0, 0, 4, 4], dtype=np.float32),
        depth_shape=(8, 8),
        proposal_to_depth_affine=np.eye(3),
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
        voxel_keys=voxels,
        coverage=coverage,
    )


def _batch(frame_id, fragments):
    diagnostics = tuple(
        RawProposalDiagnostic(
            proposal_id=view.proposal_id,
            selected=True,
            reason=None,
            coverage=view.coverage,
            elapsed_ms=0.0,
            fragment=view,
        )
        for view in fragments
    )
    ids = tuple(view.proposal_id for view in fragments)
    return PreparedRawKeyframe("scene", frame_id, ids, ids, diagnostics, 0.0)


def _resolution(frame_id, proposal_ids, proposal_tracks, active, aliases=None):
    return IdentityResolution(
        frame_id=frame_id,
        proposal_ids=tuple(proposal_ids),
        proposal_track_ids=tuple(proposal_tracks),
        active_track_ids=tuple(active),
        track_aliases=MappingProxyType(dict(aliases or {})),
    )


def test_shadow_queries_begin_past_then_commits_native_new_track():
    shadow = GrawShadow()
    token0 = shadow.begin_keyframe(0)
    result0 = shadow.finish_keyframe(
        token0,
        batch=_batch(0, [_fragment(0, 0)]),
        resolution=_resolution(0, [0], [0], [0]),
    )
    assert result0.associations == ()
    assert result0.memory_track_ids == (0,)

    token1 = shadow.begin_keyframe(25)
    result1 = shadow.finish_keyframe(
        token1,
        batch=_batch(25, [_fragment(1, 25)]),
        resolution=_resolution(25, [1], [1], [0, 1]),
    )
    assert [(item.proposal_id, item.native_track_id, item.past_track_id) for item in result1.associations] == [(1, 1, 0)]
    assert result1.memory_track_ids == (0, 1)


def test_native_matched_track_is_reserved_and_not_counterfactually_reused():
    shadow = GrawShadow()
    token0 = shadow.begin_keyframe(0)
    shadow.finish_keyframe(
        token0,
        batch=_batch(0, [_fragment(0, 0)]),
        resolution=_resolution(0, [0], [0], [0]),
    )
    token1 = shadow.begin_keyframe(25)
    result = shadow.finish_keyframe(
        token1,
        batch=_batch(25, [_fragment(1, 25)]),
        resolution=_resolution(25, [1], [0], [0], aliases={1: 0}),
    )
    assert result.reserved_past_track_ids == (0,)
    assert result.eligible_past_track_ids == ()
    assert result.candidate_proposal_ids == ()
    assert result.associations == ()


def test_current_frame_tracks_cannot_match_each_other():
    shadow = GrawShadow()
    token = shadow.begin_keyframe(0)
    result = shadow.finish_keyframe(
        token,
        batch=_batch(0, [_fragment(0, 0), _fragment(1, 0)]),
        resolution=_resolution(0, [0, 1], [0, 1], [0, 1]),
    )
    assert result.begin_track_ids == ()
    assert result.associations == ()


def test_abort_does_not_commit_memory():
    shadow = GrawShadow()
    token = shadow.begin_keyframe(0)
    shadow.abort_keyframe(token)
    assert shadow.memory_track_ids == ()


def test_inputs_are_not_mutated_and_voxel_memory_is_read_only():
    shadow = GrawShadow()
    view = _fragment(0, 0, start=-20)
    before = view.voxel_keys.copy()
    token = shadow.begin_keyframe(0)
    shadow.finish_keyframe(
        token,
        batch=_batch(0, [view]),
        resolution=_resolution(0, [0], [0], [0]),
    )
    np.testing.assert_array_equal(view.voxel_keys, before)
    assert not view.voxel_keys.flags.writeable


def test_optional_pair_evidence_reuses_the_match_batch_without_changing_v1_output():
    shadow = GrawShadow()
    token0 = shadow.begin_keyframe(0)
    first = shadow.finish_keyframe(
        token0,
        batch=_batch(0, [_fragment(0, 0)]),
        resolution=_resolution(0, [0], [0], [0]),
    )
    assert first.pair_evidence is None
    assert "pair_evidence" not in graw_result_to_dict(first)

    token1 = shadow.begin_keyframe(25)
    observed = shadow.finish_keyframe(
        token1,
        batch=_batch(25, [_fragment(1, 25)]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        collect_pair_evidence=True,
    )
    assert observed.pair_evidence is not None
    assert observed.pair_evidence.diagnostics.fail_open is False
    assert len(observed.pair_evidence.proposals) == 1
    row = observed.pair_evidence.proposals[0]
    assert row.proposal_id == 1
    assert [(item.track_id, item.intersection) for item in row.candidates] == [
        (0, 20)
    ]
    assert [item.past_track_id for item in observed.associations] == [0]
    # The evidence-only extension is intentionally absent from the legacy v1
    # artifact, including when explicitly collected for a downstream observer.
    assert "pair_evidence" not in graw_result_to_dict(observed)
