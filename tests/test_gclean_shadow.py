import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from boxfusion.gclean_shadow import (
    FRAGMENT_SOURCE,
    GcleanShadow,
    gclean_result_to_dict,
    write_gclean_shadow_diagnostics,
)
from boxfusion.observer_track_registry import IdentityResolution
from boxfusion.smov_fragments import (
    FragmentCoverage,
    PreparedKeyframe,
    ProposalDiagnostic,
    ViewFragment,
)


def _fragment(proposal_id, frame_id, start=-20, score=0.9):
    keys = np.asarray(
        [[start + index, -2, 3] for index in range(20)], dtype=np.int64
    )
    # Deliberately do not derive matcher keys from these centroids.  The
    # Gclean contract consumes the extraction-time direct integer keys.
    points = (keys.astype(np.float64) + np.asarray([0.25, 0.5, 0.75])) * 0.05
    coverage = FragmentCoverage(
        effective_stride=4,
        sampled_rays=32,
        usable_rays=24,
        component_pixels=20,
        unique_voxels=20,
        output_voxels=20,
        output_points=20,
        valid_depth_ratio=0.75,
        component_ratio=0.625,
    )
    return ViewFragment(
        proposal_id=proposal_id,
        frame_id=frame_id,
        score=score,
        crop_xyxy_depth=np.asarray([0, 0, 8, 8], dtype=np.float32),
        depth_shape=(10, 10),
        proposal_to_depth_affine=np.eye(3),
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
        points_world=points,
        voxel_keys=keys,
        coverage=coverage,
    )


def _batch(frame_id, fragments, elapsed_ms=1.25):
    diagnostics = tuple(
        ProposalDiagnostic(
            proposal_id=view.proposal_id,
            selected=True,
            reason=None,
            coverage=view.coverage,
            elapsed_ms=0.1,
            fragment=view,
        )
        for view in fragments
    )
    return PreparedKeyframe(
        scene_id="scene",
        frame_id=frame_id,
        proposal_ids=tuple(view.proposal_id for view in fragments),
        diagnostics=diagnostics,
        elapsed_ms=elapsed_ms,
    )


def _resolution(frame_id, proposal_ids, proposal_tracks, active, aliases=None):
    return IdentityResolution(
        frame_id=frame_id,
        proposal_ids=tuple(proposal_ids),
        proposal_track_ids=tuple(proposal_tracks),
        active_track_ids=tuple(active),
        track_aliases=MappingProxyType(dict(aliases or {})),
    )


def _commit_first(shadow):
    batch = _batch(0, [_fragment(0, 0)])
    token = shadow.begin_keyframe(0)
    result = shadow.finish_keyframe(
        token,
        batch=batch,
        resolution=_resolution(0, [0], [0], [0]),
    )
    return batch, result


def test_clean_fragments_query_begin_past_then_commit_current():
    shadow = GcleanShadow()
    batch0, first = _commit_first(shadow)
    assert first.associations == ()
    assert first.memory_track_ids == (0,)

    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25)]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
    )
    assert [
        (item.proposal_id, item.native_track_id, item.past_track_id)
        for item in result.associations
    ] == [(1, 1, 0)]
    assert result.begin_track_ids == (0,)
    assert result.memory_track_ids == (0, 1)
    assert result.fragment_source == FRAGMENT_SOURCE
    assert result.total_observer_elapsed_ms == pytest.approx(
        result.smov_prepare_elapsed_ms
        + result.voxel_adapter_elapsed_ms
        + result.finish_elapsed_ms
    )
    assert batch0.diagnostics[0].fragment.voxel_keys[0, 0] == -20


def test_native_reserved_track_cannot_be_reused_counterfactually():
    shadow = GcleanShadow()
    _commit_first(shadow)
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25)]),
        resolution=_resolution(25, [1], [0], [0], aliases={1: 0}),
        reserved_past_track_ids=(0,),
    )
    assert result.reserved_past_track_ids == (0,)
    assert result.eligible_past_track_ids == ()
    assert result.candidate_proposal_ids == ()
    assert result.associations == ()


def test_bad_clean_voxel_fragment_abstains_without_poisoning_memory():
    shadow = GcleanShadow()
    _commit_first(shadow)
    good = _fragment(1, 25)
    bad = SimpleNamespace(
        proposal_id=good.proposal_id,
        frame_id=good.frame_id,
        score=good.score,
        crop_xyxy_depth=good.crop_xyxy_depth,
        depth_shape=good.depth_shape,
        proposal_to_depth_affine=good.proposal_to_depth_affine,
        intrinsics=good.intrinsics,
        camera_to_world=good.camera_to_world,
        points_world=good.points_world,
        voxel_keys=np.zeros((513, 3), dtype=np.int64),
        coverage=good.coverage,
    )
    batch = PreparedKeyframe(
        scene_id="scene",
        frame_id=25,
        proposal_ids=(1,),
        diagnostics=(
            ProposalDiagnostic(1, True, None, good.coverage, 0.0, bad),
        ),
        elapsed_ms=0.5,
    )
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=batch,
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
    )
    assert result.associations == ()
    assert result.candidate_proposal_ids == ()
    assert result.memory_track_ids == (0,)
    assert result.adapter_diagnostics["converted_fragments"] == 0
    assert result.adapter_diagnostics["failure_reasons"] == {"voxel_cap": 1}
    assert result.matcher_diagnostics["fail_open"] is False


def test_shadow_is_deterministic_and_does_not_mutate_clean_inputs():
    candidate = _fragment(1, 25)
    before_keys = candidate.voxel_keys.copy()
    before_points = candidate.points_world.copy()
    signatures = []
    for _ in range(2):
        shadow = GcleanShadow()
        _commit_first(shadow)
        token = shadow.begin_keyframe(25, active_track_ids=(0,))
        result = shadow.finish_keyframe(
            token,
            batch=_batch(25, [candidate]),
            resolution=_resolution(25, [1], [1], [0, 1]),
            unmatched_retained_proposal_ids=(1,),
        )
        signatures.append(
            (
                result.begin_track_ids,
                result.reserved_past_track_ids,
                result.eligible_past_track_ids,
                result.candidate_proposal_ids,
                result.candidate_native_track_ids,
                result.associations,
                dict(result.matcher_diagnostics),
                result.memory_track_ids,
            )
        )
    assert signatures[0] == signatures[1]
    np.testing.assert_array_equal(candidate.voxel_keys, before_keys)
    np.testing.assert_array_equal(candidate.points_world, before_points)
    assert not candidate.voxel_keys.flags.writeable
    assert not candidate.points_world.flags.writeable


def test_diagnostics_declare_shadow_source_caps_and_atomic_json(tmp_path):
    shadow = GcleanShadow()
    _, result = _commit_first(shadow)
    diagnostics = shadow.diagnostics()
    assert diagnostics["schema"] == "boxfusion.gclean_shadow.v1"
    assert diagnostics["mode"] == "shadow"
    assert diagnostics["fragment_source"] == FRAGMENT_SOURCE
    assert diagnostics["caps"] == {
        "max_proposals_per_keyframe": 64,
        "max_tracks": 1024,
        "max_views_per_track": 5,
        "max_voxels_per_view": 512,
        "max_union_voxels_per_track": 1024,
        "min_voxels": 16,
    }
    assert diagnostics["timing"]["total_observer"]["count"] == 1

    record = gclean_result_to_dict(result)
    destination = tmp_path / "scene.gclean_shadow.json"
    write_gclean_shadow_diagnostics(
        destination,
        scene_id="scene",
        results=(record,),
        summary=diagnostics,
        trace_valid=True,
    )
    payload = json.loads(destination.read_text())
    assert payload["schema"] == "boxfusion.gclean_shadow.v1"
    assert payload["mode"] == "shadow"
    assert payload["fragment_source"] == FRAGMENT_SOURCE
    assert payload["frames"][0]["fragment_source"] == FRAGMENT_SOURCE
    assert payload["summary"]["caps"]["max_tracks"] == 1024
    assert payload["summary"]["timing"]["total_observer"]["count"] == 1
    assert payload["trace_valid"] is True


def test_structural_misalignment_rejects_and_remains_abortable():
    shadow = GcleanShadow()
    fragment = _fragment(1, 0)
    batch = PreparedKeyframe(
        scene_id="scene",
        frame_id=0,
        proposal_ids=(999,),
        diagnostics=(
            ProposalDiagnostic(1, True, None, fragment.coverage, 0.0, fragment),
        ),
        elapsed_ms=0.0,
    )
    token = shadow.begin_keyframe(0)
    with pytest.raises(ValueError, match="diagnostic order"):
        shadow.finish_keyframe(
            token,
            batch=batch,
            resolution=_resolution(0, [999], [0], [0]),
        )
    assert shadow.pending
    shadow.abort_keyframe(token)
    assert not shadow.pending


def test_selected_proposal_hard_cap_is_enforced_independently_of_extractor():
    fragment = _fragment(0, 0)
    diagnostics = tuple(
        ProposalDiagnostic(
            proposal_id=index,
            selected=True,
            reason=None,
            coverage=fragment.coverage,
            elapsed_ms=0.0,
            fragment=None,
        )
        for index in range(65)
    )
    batch = PreparedKeyframe(
        scene_id="scene",
        frame_id=0,
        proposal_ids=tuple(range(65)),
        diagnostics=diagnostics,
        elapsed_ms=0.0,
    )
    shadow = GcleanShadow()
    token = shadow.begin_keyframe(0)
    with pytest.raises(ValueError, match="hard cap of 64"):
        shadow.finish_keyframe(
            token,
            batch=batch,
            resolution=_resolution(0, range(65), range(65), range(65)),
        )
    shadow.abort_keyframe(token)


def test_optional_pair_evidence_keeps_default_gclean_schema_unchanged():
    shadow = GcleanShadow()
    _, first = _commit_first(shadow)
    assert first.pair_evidence is None
    assert "pair_evidence" not in gclean_result_to_dict(first)

    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    observed = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25)]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
        collect_pair_evidence=True,
    )
    assert observed.pair_evidence is not None
    assert observed.pair_evidence.diagnostics.fail_open is False
    assert observed.pair_evidence.proposals[0].proposal_id == 1
    assert observed.pair_evidence.proposals[0].candidates[0].track_id == 0
    assert "pair_evidence" not in gclean_result_to_dict(observed)
