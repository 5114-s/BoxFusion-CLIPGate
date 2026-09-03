import json
from types import MappingProxyType

import numpy as np
import pytest

from boxfusion.observer_track_registry import IdentityResolution
from boxfusion.puf_gclean_shadow import (
    CANDIDATE_SOURCE,
    FRAGMENT_SOURCE,
    PufGcleanShadow,
    puf_gclean_result_to_dict,
    write_puf_gclean_shadow_diagnostics,
)
from boxfusion.smov_fragments import (
    FragmentCoverage,
    PreparedKeyframe,
    ProposalDiagnostic,
    ViewFragment,
)


def _fragment(proposal_id, frame_id, keys, score=0.9):
    voxel_keys = np.asarray(keys, dtype=np.int64)
    points = (
        voxel_keys.astype(np.float64) + np.asarray([0.25, 0.5, 0.75])
    ) * 0.05
    coverage = FragmentCoverage(
        effective_stride=4,
        sampled_rays=max(32, len(voxel_keys)),
        usable_rays=len(voxel_keys),
        component_pixels=len(voxel_keys),
        unique_voxels=len(voxel_keys),
        output_voxels=len(voxel_keys),
        output_points=len(voxel_keys),
        valid_depth_ratio=1.0,
        component_ratio=1.0,
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
        voxel_keys=voxel_keys,
        coverage=coverage,
    )


def _line(start=0, count=20, y=0, z=0):
    return [[start + index, y, z] for index in range(count)]


def _shared_line(shared, *, offset=100):
    return _line(0, shared) + _line(offset, 20 - shared)


def _batch(frame_id, fragments):
    diagnostics = tuple(
        ProposalDiagnostic(
            proposal_id=view.proposal_id,
            selected=True,
            reason=None,
            coverage=view.coverage,
            elapsed_ms=0.0,
            fragment=view,
        )
        for view in fragments
    )
    return PreparedKeyframe(
        scene_id="scene",
        frame_id=frame_id,
        proposal_ids=tuple(view.proposal_id for view in fragments),
        diagnostics=diagnostics,
        elapsed_ms=0.0,
    )


def _resolution(frame_id, proposal_ids, proposal_tracks, active, aliases=None):
    return IdentityResolution(
        frame_id=frame_id,
        proposal_ids=tuple(proposal_ids),
        proposal_track_ids=tuple(proposal_tracks),
        active_track_ids=tuple(active),
        track_aliases=MappingProxyType(dict(aliases or {})),
    )


def _seed(shadow, fragments, track_ids):
    token = shadow.begin_keyframe(0)
    return shadow.finish_keyframe(
        token,
        batch=_batch(0, fragments),
        resolution=_resolution(
            0,
            [item.proposal_id for item in fragments],
            track_ids,
            track_ids,
        ),
    )


def test_all_positive_candidates_are_normalized_and_paper_directive_is_shadow_only():
    shadow = PufGcleanShadow()
    identical = _line(-10, 20, y=-2, z=3)
    _seed(
        shadow,
        [_fragment(0, 0, identical), _fragment(1, 0, identical)],
        [0, 1],
    )

    current = _fragment(2, 25, identical)
    before = current.voxel_keys.copy()
    token = shadow.begin_keyframe(25, active_track_ids=(0, 1))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [current]),
        resolution=_resolution(25, [2], [2], [0, 1, 2]),
        unmatched_retained_proposal_ids=(2,),
    )

    assert result.fail_open is False
    assert result.birth_enabled is False
    assert result.candidate_proposal_ids == (2,)
    assert result.candidate_native_track_ids == (2,)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert [item.past_track_id for item in row.candidates] == [0, 1]
    assert sum(item.beta for item in row.candidates) + row.beta_null == pytest.approx(1.0)
    assert row.beta_null == pytest.approx(0.4 / 2.4)
    # Equal likelihoods use the stable track ID, while Gclean's mutual/margin
    # gate accepts neither.  This proves PUF sees all candidates rather than
    # only the Gclean accepted edge set.
    assert row.argmax_past_track_id == 0
    assert row.gclean_accepted_past_track_id is None
    assert result.gclean_associations == ()
    assert [(item.proposal_id, item.past_track_id) for item in result.directives] == [
        (2, 0)
    ]
    assert result.directives[0].birth_enabled is False
    assert result.directives[0].agrees_with_gclean is None
    assert result.associations == ()  # tied best-track margin is not active-safe
    assert row.normalization_error <= 1e-12
    assert result.puf_diagnostics["max_normalization_error"] <= 1e-12
    for candidate in row.candidates:
        assert candidate.centroid_distance_m == pytest.approx(
            candidate.centroid_distance_voxels * 0.05
        )
    np.testing.assert_array_equal(current.voxel_keys, before)
    assert not current.voxel_keys.flags.writeable


@pytest.mark.parametrize(
    ("shared", "expected_beta_null", "expect_directive"),
    [(8, 0.5, True), (7, 0.4 / 0.75, False)],
)
def test_paper_gate_includes_exact_half_and_has_no_extra_confidence_threshold(
    shared, expected_beta_null, expect_directive
):
    shadow = PufGcleanShadow()
    # A common voxel core keeps the pair in broad phase; the disjoint tail
    # controls PUF's asymmetric intersection/current-proposal likelihood.
    past_keys = _shared_line(shared, offset=100)
    current_keys = _line(0, 20)
    _seed(shadow, [_fragment(0, 0, past_keys)], [0])
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25, current_keys)]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
    )
    assert result.fail_open is False
    assert result.rows[0].beta_null == pytest.approx(expected_beta_null)
    assert result.rows[0].paper_gate is expect_directive
    assert bool(result.directives) is expect_directive
    assert result.associations == ()
    if expect_directive:
        # beta_track is only 0.5 here; a hidden 0.7 gate would incorrectly
        # suppress this literature-rule counterfactual.
        assert result.directives[0].beta_track == pytest.approx(0.5)


def test_reserved_native_match_is_excluded_before_evidence_and_probability():
    shadow = PufGcleanShadow()
    keys = _line(0, 20)
    _seed(shadow, [_fragment(0, 0, keys)], [0])
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25, keys)]),
        resolution=_resolution(25, [1], [0], [0], aliases={1: 0}),
        unmatched_retained_proposal_ids=(),
    )
    assert result.fail_open is False
    assert result.rows == ()
    assert result.directives == ()
    assert result.associations == ()
    assert result.gclean_result.reserved_past_track_ids == (0,)
    assert result.gclean_result.candidate_proposal_ids == ()


def test_null_only_explicit_proposal_row_abstains():
    shadow = PufGcleanShadow()
    _seed(shadow, [_fragment(0, 0, _line(0, 20))], [0])
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25, _line(200, 20))]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
    )
    assert result.fail_open is False
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.candidates == ()
    assert row.beta_null == 1.0
    assert row.paper_gate is False
    assert result.directives == ()
    assert result.associations == ()


def test_diagnostics_and_atomic_writer_preserve_independent_schema(tmp_path):
    shadow = PufGcleanShadow()
    _seed(shadow, [_fragment(0, 0, _line(-20, 20))], [0])
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25, _line(-20, 20))]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
    )
    record = puf_gclean_result_to_dict(result)
    assert record["schema"] == "boxfusion.puf_gclean_shadow.v1"
    assert record["candidate_source"] == CANDIDATE_SOURCE
    assert record["fragment_source"] == FRAGMENT_SOURCE
    assert record["birth_enabled"] is False
    assert record["candidate_proposal_ids"] == [1]
    assert record["candidate_native_track_ids"] == [1]
    assert [item["past_track_id"] for item in record["associations"]] == [0]
    assert result.evidence_elapsed_ms >= 0.0
    assert result.probability_elapsed_ms >= 0.0
    assert result.puf_elapsed_ms == pytest.approx(
        result.evidence_elapsed_ms + result.probability_elapsed_ms
    )
    assert result.total_observer_elapsed_ms == pytest.approx(
        result.gclean_total_observer_elapsed_ms
        + result.probability_elapsed_ms
    )
    assert "gclean_result" not in record
    summary = shadow.diagnostics()
    assert summary["lambda_null"] == 0.4
    assert summary["paper_gate"] == "beta_null<=0.5_then_stable_track_argmax"
    assert summary["fail_open"] is False
    assert summary["timing"]["evidence"]["count"] == 2
    assert summary["timing"]["probability"]["count"] == 2
    assert summary["timing"]["puf_incremental"]["count"] == 2

    destination = tmp_path / "scene.puf_gclean_shadow.json"
    write_puf_gclean_shadow_diagnostics(
        destination,
        scene_id="scene",
        results=(record,),
        summary=summary,
        trace_valid=True,
    )
    payload = json.loads(destination.read_text())
    assert payload["schema"] == "boxfusion.puf_gclean_shadow.v1"
    assert payload["frame_count"] == 1
    assert payload["birth_enabled"] is False
    assert payload["fail_open"] is False
    assert payload["summary"]["gclean"]["fragment_source"] == FRAGMENT_SOURCE
    assert payload["frames"][0]["rows"][0]["candidates"][0][
        "centroid_distance_m"
    ] == pytest.approx(
        payload["frames"][0]["rows"][0]["candidates"][0][
            "centroid_distance_voxels"
        ]
        * 0.05
    )


def test_puf_wrapper_exception_fails_open_after_causal_native_commit(monkeypatch):
    shadow = PufGcleanShadow()
    keys = _line(0, 20)
    _seed(shadow, [_fragment(0, 0, keys)], [0])

    def _explode(*args, **kwargs):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(
        "boxfusion.puf_gclean_shadow.compute_puf_lite", _explode
    )
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25, keys)]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
    )
    assert result.fail_open is True
    assert result.fail_open_code == "wrapper_exception:RuntimeError"
    assert result.rows == ()
    assert result.directives == ()
    assert result.associations == ()
    assert result.gclean_result.memory_track_ids == (0, 1)
    assert shadow.memory_track_ids == (0, 1)


def test_matcher_fail_open_suppresses_probability_rows_and_active_safe_output(
    monkeypatch,
):
    from boxfusion.group3d_lite import MatchResult
    from boxfusion.group3d_lite_oracle import Diagnostics

    shadow = PufGcleanShadow()
    keys = _line(0, 20)
    _seed(shadow, [_fragment(0, 0, keys)], [0])

    def _matcher_failure(*args, **kwargs):
        return MatchResult(
            (), Diagnostics(fail_open=True, code="synthetic_matcher_failure")
        )

    monkeypatch.setattr(
        "boxfusion.graw_shadow.match_prepared_proposals", _matcher_failure
    )
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(25, [_fragment(1, 25, keys)]),
        resolution=_resolution(25, [1], [1], [0, 1]),
        unmatched_retained_proposal_ids=(1,),
    )
    assert result.evidence_diagnostics["fail_open"] is False
    assert result.fail_open is True
    assert result.fail_open_code == "gclean_matcher:synthetic_matcher_failure"
    assert result.candidate_proposal_ids == (1,)
    assert result.candidate_native_track_ids == (1,)
    assert result.rows == ()
    assert result.directives == ()
    assert result.associations == ()
    assert shadow.diagnostics()["fail_open"] is True


def test_same_past_track_conflict_excludes_the_entire_active_safe_group():
    shadow = PufGcleanShadow()
    keys = _line(0, 20)
    _seed(shadow, [_fragment(0, 0, keys)], [0])
    token = shadow.begin_keyframe(25, active_track_ids=(0,))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(
            25,
            [_fragment(1, 25, keys), _fragment(2, 25, keys)],
        ),
        resolution=_resolution(25, [1, 2], [1, 2], [0, 1, 2]),
        unmatched_retained_proposal_ids=(1, 2),
    )
    assert [(item.proposal_id, item.past_track_id) for item in result.directives] == [
        (1, 0),
        (2, 0),
    ]
    assert all(item.margin > 0.0 for item in result.directives)
    assert result.same_track_conflict_groups == 1
    assert result.same_track_conflict_directives == 2
    assert result.associations == ()


def test_ambiguous_competitor_still_blocks_a_safe_same_track_claim():
    shadow = PufGcleanShadow()
    _seed(
        shadow,
        [
            _fragment(0, 0, _line(0, 20)),
            _fragment(1, 0, _line(20, 20)),
        ],
        [0, 1],
    )
    token = shadow.begin_keyframe(25, active_track_ids=(0, 1))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(
            25,
            [
                _fragment(2, 25, _line(0, 20)),
                _fragment(3, 25, _line(10, 20)),
            ],
        ),
        resolution=_resolution(25, [2, 3], [2, 3], [0, 1, 2, 3]),
        unmatched_retained_proposal_ids=(2, 3),
    )
    assert [(item.proposal_id, item.past_track_id) for item in result.directives] == [
        (2, 0),
        (3, 0),
    ]
    margins = {item.proposal_id: item.margin for item in result.directives}
    assert margins[2] > 0.0
    assert margins[3] == pytest.approx(0.0)
    # Conflict groups are formed before the positive-margin filter.
    assert result.associations == ()


def test_distinct_tracks_are_active_safe_and_order_invariant():
    def _run(reverse):
        shadow = PufGcleanShadow()
        _seed(
            shadow,
            [
                _fragment(0, 0, _line(0, 20)),
                _fragment(1, 0, _line(100, 20)),
            ],
            [0, 1],
        )
        current = [
            _fragment(2, 25, _line(0, 20)),
            _fragment(3, 25, _line(100, 20)),
        ]
        proposal_ids = [2, 3]
        native_ids = [2, 3]
        if reverse:
            current.reverse()
            proposal_ids.reverse()
            native_ids.reverse()
        token = shadow.begin_keyframe(25, active_track_ids=(0, 1))
        result = shadow.finish_keyframe(
            token,
            batch=_batch(25, current),
            resolution=_resolution(
                25, proposal_ids, native_ids, [0, 1, 2, 3]
            ),
            unmatched_retained_proposal_ids=(2, 3),
        )
        return tuple(
            (item.proposal_id, item.native_track_id, item.past_track_id)
            for item in result.associations
        )

    assert _run(False) == ((2, 2, 0), (3, 3, 1))
    assert _run(True) == ((2, 2, 0), (3, 3, 1))


def test_conflict_group_does_not_remove_an_independent_safe_association():
    shadow = PufGcleanShadow()
    _seed(
        shadow,
        [
            _fragment(0, 0, _line(0, 20)),
            _fragment(1, 0, _line(100, 20)),
        ],
        [0, 1],
    )
    token = shadow.begin_keyframe(25, active_track_ids=(0, 1))
    result = shadow.finish_keyframe(
        token,
        batch=_batch(
            25,
            [
                _fragment(2, 25, _line(0, 20)),
                _fragment(3, 25, _line(0, 20)),
                _fragment(4, 25, _line(100, 20)),
            ],
        ),
        resolution=_resolution(
            25, [2, 3, 4], [2, 3, 4], [0, 1, 2, 3, 4]
        ),
        unmatched_retained_proposal_ids=(2, 3, 4),
    )
    assert [(item.proposal_id, item.past_track_id) for item in result.directives] == [
        (2, 0),
        (3, 0),
        (4, 1),
    ]
    assert [
        (item.proposal_id, item.native_track_id, item.past_track_id)
        for item in result.associations
    ] == [(4, 4, 1)]


def test_exact_token_is_required_and_abort_does_not_commit():
    shadow = PufGcleanShadow()
    token = shadow.begin_keyframe(0)
    with pytest.raises(RuntimeError, match="exact pending"):
        shadow.abort_keyframe(object())
    shadow.abort_keyframe(token)
    assert shadow.pending is False
    assert shadow.memory_track_ids == ()
