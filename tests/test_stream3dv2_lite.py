from __future__ import annotations

import math

import numpy as np
import pytest

from boxfusion.stream3dv2_lite import (
    LOCAL_WINDOW_KEYFRAMES,
    TrackView,
    build_track_geometry,
    continuous_evidence_score,
    policy_receipt,
    summarize_semantic_evidence,
)


_CORNER_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)


def _corners(
    center: tuple[float, float, float] = (0.1, 0.1, 1.05),
    extent: tuple[float, float, float] = (1.0, 1.0, 0.6),
) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    return center_array[None] + _CORNER_SIGNS * extent_array[None] * 0.5


def _voxel_block(
    base: tuple[int, int, int] = (0, 0, 20),
    shape: tuple[int, int, int] = (3, 3, 2),
) -> np.ndarray:
    grid = np.stack(
        np.meshgrid(
            *(np.arange(size, dtype=np.int64) for size in shape),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    keys = grid + np.asarray(base, dtype=np.int64)[None]
    # Stay strictly inside each 5 cm voxel so floating-point boundary effects
    # cannot change the synthetic topology.
    return (keys.astype(np.float64) + 0.25) * 0.05


def _view(
    source_id: str,
    ordinal: int,
    *,
    points: np.ndarray | None = None,
    box: np.ndarray | None = None,
    mask_confidence: float = 0.8,
    hb_confidence: float = 0.7,
) -> TrackView:
    return TrackView(
        source_id=source_id,
        frame_id=ordinal * 25,
        frame_ordinal=ordinal,
        mask_confidence=mask_confidence,
        hb_confidence=hb_confidence,
        points_world=_voxel_block() if points is None else points,
        hb_corners=_corners() if box is None else box,
    )


def _pmr_geometry():
    shared = _voxel_block((0, 0, 20), (3, 3, 2))
    left = _voxel_block((-2, 0, 20), (2, 3, 2))
    right = _voxel_block((3, 0, 20), (2, 3, 2))
    disconnected = _voxel_block((100, 0, 20), (3, 3, 2))
    past = _view("past", 7, points=np.concatenate((left, shared), axis=0))
    current = _view(
        "current",
        8,
        points=np.concatenate((shared, right, disconnected), axis=0),
    )
    return build_track_geometry((past, current))


def test_default_window_is_exactly_twenty_keyframe_ordinals():
    assert LOCAL_WINDOW_KEYFRAMES == 20
    stale = _view(
        "stale",
        0,
        points=_voxel_block((80, 0, 20)),
        box=_corners((4.1, 0.1, 1.05)),
    )
    boundary = _view("boundary", 1)
    current = _view("current", 20)

    geometry = build_track_geometry((current, stale, boundary))

    # At decision ordinal 20, the inclusive 20-keyframe window is [1, 20].
    assert geometry.decision_frame_ordinal == 20
    assert geometry.source_ids == ("boundary", "current")
    assert "stale" not in geometry.selected_source_ids
    assert float(np.max(geometry.refined_points[:, 0])) < 1.0

    wider = build_track_geometry(
        (current, stale, boundary), local_window_keyframes=21
    )
    assert wider.source_ids == ("stale", "boundary", "current")
    assert policy_receipt()["local_window_keyframes"] == 20


def test_scp_keeps_only_the_geometric_component_touching_latest_view():
    past = _view(
        "past",
        0,
        points=_voxel_block((0, 0, 20)),
        box=_corners((0.075, 0.075, 1.025), (0.5, 0.5, 0.5)),
    )
    distractor = _view(
        "distractor",
        1,
        points=_voxel_block((100, 0, 20)),
        box=_corners((5.075, 0.075, 1.025), (0.5, 0.5, 0.5)),
    )
    latest = _view(
        "latest",
        2,
        points=_voxel_block((3, 0, 20)),
        box=_corners((0.225, 0.075, 1.025), (0.5, 0.5, 0.5)),
    )

    geometry = build_track_geometry((distractor, latest, past))

    assert geometry.set_cover_fraction == pytest.approx(1.0)
    assert set(geometry.selected_source_ids) == {"past", "latest"}
    assert "distractor" not in geometry.selected_source_ids
    assert geometry.distinct_view_count == 2
    assert float(np.max(geometry.refined_points[:, 0])) < 1.0


def test_pmr_uses_multiview_seeds_and_drops_a_disconnected_current_blob():
    geometry = _pmr_geometry()

    assert set(geometry.selected_source_ids) == {"past", "current"}
    assert 0.0 < geometry.pmr_seed_fraction < 1.0
    assert 0.0 < geometry.pmr_retained_fraction < 1.0
    assert float(np.max(geometry.refined_points[:, 0])) < 1.0
    assert not geometry.refined_points.flags.writeable
    assert not geometry.corners.flags.writeable

    # A causal prefix is deterministic and independent of caller iteration
    # order; no state or later frame is consulted by the geometry builder.
    reverse_order = build_track_geometry(
        tuple(
            reversed(
                (
                    _view(
                        "past",
                        7,
                        points=np.concatenate(
                            (
                                _voxel_block((-2, 0, 20), (2, 3, 2)),
                                _voxel_block((0, 0, 20), (3, 3, 2)),
                            ),
                            axis=0,
                        ),
                    ),
                    _view(
                        "current",
                        8,
                        points=np.concatenate(
                            (
                                _voxel_block((0, 0, 20), (3, 3, 2)),
                                _voxel_block((3, 0, 20), (2, 3, 2)),
                                _voxel_block((100, 0, 20), (3, 3, 2)),
                            ),
                            axis=0,
                        ),
                    ),
                )
            )
        )
    )
    assert reverse_order.selected_source_ids == geometry.selected_source_ids
    np.testing.assert_array_equal(reverse_order.refined_points, geometry.refined_points)
    np.testing.assert_allclose(reverse_order.corners, geometry.corners, rtol=0.0, atol=0.0)
    assert reverse_order.preliminary_score == pytest.approx(geometry.preliminary_score)


def test_semantic_summary_uses_only_matched_views_and_reports_consensus():
    receipt = {
        "selected_view_count": 4,
        "views": [
            {
                "matched": True,
                "strong": True,
                "sam3_label": "chair",
                "sam3_score": 0.9,
                "mask_containment": 0.8,
                "box_coverage": 0.7,
                "evidence_score": 0.6,
            },
            {
                "matched": True,
                "strong": False,
                "sam3_label": "chair",
                "sam3_score": 0.7,
                "mask_containment": 0.6,
                "box_coverage": 0.5,
                "evidence_score": 0.4,
            },
            {
                "matched": True,
                "strong": True,
                "sam3_label": "table",
                "sam3_score": 0.5,
                "mask_containment": 0.4,
                "box_coverage": 0.3,
                "evidence_score": 0.2,
            },
            {
                "matched": False,
                "strong": True,
                "sam3_label": "ignored",
                "sam3_score": 1.0,
                "mask_containment": 1.0,
                "box_coverage": 1.0,
                "evidence_score": 1.0,
            },
            "non-mapping rows are ignored",
        ],
    }

    summary = summarize_semantic_evidence(receipt)

    assert summary["selected_view_count"] == 4
    assert summary["matched_view_count"] == 3
    assert summary["strong_view_count"] == 2
    assert summary["dominant_label"] == "chair"
    assert summary["dominant_label_votes"] == 2
    assert summary["label_consistency"] == pytest.approx(2.0 / 3.0)
    assert summary["median_sam3_score"] == pytest.approx(0.7)
    assert summary["median_mask_containment"] == pytest.approx(0.6)
    assert summary["median_box_coverage"] == pytest.approx(0.5)
    assert summary["median_evidence_score"] == pytest.approx(0.4)
    assert 0.0 < summary["semantic_quality"] < 1.0


def test_missing_semantics_is_explicitly_omitted_not_counted_as_zero():
    geometry = _pmr_geometry()
    empty_summary = summarize_semantic_evidence({"views": []})

    assert empty_summary == {
        "selected_view_count": 0,
        "matched_view_count": 0,
        "strong_view_count": 0,
        "dominant_label": None,
        "dominant_label_votes": 0,
        "label_consistency": 0.0,
        "median_sam3_score": 0.0,
        "median_mask_containment": 0.0,
        "median_box_coverage": 0.0,
        "median_evidence_score": 0.0,
        "semantic_quality": 0.02,
    }
    assert continuous_evidence_score(geometry, None) == pytest.approx(
        geometry.preliminary_score
    )
    assert continuous_evidence_score(geometry, empty_summary) < geometry.preliminary_score


def test_continuous_score_has_exact_semantic_and_duplication_risk_response():
    geometry = _pmr_geometry()
    semantic = {"semantic_quality": 0.81}

    without_risk = continuous_evidence_score(geometry, semantic)
    with_risk = continuous_evidence_score(
        geometry, semantic, duplication_risk=0.36
    )

    assert without_risk == pytest.approx(
        math.sqrt(geometry.preliminary_score * 0.81)
    )
    assert with_risk == pytest.approx(without_risk * 0.8)
    assert 0.0 < with_risk < without_risk <= 1.0
    assert continuous_evidence_score(
        geometry, semantic, duplication_risk=-2.0
    ) == pytest.approx(without_risk)
    assert continuous_evidence_score(
        geometry, semantic, duplication_risk=2.0
    ) == pytest.approx(without_risk * 0.01)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("source_id", ""),
        ("frame_id", -1),
        ("frame_ordinal", -1),
        ("mask_confidence", -0.01),
        ("mask_confidence", float("nan")),
        ("hb_confidence", 1.01),
        ("points_world", np.empty((0, 3), dtype=np.float64)),
        ("points_world", np.asarray([[0.0, np.inf, 1.0]], dtype=np.float64)),
        ("hb_corners", np.zeros((8, 3), dtype=np.float64)),
        ("hb_corners", np.zeros((7, 3), dtype=np.float64)),
    ],
)
def test_track_view_rejects_invalid_inputs(field, invalid):
    kwargs = {
        "source_id": "valid",
        "frame_id": 0,
        "frame_ordinal": 0,
        "mask_confidence": 0.8,
        "hb_confidence": 0.7,
        "points_world": _voxel_block(),
        "hb_corners": _corners(),
    }
    kwargs[field] = invalid

    with pytest.raises(ValueError):
        TrackView(**kwargs)


def test_geometry_and_semantic_scoring_reject_invalid_inputs():
    with pytest.raises(ValueError, match="at least one view"):
        build_track_geometry(())

    duplicate_a = _view("duplicate", 0)
    duplicate_b = _view("duplicate", 1)
    with pytest.raises(ValueError, match="source identities must be unique"):
        build_track_geometry((duplicate_a, duplicate_b))

    with pytest.raises(ValueError, match="quality terms must be finite"):
        summarize_semantic_evidence(
            {
                "views": [
                    {
                        "matched": True,
                        "strong": True,
                        "sam3_label": "chair",
                        "sam3_score": float("nan"),
                    }
                ]
            }
        )

    geometry = _pmr_geometry()
    with pytest.raises(ValueError, match="quality terms must be finite"):
        continuous_evidence_score(
            geometry, {"semantic_quality": float("nan")}
        )
