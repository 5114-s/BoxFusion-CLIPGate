import copy
import importlib.util
import itertools
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "boxfusion"
    / "fragment_stitch.py"
)
SPEC = importlib.util.spec_from_file_location(
    "boxfusion_fragment_stitch", SOURCE
)
stitch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stitch
SPEC.loader.exec_module(stitch)


def snapshot(
    track_id,
    *,
    frame,
    box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    label="chair",
    state="active",
    views=1,
    nodes=1,
    edges=0,
    points=100,
    score=0.90,
    confirmed=False,
):
    return {
        "track_id": track_id,
        "lifecycle_state": state,
        "event_frame": frame,
        "box": np.asarray(box, dtype=np.float32),
        "view_count": views,
        "node_count": nodes,
        "edge_count": edges,
        "memory_geometry_points": points,
        "mean_detector_score": score,
        "label": label,
        "graph_confirmed": confirmed,
        # Real online snapshots contain unrelated diagnostics.  Extra input
        # keys must not affect stitching.
        "rejections": {"geometry": 2},
    }


def enabled(**overrides):
    return {"enabled": True, **overrides}


def test_default_config_is_safe_and_matches_observer_contract():
    config = stitch.resolve_fragment_stitch_config()

    assert config == {
        "enabled": False,
        "minimum_pair_iou": pytest.approx(0.40),
        "minimum_pair_containment": pytest.approx(0.60),
        "maximum_center_distance": pytest.approx(0.25),
        "minimum_max_detector_score": pytest.approx(0.85),
        "minimum_mean_detector_score": pytest.approx(0.70),
        "minimum_event_frame_separation": 5,
        "require_live_member": True,
    }


def test_config_is_strict_and_detached():
    with pytest.raises(ValueError, match="Unknown"):
        stitch.resolve_fragment_stitch_config({"minimum_pair_io": 0.4})
    with pytest.raises(ValueError, match="mapping"):
        stitch.resolve_fragment_stitch_config([])

    source = {"minimum_pair_iou": 0.5}
    resolved = stitch.resolve_fragment_stitch_config(source)
    resolved["minimum_pair_iou"] = 0.2
    assert source == {"minimum_pair_iou": 0.5}


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": 1},
        {"require_live_member": "yes"},
        {"minimum_pair_iou": -0.01},
        {"minimum_pair_iou": 1.01},
        {"minimum_pair_containment": np.nan},
        {"maximum_center_distance": -0.01},
        {"maximum_center_distance": np.inf},
        {"minimum_max_detector_score": 1.01},
        {"minimum_mean_detector_score": -0.01},
        {"minimum_event_frame_separation": 0},
        {"minimum_event_frame_separation": 1.5},
        {"minimum_event_frame_separation": True},
    ],
)
def test_invalid_config_fails_fast(override):
    with pytest.raises(ValueError):
        stitch.resolve_fragment_stitch_config(override)


def test_disabled_valid_call_returns_immutable_empty_tuple():
    rows = [snapshot(1, frame=1), snapshot(2, frame=10)]
    assert stitch.build_fragment_stitch_candidates(rows) == ()


def test_iou_branch_builds_candidate_and_normalizes_label():
    rows = [
        snapshot(7, frame=1, label=" Dining_Table ", score=0.90),
        snapshot(
            3,
            frame=10,
            label="dining-table",
            box=(0.2, 0.0, 0.0, 1.0, 1.0, 1.0),
            score=0.80,
        ),
    ]

    (candidate,) = stitch.build_fragment_stitch_candidates(
        rows, enabled()
    )

    assert candidate.track_ids == (3, 7)
    assert candidate.label == "dining table"
    assert candidate.states == ("active", "active")
    assert candidate.event_frames == (10, 1)
    assert candidate.total_views == 2
    assert candidate.edge_count == 1
    assert candidate.pair_metrics[0].compatibility_branch == (
        "iou+containment"
    )
    assert candidate.min_pair_iou == pytest.approx(2.0 / 3.0)
    assert candidate.min_pair_containment == pytest.approx(0.8)
    assert candidate.max_detector_score == pytest.approx(0.90)
    assert candidate.mean_detector_score == pytest.approx(0.85)


def test_containment_branch_accepts_low_iou_only_when_centers_are_close():
    outer = snapshot(
        1,
        frame=0,
        box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        score=0.90,
    )
    inner = snapshot(
        2,
        frame=10,
        box=(0.0, 0.0, 0.0, 0.5, 0.5, 0.5),
        score=0.80,
    )

    (candidate,) = stitch.build_fragment_stitch_candidates(
        [outer, inner], enabled()
    )
    pair = candidate.pair_metrics[0]
    assert pair.iou == pytest.approx(0.125)
    assert pair.containment == pytest.approx(1.0)
    assert pair.center_distance == pytest.approx(0.0)
    assert pair.compatibility_branch == "containment"

    shifted = snapshot(
        2,
        frame=10,
        box=(0.30, 0.0, 0.0, 0.5, 0.5, 0.5),
        score=0.80,
    )
    assert (
        stitch.build_fragment_stitch_candidates(
            [outer, shifted], enabled()
        )
        == ()
    )


def test_frame_separation_and_label_compatibility_are_hard_gates():
    assert (
        stitch.build_fragment_stitch_candidates(
            [snapshot(1, frame=10), snapshot(2, frame=14)],
            enabled(),
        )
        == ()
    )
    assert (
        stitch.build_fragment_stitch_candidates(
            [
                snapshot(1, frame=10, label=""),
                snapshot(2, frame=20, label=""),
            ],
            enabled(),
        )
        == ()
    )
    assert (
        stitch.build_fragment_stitch_candidates(
            [
                snapshot(1, frame=10, label="chair"),
                snapshot(2, frame=20, label="table"),
            ],
            enabled(),
        )
        == ()
    )


def test_cluster_score_gates_require_strong_max_and_mean():
    high = snapshot(1, frame=0, score=0.84)
    medium = snapshot(2, frame=10, score=0.80)
    assert (
        stitch.build_fragment_stitch_candidates(
            [high, medium], enabled()
        )
        == ()
    )

    high = snapshot(1, frame=0, score=0.90)
    weak = snapshot(2, frame=10, score=0.40)
    assert (
        stitch.build_fragment_stitch_candidates(
            [high, weak], enabled()
        )
        == ()
    )

    (candidate,) = stitch.build_fragment_stitch_candidates(
        [high, weak],
        enabled(minimum_mean_detector_score=0.60),
    )
    assert candidate.max_detector_score == pytest.approx(0.90)
    assert candidate.mean_detector_score == pytest.approx(0.65)

    # The two thresholds are independent.  A stricter mean than max threshold
    # is valid (the max gate is simply redundant for such a configuration).
    resolved = stitch.resolve_fragment_stitch_config(
        {
            "minimum_max_detector_score": 0.70,
            "minimum_mean_detector_score": 0.80,
        }
    )
    assert resolved["minimum_mean_detector_score"] == pytest.approx(0.80)


def test_existing_confirmation_excludes_the_entire_connected_component():
    rows = [
        snapshot(1, frame=0, confirmed=False),
        snapshot(2, frame=10, confirmed=True),
        snapshot(3, frame=20, confirmed=False),
    ]
    assert stitch.build_fragment_stitch_candidates(rows, enabled()) == ()


def test_live_member_gate_can_be_disabled_explicitly():
    rows = [
        snapshot(1, frame=0, state="discarded"),
        snapshot(2, frame=10, state="absorbed"),
    ]
    assert stitch.build_fragment_stitch_candidates(rows, enabled()) == ()

    (candidate,) = stitch.build_fragment_stitch_candidates(
        rows, enabled(require_live_member=False)
    )
    assert candidate.states == ("discarded", "absorbed")

    rows[1]["lifecycle_state"] = " ARCHIVED "
    (candidate,) = stitch.build_fragment_stitch_candidates(rows, enabled())
    assert candidate.states == ("discarded", "archived")


def test_minimum_tracks_frames_and_total_views_are_enforced():
    zero_views = [
        snapshot(1, frame=0, views=0),
        snapshot(2, frame=10, views=0),
    ]
    assert (
        stitch.build_fragment_stitch_candidates(zero_views, enabled())
        == ()
    )

    same_frame = [
        snapshot(1, frame=10),
        snapshot(2, frame=10),
    ]
    assert (
        stitch.build_fragment_stitch_candidates(
            same_frame,
            enabled(minimum_event_frame_separation=1),
        )
        == ()
    )


def test_anchor_clique_does_not_close_a_transitive_three_node_chain():
    # A--B and B--C pass; A--C does not.  B has the greatest pair-IoU sum and
    # becomes the anchor, but after admitting its stronger A edge the
    # incompatible C endpoint must not enter the same candidate.
    rows = [
        snapshot(
            30,
            frame=20,
            box=(0.60, 0.0, 0.0, 1.0, 1.0, 1.0),
            score=0.90,
        ),
        snapshot(
            10,
            frame=0,
            box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            score=0.90,
        ),
        snapshot(
            20,
            frame=10,
            box=(0.20, 0.0, 0.0, 1.0, 1.0, 1.0),
            score=0.90,
        ),
    ]

    expected = None
    for permutation in itertools.permutations(rows):
        (candidate,) = stitch.build_fragment_stitch_candidates(
            list(permutation), enabled()
        )
        signature = (
            candidate.track_ids,
            candidate.event_frames,
            tuple(pair.track_ids for pair in candidate.pair_metrics),
            candidate.edge_count,
            candidate.min_pair_iou,
            candidate.min_pair_containment,
        )
        if expected is None:
            expected = signature
        else:
            assert signature == expected

    assert expected[0] == (10, 20)
    assert expected[1] == (0, 10)
    assert expected[2] == ((10, 20),)
    assert expected[3] == 1


def test_anchor_ranking_uses_declared_lexicographic_priority():
    base = [
        snapshot(
            1,
            frame=0,
            views=3,
            nodes=1,
            points=50,
            score=0.99,
        ),
        snapshot(
            2,
            frame=10,
            views=2,
            nodes=100,
            points=1000,
            score=0.99,
        ),
    ]
    (candidate,) = stitch.build_fragment_stitch_candidates(base, enabled())
    assert candidate.representative_track_id == 1

    ranking_cases = [
        (
            dict(views=3, nodes=1, points=20, score=0.90),
            dict(views=2, nodes=100, points=999, score=0.99),
            1,
        ),
        (
            dict(views=3, nodes=1, points=20, score=0.80),
            dict(views=3, nodes=100, points=10, score=0.99),
            1,
        ),
        (
            dict(views=3, nodes=1, points=20, score=0.90),
            dict(views=3, nodes=100, points=20, score=0.89),
            1,
        ),
        (
            dict(views=3, nodes=1, points=20, score=0.90),
            dict(views=3, nodes=100, points=20, score=0.90),
            1,
        ),
    ]
    for first, second, expected in ranking_cases:
        rows = [
            snapshot(1, frame=0, **first),
            snapshot(2, frame=10, **second),
        ]
        (candidate,) = stitch.build_fragment_stitch_candidates(
            rows, enabled()
        )
        assert candidate.representative_track_id == expected


def test_anchor_pair_iou_sum_precedes_fragment_quality_and_owns_box():
    # Center fragment 20 has poor standalone quality but two direct edges and
    # therefore the greatest IoU sum.  The endpoint pair is compatible here,
    # so all three form one clique and the exported diagnostic box must be the
    # anchor's detached box.
    rows = [
        snapshot(
            10,
            frame=0,
            box=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            views=10,
            points=1000,
            score=0.99,
        ),
        snapshot(
            20,
            frame=10,
            box=(0.1, 0.0, 0.0, 1.0, 1.0, 1.0),
            views=1,
            points=1,
            score=0.80,
        ),
        snapshot(
            30,
            frame=20,
            box=(0.2, 0.0, 0.0, 1.0, 1.0, 1.0),
            views=9,
            points=900,
            score=0.99,
        ),
    ]

    (candidate,) = stitch.build_fragment_stitch_candidates(rows, enabled())

    assert candidate.track_ids == (10, 20, 30)
    assert candidate.representative_track_id == 20
    np.testing.assert_array_equal(candidate.box, rows[1]["box"])
    assert candidate.edge_count == 3


def test_candidates_and_pair_metrics_are_frozen_and_box_is_read_only():
    rows = [
        snapshot(1, frame=0),
        snapshot(2, frame=10, box=(0.1, 0, 0, 1, 1, 1)),
    ]
    (candidate,) = stitch.build_fragment_stitch_candidates(rows, enabled())

    with pytest.raises(FrozenInstanceError):
        candidate.total_views = 100
    with pytest.raises(FrozenInstanceError):
        candidate.pair_metrics[0].iou = 0.0
    with pytest.raises(ValueError):
        candidate.box[0] = 100.0
    assert candidate.box.flags.writeable is False


def test_input_rows_and_arrays_are_not_mutated_or_aliased():
    rows = [
        snapshot(1, frame=0),
        snapshot(2, frame=10, box=(0.1, 0, 0, 1, 1, 1)),
    ]
    original = copy.deepcopy(rows)
    (candidate,) = stitch.build_fragment_stitch_candidates(rows, enabled())

    for actual, expected in zip(rows, original):
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]

    rows[0]["box"][0] = 99.0
    assert candidate.box[0] != pytest.approx(99.0)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda row: row.pop("track_id"), "missing"),
        (lambda row: row.update(track_id=True), "track_id"),
        (lambda row: row.update(track_id=-1), "track_id"),
        (lambda row: row.update(event_frame=-1), "event_frame"),
        (lambda row: row.update(lifecycle_state=""), "lifecycle_state"),
        (lambda row: row.update(lifecycle_state=1), "lifecycle_state"),
        (lambda row: row.update(box=[0, 0, 0, 1, 1]), "box"),
        (lambda row: row.update(box=[0, 0, 0, 1, 0, 1]), "dimensions"),
        (lambda row: row.update(box=[0, 0, 0, 1, np.nan, 1]), "finite"),
        (lambda row: row.update(view_count=-1), "view_count"),
        (lambda row: row.update(node_count=1.5), "node_count"),
        (lambda row: row.update(mean_detector_score=1.1), "score"),
        (lambda row: row.update(label=None), "label"),
        (lambda row: row.update(graph_confirmed=1), "graph_confirmed"),
    ],
)
def test_snapshot_schema_is_strict(mutator, message):
    row = snapshot(1, frame=0)
    mutator(row)
    with pytest.raises(ValueError, match=message):
        stitch.build_fragment_stitch_candidates([row])


def test_duplicate_track_ids_and_non_mapping_inputs_are_rejected():
    with pytest.raises(ValueError, match="sequence"):
        stitch.build_fragment_stitch_candidates("not snapshots")
    with pytest.raises(ValueError, match="mapping"):
        stitch.build_fragment_stitch_candidates([1])
    with pytest.raises(ValueError, match="duplicate"):
        stitch.build_fragment_stitch_candidates(
            [snapshot(1, frame=0), snapshot(1, frame=10)]
        )


def test_multiple_candidates_have_stable_label_then_representative_order():
    rows = [
        snapshot(9, frame=10, label="table"),
        snapshot(8, frame=0, label="table"),
        snapshot(4, frame=10, label="chair"),
        snapshot(3, frame=0, label="chair"),
    ]
    candidates = stitch.build_fragment_stitch_candidates(rows, enabled())

    assert tuple(candidate.label for candidate in candidates) == (
        "chair",
        "table",
    )
    assert tuple(
        candidate.representative_track_id for candidate in candidates
    ) == (3, 8)
