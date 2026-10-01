"""CPU-only regression checks for raw-observation contamination diagnostics."""

from itertools import product

import numpy as np
import pytest

from tools.true_fusion_audit_core import (
    aabb_iou,
    assess_membership,
    assign_observations,
    class_agnostic_ap,
    summarize_memberships,
)


def box(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
    return np.asarray(list(product(*zip(lo, hi))), dtype=np.float64)


def test_aabb_iou_exact_overlap_disjoint_and_empty():
    unit = box()
    double = box(hi=(2.0, 1.0, 1.0))
    distant = box(lo=(3.0, 0.0, 0.0), hi=(4.0, 1.0, 1.0))
    np.testing.assert_array_equal(aabb_iou([unit, double, distant], [unit]), [[1], [0.5], [0]])
    assert aabb_iou([], [unit]).shape == (0, 1)
    assert aabb_iou([unit], []).shape == (1, 0)
    assert aabb_iou([], []).shape == (0, 0)


@pytest.mark.parametrize("invalid", [np.full((8, 3), np.nan), np.zeros((8, 3))])
def test_aabb_iou_rejects_invalid_geometry(invalid):
    with pytest.raises(ValueError):
        aabb_iou([invalid], [box()])


def test_assignment_threshold_is_strict():
    below = np.nextafter(0.5, 0)
    above = np.nextafter(0.5, 1)
    result = assign_observations([[below], [0.5], [above]], 0.5)
    np.testing.assert_array_equal(result, [-1, -1, 0])


def test_assignment_unique_policy_does_not_hide_multiple_gt_overlaps():
    matrix = [[0.8, 0.7, 0], [0.5, 0.5, 0], [0, 0.5, 0.9], [0.1, 0.2, 0.3]]
    np.testing.assert_array_equal(assign_observations(matrix, 0.5), [-2, -1, 2, -1])
    np.testing.assert_array_equal(assign_observations(matrix, 0.5, policy="best"), [0, -1, 2, -1])
    np.testing.assert_array_equal(assign_observations([[0.9, 0.9]], 0.5), [-2])
    np.testing.assert_array_equal(assign_observations([[0.9, 0.9]], 0.5, policy="best"), [0])


def test_assignment_empty_inputs_and_invalid_policy():
    np.testing.assert_array_equal(assign_observations(np.empty((3, 0)), 0.5), [-1, -1, -1])
    assert assign_observations(np.empty((0, 2)), 0.5).shape == (0,)
    with pytest.raises(ValueError):
        assign_observations([[0.8]], 0.5, policy="other")
    with pytest.raises(ValueError):
        assign_observations([[np.nan]], 0.5)


@pytest.mark.parametrize(
    ("identities", "expected", "unknown", "ambiguous"),
    [
        ([0, 0], "clean_known", 0, 0),
        ([0, -1], "one_known_with_unresolved", 1, 0),
        ([0, -2], "one_known_with_unresolved", 0, 1),
        ([0, -1, -2], "one_known_with_unresolved", 1, 1),
        ([-1, -1], "all_unknown", 2, 0),
        ([-2, -2], "unresolved_without_unique_identity", 0, 2),
        ([-1, -2], "unresolved_without_unique_identity", 1, 1),
    ],
)
def test_unknown_and_ambiguous_are_not_counted_as_clean(identities, expected, unknown, ambiguous):
    result = assess_membership(range(len(identities)), identities, range(len(identities)))
    assert result["status"] == expected
    assert result["unknown_observations"] == unknown
    assert result["ambiguous_observations"] == ambiguous
    if unknown or ambiguous:
        assert result["status"] != "clean_known"


def test_repeated_observation_ids_do_not_manufacture_support():
    result = assess_membership([0, 0, 1, 1, 2], [4, 4, 4], [10, 10, 20])
    assert result["observation_ids"] == [0, 1, 2]
    assert result["observations"] == 3
    assert result["duplicate_memberships"] == 2
    assert result["distinct_frames"] == 2
    assert result["known_groups"]["4"]["observation_ids"] == [0, 1, 2]
    assert result["known_groups"]["4"]["frames"] == [10, 20]
    assert result["groups_with_2_frames"] == 1
    assert result["groups_with_3_frames"] == 0


def test_three_selected_views_can_be_mixed_without_two_frames_per_identity():
    result = assess_membership([0, 1, 2], [0, 0, 1], [10, 20, 30])
    assert result["status"] == "mixed_known"
    assert result["known_identities"] == 2
    assert result["groups_with_2_frames"] == 1
    assert result["groups_with_3_frames"] == 0
    assert result["minority_known_fraction"] == pytest.approx(1 / 3)
    summary = summarize_memberships([[0, 1, 2]], [0, 0, 1], [10, 20, 30])
    assert summary["mixed_known_groups"] == 1
    assert summary["mixed_with_two_2frame_groups"] == 0


def test_minority_fraction_includes_every_nonmajority_gt_not_only_runner_up():
    identities = [0, 0, 0, 0, 1, 1, 2, 3, -1, -2]
    result = assess_membership(range(10), identities, range(10))
    assert result["status"] == "mixed_known"
    assert result["minority_known_fraction"] == pytest.approx(4 / 8)
    assert result["unknown_observations"] == 1
    assert result["ambiguous_observations"] == 1


def test_same_frame_known_coexistence_is_not_cross_frame_confirmation():
    result = assess_membership([0, 1, 2, 3], [0, 1, 0, 0], [10, 10, 20, 20])
    assert result["same_frame_known_coexistence_frames"] == [10]
    assert result["known_groups"]["0"]["distinct_frames"] == 2
    assert result["known_groups"]["1"]["distinct_frames"] == 1


def test_empty_membership_and_summary_are_explicit():
    result = assess_membership([], [], [])
    assert result["status"] == "empty"
    assert result["observations"] == result["distinct_frames"] == 0
    assert result["known_groups"] == {}
    assert result["minority_known_fraction"] is None
    summary = summarize_memberships([], [], [])
    assert summary["groups"] == 0
    assert summary["details"] == []
    assert summary["fully_identified_groups"] == 0


def test_summary_denominator_excludes_empty_unknown_and_ambiguous_groups():
    summary = summarize_memberships([[], [0], [1], [2], [0, 3], [0, 1]], [0, -1, -2, 1], [0, 1, 2, 3])
    assert summary["groups"] == 6
    assert summary["fully_identified_groups"] == 2
    assert summary["mixed_known_groups"] == 1
    assert summary["status_counts"]["one_known_with_unresolved"] == 1


@pytest.mark.parametrize("ids", [[-1], [2], [0.5], [True]])
def test_invalid_membership_ids_fail_closed(ids):
    with pytest.raises(ValueError):
        assess_membership(ids, [0, 1], [0, 1])


def test_membership_requires_aligned_frame_and_identity_arrays():
    with pytest.raises(ValueError):
        assess_membership([0], [0, 1], [0])


def test_ap_duplicate_predictions_are_false_positives_not_extra_recall():
    result = class_agnostic_ap({"s": ([box(), box()], [0.9, 0.8])}, {"s": [box()]}, 0.5)
    assert (result["tp"], result["fp"], result["gt"], result["predictions"]) == (1, 1, 1, 2)
    # Low-score duplicate does not reduce interpolated AP after recall was gained.
    assert result["ap"] == pytest.approx(100 / (1 + 1e-6))


def test_ap_uses_scores_not_prediction_row_order():
    false_box = box(lo=(3, 0, 0), hi=(4, 1, 1))
    gt = {"s": [box()]}
    good = class_agnostic_ap({"s": ([false_box, box()], [0.1, 0.9])}, gt, 0.5)
    bad = class_agnostic_ap({"s": ([false_box, box()], [0.9, 0.1])}, gt, 0.5)
    assert good["ap"] == pytest.approx(100 / (1 + 1e-6))
    assert bad["ap"] == pytest.approx(50 / (1 + 1e-6))
    assert good["tp"] == bad["tp"] == 1
    assert good["fp"] == bad["fp"] == 1


def test_ap_sorts_globally_across_scenes_and_counts_gt_in_scenes_without_predictions():
    predictions = {"miss": ([box()], [0.9]), "hit": ([box()], [0.8])}
    gt = {"miss": [], "hit": [box()], "not_predicted": [box()]}
    result = class_agnostic_ap(predictions, gt, 0.5)
    assert result["gt"] == 2
    assert result["tp"] == result["fp"] == 1
    assert result["ap"] == pytest.approx(50 / (2 + 1e-6))


def test_ap_matching_is_strict_and_does_not_fallback_from_an_already_matched_best_gt():
    # This matches the anchor's greedy best-IoU policy, not maximum-cardinality matching.
    gt = {"s": [box(), box(lo=(0.2, 0, 0), hi=(1.2, 1, 1))]}
    result = class_agnostic_ap({"s": ([box(), box()], [0.9, 0.8])}, gt, 0.5)
    assert (result["tp"], result["fp"]) == (1, 1)
    at_threshold = class_agnostic_ap(
        {"s": ([box(hi=(2, 1, 1))], [0.9])}, {"s": [box()]}, 0.5
    )
    assert at_threshold["ap"] == 0
    assert (at_threshold["tp"], at_threshold["fp"]) == (0, 1)


@pytest.mark.parametrize("predictions", [{}, {"s": ([], [])}])
def test_ap_empty_predictions_are_zero_not_an_error(predictions):
    result = class_agnostic_ap(predictions, {"s": [box()]}, 0.5)
    assert result == {"ap": 0.0, "tp": 0, "fp": 0, "gt": 1, "predictions": 0}


def test_ap_no_gt_counts_all_predictions_as_false_positives():
    result = class_agnostic_ap({"s": ([box()], [0.9])}, {}, 0.5)
    assert result == {"ap": 0.0, "tp": 0, "fp": 1, "gt": 0, "predictions": 1}


@pytest.mark.parametrize("scores", [[np.nan], [], [0.1, 0.2]])
def test_ap_rejects_nonfinite_or_misaligned_scores(scores):
    with pytest.raises(ValueError):
        class_agnostic_ap({"s": ([box()], scores)}, {"s": [box()]}, 0.5)
