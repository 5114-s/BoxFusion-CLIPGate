import numpy as np
import pytest

from tools.eval_scannet_causal_dynamic_ap import aabb_overlaps, boxes_array, removal_mapping


def test_removal_mapping_collapses_duplicate_gt_references():
    removed, records = removal_mapping(np.array([[0.6, 0.1], [0.7, 0.2], [0.0, 0.5]]))
    assert removed == {0, 1}
    assert [r['duplicate_gt_reference'] for r in records] == [False, True, False]


def test_unmatched_removal_marker_fails_closed():
    with pytest.raises(ValueError, match='no GT'):
        removal_mapping(np.array([[0.49, 0.1]]))


def test_empty_boxes_and_identity_overlap():
    assert boxes_array([], 'empty').shape == (0, 8, 3)
    box = np.array([[[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]], dtype=float)
    assert aabb_overlaps(box, box)[0, 0] == 1
    assert aabb_overlaps(box, boxes_array([], 'empty')).shape == (1, 0)


def test_best_match_is_selected_not_first_eligible_gt():
    removed, records = removal_mapping(np.array([[0.55, 0.7]]))
    assert removed == {1}
    assert records[0]['eligible_gt_indices'] == [0, 1]
