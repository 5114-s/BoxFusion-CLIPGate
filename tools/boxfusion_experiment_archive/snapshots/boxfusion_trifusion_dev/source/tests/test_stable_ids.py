import numpy as np
import pytest

from boxfusion.stable_ids import resolve_fusion_stable_ids


def test_unique_minimum_path_is_exact_noop():
    groups = [[8, 3, 8], [4, 9], [12]]

    result = resolve_fusion_stable_ids(groups)

    assert result.dtype == np.int64
    assert result.tolist() == [3, 4, 12]


def test_overlapping_groups_with_distinct_minima_remain_unchanged():
    groups = [[1, 5], [2, 5], [7, 9]]

    result = resolve_fusion_stable_ids(groups)

    assert result.tolist() == [1, 2, 7]


def test_scene0221_collision_uses_group_unique_source_id():
    groups = [
        [92, 116, 129],
        [4, 7],
        [92, 116, 128],
    ]

    result = resolve_fusion_stable_ids(groups)

    assert result.tolist() == [129, 4, 92]
    assert len(set(result.tolist())) == len(result)


def test_scene0246_collision_is_deterministic_across_group_order():
    forward = resolve_fusion_stable_ids([[22, 204], [22, 208]])
    reverse = resolve_fusion_stable_ids([[22, 208], [22, 204]])

    assert forward.tolist() == [22, 208]
    assert reverse.tolist() == [208, 22]


def test_identical_groups_receive_deterministic_synthetic_fallback():
    groups = [[5, 8], [5, 8], [5, 8]]

    first = resolve_fusion_stable_ids(groups)
    second = resolve_fusion_stable_ids(groups)

    assert np.array_equal(first, second)
    assert first[0] == 5
    assert first[1] == 8
    assert first[2] >= 1 << 62
    assert len(set(first.tolist())) == 3


@pytest.mark.parametrize(
    "groups,error_type,message",
    [
        ("not-groups", TypeError, "sequence"),
        ([[]], ValueError, "must not be empty"),
        ([[True]], TypeError, "only integers"),
        ([[1.5]], TypeError, "only integers"),
        ([[-1]], ValueError, "non-negative"),
        ([[(1 << 62)]], ValueError, "source-id range"),
    ],
)
def test_invalid_fusion_groups_fail_fast(groups, error_type, message):
    with pytest.raises(error_type, match=message):
        resolve_fusion_stable_ids(groups)
