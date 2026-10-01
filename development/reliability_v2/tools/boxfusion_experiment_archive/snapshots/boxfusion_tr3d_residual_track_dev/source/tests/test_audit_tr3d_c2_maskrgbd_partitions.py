from __future__ import annotations

import pytest

from tools.audit_tr3d_c2_maskrgbd_observer import (
    _decision,
    _partition_scene_ids,
)


def _route(candidate_count: int, *, hit25: float, hit50: float, novel25: int):
    return {
        "candidate_count": candidate_count,
        "thresholds": {
            "0.25": {
                "independent_gt_hit_precision": hit25,
                "novel_oracle_tp": novel25,
            },
            "0.50": {"independent_gt_hit_precision": hit50},
        },
    }


def test_partition_scene_ids_builds_ordered_all100_and_heldout90():
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    excluded = scenes[::10]

    partitions = _partition_scene_ids(scenes, excluded)

    assert partitions["all"] == scenes
    assert partitions["heldout"] == [
        scene_id for scene_id in scenes if scene_id not in set(excluded)
    ]
    assert len(partitions["heldout"]) == 90
    assert _partition_scene_ids(scenes, None) == {"all": scenes}


def test_partition_scene_ids_rejects_noncanonical_exclusion():
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    with pytest.raises(ValueError, match="exactly 10"):
        _partition_scene_ids(scenes, scenes[:9])
    with pytest.raises(ValueError, match="absent"):
        _partition_scene_ids(scenes, [*scenes[:9], "scene9999_00"])
    with pytest.raises(ValueError, match="exactly 100"):
        _partition_scene_ids(scenes[:10], scenes[:10])


def test_decision_is_partition_local():
    passing = {
        "source_top10": _route(20, hit25=0.40, hit50=0.10, novel25=0),
        "mask2_depth": _route(10, hit25=0.60, hit50=0.30, novel25=3),
    }
    failing = {
        "source_top10": _route(20, hit25=0.55, hit50=0.10, novel25=0),
        "mask2_depth": _route(4, hit25=0.60, hit50=0.20, novel25=2),
    }

    assert _decision(passing)["pass"] is True
    assert _decision(failing)["pass"] is False

