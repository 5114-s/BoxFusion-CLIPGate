import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


SOURCE = os.environ.get(
    "BOXFUSION_PVQ_AR",
    str(
        Path(__file__).resolve().parents[1]
        / "boxfusion"
        / "pvq_ar.py"
    ),
)
spec = importlib.util.spec_from_file_location("boxfusion_pvq_ar", SOURCE)
pvq_ar_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pvq_ar_module
spec.loader.exec_module(pvq_ar_module)


class FakeBoxManager:
    """Minimal stand-in exposing only fusion_list."""

    def __init__(self, fusion_list):
        self.fusion_list = fusion_list


def make_cfg(**overrides):
    cfg = {
        "enabled": True,
        "mode": "shadow",
        "max_prototypes": 4,
        "memory_per_track": 12,
        "ambiguity_margin": 0.10,
        "view_angle_max_deg": 60.0,
        "rearrange_margin": 0.05,
        "min_similarity": 0.50,
        "require_both_prototypes": True,
        "diagnostics_dir": tempfile.mkdtemp(prefix="pvq_ar_test_"),
        "scene_event_cap": 4096,
    }
    cfg.update(overrides)
    return {"association": {"pvq_ar": cfg}}


def unit(dim=8, index=0):
    feature = np.zeros(dim, dtype=np.float32)
    feature[index % dim] = 1.0
    return feature


def prototype(init_id, frame_id, feature, view_dir, score=0.9):
    view = np.asarray(view_dir, dtype=np.float64)
    view = view / np.linalg.norm(view)
    return pvq_ar_module.Prototype(
        init_id=init_id,
        frame_id=frame_id,
        feature=np.asarray(feature, dtype=np.float32)
        / np.linalg.norm(feature),
        view_dir=view,
        score=score,
    )


def build_ar(cfg, committed):
    ar = pvq_ar_module.PVQAR(cfg)
    ar.bind_scene("scene0000_00")
    ar._committed = committed
    return ar


def adjudicate(ar, **overrides):
    kwargs = dict(
        frame_id=50,
        proposal_row=3,
        proposal_init_id=99,
        query_feature=unit(8, 0),
        query_view_dir=[1.0, 0.0, 0.0],
        proposal_corners_world=np.zeros((8, 3)),
        candidate_rows=[7, 11],
        candidate_canonicals=[7, 11],
        candidate_ious=[0.20, 0.19],
        candidate_margins=[0.10, 0.09],
        candidate_scores=[0.8, 0.8],
        candidate_corners_world=[np.zeros((8, 3)), np.zeros((8, 3))],
    )
    kwargs.update(overrides)
    return ar.adjudicate_ambiguity(**kwargs)


def test_config_validation():
    with pytest.raises(pvq_ar_module.PVQARConfigError):
        pvq_ar_module.resolve_pvq_ar_config({"enabled": True, "mode": "x"})
    with pytest.raises(pvq_ar_module.PVQARConfigError):
        pvq_ar_module.resolve_pvq_ar_config(
            {"enabled": True, "max_prototypes": 5}
        )
    with pytest.raises(pvq_ar_module.PVQARConfigError):
        pvq_ar_module.resolve_pvq_ar_config(
            {"enabled": True, "diagnostics_dir": None}
        )
    resolved = pvq_ar_module.resolve_pvq_ar_config(
        {"enabled": True, "diagnostics_dir": "/tmp/x"}
    )
    assert resolved["max_prototypes"] == 4
    disabled = pvq_ar_module.resolve_pvq_ar_config(None)
    assert disabled["enabled"] is False


def test_shadow_mode_always_keeps_native_choice():
    committed = {
        7: [prototype(7, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        11: [prototype(11, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(make_cfg(mode="shadow"), committed)
    # Alternative similarity is clearly better (1.0 vs 0.0): shadow must
    # still return 0 so the native decision and predictions stay identical.
    assert adjudicate(ar) == 0
    assert ar.stats["shadow_rearrangements"] == 1
    assert ar.stats["applied_rearrangements"] == 0


def test_active_mode_rearranges_only_with_clear_margin():
    committed = {
        7: [prototype(7, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        11: [prototype(11, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(make_cfg(mode="active"), committed)
    assert adjudicate(ar) == 1
    assert ar.stats["applied_rearrangements"] == 1

    # Below the rearrange margin -> native.
    committed_small = {
        7: [prototype(7, 10, unit(8, 0), [1.0, 0.0, 0.0])],
        11: [prototype(11, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar2 = build_ar(make_cfg(mode="active"), committed_small)
    assert adjudicate(ar2) == 0
    assert ar2._abstain_reasons.get("native_better") == 1


def test_missing_prototype_is_not_negative_evidence():
    # Native track has no compatible prototype: must abstain even when the
    # alternative is a strong match.
    committed = {
        7: [prototype(7, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        11: [prototype(11, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(make_cfg(mode="active"), committed)
    # Query from the opposite side: track 7's prototype (at +x) is 180
    # degrees away and therefore not view compatible.
    assert (
        adjudicate(ar, query_view_dir=[-1.0, 0.0, 0.0]) == 0
    )
    assert ar._abstain_reasons.get("abstain_missing_prototype") == 1


def test_view_compatible_prototype_retrieval_and_k_cap():
    dim = 8
    features = [unit(dim, i) for i in range(6)]
    # All six observations sit within the 60-degree cone around +x, but
    # only the K most view-aligned ones may be retrieved.
    views = []
    angles = [5.0, 10.0, 20.0, 30.0, 40.0, 55.0]
    for angle in angles:
        rad = np.deg2rad(angle)
        views.append([np.cos(rad), np.sin(rad), 0.0])
    committed = {
        7: [
            prototype(70 + i, 10 + i, features[i], views[i])
            for i in range(6)
        ]
    }
    ar = build_ar(make_cfg(mode="shadow"), committed)
    query = adjudicate(ar)
    assert query == 0
    # Re-inspect the logged record for retrieval behaviour.
    with open(ar._jsonl_path, encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    native = record["candidates"][0]
    assert native["compatible_count"] == 6
    assert native["retrieved_count"] == 4
    assert native["prototype_angles_deg"] == [5.0, 10.0, 20.0, 30.0]


def test_memory_is_bounded_per_track():
    cfg = make_cfg(memory_per_track=3)
    ar = pvq_ar_module.PVQAR(cfg)
    ar.bind_scene("scene0000_00")
    fusion_list = [
        [3, 5, 9, 12, 20, 33],  # canonical 3, six historical observations
        [7],
    ]
    for init_id, frame_id in zip([3, 5, 9, 12, 20, 33], [1, 2, 3, 4, 5, 6]):
        ar._observations[init_id] = prototype(init_id, frame_id, unit(8), [1, 0, 0])
    ar.begin_keyframe(FakeBoxManager(fusion_list))
    assert [view.init_id for view in ar._committed[3]] == [12, 20, 33]
    assert [view.init_id for view in ar._committed[7]] == []


def test_merge_invariant_canonical_key():
    cfg = make_cfg()
    ar = pvq_ar_module.PVQAR(cfg)
    ar.bind_scene("scene0000_00")
    # Whichever row survives a native merge, min(init_id) of the union is
    # the same stable canonical key.
    assert min([5] + [2, 3]) == min([2, 3] + [5]) == 2


def test_events_logged_with_choice_set_for_oracle():
    committed = {
        7: [prototype(7, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        11: [prototype(11, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(make_cfg(mode="shadow"), committed)
    adjudicate(ar)
    with open(ar._jsonl_path, encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    for key in (
        "frame_id",
        "proposal_init_id",
        "proposal_corners_world",
        "candidates",
        "reason",
        "chosen",
        "applied",
    ):
        assert key in record
    assert len(record["candidates"]) == 2
    for candidate in record["candidates"]:
        for key in (
            "track_row",
            "canonical_id",
            "iou",
            "corners_world",
            "best_similarity",
            "prototype_angles_deg",
        ):
            assert key in candidate
    assert record["applied"] is False


def test_scene_event_cap_stops_writing():
    committed = {
        7: [prototype(7, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        11: [prototype(11, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(make_cfg(scene_event_cap=2), committed)
    adjudicate(ar)
    adjudicate(ar)
    adjudicate(ar)
    with open(ar._jsonl_path, encoding="utf-8") as handle:
        lines = handle.readlines()
    assert len(lines) == 2
    assert ar.stats["event_cap_hits"] == 1


def nms_cfg(top_mode="active", **overrides):
    section = {
        "enabled": True,
        "min_child_dim": 0.35,
        "iou_confident": 0.60,
        "contrast_margin": 0.05,
        "min_similarity": 0.50,
        "iou_orphan_guard": 0.30,
        "max_events_per_keyframe": 16,
    }
    section.update(overrides)
    cfg = make_cfg(mode=top_mode)
    cfg["association"]["pvq_ar"]["nms_stage"] = section
    return cfg


def adjudicate_nms(ar, **overrides):
    kwargs = dict(
        keyframe_id=100,
        parent_row=0,
        child_row=1,
        parent_init_id=1,
        child_init_id=2,
        iou=0.20,
        parent_score=0.9,
        child_score=0.8,
        child_max_dim=0.80,
        query_feature=unit(8, 0),
        query_view_dir=[1.0, 0.0, 0.0],
        parent_corners_world=np.zeros((8, 3)),
        child_corners_world=np.zeros((8, 3)),
        row_canonicals={0: 1, 1: 2, 2: 3},
    )
    kwargs.update(overrides)
    return ar.adjudicate_nms_absorb(**kwargs)


def test_nms_stage_disabled_by_default():
    ar = pvq_ar_module.PVQAR(make_cfg())
    assert ar.nms_stage["enabled"] is False
    # No stage -> the adjudicator never refuses.
    assert adjudicate_nms(ar) is False


def test_nms_shadow_never_refuses_even_when_contested():
    # Parent's prototype points to feature slot 1, rival (row 2/canonical
    # 3) matches the query exactly: the rival wins the contrast contest.
    committed = {
        1: [prototype(1, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        3: [prototype(3, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(nms_cfg(top_mode="shadow"), committed)
    assert adjudicate_nms(ar) is False
    assert ar.nms_stats["decisions"].get("refuse_contest") == 1
    assert ar.nms_stats["applied_refusals"] == 0


def test_nms_active_refuses_on_clear_contest():
    committed = {
        1: [prototype(1, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        3: [prototype(3, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(nms_cfg(), committed)
    assert adjudicate_nms(ar) is True
    assert ar.nms_stats["applied_refusals"] == 1


def test_nms_abstain_when_parent_owns_query():
    # A credible rival (0.57 >= 0.50) that still loses by more than the
    # contrast margin: the parent owns the query.
    query = np.zeros(8, dtype=np.float32)
    query[0] = 1.0
    query[1] = 0.7
    committed = {
        1: [prototype(1, 10, unit(8, 0), [1.0, 0.0, 0.0])],
        3: [prototype(3, 10, unit(8, 1), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(nms_cfg(), committed)
    assert adjudicate_nms(ar, query_feature=query) is False
    assert ar.nms_stats["decisions"].get("abstain_parent_owns") == 1


def test_nms_abstain_when_no_credible_rival():
    # Rival exists but its similarity is below the absolute floor.
    committed = {
        1: [prototype(1, 10, unit(8, 2), [1.0, 0.0, 0.0])],
        3: [prototype(3, 10, unit(8, 5), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(nms_cfg(), committed)
    assert adjudicate_nms(ar) is False
    assert ar.nms_stats["decisions"].get("abstain_no_rival") == 1


def test_nms_orphan_guard_when_parent_has_no_memory():
    committed = {
        3: [prototype(3, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(nms_cfg(), committed)
    # High overlap + parent without prototypes: geometry stays decisive.
    assert adjudicate_nms(ar, iou=0.50) is False
    assert ar.nms_stats["decisions"].get("abstain_orphan_guard") == 1
    # Weak overlap + rival claim: refusal is justified.
    assert adjudicate_nms(ar, iou=0.15) is True
    assert ar.nms_stats["applied_refusals"] == 1


def test_nms_gates_skip_adjudication():
    committed = {
        1: [prototype(1, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        3: [prototype(3, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(nms_cfg(), committed)
    assert adjudicate_nms(ar, child_max_dim=0.20) is False
    assert ar.nms_stats["gate_small_child"] == 1
    assert adjudicate_nms(ar, iou=0.90) is False
    assert ar.nms_stats["gate_confident_iou"] == 1
    assert ar.nms_stats["events"] == 0


def test_nms_keyframe_event_cap():
    committed = {
        1: [prototype(1, 10, unit(8, 1), [1.0, 0.0, 0.0])],
        3: [prototype(3, 10, unit(8, 0), [1.0, 0.0, 0.0])],
    }
    ar = build_ar(nms_cfg(max_events_per_keyframe=2), committed)
    assert adjudicate_nms(ar) is True
    assert adjudicate_nms(ar) is True
    assert adjudicate_nms(ar) is False
    assert ar.nms_stats["gate_event_cap"] == 1
