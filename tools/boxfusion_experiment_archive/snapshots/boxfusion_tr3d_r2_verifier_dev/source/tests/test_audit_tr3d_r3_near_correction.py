from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.audit_tr3d_r3_near_correction import (
    SceneCounterfactual,
    _create_only,
    _partition_counterfactuals,
    _validate_optional_export_report_paths,
    evaluate_counterfactuals,
    fixed_rule_scores,
    per_anchor_oracle_upper_bound,
    replacement_boxes,
    replacement_rows_for_rule,
    scored_detection_metrics,
    select_one_per_anchor,
    validate_scene,
)


def _boxes(*centers: float) -> np.ndarray:
    return np.asarray(
        [[center - 0.5, 0.0, 0.0, center + 0.5, 1.0, 1.0] for center in centers],
        dtype=np.float64,
    )


def _scene(
    *,
    anchor_boxes: np.ndarray | None = None,
    gt_boxes: np.ndarray | None = None,
    candidate_boxes: np.ndarray | None = None,
    proposal_ids: np.ndarray | None = None,
    anchor_indices: np.ndarray | None = None,
    anchor_iou: np.ndarray | None = None,
    tr3d_score: np.ndarray | None = None,
    depth_available: np.ndarray | None = None,
    depth_quality: np.ndarray | None = None,
    feature_available: np.ndarray | None = None,
    feature_cosine: np.ndarray | None = None,
) -> SceneCounterfactual:
    anchors = _boxes(0.0, 4.0) if anchor_boxes is None else anchor_boxes
    candidates = _boxes(0.0, 1.0, 4.0) if candidate_boxes is None else candidate_boxes
    count = len(candidates)
    return SceneCounterfactual(
        scene_id="scene0000_00",
        anchor_boxes=np.asarray(anchors, dtype=np.float64),
        anchor_scores=np.asarray([0.9, 0.8][: len(anchors)], dtype=np.float64),
        gt_boxes=np.asarray(_boxes(1.0, 4.0) if gt_boxes is None else gt_boxes, dtype=np.float64),
        candidate_boxes=np.asarray(candidates, dtype=np.float64),
        proposal_ids=np.asarray(
            np.arange(10, 10 + count) if proposal_ids is None else proposal_ids,
            dtype=np.int64,
        ),
        anchor_indices=np.asarray(
            [0, 0, 1][:count] if anchor_indices is None else anchor_indices,
            dtype=np.int64,
        ),
        anchor_iou=np.asarray(
            [0.9, 0.4, 0.8][:count] if anchor_iou is None else anchor_iou,
            dtype=np.float64,
        ),
        tr3d_score=np.asarray(
            [0.4, 0.8, 0.7][:count] if tr3d_score is None else tr3d_score,
            dtype=np.float64,
        ),
        depth_available=np.asarray(
            [True, True, True][:count]
            if depth_available is None
            else depth_available,
            dtype=np.bool_,
        ),
        depth_quality=np.asarray(
            [0.9, 0.2, 0.8][:count] if depth_quality is None else depth_quality,
            dtype=np.float64,
        ),
        feature_available=np.asarray(
            [True, False, True][:count]
            if feature_available is None
            else feature_available,
            dtype=np.bool_,
        ),
        feature_cosine=np.asarray(
            [0.8, 0.0, 0.6][:count] if feature_cosine is None else feature_cosine,
            dtype=np.float64,
        ),
    )


def test_selection_is_one_per_anchor_and_ties_use_proposal_id() -> None:
    scene = _scene(
        proposal_ids=np.asarray([12, 11, 10]),
        tr3d_score=np.asarray([0.8, 0.8, 0.7]),
    )
    selected = select_one_per_anchor(scene, scene.tr3d_score)
    assert scene.proposal_ids[selected].tolist() == [11, 10]
    replacement = replacement_boxes(scene, selected)
    np.testing.assert_array_equal(replacement[0], scene.candidate_boxes[1])
    np.testing.assert_array_equal(replacement[1], scene.candidate_boxes[2])


def test_primary_rule_requires_score_above_frozen_anchor() -> None:
    scene = _scene(tr3d_score=np.asarray([0.95, 0.8, 0.79]))
    selected = select_one_per_anchor(scene, scene.tr3d_score)
    applied = replacement_rows_for_rule(
        scene, selected, "tr3d_score_gt_anchor_score"
    )
    assert scene.anchor_indices[applied].tolist() == [0]


def test_fixed_rules_are_exact_and_feature_absence_is_neutral() -> None:
    scene = _scene()
    scores = fixed_rule_scores(scene)
    np.testing.assert_allclose(scores["score_anchor_iou"], [0.36, 0.32, 0.56])
    np.testing.assert_allclose(scores["score_depth_quality"], [0.36, 0.16, 0.56])
    expected_missing = 0.8 * (0.5 + 0.5 * 0.4) * (0.75 + 0.25 * 0.2) * (0.75 + 0.25 * 0.5)
    assert scores["fixed_joint"][1] == pytest.approx(expected_missing)


def test_validation_rejects_nonzero_missing_feature_sentinel() -> None:
    scene = _scene(feature_cosine=np.asarray([0.8, 0.2, 0.6]))
    with pytest.raises(ValueError, match="zero sentinel"):
        validate_scene(scene)


def test_validation_rejects_nonzero_missing_depth_sentinel() -> None:
    scene = _scene(
        depth_available=np.asarray([True, False, True]),
        depth_quality=np.asarray([0.9, 0.2, 0.8]),
    )
    with pytest.raises(ValueError, match="depth quality"):
        validate_scene(scene)


def test_scored_metrics_preserve_global_score_order() -> None:
    # A high-scored false positive before the only true positive gives AP=0.5.
    values = scored_detection_metrics(
        [("scene0000_00", _boxes(10.0, 0.0), np.asarray([0.9, 0.8]), _boxes(0.0))],
        0.50,
    )
    assert values["matched_tp"] == 1
    assert values["average_precision"] == pytest.approx(0.5, abs=1e-6)


def test_scored_matching_mirrors_duplicate_gt_behavior() -> None:
    # Both rows prefer GT0. The second is an FP even though it overlaps GT1,
    # matching evaluation/utils/eval_det.py semantics.
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.1, 0.0, 0.0, 1.2, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    gt = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.8, 0.0, 0.0, 1.8, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    values = scored_detection_metrics(
        [("scene0000_00", boxes, np.asarray([0.9, 0.8]), gt)], 0.25
    )
    assert values["matched_tp"] == 1


def test_per_anchor_oracle_enforces_one_slot_per_anchor() -> None:
    # Two perfect candidates share one anchor. They cannot both count after a
    # one-to-one replacement, although the unrestricted union could count 2.
    scene = _scene(
        anchor_boxes=_boxes(10.0),
        gt_boxes=_boxes(0.0, 2.0),
        candidate_boxes=_boxes(0.0, 2.0),
        proposal_ids=np.asarray([1, 2]),
        anchor_indices=np.asarray([0, 0]),
        anchor_iou=np.asarray([0.1, 0.1]),
        tr3d_score=np.asarray([0.95, 0.85]),
        depth_available=np.asarray([True, True]),
        depth_quality=np.asarray([1.0, 1.0]),
        feature_available=np.asarray([False, False]),
        feature_cosine=np.asarray([0.0, 0.0]),
    )
    assert per_anchor_oracle_upper_bound(scene, 0.5) == 1


def test_end_to_end_counterfactual_reports_replacement_and_union() -> None:
    # G0 misses GT0. The highest-score candidate for anchor0 fixes it while
    # anchor1 remains correct, so every fixed rule should gain one TP50.
    scene = _scene()
    report = evaluate_counterfactuals([scene])
    baseline = report["baseline"]["0.50"]
    assert baseline["scored"]["matched_tp"] == 1
    assert report["all_near_union_oracle"]["0.50"]["delta_matches"] == 1
    assert report["per_anchor_gt_oracle_upper_bound"]["0.50"]["delta_matches"] == 1
    score = report["fixed_rules"]["tr3d_score"]["thresholds"]["0.50"]
    assert score["replacement"]["delta_scored_tp"] == 1
    assert score["add_one"]["delta_maximum_matches"] == 1
    assert score["add_one"]["conservative_source_score"] == 0.0
    assert score["add_one"]["scored_supplementary_only"]["matched_tp"] == 2
    assert score["candidate_hits"]["hits"] == 2


def test_gate_requires_cross50_precision_coverage_and_ap_gain() -> None:
    # Three scenes x two safe crossings satisfy the frozen held-out gate.
    source = _scene(
        anchor_boxes=_boxes(10.0, 12.0),
        gt_boxes=_boxes(0.0, 2.0),
        candidate_boxes=_boxes(0.0, 2.0),
        proposal_ids=np.asarray([1, 2]),
        anchor_indices=np.asarray([0, 1]),
        anchor_iou=np.asarray([0.1, 0.1]),
        tr3d_score=np.asarray([0.95, 0.85]),
        depth_available=np.asarray([True, True]),
        depth_quality=np.asarray([1.0, 1.0]),
        feature_available=np.asarray([False, False]),
        feature_cosine=np.asarray([0.0, 0.0]),
    )
    scenes = [
        SceneCounterfactual(
            **{**source.__dict__, "scene_id": f"scene{index:04d}_00"}
        )
        for index in range(3)
    ]
    report = evaluate_counterfactuals(scenes)
    assert report["pre_registered_gate"]["rules"]["tr3d_score"]["pass"]
    assert report["pre_registered_gate"]["primary_rule_pass"]
    assert report["pre_registered_gate"]["pass"]
    primary = report["pre_registered_gate"]["rules"][
        "tr3d_score_gt_anchor_score"
    ]
    assert primary["cross50_gain_minus_loss"] == 6
    assert primary["cross50_replacement_precision"] == 1.0
    assert primary["cross50_positive_scene_coverage"] == 3
    assert not report["pre_registered_gate"]["direct_activation_authorized"]
    assert not report["pre_registered_gate"]["validation_threshold_or_rule_selection_permitted"]


def test_report_write_is_atomic_create_only(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    _create_only(path, {"ok": True})
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="immutable R3"):
        _create_only(path, {"ok": False})
    assert '"ok": true' in path.read_text(encoding="utf-8")


def test_fixed10_is_veto_only_and_cannot_authorize_calibration() -> None:
    scenes = []
    for index in range(10):
        source = _scene()
        scenes.append(
            SceneCounterfactual(
                **{**source.__dict__, "scene_id": f"scene{index:04d}_00"}
            )
        )
    report = _partition_counterfactuals(scenes, None)
    assert report["mode"] == "development_fixed10_veto_only"
    assert report["decision_partition"] is None
    assert not report["heldout_gate_authoritative"]


def test_full100_partition_uses_exact_heldout90_difference() -> None:
    scenes = []
    for index in range(100):
        source = _scene()
        scenes.append(
            SceneCounterfactual(
                **{**source.__dict__, "scene_id": f"scene{index:04d}_00"}
            )
        )
    development = [scene.scene_id for scene in scenes[::10]]
    report = _partition_counterfactuals(scenes, development)
    assert report["decision_partition"] == "heldout90"
    assert report["heldout_gate_authoritative"]
    assert len(report["development_scene_ids"]) == 10
    assert len(report["heldout_scene_ids"]) == 90
    assert not set(report["development_scene_ids"]) & set(
        report["heldout_scene_ids"]
    )


def test_real_v2_export_keeps_r2_report_paths_only_in_input_reports() -> None:
    root = Path(__file__).resolve().parents[1]
    export_path = root / "reports/tr3d_r3/r3_near_fixed10_v2/export_report.json"
    if not export_path.is_file():
        pytest.skip("real immutable R3 v2 export is not present")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    assert "r2a_export_report" not in export
    assert "r2b_export_report" not in export
    input_reports = _validate_optional_export_report_paths(
        export,
        r2a_enabled=True,
        r2b_enabled=True,
        r2a_export_report=(
            root / "reports/tr3d_r2a/r2a_depth_fixed10_v3/export_report.json"
        ),
        r2b_export_report=(
            root / "reports/tr3d_r2b/r2b_dino_fixed10_v1/export_report.json"
        ),
    )
    assert input_reports["r2a_enabled"] is True
    assert input_reports["r2b_enabled"] is True
