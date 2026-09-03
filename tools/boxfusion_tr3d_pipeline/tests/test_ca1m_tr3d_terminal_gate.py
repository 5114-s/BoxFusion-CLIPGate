from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.ca1m_native_b6_observer import FEATURE_NAMES as NATIVE_FEATURE_NAMES
from boxfusion.ca1m_tr3d_terminal import (
    BOX_MODE,
    COORDINATE_FRAME,
    CORNER_SEMANTICS,
    SCHEMA as TERMINAL_SCHEMA,
    associate_terminal_candidates,
)
from boxfusion.ca1m_tr3d_terminal_gate import (
    BENEFIT_TARGET,
    CA1MTerminalGatePolicy,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    POLICY_SCHEMA,
    QUALITY_TARGET,
    RELATION_FEATURE_NAMES,
    SELECTION_RULE,
    build_terminal_gate_features,
    select_terminal_replacements,
)


SIGNS = np.asarray(
    [
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ],
    dtype=np.float32,
)


def corners(center, extent):
    return np.asarray(center, dtype=np.float32) + SIGNS * (
        np.asarray(extent, dtype=np.float32) * 0.5
    )


def cache(*, materialized=True):
    anchors = np.stack(
        (
            corners([0, 0, 0], [2, 2, 2]),
            corners([10, 0, 0], [2, 2, 2]),
        )
    ).astype(np.float32)
    candidates = np.stack(
        (
            corners([10.05, 0, 0], [2, 2, 2]),
            corners([0.05, 0, 0], [2, 2, 2]),
            corners([0.10, 0, 0], [1.2, 1.2, 1.2]),
            corners([30, 0, 0], [2, 2, 2]),
        )
    ).astype(np.float32)
    anchor_scores = np.asarray([0.40, 0.60], dtype=np.float32)
    candidate_scores = np.asarray([0.90, 0.80, 0.70, 0.60], dtype=np.float32)
    association = associate_terminal_candidates(
        anchor_corners=anchors,
        anchor_scores=anchor_scores,
        candidate_corners=candidates,
        candidate_scores=candidate_scores,
        near_iou=0.15,
    )
    return {
        "schema": np.asarray(TERMINAL_SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "observer_only": np.asarray(True, dtype=np.bool_),
        "mutation_enabled": np.asarray(False, dtype=np.bool_),
        "ground_truth_access": np.asarray(False, dtype=np.bool_),
        "scene_id": np.asarray("48018894"),
        "coordinate_frame": np.asarray(COORDINATE_FRAME),
        "box_mode": np.asarray(BOX_MODE),
        "corner_semantics": np.asarray(CORNER_SEMANTICS),
        "adapter_mode": np.asarray("genuine"),
        "near_iou": np.asarray(0.15, dtype=np.float64),
        "anchor_corners": anchors,
        "anchor_scores": anchor_scores,
        "candidate_corners": candidates,
        "candidate_scores": candidate_scores,
        "candidate_point_count": np.asarray([80, 40, 10, 1], dtype=np.int64),
        "candidate_labels": np.zeros(4, dtype=np.int64),
        "point_count": np.asarray(100, dtype=np.int64),
        "best_anchor_indices": association.best_anchor_indices,
        "best_anchor_iou": association.best_anchor_iou,
        "best_anchor_center_distance_m": association.best_anchor_center_distance_m,
        "near_mask": association.near_mask,
        "materialized_active_verified": np.asarray(materialized, dtype=np.bool_),
    }


def evidence():
    anchor = np.vstack(
        (
            np.linspace(0.01, 0.14, 14),
            np.linspace(0.21, 0.34, 14),
        )
    ).astype(np.float32)
    candidate = np.vstack(
        tuple(np.linspace(0.40 + row * 0.10, 0.53 + row * 0.10, 14) for row in range(4))
    ).astype(np.float32)
    candidate[:, 0] = np.asarray([0.90, 0.80, 0.70, 0.60], dtype=np.float32)
    return anchor, candidate


def gate(values=None, *, threshold=0.5):
    weights = np.zeros(len(FEATURE_NAMES), dtype=float)
    for name, weight in (values or {}).items():
        weights[FEATURE_NAMES.index(name)] = weight
    return {
        "weights": weights.tolist(),
        "feature_mean": np.zeros(len(FEATURE_NAMES)).tolist(),
        "feature_scale": np.ones(len(FEATURE_NAMES)).tolist(),
        "bias": 0.0,
        "probability_threshold": threshold,
    }


def policy_payload(*, authorized=True, quality=None, benefit=None, maximum=8):
    forbidden = [f"{70_000_000 + index:08d}" for index in range(107)]
    return {
        "schema": POLICY_SCHEMA,
        "complete": True,
        "activation_authorized": authorized,
        "train_only": True,
        "scene_group_split": True,
        "ground_truth_used_only_for_training": True,
        "candidate_collection_ground_truth_access": False,
        "validation_predictions_used_for_training": False,
        "validation_scene_access": False,
        "one_time_audit_passed": True,
        "geometry_only": True,
        "preserve_anchor_scores": True,
        "preserve_row_order": True,
        "preserve_row_count": True,
        "clip_semantics_unchanged": True,
        "dataset": "ca1m",
        "observer_schema": TERMINAL_SCHEMA,
        "native_feature_names": list(NATIVE_FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_schema": FEATURE_SCHEMA,
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "selection_rule": SELECTION_RULE,
        "gate_train_fold_ids": [2, 3, 4],
        "calibration_fold_ids": [0],
        "one_time_audit_fold_ids": [1],
        "validation_overlap_count": 0,
        "training_scene_ids": ["48018894", "49739821"],
        "forbidden_validation_scene_ids": forbidden,
        "training_data_sha256": "1" * 64,
        "observer_audit_sha256": "2" * 64,
        "training_scene_list_sha256": "3" * 64,
        "forbidden_validation_scene_list_sha256": "4" * 64,
        "near_iou": 0.15,
        "max_replacements_per_scene": maximum,
        "quality25_gate": quality or gate(),
        "benefit05_gate": benefit or gate(),
    }


def write_policy(path: Path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    path.chmod(0o444)
    return CA1MTerminalGatePolicy.load(path)


def test_feature_schema_is_anchor14_candidate14_then_twelve_relations():
    assert len(NATIVE_FEATURE_NAMES) == 14
    assert len(FEATURE_NAMES) == 40
    assert FEATURE_NAMES[:14] == tuple(f"anchor_{name}" for name in NATIVE_FEATURE_NAMES)
    assert FEATURE_NAMES[14:28] == tuple(
        f"candidate_{name}" for name in NATIVE_FEATURE_NAMES
    )
    assert FEATURE_NAMES[28:] == RELATION_FEATURE_NAMES


def test_feature_builder_is_row_aligned_finite_and_gt_free():
    value = cache()
    anchor, candidate = evidence()
    frozen = {key: np.array(item, copy=True) for key, item in value.items()}
    result = build_terminal_gate_features(
        value, anchor_native_evidence=anchor, candidate_native_evidence=candidate
    )
    assert result.scene_id == "48018894"
    assert result.candidate_rows.tolist() == [0, 1, 2]
    assert result.anchor_indices.tolist() == [1, 0, 0]
    assert result.features.shape == (3, 40)
    assert result.features.dtype == np.float32
    assert np.isfinite(result.features).all()
    assert np.array_equal(result.features[1, :14], anchor[0])
    assert np.array_equal(result.features[1, 14:28], candidate[1])
    relation = dict(zip(RELATION_FEATURE_NAMES, result.features[1, 28:]))
    assert relation["candidate_minus_anchor_score"] == pytest.approx(0.40)
    assert relation["candidate_point_support_fraction"] == pytest.approx(0.40)
    assert relation["candidate_global_rank_fraction"] == pytest.approx(1 / 3)
    assert relation["candidate_anchor_group_rank_fraction"] == pytest.approx(0.0)
    assert relation["log1p_anchor_group_size"] == pytest.approx(np.log1p(2))
    assert relation["candidate_score_minus_best_sibling"] == pytest.approx(0.10)
    for key in value:
        assert np.array_equal(np.asarray(value[key]), frozen[key])


def test_policy_loader_requires_immutable_authorized_split_manifest(tmp_path):
    target = tmp_path / "policy.json"
    policy = write_policy(target, policy_payload())
    assert policy.activation_authorized is True
    assert policy.max_replacements_per_scene == 8
    assert policy.quality25_gate.weights.shape == (40,)
    assert not policy.quality25_gate.weights.flags.writeable

    unauthorized = tmp_path / "unauthorized.json"
    payload = policy_payload(authorized=False)
    unauthorized.write_text(json.dumps(payload))
    unauthorized.chmod(0o444)
    with pytest.raises(ValueError, match="activation_authorized"):
        CA1MTerminalGatePolicy.load(unauthorized)

    wrong_split = tmp_path / "wrong_split.json"
    payload = policy_payload()
    payload["one_time_audit_fold_ids"] = [0]
    wrong_split.write_text(json.dumps(payload))
    wrong_split.chmod(0o444)
    with pytest.raises(ValueError, match="provenance/schema"):
        CA1MTerminalGatePolicy.load(wrong_split)


def test_selection_uses_benefit_then_quality_and_only_returns_indices(tmp_path):
    # Equal benefit logits; candidate native detector score breaks the anchor-0
    # tie through quality even though candidate row 2 has the lower TR3D score.
    quality = gate({"candidate_support_given_depth": 8.0}, threshold=0.5)
    benefit = gate(threshold=0.5)
    policy = write_policy(
        tmp_path / "policy.json",
        policy_payload(quality=quality, benefit=benefit, maximum=8),
    )
    anchor, candidate = evidence()
    result = select_terminal_replacements(
        cache(),
        anchor_native_evidence=anchor,
        candidate_native_evidence=candidate,
        policy=policy,
    )
    assert result.anchor_indices.tolist() == [0, 1]
    assert result.candidate_rows.tolist() == [2, 0]
    assert result.evaluated_count == 3
    assert result.eligible_count == 3
    assert not hasattr(result, "corners")


def test_selection_requires_materialized_b6_identity(tmp_path):
    policy = write_policy(tmp_path / "policy.json", policy_payload())
    anchor, candidate = evidence()
    with pytest.raises(ValueError, match="verified active B6"):
        select_terminal_replacements(
            cache(materialized=False),
            anchor_native_evidence=anchor,
            candidate_native_evidence=candidate,
            policy=policy,
        )


def test_feature_builder_rejects_tampered_association():
    value = cache()
    value["best_anchor_indices"] = value["best_anchor_indices"].copy()
    value["best_anchor_indices"][0] = 0
    anchor, candidate = evidence()
    with pytest.raises(ValueError, match="differs from recomputation"):
        build_terminal_gate_features(
            value,
            anchor_native_evidence=anchor,
            candidate_native_evidence=candidate,
        )


def test_feature_builder_rejects_candidate_native_row_mismatch():
    value = cache()
    anchor, candidate = evidence()
    candidate = candidate[[1, 0, 2, 3]]
    with pytest.raises(ValueError, match="native detector_score"):
        build_terminal_gate_features(
            value,
            anchor_native_evidence=anchor,
            candidate_native_evidence=candidate,
        )
