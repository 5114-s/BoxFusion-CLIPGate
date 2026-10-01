from __future__ import annotations

import numpy as np
import pytest

from tools.run_scannet_raw_boxer_clip_vocab_shadow_full100 import (
    ClipVocabShadowError,
    _crop_rgb,
    _index_owl_rows,
    _prefilter_reason,
    _resolve_evidence,
    _resolve_owl_target_group,
    _score_summary,
    _track_summary,
)


def _receipt(**overrides):
    row = {
        "min_pairwise_aabb_iou": 0.5,
        "max_pairwise_center_distance_m": 0.1,
        "first_last_frame_span": 50,
        "max_camera_baseline_m": 0.2,
        "max_view_ray_span_deg": 12.0,
        "min_medoid_aabb_extent_m": 0.3,
        "max_native_aabb_iou": 0.0,
        "max_candidate_in_native_containment": 0.0,
        "max_native_in_candidate_containment": 0.0,
    }
    row.update(overrides)
    return row


def _owl(frame, name, sem, bbox):
    return {
        "time_ns": str(frame),
        "frame_id": "0",
        "img_width": "960",
        "img_height": "960",
        "x1": str(bbox[0]),
        "y1": str(bbox[1]),
        "x2": str(bbox[2]),
        "y2": str(bbox[3]),
        "name": name,
        "sem_id": str(sem),
        "prob": "0.7",
    }


def test_exact_raw_to_owl_mapping_uses_boxer_instance_not_label_order():
    owl_rows = [
        _owl(25, "chair", 9, (0, 0, 10, 10)),
        _owl(25, "chair", 9, (20, 20, 40, 40)),
        _owl(25, "chair", 9, (50, 50, 80, 80)),
    ]
    raw_rows = [
        {
            "time_ns": "25",
            "name": "chair",
            "instance": "2",
            "sem_id": "9",
            "prob": "0.8",
        }
    ]
    resolved = _resolve_evidence(
        raw_rows=raw_rows,
        owl_by_frame=_index_owl_rows(owl_rows),
        raw_source_row=0,
        expected_frame_id=25,
    )
    assert resolved["boxer_instance"] == 2
    assert resolved["owl_bbox_xyxy"] == [50.0, 50.0, 80.0, 80.0]


def test_exact_mapping_rejects_semantic_provenance_mismatch():
    raw_rows = [
        {
            "time_ns": "25",
            "name": "chair",
            "instance": "0",
            "sem_id": "9",
            "prob": "0.8",
        }
    ]
    with pytest.raises(ClipVocabShadowError, match="provenance mismatch"):
        _resolve_evidence(
            raw_rows=raw_rows,
            owl_by_frame=_index_owl_rows([_owl(25, "table", 10, (0, 0, 10, 10))]),
            raw_source_row=0,
            expected_frame_id=25,
        )


def test_frozen_prefilter_selects_pre_nms_and_pre_cap_insertion_set():
    assert _prefilter_reason(_receipt(decision="accepted")) is None
    assert _prefilter_reason(_receipt(decision="self_nms")) is None
    assert _prefilter_reason(_receipt(decision="scene_cap")) is None
    assert (
        _prefilter_reason(_receipt(decision="semantic_inconsistent"))
        == "prior_v2_semantic_inconsistent"
    )


def test_crop_uses_floor_ceil_and_clamps():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    crop = np.asarray(_crop_rgb(image, (-1.2, 0.8, 3.2, 9.0)))
    assert crop.shape == (4, 4, 3)
    np.testing.assert_array_equal(crop, image[0:4, 0:4])


def test_score_summary_reports_all_target_non_target_and_margin():
    vocabulary = ["target_a", "other", "target_b", "other_2"]
    result = _score_summary(
        np.asarray([0.2, 0.8, 0.7, 0.1]),
        vocabulary,
        target_indices=[0, 2],
        non_target_indices=[1, 3],
    )
    assert result["all_vocab_top1_name"] == "other"
    assert result["all_vocab_top1_is_target"] is False
    assert result["target_best_name"] == "target_b"
    assert result["non_target_best_name"] == "other"
    assert result["target_non_target_margin"] == pytest.approx(-0.1)


def test_track_gate_requires_three_same_owl_alias_and_two_matching_clip_votes():
    evidence = []
    for index in range(3):
        evidence.append(
            {
                "owl_exact_target_alias_groups": ["chair"],
                "all_vocab_top1_is_target": index < 2,
                "all_vocab_top1_target_alias_groups": ["chair"] if index < 2 else [],
                "target_best_name": "chair",
                "target_best_cosine": 0.25,
                "target_non_target_margin": 0.01,
            }
        )
    summary = _track_summary(evidence)
    assert summary["owl_collapsed_target_group"] == "chair"
    assert summary["clip_top1_same_group_as_owl_votes"] == 2
    assert summary["gate_pass"] is True

    evidence[2]["owl_exact_target_alias_groups"] = ["table"]
    rejected = _track_summary(evidence)
    assert rejected["gate_pass"] is False
    assert "owl_exact_alias_all_three_same_group" in rejected["gate_rejection_reasons"]


@pytest.mark.parametrize(
    ("owl_name", "expected"),
    [
        ("bookcase", "cabinet_or_bookshelf"),
        ("file_cabinet", "cabinet_or_bookshelf"),
        ("trash_can", "garbage_bin"),
        ("bathtub", "bathtub"),
        ("bath-tub", "bathtub"),
        ("urinal", "toilet"),
        ("shower_curtain", "curtain"),
        ("monitor", None),
        ("plastic_bag", None),
    ],
)
def test_wordnet_owl_names_use_separate_fixed_alias_table(owl_name, expected):
    assert _resolve_owl_target_group(owl_name) == expected
