import importlib.util
import os

import numpy as np


SOURCE = os.environ.get(
    "BOXFUSION_INSTANCES",
    (
        "/data/ZhaoX/BoxFusion/upstream_clean/"
        "BoxFusion_scorefix/boxfusion/instances.py"
    ),
)
spec = importlib.util.spec_from_file_location("boxfusion_instances", SOURCE)
instances = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instances)


BASE_CFG = {
    "enabled": True,
    "geometry_min_iou": 0.08,
    "hard_geometry_iou": 0.45,
    "low_similarity": 0.45,
    "high_similarity": 0.75,
    "max_iou_penalty": 0.10,
    "max_iou_bonus": 0.0,
    "confidence_floor": 0.35,
    "confidence_full": 0.75,
}


def decide(geometry, similarity, query_score=0.9, candidate_score=0.9, **cfg):
    gate_cfg = dict(BASE_CFG)
    gate_cfg.update(cfg)
    geometry = np.asarray(geometry, dtype=np.float32)
    similarity = np.asarray(similarity, dtype=np.float32)
    scores = np.full_like(geometry, candidate_score)
    return instances.appearance_gate_decisions(
        geometry,
        similarity,
        query_score,
        scores,
        base_threshold=0.10,
        gate_cfg=gate_cfg,
    )


def test_disabled_gate_is_exact_baseline_iou_decision():
    result = instances.appearance_gate_decisions(
        [0.09, 0.10, 0.11],
        None,
        0.9,
        [0.9, 0.9, 0.9],
        base_threshold=0.10,
        gate_cfg={"enabled": False},
    )
    np.testing.assert_array_equal(result["accepted"], [False, False, True])


def test_low_similarity_softly_protects_but_does_not_hard_veto():
    result = decide([0.15, 0.25], [0.20, 0.20])
    np.testing.assert_allclose(result["thresholds"], [0.20, 0.20])
    np.testing.assert_array_equal(result["accepted"], [False, True])
    np.testing.assert_array_equal(result["protected"], [True, False])


def test_low_confidence_falls_back_to_original_geometry():
    result = decide(
        [0.15],
        [0.20],
        query_score=0.35,
        candidate_score=0.9,
    )
    np.testing.assert_allclose(result["reliability"], [0.0])
    np.testing.assert_allclose(result["thresholds"], [0.10])
    np.testing.assert_array_equal(result["accepted"], [True])


def test_hard_geometry_overrides_low_similarity():
    result = decide(
        [0.50],
        [0.10],
        max_iou_penalty=0.60,
    )
    np.testing.assert_array_equal(result["accepted"], [True])
    np.testing.assert_array_equal(result["hard_overrides"], [True])


def test_positive_bonus_is_optional_and_respects_geometry_minimum():
    result = decide(
        [0.07, 0.09],
        [0.90, 0.90],
        max_iou_bonus=0.03,
    )
    np.testing.assert_array_equal(result["accepted"], [False, True])
    np.testing.assert_array_equal(result["promoted"], [False, True])


def test_cosine_similarity_normalizes_inputs():
    result = instances.cosine_similarity_to_many(
        [2.0, 0.0],
        [[4.0, 0.0], [0.0, 3.0], [-1.0, 0.0]],
    )
    np.testing.assert_allclose(result, [1.0, 0.0, -1.0], atol=1e-6)


def test_stage_config_overrides_shared_thresholds_only_for_that_stage():
    config = {
        "enabled": True,
        "low_similarity": 0.45,
        "high_similarity": 0.75,
        "spatial": {
            "low_similarity": 0.65,
            "high_similarity": 0.85,
        },
    }
    spatial = instances.resolve_appearance_gate_config(config, "spatial")
    correspondence = instances.resolve_appearance_gate_config(
        config, "correspondence"
    )
    assert spatial["low_similarity"] == 0.65
    assert spatial["high_similarity"] == 0.85
    assert correspondence["low_similarity"] == 0.45
    assert correspondence["high_similarity"] == 0.75
