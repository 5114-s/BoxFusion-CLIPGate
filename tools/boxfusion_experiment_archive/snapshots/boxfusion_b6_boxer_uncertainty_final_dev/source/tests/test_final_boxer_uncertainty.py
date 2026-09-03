from pathlib import Path

import numpy as np
import pytest

from boxfusion.box_fusion import BoxFusion
from boxfusion.boxer_uncertainty import (
    fixed_topk_uncertainty_reweighting,
    resolve_final_boxer_uncertainty_config,
)


def _base_selection():
    return {
        "weights": np.asarray([0.8, 0.4, 1.0, 0.2], dtype=np.float32),
        "selected_indices": np.asarray([2, 0, 1], dtype=np.int64),
        "selected_weights": np.asarray([1.0, 0.8, 0.4], dtype=np.float32),
    }


def test_fixed_topk_reweighting_never_reselects_or_reorders():
    adjusted = fixed_topk_uncertainty_reweighting(
        _base_selection(),
        np.asarray([0.5, 0.25, 0.9, 0.01], dtype=np.float32),
        np.asarray([True, False, True, True]),
        {"minimum_confidence": 0.05, "confidence_power": 1.0},
    )

    assert adjusted["selected_indices"].tolist() == [2, 0, 1]
    assert adjusted["ranked_indices"].tolist() == [2, 0, 1]
    assert not bool(adjusted["selection_changed"])
    assert not bool(adjusted["ranking_changed"])
    # Selected Boxer rows 0 and 2 are weighted; CuTR row 1 is neutral.
    assert adjusted["uncertainty_factors"].tolist() == pytest.approx(
        [0.5, 1.0, 0.9, 1.0]
    )
    # Row 3 is not in the frozen Top-K and therefore cannot affect fusion.
    assert adjusted["uncertainty_weights"][3] == pytest.approx(0.2)
    assert bool(adjusted["effective_weights_changed"])


def test_uniform_selected_factors_cancel_under_mean_normalization():
    adjusted = fixed_topk_uncertainty_reweighting(
        _base_selection(),
        np.asarray([0.5, 0.2, 0.5, 0.1], dtype=np.float32),
        np.asarray([True, False, True, True]),
        {"minimum_confidence": 0.05, "confidence_power": 1.0},
    )
    # Row 1 is CuTR, so this is not uniform over all selected rows.
    assert bool(adjusted["effective_weights_changed"])

    adjusted_uniform = fixed_topk_uncertainty_reweighting(
        _base_selection(),
        np.asarray([0.5, 0.5, 0.5, 0.1], dtype=np.float32),
        np.asarray([True, True, True, True]),
        {"minimum_confidence": 0.05, "confidence_power": 1.0},
    )
    assert not bool(adjusted_uniform["effective_weights_changed"])


def test_final_config_is_separate_and_requires_diagnostics(tmp_path):
    cfg = resolve_final_boxer_uncertainty_config({})
    assert cfg["mode"] == "disabled"
    with pytest.raises(ValueError, match="diagnostics_dir"):
        resolve_final_boxer_uncertainty_config(
            {"final_boxer_uncertainty": {"mode": "observer"}}
        )
    cfg = resolve_final_boxer_uncertainty_config(
        {
            "final_boxer_uncertainty": {
                "mode": "active",
                "diagnostics_dir": str(tmp_path),
            }
        }
    )
    assert cfg["mode"] == "active"
    assert cfg["diagnostics_dir"] == str(tmp_path)


def _stats():
    return {
        "recipes": 1,
        "output_rows": 0,
        "eligible_rows": 0,
        "matched_rows": 0,
        "weight_changed_rows": 0,
        "optimized_rows": 0,
        "applied_rows": 0,
        "selection_changed_rows": 0,
        "ranking_changed_rows": 0,
        "scene_fallback": 0,
        "runtime_ms": 0.0,
        "rejects": {},
    }


def _final_fuser(mode, tmp_path: Path):
    fuser = BoxFusion.__new__(BoxFusion)
    fuser.final_boxer_uncertainty_cfg = {
        "mode": mode,
        "confidence_power": 1.0,
        "minimum_confidence": 0.05,
        "diagnostics_dir": str(tmp_path),
    }
    fuser._uncertainty_scene_id = "scene_test"
    fuser._final_uncertainty_records = []
    fuser._final_uncertainty_contract = None
    fuser.final_boxer_uncertainty_stats = _stats()
    selection = {
        "weights": np.asarray([1.0, 0.8, 0.6], dtype=np.float32),
        "selected_indices": np.asarray([0, 1, 2], dtype=np.int64),
        "selected_weights": np.asarray([1.0, 0.8, 0.6], dtype=np.float32),
    }
    fuser._final_uncertainty_recipes = [
        {
            "sequence": 0,
            "global_box_index_at_fusion": 0,
            "source_group": (10, 11, 12),
            "candidate_indices": np.asarray([10, 11, 12]),
            "selected_source_indices": np.asarray([10, 11, 12]),
            "base_selection": selection,
            "projected_corners": np.zeros((3, 8, 2), dtype=np.float32),
            "camera_poses": np.repeat(
                np.eye(4, dtype=np.float32)[None], 3, axis=0
            ),
            "boxer_confidence": np.asarray(
                [0.95, 0.55, 0.25], dtype=np.float32
            ),
            "boxer_geometry_applied": np.asarray([True, True, True]),
        }
    ]

    def fake_optimize(**kwargs):
        candidate = np.asarray(
            kwargs["initial_xyzlwh"], dtype=np.float32
        ).copy()
        candidate[0] += 0.05
        return (
            candidate,
            np.asarray(kwargs["fixed_rotation"], dtype=np.float32).copy(),
            True,
            2,
        )

    fuser._optimize_final_uncertainty_candidate = fake_optimize
    return fuser


@pytest.mark.parametrize("mode,expect_changed", [("observer", False), ("active", True)])
def test_final_route_changes_only_active_geometry(
    mode, expect_changed, tmp_path
):
    fuser = _final_fuser(mode, tmp_path)
    global_box = np.asarray([[0.0, 0.0, 0.0, 1.0, 1.2, 0.8]], dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)[None]
    baseline = fuser._obb_corners_numpy(global_box[0], rotation[0])[None]
    scores = np.asarray([0.73125], dtype=np.float32)
    score_bytes = scores.tobytes()

    output = fuser.apply_final_boxer_uncertainty(
        baseline_corners=baseline,
        scores=scores,
        source_indices=np.asarray([0], dtype=np.int64),
        stable_ids=np.asarray([10], dtype=np.int64),
        global_xyzlwh=global_box,
        global_rotations=rotation,
        global_stable_ids=np.asarray([10], dtype=np.int64),
        frozen_fusion_groups=((10, 11, 12),),
        minimum_extent=0.4,
    )

    assert (not np.array_equal(output, baseline)) is expect_changed
    assert output.shape == baseline.shape
    assert scores.tobytes() == score_bytes
    assert fuser.final_boxer_uncertainty_stats["selection_changed_rows"] == 0
    assert fuser.final_boxer_uncertainty_stats["ranking_changed_rows"] == 0
    assert fuser._final_uncertainty_contract["protected_fields_equal"]
    if mode == "observer":
        assert fuser.final_boxer_uncertainty_stats["applied_rows"] == 0
    else:
        assert fuser.final_boxer_uncertainty_stats["applied_rows"] == 1

    diagnostic = fuser._write_final_boxer_uncertainty_diagnostics()
    assert Path(diagnostic).is_file()


def test_final_route_fails_closed_on_identity_mismatch(tmp_path):
    fuser = _final_fuser("active", tmp_path)
    global_box = np.asarray([[0.0, 0.0, 0.0, 1.0, 1.2, 0.8]], dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)[None]
    baseline = fuser._obb_corners_numpy(global_box[0], rotation[0])[None]
    output = fuser.apply_final_boxer_uncertainty(
        baseline_corners=baseline,
        scores=np.asarray([0.5], dtype=np.float32),
        source_indices=np.asarray([0], dtype=np.int64),
        stable_ids=np.asarray([99], dtype=np.int64),
        global_xyzlwh=global_box,
        global_rotations=rotation,
        global_stable_ids=np.asarray([10], dtype=np.int64),
        frozen_fusion_groups=((10, 11, 12),),
        minimum_extent=0.4,
    )
    assert np.array_equal(output, baseline)
    assert fuser.final_boxer_uncertainty_stats["applied_rows"] == 0
    assert fuser.final_boxer_uncertainty_stats["rejects"] == {
        "stable_id_mismatch": 1
    }
