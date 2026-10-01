from __future__ import annotations

import hashlib
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from boxfusion.online_ablation import apply_online_ablation_profile
from boxfusion.online_refinement import resolve_online_refinement_config
from boxfusion.quality_score import QUALITY_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ca1m_c2_b6_zero_shot_observer.yaml"
CONFIG_SHA = "310754ab8b6aa5fcfe378736e8668fed6c5d78468005f5396dbfd480da4068ad"


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit_tool = load_tool(
    "ca1m_c2_identity_tool", "audit_ca1m_c2_b6_zero_shot_observer.py"
)
counterfactual_tool = load_tool(
    "ca1m_c2_counterfactual_tool",
    "evaluate_ca1m_b6_zero_shot_counterfactual.py",
)


def test_ca1m_c2_profile_is_strict_identity_observer() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA
    raw = yaml.safe_load(CONFIG.read_text())
    profiled = apply_online_ablation_profile(raw, "quality_observer")
    online = resolve_online_refinement_config(profiled)
    assert profiled["dataset"] == "CA1M"
    assert profiled["detection"]["score_thresh"] == 0.4
    assert profiled["data"]["gap"] == 20
    assert profiled["lifting"]["backend"] == "boxer"
    assert profiled["lifting"]["boxer"]["selective_gate"] == {
        "enabled": True,
        "max_center_shift_m": 0.10,
        "min_volume_ratio": 0.50,
        "max_volume_ratio": 2.00,
    }
    assert online["scannet_axis_aligned_only"] is False
    assert online["appearance_memory"]["enabled"] is True
    assert online["quality"]["enabled"] is False
    assert online["refit"]["enabled"] is False
    assert online["box_refiner"]["enabled"] is False
    assert online["supplemental_output"]["enabled"] is False
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert online["output_filter"]["minimum_extent"] == 0.0


def prediction_rows(score: float = 0.7):
    corners = np.arange(24, dtype=np.float32).reshape(8, 3)
    return [(0, corners, score)]


def test_prediction_identity_rejects_any_score_or_obb_change() -> None:
    anchor = prediction_rows()
    assert audit_tool.compare_predictions("1", anchor, prediction_rows())[
        "semantic_identity"
    ]
    with pytest.raises(ValueError, match="score differs"):
        audit_tool.compare_predictions("1", anchor, prediction_rows(0.71))
    changed = prediction_rows()
    changed[0][1][0, 0] += np.float32(1e-6)
    with pytest.raises(ValueError, match="OBB corners differ"):
        audit_tool.compare_predictions("1", anchor, changed)


def test_counterfactual_pickle_keeps_geometry_and_is_create_only(tmp_path: Path) -> None:
    rows = prediction_rows()
    path = tmp_path / "1_boxes.pkl"
    counterfactual_tool.save_prediction_atomic(path, rows)
    loaded = counterfactual_tool.load_prediction(path)
    assert loaded[0][0] == 0
    assert np.array_equal(loaded[0][1], rows[0][1])
    assert loaded[0][2] == rows[0][2]
    with pytest.raises(FileExistsError):
        counterfactual_tool.save_prediction_atomic(path, rows)


def test_diagnostic_schema_and_mapping(tmp_path: Path) -> None:
    path = tmp_path / "1_tracks.npz"
    names = np.asarray(QUALITY_FEATURE_NAMES, dtype=np.str_)
    features = np.full((1, len(names)), 0.5, dtype=np.float32)
    features[0, 0] = 0.7
    summary = {
        "enabled": True,
        "candidate_ttl_clock": "provider_call",
        "candidate_archived_total": 0,
        "supplemental_output": 0,
        "refits_accepted": 0,
        "neural_refits_accepted": 0,
        "provider_calls": 1,
        "provider_seconds": 0.1,
        "appearance_seconds": 0.2,
        "geometry_seconds": 0.3,
    }
    np.savez_compressed(
        path,
        scene_id=np.asarray("1"),
        boxes=audit_tool.corners_to_center_size(
            np.stack([prediction_rows()[0][1]], axis=0)
        ),
        scores=np.asarray([0.7], dtype=np.float32),
        quality_features=features,
        points=np.zeros((1, 8, 3), dtype=np.float32),
        point_mask=np.ones((1, 8), dtype=bool),
        source_indices=np.asarray([0], dtype=np.int64),
        track_ids=np.asarray([5], dtype=np.int64),
        result_indices=np.asarray([0], dtype=np.int64),
        labels=np.asarray([""], dtype=np.str_),
        quality_feature_names=names,
        summary_json=np.asarray(__import__("json").dumps(summary)),
    )
    result = audit_tool.audit_diagnostic("1", path, prediction_rows())
    assert result["observed_rows"] == 1
    assert result["coverage"] == 1.0

    bad_path = tmp_path / "bad_tracks.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["boxes"] = values["boxes"].copy()
    values["boxes"][0, 0] += np.float32(1e-3)
    np.savez_compressed(bad_path, **values)
    with pytest.raises(ValueError, match="do not map to observer OBB"):
        audit_tool.audit_diagnostic("1", bad_path, prediction_rows())


def test_metric_parser_requires_three_complete_thresholds() -> None:
    text = "\n".join(
        f"eval {name}: {value:.6f}"
        for value in (0.1, 0.2, 0.3)
        for name in ("mAP", "APrec", "ARecall")
    )
    metrics = counterfactual_tool.parse_metrics(text)
    assert set(metrics) == {"0.15", "0.25", "0.50"}
    with pytest.raises(ValueError, match="expected 9"):
        counterfactual_tool.parse_metrics("eval mAP: 0.1")


def test_runtime_log_requires_one_valid_summary(tmp_path: Path) -> None:
    path = tmp_path / "scene.log"
    path.write_text("Cost: 10.00 s Average FPS: 12.50\n")
    result = audit_tool.audit_runtime_log(path)
    assert result["cost_seconds"] == 10.0
    assert result["average_fps"] == 12.5
    assert result["frame_equivalent"] == 125.0
    path.write_text("Cost: 1.00 s Average FPS: 2.00\nCost: 2.00 s Average FPS: 3.00\n")
    with pytest.raises(ValueError, match="expected one"):
        audit_tool.audit_runtime_log(path)
