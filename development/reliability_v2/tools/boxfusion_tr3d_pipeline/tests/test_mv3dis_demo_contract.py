from __future__ import annotations

import ast
from pathlib import Path

import yaml

from boxfusion.mv3dis_depth_lite import resolve_mv3dis_depth_lite_config


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo.py"
CONFIG = (
    ROOT
    / "config"
    / "scannet_qim_puf_arbitration_mv3dis_shadow.yaml"
)


def _nested_function(name: str) -> ast.FunctionDef:
    tree = ast.parse(DEMO.read_text())
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_frozen_real_stream_config_is_shadow_only():
    cfg = yaml.safe_load(CONFIG.read_text())
    resolved = resolve_mv3dis_depth_lite_config(cfg["mv3dis_depth_lite"])
    assert resolved == {
        "enabled": True,
        "observer_only": True,
        "max_guides_per_track": 5,
        "max_depth_frames": 80,
        "max_proposals": 256,
        "max_qim_candidates": 3,
        "projection_budget_points": 8192,
        "points_per_projection": 64,
        "frame_visibility_threshold": 0.30,
        "box_visibility_threshold": 0.90,
        "candidate_dominance_threshold": 0.90,
        "min_history_views": 2,
        "alpha": 0.05,
        "max_diagnostic_examples": 1024,
    }
    assert "reliable_views" not in cfg["box_fusion"]


def test_mv_query_has_no_puf_or_semantic_inputs():
    node = _nested_function("query_mv3dis_depth_lite")
    parameter_names = {argument.arg for argument in node.args.args}
    assert parameter_names == {
        "frame_index",
        "scene_identifier",
        "current_predictions",
        "current_sample",
        "qim_batch",
    }
    loaded_names = {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }
    assert "puf_lite" not in loaded_names
    assert "puf_batch" not in loaded_names
    assert "clip_model" not in loaded_names
    assert "text_prompt" not in loaded_names


def test_mv_commit_uses_native_post_association_trace_only():
    node = _nested_function("commit_mv3dis_depth_lite")
    calls = {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    attributes = {
        child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
    }
    assert "derive_committed_track_ids" in calls
    assert "commit" in attributes
    loaded_names = {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    }
    assert "association_events" in loaded_names
    assert "native_targets" in loaded_names
    assert "puf_lite" not in loaded_names


def test_query_precedes_clip_and_native_association_and_summary_is_emitted():
    source = DEMO.read_text()
    loop_start = source.index(
        "pred_instances.pred_boxes_3d.transform2world"
    )
    query = source.index("mv3dis_query_context = query_mv3dis_depth_lite", loop_start)
    clip = source.index("appearance_gate_cfg =", query)
    association = source.index("Instances3D.spatial_association", clip)
    commit = source.index("commit_moon_qim_lite(", association)
    assert loop_start < query < clip < association < commit
    assert "MV3DIS-Depth-lite S0 shadow JSON | " in source

