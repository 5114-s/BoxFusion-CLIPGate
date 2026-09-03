import ast
from pathlib import Path

import yaml

from boxfusion.cutr_residual_cross_view_r1 import (
    build_cutr_residual_cross_view_r1,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "demo.py"
S0_CONFIG_PATH = ROOT / "config" / "scannet_cutr_residual_shadow.yaml"
R1_CONFIG_PATH = ROOT / "config" / "scannet_cutr_residual_r1_shadow.yaml"


def _source():
    return DEMO_PATH.read_text(encoding="utf-8")


def _function_source(name):
    source = _source()
    tree = ast.parse(source)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def test_r1_config_is_frozen_s0_plus_one_block():
    s0 = yaml.safe_load(S0_CONFIG_PATH.read_text(encoding="utf-8"))
    r1 = yaml.safe_load(R1_CONFIG_PATH.read_text(encoding="utf-8"))
    section = r1.pop("cutr_residual_cross_view_r1")
    assert r1 == s0
    assert section == {
        "enabled": True,
        "observer_only": True,
        "descriptor_dim": 256,
        "descriptor_cosine": 0.80,
        "translation_gap_m": 0.80,
        "rotation_gap_deg": 30.0,
        "depth_alpha": 0.05,
        "frame_visibility": 0.30,
        "box_visibility": 0.90,
        "min_component_nodes": 3,
        "min_component_edges": 2,
        "max_nodes_per_track": 5,
        "projection_budget_points": 8192,
        "max_receipts": 1024,
    }
    assert build_cutr_residual_cross_view_r1(
        yaml.safe_load(R1_CONFIG_PATH.read_text(encoding="utf-8"))
    ).enabled


def test_r1_consumes_row_aligned_base_assignments_only():
    helper = _function_source("observe_cutr_residual_keyframe")
    assert "base_result = cutr_residual_birth_lite.observe(" in helper
    assert "for assignment in base_result.assignments" in helper
    assert "side_row_by_raw[assignment.raw_index]" in helper
    assert "if len(assigned_side_rows) > 64:" in helper
    assert helper.index("base_result = cutr_residual_birth_lite.observe(") < helper.index(
        "assigned_side_rows = tuple("
    )


def test_r1_copies_descriptor_and_raw_box_after_assignment_cap():
    helper = _function_source("observe_cutr_residual_keyframe")
    assert "residual_predictions.object_desc[selector]" in helper
    assert "residual_predictions.pred_boxes[selector]" in helper
    assert "pred_proj_xy[selector]" not in helper
    assert "world_boxes.tensor[selector, :3]" in helper
    assert "world_boxes.tensor[selector, 3:6]" in helper
    assert "world_boxes.R[selector]" in helper


def test_r1_uses_current_aligned_depth_intrinsics_and_world_pose():
    helper = _function_source("observe_cutr_residual_keyframe")
    assert 'sample["wide"]["depth"][-1].numpy()' in helper
    assert 'sample["sensor_info"].wide.depth.K[-1].numpy()' in helper
    assert "sample_depth_guide_points_batch(" in helper
    assert "camera_to_world = np.array(" in helper
    assert "sample[\"sensor_info\"].gt.depth.K" not in helper


def test_r1_batch_and_row_failures_abstain_without_skipping_observe():
    helper = _function_source("observe_cutr_residual_keyframe")
    assert '"r1_batch_extraction_failed"' in helper
    assert '"invalid_depth_guide"' in helper
    assert '"invalid_r1_evidence"' in helper
    assert "cutr_residual_cross_view_r1.observe(" in helper
    observe_index = helper.index("cutr_residual_cross_view_r1.observe(")
    assert helper.index("if assigned_side_rows:") < observe_index
    assert helper.index("evidence_rows = []") < observe_index


def test_r1_native_guard_covers_all_array_valued_cutr_fields():
    helper = _function_source("observe_cutr_residual_keyframe")
    snapshot = _function_source("snapshot_native_fields")
    assert "instances.get_fields().items()" in snapshot
    assert "value.tensor.detach().cpu().numpy()" in snapshot
    assert "value.R.detach().cpu().numpy()" in snapshot
    assert "native_before = snapshot_native_fields(native_predictions)" in helper
    assert "native_after = snapshot_native_fields(native_predictions)" in helper
    assert "np.array_equal(before, after)" in helper


def test_r1_is_true_keyframe_only_and_terminal_stale_cannot_observe():
    source = _source()
    keyframe = source.index("# only process keyframes")
    observe = source.index("observe_cutr_residual_keyframe(", keyframe)
    native_empty = source.index("if len(pred_instances)==0:", observe)
    guard = source.rfind("if has_current_cutr_proposals:", keyframe, observe)
    assert keyframe < guard < observe < native_empty
    call = source[observe:native_empty]
    assert "sample," in call


def test_r1_terminal_close_is_subset_only_after_s0_close():
    source = _source()
    s0_close = source.index("residual_close = cutr_residual_birth_lite.close(")
    r1_close = source.index("cutr_residual_cross_view_r1.close(", s0_close)
    r1_json = source.index(
        '"CuTR-residual-cross-view-R1 shadow JSON | "', r1_close
    )
    region = source[r1_close:r1_json]
    assert s0_close < r1_close < r1_json
    assert "residual_close" in region
    assert "np.concatenate" not in region
    assert "Instances3D.cat" not in region
    assert '"native_export_appended": False' in region

