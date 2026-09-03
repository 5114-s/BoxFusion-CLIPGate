import ast
from pathlib import Path

import numpy as np
import torch
import yaml

from boxfusion.boxes import BoxDOF, GeneralInstance3DBoxes
from boxfusion.cutr_residual_birth_lite import partition_scores


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "demo.py"
BASE_CONFIG_PATH = ROOT / "config" / "scannet_eval.yaml"
SHADOW_CONFIG_PATH = (
    ROOT / "config" / "scannet_cutr_residual_shadow.yaml"
)
DOC_PATH = ROOT / "docs" / "CUTR_RESIDUAL_BIRTH_LITE.md"


def _source():
    return DEMO_PATH.read_text(encoding="utf-8")


def _function_source(name):
    source = _source()
    tree = ast.parse(source)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def test_shadow_config_is_scannet_eval_plus_one_enabled_block():
    baseline = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    shadow = yaml.safe_load(SHADOW_CONFIG_PATH.read_text(encoding="utf-8"))
    residual = shadow.pop("cutr_residual_birth_lite")
    assert shadow == baseline
    assert residual == {
        "enabled": True,
        "observer_only": True,
        "score_floor": 0.10,
        "score_ceiling": baseline["detection"]["score_thresh"],
        "max_tracks": 1024,
        "max_observations_per_frame": 64,
    }


def test_demo_builds_observer_and_rejects_cache_or_threshold_drift():
    source = _source()
    assert "build_cutr_residual_birth_lite(cfg)" in source
    assert "if proposal_cache is not None:" in source
    assert (
        'cfg["cutr_residual_birth_lite"]["score_ceiling"]'
        in source
    )
    assert 'cfg["detection"]["score_thresh"]' in source
    assert "configured_ceiling != native_threshold" in source


def test_live_raw_output_is_copied_before_unchanged_native_threshold():
    source = _source()
    live_start = source.index('source_attempt_id = "primary"')
    retry_start = source.index('source_attempt_id = "retry"', live_start)
    primary = source[live_start:retry_start]
    assert "raw_pred_instances = model(packaged)[0]" in primary
    assert "make_cutr_residual_side(raw_pred_instances, image)" in primary
    assert "pred_instances = raw_pred_instances" in primary
    assert "pred_instances.scores >= float(score_thresh)" in primary
    assert primary.index("make_cutr_residual_side") < primary.index(
        "pred_instances.scores >= float(score_thresh)"
    )
    assert primary.index(
        "pred_instances.scores >= float(score_thresh)"
    ) < primary.index("apply_lifting_if_configured")


def test_side_partition_keeps_raw_rows_separate_and_only_mirrors_filters():
    helper = _function_source("make_cutr_residual_side")
    assert "partition_scores(" in helper
    assert "partition.residual_indices" in helper
    assert "raw_predictions[row_selector].clone()" in helper
    assert "check_uv_bounds" in helper
    assert "check_floor_mask" in helper
    for forbidden in (
        "apply_lifting_if_configured",
        "text_prompt",
        "query_moon_qim_lite",
        "Box_Fuser",
        "boxfusion(",
    ):
        assert forbidden not in helper


def test_frame_zero_retry_explicitly_drops_primary_side_and_repartitions():
    source = _source()
    retry_start = source.index('source_attempt_id = "retry"')
    retry_end = source.index(
        "if proposal_cache is not None and proposal_cache.is_record",
        retry_start,
    )
    retry = source[retry_start:retry_end]
    reset = retry.index("cutr_residual_predictions = None")
    model_call = retry.index("raw_pred_instances = model(packaged)[0]")
    repartition = retry.index("make_cutr_residual_side(")
    assert reset < model_call < repartition
    assert "actual_native_threshold=float(" in retry
    assert "apply_floor_filter=False" in retry
    assert (
        "pred_instances.scores\n"
        "                        >= float(cfg['detection']['score_thresh']/4)"
        in retry
    )

    retry_cutoff = 0.5 / 4
    scores = np.asarray([0.099, 0.10, 0.1249, 0.125, 0.50])
    partition = partition_scores(scores, score_ceiling=retry_cutoff)
    actual_native_rows = set(
        np.flatnonzero(scores >= retry_cutoff).tolist()
    )
    assert partition.residual_indices == (1, 2)
    assert set(partition.residual_indices).isdisjoint(actual_native_rows)


def test_side_state_resets_per_frame_and_terminal_stale_is_not_observed():
    source = _source()
    loop = source.index("for sample in dataset:")
    sample_id = source.index("sample_video_id =", loop)
    assert loop < source.index("cutr_residual_predictions = None", loop) < sample_id
    assert (
        loop
        < source.index("cutr_residual_raw_row_indices = np.empty", loop)
        < sample_id
    )

    keyframe_branch = source.index("# only process keyframes", loop)
    observe = source.index("observe_cutr_residual_keyframe(", keyframe_branch)
    native_empty = source.index("if len(pred_instances)==0:", keyframe_branch)
    assert keyframe_branch < observe < native_empty
    guard = source.rfind("if has_current_cutr_proposals:", keyframe_branch, observe)
    assert guard != -1
    assert "has_current_cutr_proposals = count % gap == 0" in source[
        keyframe_branch:observe
    ]


def test_observer_world_transforms_clone_and_guards_rng_and_native_arrays():
    helper = _function_source("observe_cutr_residual_keyframe")
    assert "np.array(frame_pose, dtype=np.float32, copy=True)" in helper
    assert "residual_predictions.pred_boxes_3d.transform2world(side_pose)" in helper
    assert "ResidualObservation(" in helper
    assert "raw_index=int(raw_row_indices[row])" in helper
    assert "with preserved_observer_rng_state():" in helper
    assert "cutr_residual_birth_lite.observe(" in helper
    assert "np.array_equal(before, after)" in helper


def test_nonempty_float32_side_pose_executes_general_box_world_transform():
    """Runtime regression for the float64/float32 matmul crash."""

    for count in (0, 1, 2):
        tensor = torch.tensor(
            [[0.1, 0.2, 2.0, 1.0, 1.2, 0.8]] * count,
            dtype=torch.float32,
        ).reshape(count, 6)
        rotations = torch.eye(3, dtype=torch.float32).repeat(count, 1, 1)
        boxes = GeneralInstance3DBoxes(tensor, rotations, dof=BoxDOF.All)
        native_tensor = boxes.tensor.clone()
        pose_batch = np.repeat(
            np.eye(4, dtype=np.float32)[None, :, :], count, axis=0
        )
        boxes.transform2world(pose_batch)
        assert tuple(boxes.corners.shape) == (count, 8, 3)
        torch.testing.assert_close(boxes.tensor, native_tensor, rtol=0.0, atol=0.0)


def test_terminal_close_is_after_native_postprocess_and_never_appends():
    source = _source()
    final = source.index("# save global boxes for evaluation")
    postprocess = source.index("boxes_3d, valid_mask = post_process(", final)
    close = source.index("cutr_residual_birth_lite.close(", postprocess)
    machine_json = source.index(
        '"CuTR-residual-birth-lite shadow JSON | "', close
    )
    assert final < postprocess < close < machine_json
    close_region = source[close:machine_json]
    assert "native_corners=expected_native_corners.copy()" in close_region
    assert "native_scores=expected_native_scores.copy()" in close_region
    assert "np.concatenate" not in close_region
    assert "Instances3D.cat" not in close_region
    assert '"native_export_appended": False' in close_region


def test_documentation_freezes_cache_caveat_and_shadow_ap_contract():
    text = DOC_PATH.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "[0.10, 0.50)" in text
    assert "Proposal-cache record" in text
    assert "terminal stale-frame" in text
    assert "32.53 FPS" in text and "32.50 FPS" in text
    assert "0.277389" in text and "0.030303" in text
    assert "none of the six candidates added a true positive" in text
    assert "must not be expanded to fixed-10" in normalized
    assert "does **not** append" in text
