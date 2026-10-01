from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo.py"
CORE = ROOT / "boxfusion" / "openbox_smov_r2.py"
BASE_CONFIG = ROOT / "config" / "scannet_eval.yaml"
R2_CONFIG = ROOT / "config" / "scannet_openbox_smov_r2_shadow.yaml"
DOC = ROOT / "docs" / "OPENBOX_SMOV_R2_SHADOW.md"


def _source() -> str:
    return DEMO.read_text(encoding="utf-8")


def _function_source(name: str) -> str:
    source = _source()
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1
    node = matches[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def test_config_is_scannet_eval_plus_training_free_shadow_block():
    baseline = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    shadow = yaml.safe_load(R2_CONFIG.read_text(encoding="utf-8"))
    r2 = shadow.pop("openbox_smov_r2")
    assert shadow == baseline
    assert r2["enabled"] is True
    assert r2["observer_only"] is True
    assert r2["max_views_per_track"] == 5
    assert r2["max_points_per_view"] == 512
    assert r2["max_points_per_track"] == 1024
    assert r2["min_views"] == 3
    assert r2["translation_gap_m"] == 0.80
    assert r2["rotation_gap_deg"] == 30.0
    assert r2["min_points"] == 192
    assert "checkpoint" not in str(r2).lower()
    assert "train" not in str(r2).lower()
    assert "proposal_cache" not in shadow
    assert "online_refinement" not in shadow

    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    defaults_node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_CONFIG"
            for target in node.targets
        )
    )
    expected = ast.literal_eval(defaults_node)
    expected["enabled"] = True
    expected["diagnostics"] = {
        "root": "./diagnostics/openbox_smov_r2/visibility_v2"
    }
    assert r2 == expected


def test_demo_builds_isolated_observer_and_rejects_mutating_paths():
    source = _source()
    assert "build_openbox_smov_r2(cfg)" in source
    assert (
        "moon_qim_lite.enabled or openbox_smov_r2.enabled" in source
    )
    assert "OpenBox-SMOV R2 shadow and online refinement" in source
    assert "proposal cache record/replay is forbidden" in source
    assert "diagnostics root must differ" in source


def test_demo_has_explicit_create_only_r2_diagnostics_override():
    source = _source()
    assert '"--openbox-smov-r2-diagnostics-root"' in source
    assert (
        "args.openbox_smov_r2_diagnostics_root" in source
    )
    assert (
        'r2_cfg.setdefault("diagnostics", {})["root"]' in source
    )
    assert 'requires an "' in source
    assert '"enabled OpenBox-SMOV R2 observer"' in source


def test_prepare_is_before_native_association_and_has_no_semantic_input():
    source = _source()
    transform = source.index(
        "pred_instances.pred_boxes_3d.transform2world"
    )
    prepare = source.index(
        "openbox_smov_r2_batch = prepare_openbox_smov_r2_keyframe(",
        transform,
    )
    association = source.index(
        "Instances3D.spatial_association", prepare
    )
    fusion = source.index("Box_Fuser.boxfusion", association)
    commit = source.index(
        "commit_openbox_smov_r2(openbox_smov_r2_batch)", fusion
    )
    moon_commit = source.index("commit_moon_qim_lite(", commit)
    clear = source.index("box_manager.merge_log.clear()", moon_commit)
    assert transform < prepare < association < fusion < commit < moon_commit < clear

    helper = _function_source("prepare_openbox_smov_r2_keyframe")
    assert "boxes_xyxy=raw_boxes_xyxy.copy()" in helper
    assert "proposal_scores=proposal_scores.copy()" in helper
    assert "proposal_image_shape=tuple(" in helper
    assert 'current_sample["wide"]["depth"][-1]' in helper
    assert "wide.depth.K[-1]" in helper
    assert "depth_m=depth_meters.copy()" in helper
    assert "camera_to_world=camera_to_world.copy()" in helper
    assert "previous_fusion_groups=previous_groups" in helper
    for forbidden in (
        "clip_model",
        "text_features",
        "text_prompt",
        "categories",
        "projected_boxes",
        "sample_depth_guide_points_batch",
    ):
        assert forbidden not in helper


def test_demo_keywords_match_the_core_two_phase_api():
    source = _source()
    tree = ast.parse(source)

    def call_keywords(method: str) -> set[str]:
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "openbox_smov_r2"
        ]
        assert len(calls) == 1
        return {keyword.arg for keyword in calls[0].keywords}

    assert call_keywords("prepare_keyframe") == {
        "scene_id",
        "frame_id",
        "proposal_ids",
        "boxes_xyxy",
        "proposal_scores",
        "proposal_image_shape",
        "depth_m",
        "intrinsics",
        "camera_to_world",
        "previous_fusion_groups",
    }
    assert call_keywords("prepare_abstain") == {
        "scene_id",
        "frame_id",
        "proposal_ids",
        "previous_fusion_groups",
        "reason",
    }
    assert call_keywords("commit_keyframe") == {
        "current_fusion_groups",
        "association_events",
    }
    assert call_keywords("finalize_shadow") == {
        "scene_id",
        "native_corners",
        "native_scores",
        "stable_ids",
    }


def test_only_true_cutr_keyframes_prepare_and_commit_views():
    source = _source()
    keyframe = source.index("# only process keyframes")
    flag = source.index(
        "has_current_cutr_proposals = count % gap == 0", keyframe
    )
    prepare = source.index(
        "openbox_smov_r2_batch = prepare_openbox_smov_r2_keyframe(",
        flag,
    )
    commit = source.index(
        "commit_openbox_smov_r2(openbox_smov_r2_batch)", prepare
    )
    assert source.rfind("if has_current_cutr_proposals:", flag, prepare) != -1
    assert source.rfind("if has_current_cutr_proposals:", prepare, commit) != -1
    assert "prepare_openbox_smov_r2_abstain(" in source[flag:prepare]
    assert '"empty_current_cutr_proposals"' in source[flag:prepare]


def test_prepare_commit_and_terminal_are_structurally_isolated_identity_guards():
    prepare = _function_source("prepare_openbox_smov_r2_keyframe")
    abstain = _function_source("prepare_openbox_smov_r2_abstain")
    commit = _function_source("commit_openbox_smov_r2")
    source = _source()
    # R2 is deterministic NumPy.  Snapshotting every CUDA RNG stream here
    # creates synchronization latency without protecting any random call.
    assert "preserved_observer_rng_state" not in prepare
    assert "preserved_observer_rng_state" not in abstain
    assert "preserved_observer_rng_state" not in commit
    assert "proposal_ids.copy()" in prepare
    assert "raw_boxes_xyxy.copy()" in prepare
    assert "proposal_scores.copy()" in prepare
    assert "depth_meters.copy()" in prepare
    assert "np.array_equal(proposal_ids, expected_ids)" in prepare
    assert "np.array_equal(raw_boxes_xyxy, expected_boxes)" in prepare
    assert "np.array_equal(proposal_scores, expected_scores)" in prepare
    assert "np.array_equal(depth_meters, expected_depth)" in prepare
    assert '"winner_members": winner' in commit
    assert '"loser_members": loser' in commit
    assert "after_groups != current_groups" in commit
    assert "after_event_signature != native_event_signature" in commit
    assert "openbox_smov_r2_result.native_corners" in source
    assert "openbox_smov_r2_result.native_scores" in source
    assert "openbox_smov_r2_result.stable_ids" in source
    assert "geometry/score/order identity" in source


def test_r2_core_has_no_random_or_torch_surface():
    source = CORE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "random" not in imported_roots
    assert "torch" not in imported_roots
    assert "np.random" not in source


def test_postprocess_mask_aligns_ids_and_shadow_never_replaces_native_boxes():
    source = _source()
    postprocess = source.index("boxes_3d, valid_mask = post_process(")
    mask_ids = source.index(
        "openbox_smov_r2_stable_ids[valid_mask]", postprocess
    )
    finalize = source.index("openbox_smov_r2.finalize_shadow(", mask_ids)
    sidecar = source.index(
        "save_r2_shadow_sidecar_create_only(", finalize
    )
    native_save = source.index(
        "save_prediction_create_only(boxes_3d, scores, output_path)",
        sidecar,
    )
    assert postprocess < mask_ids < finalize < sidecar < native_save
    region = source[finalize:native_save]
    assert "counterfactual_corners" not in region
    assert "boxes_3d = openbox_smov_r2" not in source
    assert "scores = openbox_smov_r2" not in source
    assert "OpenBox-SMOV R2 shadow JSON | " in source


def test_documentation_freezes_online_shadow_and_no_gain_claim():
    text = DOC.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "training-free" in text
    assert "Two-phase online contract" in text
    assert "legacy terminal replay is never treated as a new view" in text
    assert "create-only `.npz`" in text
    assert "`boxes_3d` is never assigned from the R2 result" in text
    assert "shadow AP is intentionally identical to the control" in normalized
    assert "CLIP follows the original code path unchanged" in normalized
    assert "FPS ratio" in text and "0.95" in text
