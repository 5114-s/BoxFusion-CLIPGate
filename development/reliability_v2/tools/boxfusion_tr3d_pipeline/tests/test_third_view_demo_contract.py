from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo.py"
CONFIG = ROOT / "config/scannet_qim_puf_arbitration_third_view_shadow.yaml"


def test_demo_commits_third_view_only_inside_true_cutr_commit_path():
    source = DEMO.read_text()
    assert "build_third_view_birth_lite(cfg)" in source
    assert "third-view birth-lite requires enabled Moon-QIM-lite" in source
    helper_start = source.index("    def commit_moon_qim_lite(")
    helper_end = source.index("    def observe_online_keyframe(", helper_start)
    helper = source[helper_start:helper_end]
    registry = helper.index("stable_ids = moon_qim_identity.update")
    observe = helper.index("third_view_birth_lite.observe(")
    clear = helper.index("box_manager.merge_log.clear()")
    assert registry < observe < clear
    assert "third_view_birth_lite.finalize(" in helper
    assert "if not all(third_view_result.keep_mask)" in helper

    # Both runtime call sites are guarded by the explicit real-CuTR flag. The
    # terminal branch may reuse stale pred_instances and must never commit.
    assert source.count("if has_current_cutr_proposals:") >= 2
    assert source.count("commit_moon_qim_lite(") == 3  # definition + 2 calls
    assert "Third-view-birth-lite shadow JSON | " in source
    assert "build_side_birth_probation_lite(cfg)" in source
    assert "requires enabled PUF arbitration-lite" in source
    assert "if row.action != \"birth\"" in helper
    assert "derive_committed_track_ids(" in helper
    assert "side_birth_probation_lite.observe_true_cutr_keyframe(" in helper
    assert "native_target_kind=native_kind" in helper
    assert "Side-birth-probation-lite shadow JSON | " in source


def test_demo_validates_dense_source_lineage_and_never_applies_side_mask():
    source = DEMO.read_text()
    assert "per_frame_ins.init_id[source_index]" in source
    assert "per_frame_ins.frame_id[source_index]" in source
    assert "not np.array_equal(selected_init_ids, expected_init_ids)" in source
    assert "would_admit_side_candidate_mask" not in source
    assert "boxes_3d = boxes_3d[third_view" not in source
    assert "scores = scores[third_view" not in source
    assert "boxes_3d = boxes_3d[side_birth" not in source
    assert "scores = scores[side_birth" not in source


def test_shadow_config_freezes_three_views_and_keeps_mv3dis_isolated():
    text = CONFIG.read_text()
    assert "third_view_birth_lite:" in text
    assert "min_distinct_source_frames: 3" in text
    assert "observer_only: true" in text
    assert "side_birth_probation_lite:" in text
    assert "min_distinct_keyframes: 3" in text
    assert "max_missed_keyframes: 10" in text
    # This is an isolated third-view ablation after the unsafe MV3DIS S0 veto
    # was rejected; its overhead/evidence must be measured independently.
    assert "mv3dis_depth_lite:" not in text
