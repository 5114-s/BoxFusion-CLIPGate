from pathlib import Path

from tools.build_oriented_refiner_dataset import (
    STRICT_PROVENANCE_EXPECTED,
    strict_provenance_for_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def test_g0_provenance_is_explicit_and_preserves_legacy_default():
    legacy = strict_provenance_for_profile("b5v2_memory_observer")
    g0 = strict_provenance_for_profile("sgcdet_sparse_observer")
    assert legacy == STRICT_PROVENANCE_EXPECTED
    assert g0["online_ablation_profile"] == "sgcdet_sparse_observer"
    assert g0["mutation_quality_enabled"] is True
    assert g0["box_refiner_coordinate_frame"] == "world_aabb"
    differing = {name for name in legacy if legacy[name] != g0[name]}
    assert differing == {
        "online_ablation_profile",
        "mutation_quality_enabled",
        "box_refiner_coordinate_frame",
    }


def test_runner_appends_train_overrides_after_optional_array_initialization():
    source = (ROOT / "scripts/run_scannet_online_refinement.sh").read_text()
    initialization = source.index("local optional_args=()")
    cache_append = source.index(
        "optional_args+=(\n                --proposal-cache-mode",
        initialization,
    )
    supplemental_append = source.index(
        "optional_args+=(\n                --online-supplemental-cache-directory",
        initialization,
    )
    assert initialization < cache_append < supplemental_append
    assert source.count("--proposal-cache-mode") == 1
    assert source.count("--online-supplemental-cache-directory") == 1


def test_train_collection_is_isolated_and_prediction_preserving():
    source = (
        ROOT / "scripts/collect_scannet_b6_g0_sgcdet_train.sh"
    ).read_text()
    required = (
        'BOXFUSION_ONLINE_ABLATION_PROFILE="sgcdet_sparse_observer"',
        'BOXFUSION_PROPOSAL_CACHE_MODE_OVERRIDE="disabled"',
        'BOXFUSION_SKIP_EVALUATION="1"',
        'BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="0.10"',
        'BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="0.50"',
        'BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="2.00"',
    )
    for token in required:
        assert token in source


def test_retrained_active_is_ap50_qualified_before_publication():
    source = (
        ROOT / "scripts/train_scannet_b6_g0_sgcdet_refiner.sh"
    ).read_text()
    for token in (
        "audit_sgcdet_training_potential.py",
        "--verify-existing",
        "--selection-metric ap50_proxy",
        "--minimum-validation-cross-success 1",
        "--maximum-validation-drop50-rate 0.01",
    ):
        assert token in source


def test_collection_audit_locks_config_and_full_output_identity():
    source = (
        ROOT / "tools/audit_g0_sgcdet_train_collection.py"
    ).read_text()
    for token in (
        "EXPECTED_CONFIG_SHA256",
        "output_pre_geometry_corners",
        "output_post_geometry_corners",
        "exported prediction order/geometry drifted",
        "refusing to re-sign changed training data",
        "immutable collection manifest already exists",
        "proposal_count == 0",
        "boxer_inference_calls",
    ):
        assert token in source

    collector = (
        ROOT / "scripts/collect_scannet_b6_g0_sgcdet_train.sh"
    ).read_text()
    assert 'if [[ -e "$MANIFEST" ]]' in collector
    assert "AUDIT_MODE+=(--verify-existing)" in collector


def test_paired_validation_uses_an_isolated_configurable_namespace():
    source = (
        ROOT / "scripts/run_scannet_b6_g0_sgcdet_retrained_paired.sh"
    ).read_text()
    assert 'BOXFUSION_RETRAINED_PAIR_RUN_ID:-v2' in source
    assert 'G0_TAG="g0_retrained_frozen_fixed10_${PAIR_RUN_ID}"' in source
    assert 'ACTIVE_TAG="g0_retrained_active_fixed10_${PAIR_RUN_ID}"' in source
