from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scannet_udc_mobilesam_full100 as module
from boxfusion.udc_mobilesam import UDCPrompt


def _prompt(
    *, center=(0.0, 0.0, 1.0), component_id=1, box=(100, 100, 140, 140)
) -> UDCPrompt:
    center = np.asarray(center, dtype=np.float64)
    lower = center - np.asarray([0.10, 0.10, 0.10])
    upper = center + np.asarray([0.10, 0.10, 0.10])
    return UDCPrompt(
        rank=0,
        component_id=component_id,
        box_xyxy=np.asarray(box, dtype=np.float32),
        grid_pixel_count=24,
        source_pixels_yx=np.asarray(
            [[100 + 4 * (index // 6), 100 + 4 * (index % 6)] for index in range(24)],
            dtype=np.int64,
        ),
        voxel_keys=np.asarray([[0, 0, 20], [1, 0, 20]], dtype=np.int64),
        world_q02=lower,
        world_q98=upper,
        world_extent=upper - lower,
        world_diagonal_m=float(np.linalg.norm(upper - lower)),
        world_volume_m3=float(np.prod(upper - lower)),
    )


def test_prompt_expansion_and_fixed_shape_gates() -> None:
    box, reason = module._expanded_prompt_box([100.0, 100.0, 140.0, 140.0])
    np.testing.assert_array_equal(box, [92.0, 92.0, 149.0, 149.0])
    assert reason is None

    _, reason = module._expanded_prompt_box([0.0, 0.0, 639.0, 479.0])
    assert reason == "prompt_area"


def test_mask_is_clipped_to_prompt_before_seed_gate() -> None:
    mask = np.ones((module.IMAGE_HEIGHT, module.IMAGE_WIDTH), dtype=bool)
    clipped = module._clip_mask_to_prompt(mask, [10.2, 20.2, 30.1, 40.1])
    assert clipped.sum() == (31 - 10) * (41 - 20)
    assert clipped[20, 10]
    assert not clipped[19, 10]
    assert not clipped[20, 31]


def test_mobilesam_gate_checks_iou_area_and_component_seed_coverage() -> None:
    source = np.asarray([[100, 100], [104, 104], [108, 108], [112, 112]])
    mask = np.zeros((module.IMAGE_HEIGHT, module.IMAGE_WIDTH), dtype=bool)
    mask[90:120, 90:120] = True
    metrics, reason = module._mask_gate(mask, 0.90, source)
    assert reason is None
    assert metrics["seed_coverage"] == 1.0

    mask[source[2:, 0], source[2:, 1]] = False
    _, reason = module._mask_gate(mask, 0.90, source)
    # Exactly one half is allowed by the frozen >= 0.50 rule.
    assert reason is None
    mask[source[1, 0], source[1, 1]] = False
    _, reason = module._mask_gate(mask, 0.90, source)
    assert reason == "mobilesam_seed_coverage"
    _, reason = module._mask_gate(np.ones_like(mask), 0.90, source)
    assert reason == "mobilesam_mask_area"


def test_near_component_filter_uses_signed_floor_and_ten_centimetres() -> None:
    raw = np.asarray([[-1, 0, 0]], dtype=np.int64)
    points = np.asarray(
        [
            [-0.001, 0.001, 0.001],  # signed-floor key [-1,0,0]
            [0.099, 0.001, 0.001],   # key [1,0,0], Chebyshev distance 2
            [0.151, 0.001, 0.001],   # key [3,0,0], distance 4
        ],
        dtype=np.float32,
    )
    retained = module._near_component_points(points, raw)
    np.testing.assert_array_equal(retained, points[:2])


def test_raw_tracker_is_past_only_one_to_one_and_expires_by_source_frame() -> None:
    tracker = module._RawTrackManager()
    one = _prompt(center=(0.0, 0.0, 1.0), component_id=1)
    ids, first = tracker.update(0, [one], [one.voxel_keys])
    assert ids == [0]
    assert first["created_count"] == 1

    nearby = _prompt(center=(0.02, 0.0, 1.0), component_id=2)
    ids, second = tracker.update(25, [nearby], [nearby.voxel_keys])
    assert ids == [0]
    assert second["created_count"] == 0

    far = _prompt(center=(2.0, 0.0, 1.0), component_id=3)
    ids, _ = tracker.update(300, [far], [far.voxel_keys])
    assert ids == [1]
    assert tracker.expired_count == 1


def test_new_raw_track_unique_lexicographic_voxel_cap_is_enforced() -> None:
    tracker = module._RawTrackManager()
    prompt = _prompt()
    # Reverse order plus duplicates exercises both deterministic sorting and
    # the initial-track branch (the update branch already had a cap).
    unique = np.asarray(
        [[x, y, z] for x in range(18) for y in range(18) for z in range(16)],
        dtype=np.int64,
    )
    rows = np.concatenate((unique[::-1], unique[100:300]), axis=0)
    ids, _ = tracker.update(0, [prompt], [rows])
    expected = np.unique(rows, axis=0)[: module.RAW_TRACK_VOXEL_CAP]
    assert ids == [0]
    assert len(tracker.tracks[0].voxels) == module.RAW_TRACK_VOXEL_CAP
    np.testing.assert_array_equal(tracker.tracks[0].voxels, expected)


def test_raw_identity_directly_owns_first_three_accepted_lifts() -> None:
    histories = {}

    def observation(raw_track_id, frame_id, observation_id):
        return {
            "raw_track_id": raw_track_id,
            "frame_id": frame_id,
            "observation_id": observation_id,
        }

    assert module._append_raw_track_observation(
        histories, observation(7, 0, 0)
    ) is None
    assert module._append_raw_track_observation(
        histories, observation(8, 25, 1)
    ) is None
    assert module._append_raw_track_observation(
        histories, observation(7, 50, 2)
    ) is None
    first_three = module._append_raw_track_observation(
        histories, observation(7, 100, 3)
    )
    assert first_three is not None
    assert [row["observation_id"] for row in first_three] == [0, 2, 3]
    assert {row["raw_track_id"] for row in first_three} == {7}


def test_runtime_provider_distribution_contains_prompted_measured_frames_only() -> None:
    preprocessing = []
    provider = []
    complete = []
    common = {
        "preprocessing": preprocessing,
        "provider_and_lifting": provider,
        "complete": complete,
    }
    module._append_runtime_sample(
        **common,
        preprocess_ms=1.0,
        provider_and_lifting_ms=0.0,
        total_ms=2.0,
        mobilesam_prompted=False,
        warmup_excluded=False,
    )
    module._append_runtime_sample(
        **common,
        preprocess_ms=3.0,
        provider_and_lifting_ms=100.0,
        total_ms=104.0,
        mobilesam_prompted=True,
        warmup_excluded=True,
    )
    module._append_runtime_sample(
        **common,
        preprocess_ms=4.0,
        provider_and_lifting_ms=20.0,
        total_ms=25.0,
        mobilesam_prompted=True,
        warmup_excluded=False,
    )
    assert preprocessing == [1.0, 4.0]
    assert provider == [20.0]
    # The no-prompt measured frame remains in complete incremental runtime.
    assert complete == [2.0, 25.0]


def test_confirmation_receipt_has_materializer_contract_and_passes_geometry() -> None:
    grid = np.asarray(
        [
            [x, y, z]
            for x in np.arange(-0.10, 0.101, 0.05)
            for y in np.arange(-0.10, 0.101, 0.05)
            for z in np.arange(0.90, 1.051, 0.05)
        ],
        dtype=np.float32,
    )
    observations = []
    cameras = [(-0.30, 0.0, 0.0), (0.0, 0.30, 0.0), (0.30, 0.0, 0.0)]
    for observation_id, (frame_id, camera) in enumerate(zip((0, 25, 50), cameras)):
        observations.append(
            {
                "observation_id": observation_id,
                "raw_track_id": 7,
                "frame_id": frame_id,
                "predicted_iou": 0.90,
                "lower": grid.min(axis=0).astype(np.float64),
                "upper": grid.max(axis=0).astype(np.float64),
                "points_world": grid,
                "camera_center": np.asarray(camera, dtype=np.float64),
            }
        )
    receipt = module._confirm_receipt(
        scene="scene0000_00",
        raw_track_id=7,
        tracker_track_id=7,
        observations=observations,
    )
    assert receipt["track_id"] == receipt["raw_track_id"] == 7
    assert receipt["confirmation_frame_id"] == receipt["evidence_frame_ids"][2]
    assert receipt["mean_predicted_iou"] == 0.9
    assert receipt["pre_novelty_pass"]
    assert np.all(np.asarray(receipt["fused_obb"]["extent_xyz"]) > 0)
    assert np.asarray(receipt["fused_obb"]["corners_world"]).shape == (8, 3)


def test_plan_only_reads_only_scene_list_and_sealed_manifests(tmp_path: Path) -> None:
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    root = tmp_path / "cache" / scene
    root.mkdir(parents=True)
    records = [
        {"frame_id": 0, "count": 2, "sha256": "a" * 64},
        {"frame_id": 25, "count": 0, "sha256": "b" * 64},
    ]
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "recorded_frame_ids": [0, 25],
                "record_count": 2,
                "proposal_count": 2,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    plan = module.run_shadow(
        schedule_root=tmp_path / "cache",
        scene_root=tmp_path / "missing-rgbd-is-not-opened",
        scene_list_path=scene_list,
        checkpoint=tmp_path / "missing-checkpoint-is-not-opened",
        output_root=tmp_path / "not-written",
        device="cuda:0",
        expected_scene_count=1,
        plan_only=True,
    )
    assert plan["schema"] == module.SCHEMA
    assert plan["keyframe_count"] == 2
    assert plan["cutr_box_count"] == 2
    assert not (tmp_path / "not-written").exists()


def test_public_runner_has_no_gt_evaluator_or_terminal_prediction_surface() -> None:
    parameters = set(inspect.signature(module.run_shadow).parameters)
    forbidden = {
        "gt",
        "ground_truth",
        "annotation",
        "annotation_root",
        "evaluator",
        "baseline_root",
        "native_root",
        "prediction_root",
    }
    assert not (parameters & forbidden)
    option_strings = {
        option
        for action in module._parser()._actions
        for option in action.option_strings
    }
    assert not any(
        token in option
        for option in option_strings
        for token in ("gt", "annot", "eval", "baseline", "native", "prediction")
    )
    source = inspect.getsource(module._process_scene)
    assert "PastOnlyTargetTracker" not in source
    assert module.CONFIRM_POLICY["association"] == "raw_track_identity_first3"
    assert module.CONFIRM_POLICY["secondary_target_masklift_tracker"] is False
