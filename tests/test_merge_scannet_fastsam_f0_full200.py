from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import merge_scannet_fastsam_f0_full200 as merge


def _dump(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contracts() -> dict[str, object]:
    return dict(merge.EXPECTED_SHARD_CONTRACTS)


def _environment(shard_index: int) -> dict[str, object]:
    device = f"cuda:{shard_index}"
    return {
        "production_cuda_required": True,
        "dependency_injected_provider": False,
        "conda_environment": "boxfusion-online",
        "python_version": "3.10.0",
        "torch_version": merge.EXPECTED_TORCH_VERSION,
        "torch_cuda_version": merge.EXPECTED_TORCH_CUDA_VERSION,
        "opencv_version": merge.EXPECTED_OPENCV_VERSION,
        "ultralytics_version": merge.EXPECTED_ULTRALYTICS_VERSION,
        "device": device,
        "cuda_available": True,
        "gpu_name": merge.EXPECTED_GPU_NAME,
        "gpu_uuid": merge.EXPECTED_GPU_UUID_BY_LOGICAL_DEVICE[device],
        "compute_capability": list(merge.EXPECTED_COMPUTE_CAPABILITY),
        "cuda_visible_devices": None,
        "cuda_synchronization_contract": merge.EXPECTED_CUDA_SYNCHRONIZATION_CONTRACT,
    }


def _provider_timing(device: str) -> dict[str, object]:
    return {
        "device": device,
        "started_ns": 1,
        "prediction_finished_ns": 80_000_001,
        "finished_ns": 90_000_001,
        "prediction_seconds": 0.08,
        "prediction_ms": 80.0,
        "extraction_seconds": 0.01,
        "extraction_ms": 10.0,
        "total_seconds": 0.09,
        "total_ms": 90.0,
        "cuda_synchronized": True,
        "memory_allocated_before_bytes": 100,
        "memory_allocated_after_bytes": 200,
        "memory_reserved_before_bytes": 300,
        "memory_reserved_after_bytes": 400,
        "max_memory_allocated_bytes": 1024**3,
        "max_memory_reserved_bytes": 2 * 1024**3,
    }


def _execution_identity() -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    checkpoint_path = merge.EXPECTED_CHECKPOINT_PATH.resolve()
    checkpoint = {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    sources = {}
    for name, raw_path in merge.EXPECTED_SOURCE_PATHS.items():
        path = raw_path.resolve()
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return checkpoint, sources


def _funnel(*, selected: bool, raw_count: int = 1) -> dict[str, object]:
    masks = []
    for raw_index in range(raw_count):
        is_selected = selected and raw_index == 0
        masks.append(
            {
                "raw_index": raw_index,
                "confidence": 0.9,
                "pixel_count": 400,
                "valid_pixel_count": 400,
                "residual_pixel_count": 400,
                "valid_ratio": 1.0,
                "residual_ratio": 1.0,
                "pre_dedup_eligible": True,
                "deduplicated": False,
                "lifted": is_selected,
                "selected": is_selected,
                "rank": 0 if is_selected else None,
                "decision": "selected" if is_selected else "too_few_unique_voxels",
            }
        )
    selected_count = int(selected and raw_count > 0)
    return {
        "input_mask_count": raw_count,
        "input_explained_box_count": 0,
        "explained_union_pixels": 0,
        "pre_dedup_eligible_count": raw_count,
        "deduplicated_count": 0,
        "post_dedup_count": raw_count,
        "lifting_eligible_count": selected_count,
        "selected_count": selected_count,
        "cap_rejected_count": 0,
        "rejection_counts": {
            "selected": selected_count,
            "too_few_unique_voxels": raw_count - selected_count,
        },
        "masks": masks,
        "candidates": (
            [{"raw_index": 0, "rank": 0}] if selected_count else []
        ),
    }


def _make_full200(tmp_path: Path) -> tuple[Path, list[Path]]:
    scenes = [f"scene{index:04d}_00" for index in range(merge.EXPECTED_SCENES)]
    scene_list = tmp_path / "full200.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    scene_list_hash = hashlib.sha256(scene_list.read_bytes()).hexdigest()
    signature = "a" * 64
    rows_by_shard: list[list[dict[str, object]]] = [[], []]
    indices_by_shard: list[list[int]] = [[], []]
    order_by_shard: list[list[str]] = [[], []]
    provider_calls_by_shard = [0, 0]
    checkpoint, sources = _execution_identity()

    # 141 * 65 + 59 * 64 == the sealed composite-schedule total 12,941.
    for index, scene in enumerate(scenes):
        keyframes = 65 if index < 141 else 64
        accepted = 8
        shard_index = index % 2
        first_call_index = provider_calls_by_shard[shard_index]
        warmups = sum(
            call_index < merge.SHARD_WARMUP_SUCCESSFUL_CALLS
            for call_index in range(first_call_index, first_call_index + keyframes)
        )
        counts = {
            "keyframes": keyframes,
            "successful_frames": keyframes,
            "invalid_pose_frames": 0,
            "non_upright_producer_frames": 0,
            "cutr_boxes": 0,
            "raw_masks": keyframes,
            "pre_dedup_eligible_masks": keyframes,
            "deduplicated_masks": 0,
            "lifting_eligible_masks": accepted,
            "accepted_lifts": accepted,
            "cap_rejected_masks": 0,
            "cap_saturated_frames": 0,
            "provider_max_det_saturated_frames": 0,
            "warmup_excluded_successful_frames": warmups,
        }
        frames = []
        for position in range(keyframes):
            call_index = provider_calls_by_shard[shard_index]
            provider_calls_by_shard[shard_index] += 1
            frames.append(
                {
                    "frame_id": position * 25,
                    "inputs": {
                        "cutr_box_count": 0,
                        "producer_orientation": 0,
                        "cutr_cache_image_size": [480, 640],
                        "current_pose_valid": True,
                        "f0_pose_forward_filled": False,
                    },
                    "successful": True,
                    "abstention": None,
                    "provider_timing": _provider_timing(f"cuda:{shard_index}"),
                    "funnel": _funnel(selected=position < accepted),
                    "runtime": {
                        "provider_ms": 100.0,
                        "core_ms": 50.0,
                        "complete_ms": 150.0,
                        "receipt_total_ms": 160.0,
                        "provider_call_index_in_shard": call_index,
                        "warmup_excluded": (
                            call_index < merge.SHARD_WARMUP_SUCCESSFUL_CALLS
                        ),
                    },
                }
            )
        frame_ids = [frame["frame_id"] for frame in frames]
        schedule_path = tmp_path / "schedules" / scene / "manifest.json"
        schedule_hash = _dump(
            schedule_path,
            {
                "recorded_frame_ids": frame_ids,
                "record_count": keyframes,
                "records": [{"frame_id": frame_id} for frame_id in frame_ids],
                "producer_fingerprint": "d" * 64,
            },
        )
        frame_ledger_hash = merge._canonical_json_sha256(frame_ids)
        sidecar = {
            "schema": merge.SCENE_SCHEMA,
            "protocol_id": merge.EXPECTED_PROTOCOL_ID,
            "complete": True,
            "scene_id": scene,
            "scene_index": index,
            "run_signature_sha256": signature,
            "frame_id_ledger_sha256": frame_ledger_hash,
            "environment_sha256": merge._canonical_json_sha256(
                _environment(shard_index)
            ),
            "contracts": _contracts(),
            "checkpoint": checkpoint,
            "sources": sources,
            "schedule": {
                "root": str((tmp_path / "schedules").resolve()),
                "manifest_path": str(schedule_path.resolve()),
                "manifest_sha256": schedule_hash,
                "keyframe_count": keyframes,
            },
            "frames": frames,
            "summary": {
                "keyframe_count": keyframes,
                "counts": counts,
                "memory": {
                    "cpu_peak_rss_bytes": 2 * 1024**3,
                    "gpu_peak_memory_bytes": 3 * 1024**3,
                },
            },
        }
        sidecar_path = tmp_path / "scenes" / f"{scene}.json"
        digest = _dump(sidecar_path, sidecar)
        indices_by_shard[shard_index].append(index)
        order_by_shard[shard_index].append(scene)
        rows_by_shard[shard_index].append(
            {
                "scene_id": scene,
                "scene_index": index,
                "sidecar_path": f"../scenes/{scene}.json",
                "sidecar_sha256": digest,
                "frame_id_ledger_sha256": frame_ledger_hash,
                "schedule_path": str(schedule_path.resolve()),
                "schedule_root": str((tmp_path / "schedules").resolve()),
                "schedule_sha256": schedule_hash,
                "keyframe_count": keyframes,
                "counts": counts,
                "cpu_peak_rss_bytes": 2 * 1024**3,
                "gpu_peak_memory_bytes": 3 * 1024**3,
                "resumed": False,
            }
        )

    manifests = []
    for shard_index in range(2):
        manifest = {
            "schema": merge.SHARD_SCHEMA,
            "protocol_id": merge.EXPECTED_PROTOCOL_ID,
            "mode": "shadow",
            "complete": True,
            "run_signature_sha256": signature,
            "contracts": _contracts(),
            "environment": _environment(shard_index),
            "environment_sha256": merge._canonical_json_sha256(
                _environment(shard_index)
            ),
            "checkpoint": checkpoint,
            "sources": sources,
            "resume_rewarm_calls": 0,
            "resume_rewarm": {
                "required": False,
                "reason": "fresh_or_resume_without_completed_scene",
                "completed_scene_count": 0,
                "pending_scene_count": len(rows_by_shard[shard_index]),
                "call_count": 0,
                "all_successful": True,
                "excluded_from_scene_counts": True,
                "excluded_from_capacity": True,
                "excluded_from_runtime_distributions": True,
                "calls": [],
            },
            "scene_list": {
                "path": str(scene_list.resolve()),
                "sha256": scene_list_hash,
                "exact_scene_count": merge.EXPECTED_SCENES,
            },
            "shard": {
                "index": shard_index,
                "count": 2,
                "scene_indices": indices_by_shard[shard_index],
                "scene_order": order_by_shard[shard_index],
            },
            "full200_keyframe_count": merge.EXPECTED_KEYFRAMES,
            "expected_execution_census": {
                "sha256": merge.EXPECTED_EXECUTION_CENSUS_SHA256,
                "counts": dict(merge.EXPECTED_SHARD_EXECUTION_COUNTS[shard_index]),
            },
            "shard_keyframe_count": sum(
                row["keyframe_count"] for row in rows_by_shard[shard_index]
            ),
            "totals": {
                key: sum(row["counts"].get(key, 0) for row in rows_by_shard[shard_index])
                for key in sorted(
                    {
                        name
                        for row in rows_by_shard[shard_index]
                        for name in row["counts"]
                    }
                )
            },
            "scenes": rows_by_shard[shard_index],
        }
        manifest["totals"]["candidate_scene_count"] = len(rows_by_shard[shard_index])
        manifest["totals"]["cap_saturation_ratio"] = 0.0
        manifest["totals"]["provider_max_det_saturation_ratio"] = 0.0
        path = tmp_path / "shards" / f"shard-{shard_index:03d}-of-002.json"
        _dump(path, manifest)
        manifests.append(path)
    return scene_list, manifests


def _one_scene_inputs(tmp_path: Path, *, duplicate_frame: bool = False):
    scene = "scene0000_00"
    signature = "b" * 64
    frames = [
        {
            "frame_id": 0,
            "inputs": {
                "cutr_box_count": 0,
                "producer_orientation": 0,
                "cutr_cache_image_size": [480, 640],
                "current_pose_valid": True,
                "f0_pose_forward_filled": False,
            },
            "successful": True,
            "abstention": None,
            "funnel": _funnel(selected=True),
            "runtime": {
                "provider_ms": 10.0,
                "core_ms": 10.0,
                "complete_ms": 20.0,
                "receipt_total_ms": 22.0,
                "provider_call_index_in_shard": 0,
                "warmup_excluded": True,
            },
        },
        {
            "frame_id": 0 if duplicate_frame else 25,
            "inputs": {
                "cutr_box_count": 0,
                "producer_orientation": 0,
                "cutr_cache_image_size": [480, 640],
                "current_pose_valid": True,
                "f0_pose_forward_filled": False,
            },
            "successful": True,
            "abstention": None,
            "funnel": _funnel(selected=False),
            "runtime": {
                "provider_ms": 11.0,
                "core_ms": 10.0,
                "complete_ms": 21.0,
                "receipt_total_ms": 23.0,
                "provider_call_index_in_shard": 1,
                "warmup_excluded": True,
            },
        },
    ]
    counts = {
        "keyframes": 2,
        "successful_frames": 2,
        "invalid_pose_frames": 0,
        "non_upright_producer_frames": 0,
        "cutr_boxes": 0,
        "raw_masks": 2,
        "pre_dedup_eligible_masks": 2,
        "deduplicated_masks": 0,
        "lifting_eligible_masks": 1,
        "accepted_lifts": 1,
        "cap_rejected_masks": 0,
        "cap_saturated_frames": 0,
        "provider_max_det_saturated_frames": 0,
        "warmup_excluded_successful_frames": 2,
    }
    sidecar = {
        "schema": merge.SCENE_SCHEMA,
        "protocol_id": "test-protocol",
        "complete": True,
        "scene_id": scene,
        "scene_index": 0,
        "run_signature_sha256": signature,
        "frames": frames,
        "summary": {
            "counts": counts,
            "memory": {"cpu_peak_rss_bytes": 10, "gpu_peak_memory_bytes": 20},
        },
    }
    frame_ids = [frame["frame_id"] for frame in frames]
    schedule_path = tmp_path / "schedules" / scene / "manifest.json"
    schedule_hash = _dump(
        schedule_path,
        {
            "recorded_frame_ids": frame_ids,
            "record_count": len(frame_ids),
            "records": [{"frame_id": frame_id} for frame_id in frame_ids],
            "producer_fingerprint": "d" * 64,
        },
    )
    frame_ledger_hash = merge._canonical_json_sha256(frame_ids)
    sidecar["frame_id_ledger_sha256"] = frame_ledger_hash
    sidecar["schedule"] = {
        "root": str((tmp_path / "schedules").resolve()),
        "manifest_path": str(schedule_path.resolve()),
        "manifest_sha256": schedule_hash,
        "keyframe_count": len(frame_ids),
    }
    sidecar_path = tmp_path / "scenes" / f"{scene}.json"
    digest = _dump(sidecar_path, sidecar)
    manifest_path = tmp_path / "shards" / "shard.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    row = {
        "sidecar_path": f"../scenes/{scene}.json",
        "sidecar_sha256": digest,
        "frame_id_ledger_sha256": frame_ledger_hash,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_root": str((tmp_path / "schedules").resolve()),
        "schedule_sha256": schedule_hash,
        "keyframe_count": 2,
        "counts": counts,
        "cpu_peak_rss_bytes": 10,
        "gpu_peak_memory_bytes": 20,
    }
    return row, manifest_path, scene, signature, sidecar_path


def test_build_full200_rehashes_and_aggregates_exact_composite_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "FastSAM.pt"
    checkpoint.write_bytes(b"frozen-fastsam")
    monkeypatch.setattr(merge, "EXPECTED_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(merge, "EXPECTED_CHECKPOINT_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(
        merge,
        "EXPECTED_CHECKPOINT_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    source_paths = {}
    for name in ("runner", "core", "provider"):
        path = tmp_path / "identity" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        source_paths[name] = path
    monkeypatch.setattr(merge, "EXPECTED_SOURCE_PATHS", source_paths)
    scene_list, manifests = _make_full200(tmp_path)
    monkeypatch.setattr(
        merge,
        "EXPECTED_SCENE_LIST_SHA256",
        hashlib.sha256(scene_list.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(merge, "EXPECTED_NON_UPRIGHT_KEYFRAMES", {})
    monkeypatch.setattr(merge, "EXPECTED_INVALID_CURRENT_POSE_FRAMES", 0)
    monkeypatch.setattr(merge, "EXPECTED_NON_UPRIGHT_FRAME_COUNT", 0)
    monkeypatch.setattr(merge, "EXPECTED_SUCCESSFUL_PROVIDER_FRAMES", 12_941)
    synthetic_shard_counts = {}
    synthetic_census = []
    scenes = scene_list.read_text(encoding="utf-8").splitlines()
    for index, scene in enumerate(scenes):
        keyframes = 65 if index < 141 else 64
        shard_index = index % 2
        bucket = synthetic_shard_counts.setdefault(
            shard_index,
            {
                "keyframes": 0,
                "invalid_pose_frames": 0,
                "non_upright_producer_frames": 0,
                "successful_frames": 0,
            },
        )
        bucket["keyframes"] += keyframes
        bucket["successful_frames"] += keyframes
        synthetic_census.extend(
            {
                "scene_index": index,
                "scene_id": scene,
                "frame_id": position * 25,
                "current_pose_valid": True,
                "sealed_non_upright_orientation": 0,
                "provider_success": True,
            }
            for position in range(keyframes)
        )
    synthetic_census_sha = merge._canonical_json_sha256(synthetic_census)
    monkeypatch.setattr(
        merge, "EXPECTED_SHARD_EXECUTION_COUNTS", synthetic_shard_counts
    )
    monkeypatch.setattr(
        merge, "EXPECTED_EXECUTION_CENSUS_SHA256", synthetic_census_sha
    )
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        shard_index = manifest["shard"]["index"]
        manifest["expected_execution_census"] = {
            "sha256": synthetic_census_sha,
            "counts": synthetic_shard_counts[shard_index],
        }
        _dump(path, manifest)
    monkeypatch.setattr(
        merge,
        "_validate_shared_run_signature",
        lambda **_kwargs: {"sha256": "a" * 64, "payload_sha256": "a" * 64},
    )
    receipt = merge.build_full200_receipt(
        scene_list_path=scene_list, shard_manifest_paths=manifests[::-1]
    )
    assert receipt["coverage"]["scene_count"] == 200
    assert receipt["coverage"]["unique_keyframe_count"] == 12_941
    assert receipt["capacity"]["accepted_lifts"] == 1_600
    assert receipt["capacity"]["accepted_scene_count"] == 200
    assert receipt["runtime"]["provider_ms"]["distribution"]["p95"] == 100.0
    assert receipt["runtime"]["complete_ms"]["distribution"]["p95"] == 150.0
    assert receipt["runtime"]["amortized_complete_ms_per_source_frame"] == 6.0
    assert receipt["memory"]["gpu_peak_memory_bytes"] == 3 * 1024**3
    quality = receipt["no_gt_mask_quality_histograms"]
    assert quality["raw_fastsam_confidence"]["sample_count"] == 12_941
    assert quality["mask_pixel_area"]["sample_count"] == 12_941
    assert quality["raw_mask_count_per_successful_frame"]["sample_count"] == 12_941
    assert quality["selected_lifts_per_successful_frame"]["sample_count"] == 12_941
    assert not quality["raw_samples_included"]
    assert not quality["ground_truth_used"]
    assert receipt["capacity"]["provider_max_det_saturated_frames"] == 0
    assert len(receipt["runtime"]["samples_ms"]["provider"]) == 12_935
    assert len(receipt["runtime"]["samples_ms"]["receipt_total"]) == 12_941
    assert receipt["overall_pass"]
    assert len(receipt["scenes"]) == 200
    assert [row["index"] for row in receipt["inputs"]["shards"]] == [0, 1]

    output = merge.publish_create_only(tmp_path / "merged", receipt)
    assert output.name == merge.OUTPUT_NAME
    before = output.read_bytes()
    with pytest.raises(merge.F0MergeError, match="already exists"):
        merge.publish_create_only(tmp_path / "merged", receipt)
    assert output.read_bytes() == before


def test_scene_sidecar_rejects_duplicate_frames(tmp_path: Path) -> None:
    row, manifest, scene, signature, _ = _one_scene_inputs(
        tmp_path, duplicate_frame=True
    )
    with pytest.raises(merge.F0MergeError, match="duplicate or non-increasing"):
        merge._validate_scene_sidecar(
            row,
            manifest,
            expected_scene=scene,
            expected_index=0,
            expected_protocol_id="test-protocol",
            expected_run_signature=signature,
        )


def test_scene_sidecar_is_rehashed_before_json_is_trusted(tmp_path: Path) -> None:
    row, manifest, scene, signature, sidecar = _one_scene_inputs(tmp_path)
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(merge.F0MergeError, match="rehash failed"):
        merge._validate_scene_sidecar(
            row,
            manifest,
            expected_scene=scene,
            expected_index=0,
            expected_protocol_id="test-protocol",
            expected_run_signature=signature,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"shadow_only": False}, "shadow-only"),
        ({"birth_enabled": True}, "birth-disabled"),
        ({"ground_truth_access": True}, "no-ground-truth"),
        ({"terminal_native_prediction_access": True}, "no-terminal-native"),
    ],
)
def test_contract_violations_are_structural_failures(change, message) -> None:
    contracts = _contracts()
    contracts.update(change)
    with pytest.raises(merge.F0MergeError, match=message):
        merge._validate_contracts(contracts, "test")


def test_fixed_gates_keep_strict_max_and_inclusive_other_boundaries() -> None:
    assert merge._gate(1500, ">=", 1500)["passed"]
    assert merge._gate(0.25, "<=", 0.25)["passed"]
    assert not merge._gate(833.33, "<", 833.33)["passed"]
    assert merge._gate(833.329, "<", 833.33)["passed"]


def test_scene_list_sha_and_protocol_id_are_frozen_fail_closed(tmp_path: Path) -> None:
    scene_list = tmp_path / "wrong-full200.txt"
    scene_list.write_text(
        "\n".join(f"scene{index:04d}_00" for index in range(200)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(merge.F0MergeError, match="scene-list SHA-256"):
        merge._read_scene_list(scene_list)
    assert merge.EXPECTED_PROTOCOL_ID == (
        "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"
    )


def test_no_gt_quality_histograms_validate_ranges_and_provider_saturation() -> None:
    funnel = _funnel(selected=True, raw_count=100)
    frame = {
        "frame_id": 0,
        "inputs": {
            "cutr_box_count": 0,
            "producer_orientation": 0,
            "cutr_cache_image_size": [480, 640],
            "current_pose_valid": True,
            "f0_pose_forward_filled": False,
        },
        "successful": True,
        "abstention": None,
        "funnel": funnel,
    }
    counts = {
        "keyframes": 1,
        "successful_frames": 1,
        "invalid_pose_frames": 0,
        "non_upright_producer_frames": 0,
        "cutr_boxes": 0,
        "raw_masks": 100,
        "pre_dedup_eligible_masks": 100,
        "deduplicated_masks": 0,
        "lifting_eligible_masks": 1,
        "accepted_lifts": 1,
        "cap_rejected_masks": 0,
        "cap_saturated_frames": 0,
        "provider_max_det_saturated_frames": 1,
    }
    result = merge._validate_frame_funnels([frame], "scene", counts)
    assert result["provider_max_det_saturated_frames"] == 1
    assert result["histograms"]["raw_confidence"]["sample_count"] == 100
    assert result["histograms"]["raw_masks_per_successful_frame"]["sample_count"] == 1
    assert result["histograms"]["raw_masks_per_successful_frame"]["overflow_count"] == 0
    assert all("samples" not in histogram for histogram in result["histograms"].values())

    funnel["masks"][0]["residual_ratio"] = 1.01
    with pytest.raises(merge.F0MergeError, match=r"must be in \[0,1\]"):
        merge._validate_frame_funnels([frame], "scene", counts)


def test_funnel_mask_count_and_complete_runtime_are_cross_checked() -> None:
    funnel = _funnel(selected=False)
    funnel["input_mask_count"] = 2
    frame = {
        "frame_id": 0,
        "inputs": {
            "cutr_box_count": 0,
            "producer_orientation": 0,
            "cutr_cache_image_size": [480, 640],
            "current_pose_valid": True,
            "f0_pose_forward_filled": False,
        },
        "successful": True,
        "abstention": None,
        "funnel": funnel,
    }
    with pytest.raises(merge.F0MergeError, match="funnel list counts differ"):
        merge._validate_frame_funnels(
            [frame],
            "scene",
            {
                "keyframes": 1,
                "successful_frames": 1,
                "invalid_pose_frames": 0,
                "non_upright_producer_frames": 0,
                "cutr_boxes": 0,
                "raw_masks": 2,
                "pre_dedup_eligible_masks": 1,
                "deduplicated_masks": 0,
                "lifting_eligible_masks": 0,
                "accepted_lifts": 0,
                "cap_rejected_masks": 0,
                "cap_saturated_frames": 0,
            },
        )

    runtime_frame = {
        "successful": True,
        "runtime": {
            "provider_ms": 10.0,
            "core_ms": 5.0,
            "complete_ms": 14.0,
            "receipt_total_ms": 20.0,
            "provider_call_index_in_shard": 3,
            "warmup_excluded": False,
        },
    }
    with pytest.raises(merge.F0MergeError, match="below provider plus core"):
        merge._extract_runtime_from_frames([runtime_frame], "scene")


def test_non_upright_cache_coordinate_abstention_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = ("scene0246_00", 1900)
    monkeypatch.setattr(merge, "EXPECTED_NON_UPRIGHT_KEYFRAMES", {key: 1})
    frame = {
        "frame_id": 1900,
        "inputs": {
            "cutr_box_count": 2,
            "producer_orientation": 1,
            "cutr_cache_image_size": [640, 480],
            "current_pose_valid": True,
            "f0_pose_forward_filled": False,
        },
        "successful": False,
        "abstention": "non_upright_cache_coordinate_frame",
        "funnel": None,
    }
    counts = {
        "keyframes": 1,
        "successful_frames": 0,
        "invalid_pose_frames": 0,
        "non_upright_producer_frames": 1,
        "cutr_boxes": 2,
        "raw_masks": 0,
        "pre_dedup_eligible_masks": 0,
        "deduplicated_masks": 0,
        "lifting_eligible_masks": 0,
        "accepted_lifts": 0,
        "cap_rejected_masks": 0,
        "cap_saturated_frames": 0,
    }
    result = merge._validate_frame_funnels([frame], key[0], counts)
    assert result["seen_non_upright_keyframes"] == [key]
    frame["abstention"] = "invalid_current_pose"
    with pytest.raises(merge.F0MergeError, match="sealed non-upright abstention"):
        merge._validate_frame_funnels([frame], key[0], counts)


def test_production_environment_and_provider_timing_fail_closed() -> None:
    manifest = {"environment": _environment(0)}
    environment, digest = merge._validate_environment(manifest, shard_index=0)
    assert digest == merge._canonical_json_sha256(environment)
    timing = _provider_timing("cuda:0")
    merge._validate_provider_timing(
        timing,
        expected_device="cuda:0",
        external_provider_ms=100.0,
        label="test",
    )
    manifest["environment"]["torch_version"] = "wrong"
    with pytest.raises(merge.F0MergeError, match="torch_version"):
        merge._validate_environment(manifest, shard_index=0)
    manifest = {"environment": _environment(0)}
    manifest["environment"]["undeclared_hardware_field"] = "forbidden"
    with pytest.raises(merge.F0MergeError, match="receipt keys differ"):
        merge._validate_environment(manifest, shard_index=0)
    timing["cuda_synchronized"] = False
    with pytest.raises(merge.F0MergeError, match="not synchronized"):
        merge._validate_provider_timing(
            timing,
            expected_device="cuda:0",
            external_provider_ms=100.0,
            label="test",
        )


def test_shared_signature_is_reconstructed_from_every_sealed_input(
    tmp_path: Path,
) -> None:
    scene_root = tmp_path / "raw-scenes"
    scene_root.mkdir()
    environments = [_environment(0), _environment(1)]
    policy = {"core_schema": "core.test.v1", "core": {"voxel_size_m": 0.05}}
    manifests = [
        {
            "scene_root": str(scene_root),
            "environment": environment,
            "policy": json.loads(json.dumps(policy)),
        }
        for environment in environments
    ]
    ordered_scenes = [
        {
            "scene_id": "scene0000_00",
            "schedule": {
                "root": str(tmp_path / "schedule-root"),
                "sha256": "a" * 64,
                "producer_fingerprint": "b" * 64,
            },
        }
    ]
    checkpoint = {"path": "/frozen/FastSAM.pt", "bytes": 1, "sha256": "c" * 64}
    sources = {
        "runner": {"path": "/frozen/runner.py", "sha256": "d" * 64},
        "core": {"path": "/frozen/core.py", "sha256": "e" * 64},
        "provider": {"path": "/frozen/provider.py", "sha256": "f" * 64},
    }
    environment_protocol = {
        key: value
        for key, value in environments[0].items()
        if key not in {"device", "gpu_uuid"}
    }
    payload = {
        "protocol_id": merge.EXPECTED_PROTOCOL_ID,
        "scene_list_sha256": "1" * 64,
        "scene_order": ["scene0000_00"],
        "scene_root": str(scene_root.resolve()),
        "schedule_manifests": [
            {
                "scene_id": "scene0000_00",
                "root": str(tmp_path / "schedule-root"),
                "sha256": "a" * 64,
                "producer_fingerprint": "b" * 64,
            }
        ],
        "checkpoint": checkpoint,
        "sources": sources,
        "environment_protocol": environment_protocol,
        "core_schema": "core.test.v1",
        "core_policy": {"voxel_size_m": 0.05},
    }
    signature = merge._canonical_json_sha256(payload)
    result = merge._validate_shared_run_signature(
        manifests=manifests,
        ordered_scenes=ordered_scenes,
        exact_scene_order=["scene0000_00"],
        scene_list_sha256="1" * 64,
        checkpoint=checkpoint,
        sources=sources,
        expected_signature=signature,
    )
    assert result["sha256"] == signature

    manifests[1]["policy"]["core"]["voxel_size_m"] = 0.10
    with pytest.raises(merge.F0MergeError, match="core policy"):
        merge._validate_shared_run_signature(
            manifests=manifests,
            ordered_scenes=ordered_scenes,
            exact_scene_order=["scene0000_00"],
            scene_list_sha256="1" * 64,
            checkpoint=checkpoint,
            sources=sources,
            expected_signature=signature,
        )


def test_partial_resume_requires_exact_unrecorded_three_call_rewarm(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "0.jpg"
    rgb.write_bytes(b"sealed-rgb")
    rgb_hash = hashlib.sha256(rgb.read_bytes()).hexdigest()
    calls = [
        {
            "ordinal": ordinal,
            "success": True,
            "wall_ms": 100.0,
            "raw_mask_count": 10,
            "masks_sha256": "a" * 64,
            "confidences_sha256": "b" * 64,
            "boxes_xyxy_sha256": "c" * 64,
            "provider_timing": _provider_timing("cuda:0"),
        }
        for ordinal in range(3)
    ]
    receipt = {
        "required": True,
        "reason": "cold_resume_with_completed_prefix_and_pending_suffix",
        "completed_scene_count": 1,
        "pending_scene_count": 1,
        "scene_id": "scene0001_00",
        "frame_id": 0,
        "rgb_path": str(rgb),
        "rgb_sha256": rgb_hash,
        "call_count": 3,
        "all_successful": True,
        "excluded_from_scene_counts": True,
        "excluded_from_capacity": True,
        "excluded_from_runtime_distributions": True,
        "calls": calls,
    }
    manifest = {
        "resume_rewarm_calls": 3,
        "resume_rewarm": receipt,
    }
    rows = [{"resumed": True}, {"resumed": False}]
    scenes = [
        {
            "scene_id": "scene0000_00",
            "first_frame": {"frame_id": 0, "rgb_path": None, "rgb_sha256": None},
        },
        {
            "scene_id": "scene0001_00",
            "first_frame": {
                "frame_id": 0,
                "rgb_path": str(rgb),
                "rgb_sha256": rgb_hash,
            },
        },
    ]
    result = merge._validate_resume_rewarm(
        manifest=manifest,
        manifest_scene_rows=rows,
        validated_scenes=scenes,
        expected_device="cuda:0",
        shard_index=0,
    )
    assert result["required"] and result["call_count"] == 3
    receipt["excluded_from_runtime_distributions"] = False
    with pytest.raises(merge.F0MergeError, match="must be true"):
        merge._validate_resume_rewarm(
            manifest=manifest,
            manifest_scene_rows=rows,
            validated_scenes=scenes,
            expected_device="cuda:0",
            shard_index=0,
        )


def test_schedule_manifest_is_rehashed_and_frame_ledger_is_exact(tmp_path: Path) -> None:
    row, manifest, scene, signature, _ = _one_scene_inputs(tmp_path)
    schedule = Path(row["schedule_path"])
    schedule.write_text(schedule.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(merge.F0MergeError, match="schedule manifest rehash"):
        merge._validate_scene_sidecar(
            row,
            manifest,
            expected_scene=scene,
            expected_index=0,
            expected_protocol_id="test-protocol",
            expected_run_signature=signature,
        )


def test_cli_surface_has_no_gt_evaluator_or_prediction_pickle_input() -> None:
    actions = merge._parser()._actions
    destinations = {action.dest for action in actions}
    assert destinations == {"help", "scene_list", "shard_manifest", "output_dir"}
    source = Path(merge.__file__).read_text(encoding="utf-8")
    assert "import pickle" not in source
    assert "import evaluator" not in source
