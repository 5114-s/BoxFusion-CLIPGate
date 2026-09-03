from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/audit_scannet_sam2_n0a_extra100_replay.py"
)
SPEC = importlib.util.spec_from_file_location("audit_scannet_sam2_n0a", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _provider_config_receipt(**overrides):
    config = {
        "source_root": "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2",
        "config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint_path": (
            "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/"
            "checkpoints/sam2.1_hiera_large.pt"
        ),
        "source_file_glob": "sam2/**/*.py",
        "source_file_count": 23,
        "source_tree_sha256": (
            "cc5a594bab1508ab69cbedfbb83ba8e226f848dd142a3deba8c195ee1e2469cf"
        ),
        "config_sha256": (
            "545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636"
        ),
        "checkpoint_bytes": 898_083_611,
        "checkpoint_sha256": (
            "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
        ),
        "device": "cuda",
        "apply_postprocessing": True,
        "autocast_dtype": "bfloat16",
        "multimask_output": True,
        "return_logits": False,
        "normalize_coords": True,
        "mask_threshold": 0.0,
        "max_boxes_per_frame": 16,
        "multimask_hypotheses": 3,
    }
    config.update(overrides)
    return config


def _environment_receipt(provider_config=None):
    config = (
        _provider_config_receipt()
        if provider_config is None
        else dict(provider_config)
    )
    return {
        "preflight": {
            "conda_environment": "gsam2_env",
            "versions": {
                "python": "3.10.19",
                "torch": "2.5.1+cu121",
                "torchvision": "0.20.1+cu121",
                "numpy": "2.2.6",
                "opencv": "4.13.0",
                "hydra": "1.3.2",
                "omegaconf": "2.3.0",
                "pillow": "12.0.0",
            },
            "gpu": {
                "logical_index": 0,
                "physical_index": 3,
                "uuid": "GPU-fixture",
                "name": "NVIDIA GeForce RTX 3090",
                "compute_capability": [8, 6],
                "total_memory_bytes": 25_429_606_400,
            },
            "determinism": {
                "seed": 0,
                "pythonhashseed": "0",
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": True,
                "deterministic_algorithms_warn_only": True,
                "registered_nondeterministic_warning": "aten::cumsum_cuda",
                "warning_policy_id": audit.WARNING_POLICY_ID,
                "expected_warning_count_per_nonempty_forward": 2,
                "bitwise_replay_required": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
            },
            "provider_config": config,
        },
        "platform": "Linux-fixture",
        "cuda_visible_devices": "3",
    }


def _warning_row(
    line: int,
    *,
    category=UserWarning,
    message=None,
    filename: str | None = None,
):
    return warnings.WarningMessage(
        UserWarning(audit.EXPECTED_WARNING_MESSAGE) if message is None else message,
        category,
        str(audit.EXPECTED_WARNING_SOURCE) if filename is None else filename,
        line,
    )


def _non_authorizing_mirror_receipt(case_hash: str = "a" * 64):
    return {
        "authorizing": False,
        "evidence_strength": "non_authorizing_provider_core_mirror",
        "fresh_worker_execution": True,
        "actual_sam2_provider_and_core_executed": True,
        "complete_current_frame_batch_replayed": True,
        "source_count": 1,
        "real_dataset_file_mutation_performed": False,
        "mirrored_future_files_add_alter_delete_performed": True,
        "directory_search_performed": False,
        "future_path_passed_to_provider_or_core": False,
        "future_paths_absent_after_fixture": True,
        "warning_policy": audit._expected_warning_policy(),
        "provider_forward_count": 4,
        "authenticated_warning_count": 8,
        "expected_warning_count": 8,
        "warning_count_formula": "2 * provider_forward_count",
        "per_forward_exact_warning_pair_passed": True,
        "baseline_batch_result_sha256": case_hash,
        "case_batch_result_sha256": {
            "future_files_added": case_hash,
            "future_files_altered": case_hash,
            "future_files_deleted": case_hash,
        },
        "overall_pass": True,
    }


def _runner_level_future_receipt(case_hash: str = "c" * 64):
    operations = {
        "baseline": [],
        "referenced_future_changed": ["referenced_future_rgb_changed"],
        "unreferenced_future_added": ["unreferenced_future_file_added"],
        "unreferenced_future_altered": [
            "unreferenced_future_file_added",
            "unreferenced_future_file_altered",
        ],
        "unreferenced_future_deleted": [
            "unreferenced_future_file_added",
            "unreferenced_future_file_deleted",
        ],
    }
    unreferenced_exists = {
        "baseline": False,
        "referenced_future_changed": False,
        "unreferenced_future_added": True,
        "unreferenced_future_altered": True,
        "unreferenced_future_deleted": False,
    }
    cases = {}
    for pid, name in enumerate(audit.FUTURE_CASES, start=1001):
        data_root = f"/tmp/n0a-future-test/{name}"
        current_paths = [
            f"{data_root}/current_rgb.jpg",
            f"{data_root}/current_depth.png",
            f"{data_root}/current_pose.txt",
        ]
        future_paths = [
            f"{data_root}/future_rgb.jpg",
            f"{data_root}/future_depth.png",
            f"{data_root}/future_pose.txt",
        ]
        unreferenced_path = f"{data_root}/future_unreferenced.bin"
        mutation = {
            "operations": operations[name],
            "unreferenced_exists_at_runner_start": unreferenced_exists[name],
            "referenced_future_rgb_original_sha256": "d" * 64,
            "referenced_future_rgb_sealed_sha256": "d" * 64,
            "sidecar_sha256_after_seal_update": "f" * 64,
            "unreferenced_sha256_at_runner_start": (
                "9" * 64 if unreferenced_exists[name] else None
            ),
        }
        if name == "referenced_future_changed":
            mutation.update(
                {
                    "referenced_future_rgb_original_sha256": "d" * 64,
                    "referenced_future_rgb_sealed_sha256": "e" * 64,
                }
            )
        row = {
            "schema": audit.FUTURE_CASE_SCHEMA,
            "complete": True,
            "case_name": name,
            "fresh_process": True,
            "worker_pid": pid,
            "request_nonce": hashlib.sha256(name.encode("ascii")).hexdigest(),
            "request_file_sha256": hashlib.sha256(
                f"request:{name}".encode("ascii")
            ).hexdigest(),
            "actual_default_runner_sam2_core": True,
            "provider_or_frame_loader_injected": False,
            "environment_gpu_determinism_authenticated": True,
            "warning_policy": audit._expected_warning_policy(),
            "provider_forward_count": 1,
            "authenticated_warning_count": 2,
            "expected_warning_count": 2,
            "per_forward_exact_warning_pair_passed": True,
            "complete_current_frame_batch_replayed": True,
            "current_batch": {
                "current_frame_id": 0,
                "source_count": 1,
                "source_ids": ["fixture/source"],
                "complete_source_rows": [{"source_id": "fixture/source"}],
                "evidence_arrays": [
                    {
                        "name": "mask_packbits",
                        "dtype": "|u1",
                        "shape": [1, audit.MASK_PACKED_BYTES],
                        "sha256": "8" * 64,
                    }
                ],
                "complete_current_batch_sha256": case_hash,
            },
            "mutation": mutation,
            "path_access_instrumentation": {
                "python_audit_hook_events": [
                    "open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2"
                ],
                "native_cv2_imread_wrapped_without_loader_injection": True,
                "events": [
                    {"event": "open", "path": path} for path in current_paths
                ],
                "fixture_data_root": data_root,
                "expected_current_rgb_depth_pose_paths": current_paths,
                "expected_future_rgb_depth_pose_paths": future_paths,
                "expected_unreferenced_future_path": unreferenced_path,
                "current_rgb_depth_pose_accessed": True,
                "future_rgb_depth_pose_or_unreferenced_access_events": [],
                "fixture_data_directory_enumeration_events": [],
                "stronger_no_future_open_at_any_time_passed": True,
            },
            "overall_pass": True,
        }
        row["content_sha256"] = audit._canonical_json_sha256(row)
        cases[name] = row
    receipt = {
        "authorizing": True,
        "evidence_strength": "fresh_real_runner_two_frame_file_perturbation",
        "case_names": list(audit.FUTURE_CASES),
        "fresh_distinct_process_count": len(audit.FUTURE_CASES),
        "actual_default_runner_sam2_core_every_case": True,
        "complete_current_frame_batch_replayed_every_case": True,
        "warning_policy": audit._expected_warning_policy(),
        "provider_forward_count": len(audit.FUTURE_CASES),
        "authenticated_warning_count": 2 * len(audit.FUTURE_CASES),
        "expected_warning_count": 2 * len(audit.FUTURE_CASES),
        "warning_count_formula": "2 * provider_forward_count",
        "per_forward_exact_warning_pair_passed": True,
        "baseline_current_batch_sha256": case_hash,
        "case_current_batch_sha256": {
            name: case_hash for name in audit.FUTURE_CASES
        },
        "referenced_future_content_changed_and_sidecar_resealed": True,
        "unreferenced_future_add_alter_delete_executed": True,
        "future_rgb_depth_pose_opened_at_any_time": False,
        "fixture_data_directory_enumerated": False,
        "cases": cases,
        "overall_pass": True,
    }
    receipt["content_sha256"] = audit._canonical_json_sha256(receipt)
    return receipt


def _exact_replay_receipt(*, mirrored=None, runner_level=None):
    receipt = {
        "schema": audit.WORKER_SCHEMA,
        "complete": True,
        "overall_pass": True,
        "fresh_process": True,
        "same_gpu_uuid": "GPU-fixture",
        "selector": "sha256(source_id.encode('ascii'))[:2]_big_endian_lt_0x0290",
        "first_full_scene_count": 50,
        "full_batch_for_sampled_frame": True,
        "scheduled_frame_batch_count": 1,
        "scheduled_source_comparison_count": 1,
        "compared_source_count": 1,
        "warning_policy": audit._expected_warning_policy(),
        "provider_forward_count": 1,
        "authenticated_warning_count": 2,
        "expected_warning_count": 2,
        "warning_count_formula": "2 * provider_forward_count",
        "per_forward_exact_warning_pair_passed": True,
        "mismatch_count": 0,
        "mismatches": [],
        "global_inputs_assets_unchanged": True,
        "global_inputs_assets_before_sha256": "b" * 64,
        "global_inputs_assets_after_sha256": "b" * 64,
    }
    if mirrored is not None:
        receipt["mirrored_future_only_file_perturbation"] = mirrored
    if runner_level is not None:
        receipt["runner_level_future_perturbation"] = runner_level
    return receipt


def _source(source_id: str, scene_position: int, rank: int = 0):
    all_ious = np.asarray([0.1, 0.9, 0.3], dtype="<f4")
    return audit.SourceRecord(
        scene_position=scene_position,
        scene_index=100 + scene_position,
        scene_id=f"scene{scene_position:04d}_00",
        frame_ordinal=0,
        frame_id=0,
        rank=rank,
        raw_index=rank,
        source_id=source_id,
        identity={"source_id": source_id},
        prompt_box=[0.0, 0.0, 20.0, 20.0],
        h0={},
        expected_selected_index=1,
        expected_selected_iou_bytes=all_ious[1].tobytes(),
        expected_all_iou_bytes=all_ious.tobytes(),
        expected_mask_sha256=hashlib.sha256(
            np.packbits(np.zeros((480, 640), dtype=bool).reshape(-1), bitorder="little").tobytes()
        ).hexdigest(),
        expected_result_sha256="a" * 64,
        expected_valid=False,
        expected_abstention_reason="fixture_abstain",
        expected_nontrivial=False,
    )


def _frame(scene_position: int, sources):
    return audit.FrameRecord(
        scene_position=scene_position,
        scene_index=100 + scene_position,
        scene_id=f"scene{scene_position:04d}_00",
        frame_ordinal=0,
        frame_id=0,
        rgb={}, depth={}, pose={}, intrinsic={}, sources=list(sources),
    )


def test_sample_selector_is_exact_big_endian_sha_prefix():
    ids = [f"scene0000_00/frame_{index:06d}/raw_{index % 16:03d}" for index in range(500)]
    expected = [
        int.from_bytes(hashlib.sha256(value.encode("ascii")).digest()[:2], "big") < 0x0290
        for value in ids
    ]
    assert [audit.replay_sample_selected(value) for value in ids] == expected
    assert any(expected)
    assert not all(expected)


def test_frozen_producer_and_protocol_source_pins_match_workspace():
    root = Path(__file__).resolve().parents[1]
    expected = {
        root / "tools/run_scannet_sam2_n0a_extra100.py": audit.EXPECTED_RUNNER_SHA256,
        root / "docs/N0A_SAM2_IMAGE_MASKLIFT_EXTRA100_PROTOCOL_FREEZE.md": audit.EXPECTED_PROTOCOL_SHA256,
        root / "boxfusion/sam2_masklift_n0a.py": audit.EXPECTED_CORE_SHA256,
        root / "boxfusion/sam2_boxprompt_provider.py": audit.EXPECTED_PROVIDER_SHA256,
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_provider_config_is_compared_to_constants_before_opening_paths(monkeypatch):
    def unexpected_path_open(*_args, **_kwargs):
        raise AssertionError("provider paths must not open before constant validation")

    monkeypatch.setattr(audit, "_regular_file", unexpected_path_open)
    expected = _provider_config_receipt()
    assert audit._validate_provider_config(expected, "fixture") == expected

    tampered = _provider_config_receipt(checkpoint_path="/tmp/not-the-frozen-model.pt")
    with pytest.raises(audit.N0AAuditError, match="provider configuration differs"):
        audit._validate_provider_config(tampered, "fixture")


def test_environment_receipt_authenticates_complete_frozen_policy():
    provider_config = _provider_config_receipt()
    receipt = _environment_receipt(provider_config)
    assert audit._validate_environment_receipt(
        receipt,
        provider_config=provider_config,
        label="fixture shard",
    ) == "GPU-fixture"

    wrong_version = copy.deepcopy(receipt)
    wrong_version["preflight"]["versions"]["torch"] = "2.5.1"
    with pytest.raises(audit.N0AAuditError, match="environment policy"):
        audit._validate_environment_receipt(
            wrong_version,
            provider_config=provider_config,
            label="fixture shard",
        )

    injected_mode = {
        "preflight": {
            "test_injection_mode": True,
            "production_environment_verified": False,
            "python_version": "3.10.19",
            "numpy_version": "2.2.6",
        },
        "platform": "Linux-fixture",
        "cuda_visible_devices": None,
    }
    with pytest.raises(audit.N0AAuditError, match="production preflight"):
        audit._validate_environment_receipt(
            injected_mode,
            provider_config=provider_config,
            label="fixture shard",
        )


def test_schedule_replays_complete_batch_but_compares_only_selected_subset():
    prefix = [_source("prefix/a", 0, 0), _source("prefix/b", 0, 1)]
    sampled_id = next(
        f"tail/{index}" for index in range(10000) if audit.replay_sample_selected(f"tail/{index}")
    )
    nonsampled_id = next(
        f"tail/no/{index}" for index in range(10000) if not audit.replay_sample_selected(f"tail/no/{index}")
    )
    tail = [_source(sampled_id, 1, 0), _source(nonsampled_id, 1, 1)]
    schedule = audit.replay_schedule([_frame(0, prefix), _frame(1, tail)], first_full_scenes=1)
    assert len(schedule) == 2
    assert schedule[0][1] == (0, 1)
    assert schedule[1][1] == (0,)
    assert len(schedule[1][0].sources) == 2


def test_future_only_fixture_is_honestly_scoped_and_causal():
    receipt = audit.future_only_isolation_fixture()
    assert receipt["overall_pass"] is True
    assert receipt["evidence_strength"] == "isolated_contract_fixture_only"
    assert receipt["real_dataset_future_file_mutation_performed"] is False
    assert receipt["real_dataset_future_invariance_claimed"] is False
    assert receipt["directory_search_performed"] is False
    assert set(receipt["earlier_hash_after_by_case"].values()) == {
        receipt["earlier_hash_before_sha256"]
    }


def test_mirrored_future_file_perturbation_is_exact_but_non_authorizing(tmp_path):
    import cv2

    rgb_path = tmp_path / "current.jpg"
    depth_path = tmp_path / "current.png"
    pose_path = tmp_path / "current_pose.txt"
    intrinsic_path = tmp_path / "intrinsic.txt"
    assert cv2.imwrite(str(rgb_path), np.zeros((480, 640, 3), dtype=np.uint8))
    assert cv2.imwrite(str(depth_path), np.full((480, 640), 1000, dtype=np.uint16))
    np.savetxt(pose_path, np.eye(4, dtype=np.float64))
    intrinsic = np.asarray([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]])
    np.savetxt(intrinsic_path, intrinsic)

    identity = {
        "scene_index": 100,
        "scene_id": "scene_fixture_00",
        "frame_ordinal": 0,
        "frame_id": 0,
        "rank": 0,
        "raw_index": 0,
        "mask_sha256": "1" * 64,
        "points_and_voxel_keys_sha256": "2" * 64,
        "source_id": "scene_fixture_00/frame_000000/raw_000",
    }
    source = _source(identity["source_id"], 0)
    source.identity = identity
    source.h0 = {
        "valid": True,
        "world_q02": [-0.2, -0.2, 0.8],
        "world_q98": [0.2, 0.2, 1.2],
    }

    def seal(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    frame = audit.FrameRecord(
        scene_position=0,
        scene_index=100,
        scene_id="scene_fixture_00",
        frame_ordinal=0,
        frame_id=0,
        rgb=seal(rgb_path),
        depth=seal(depth_path),
        pose=seal(pose_path),
        intrinsic=seal(intrinsic_path),
        sources=[source],
    )
    bundle = SimpleNamespace(frames=[frame])

    class Provider:
        def predict(self, image_rgb, boxes_xyxy):
            assert image_rgb.shape == (480, 640, 3)
            assert boxes_xyxy.shape == (1, 4)
            for line in audit.EXPECTED_WARNING_LINES:
                warnings.warn_explicit(
                    audit.EXPECTED_WARNING_MESSAGE,
                    UserWarning,
                    str(audit.EXPECTED_WARNING_SOURCE),
                    line,
                )
            mask = np.zeros((1, 480, 640), dtype=bool)
            mask[:, 180:300, 250:390] = True
            return SimpleNamespace(
                masks=mask,
                selected_hypothesis_indices=np.asarray([1], dtype=np.int64),
                predicted_ious=np.asarray([0.9], dtype=np.float32),
                all_predicted_ious=np.asarray([[0.1, 0.9, 0.3]], dtype=np.float32),
            )

    receipt = audit.perform_mirrored_future_perturbation(
        bundle, provider_factory=Provider
    )
    assert receipt["overall_pass"] is True
    assert receipt["authorizing"] is False
    assert receipt["evidence_strength"] == "non_authorizing_provider_core_mirror"
    assert receipt["actual_sam2_provider_and_core_executed"] is True
    assert receipt["mirrored_future_files_add_alter_delete_performed"] is True
    assert receipt["real_dataset_file_mutation_performed"] is False
    assert receipt["directory_search_performed"] is False
    assert receipt["warning_policy"] == audit._expected_warning_policy()
    assert receipt["provider_forward_count"] == 4
    assert receipt["authenticated_warning_count"] == 8
    assert receipt["expected_warning_count"] == 8
    assert receipt["warning_count_formula"] == "2 * provider_forward_count"
    assert receipt["per_forward_exact_warning_pair_passed"] is True
    assert "allowed_warning_unique_messages" not in receipt
    assert len(set(receipt["case_batch_result_sha256"].values())) == 1
    assert next(iter(receipt["case_batch_result_sha256"].values())) == receipt["baseline_batch_result_sha256"]


def test_exact_warning_rows_require_the_complete_ordered_tuple():
    evidence = audit._validate_exact_warning_rows(
        [_warning_row(143), _warning_row(144)], label="fixture forward"
    )
    assert evidence == audit._expected_warning_evidence()


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([], "exactly two"),
        ([_warning_row(143)], "exactly two"),
        ([_warning_row(143), _warning_row(144), _warning_row(144)], "exactly two"),
        ([_warning_row(144), _warning_row(143)], "identity/line"),
        ([_warning_row(143), _warning_row(143)], "identity/line"),
        (
            [_warning_row(143, category=RuntimeWarning), _warning_row(144)],
            "category/message type",
        ),
        (
            [
                _warning_row(
                    143,
                    message=RuntimeWarning(audit.EXPECTED_WARNING_MESSAGE),
                ),
                _warning_row(144),
            ],
            "category/message type",
        ),
        (
            [
                _warning_row(
                    143,
                    message=UserWarning("prefix " + audit.EXPECTED_WARNING_MESSAGE),
                ),
                _warning_row(144),
            ],
            "message differs",
        ),
        (
            [_warning_row(143, filename="/tmp/position_encoding.py"), _warning_row(144)],
            "source path",
        ),
    ],
)
def test_exact_warning_rows_fail_closed(rows, error):
    with pytest.raises(audit.N0AAuditError, match=error):
        audit._validate_exact_warning_rows(rows, label="fixture forward")


def test_final_decision_priority_and_exact_strings():
    assert audit.final_decision(integrity_pass=False, capacity_pass=True, runtime_pass=True) == audit.DISCARD_DECISION
    assert audit.final_decision(integrity_pass=True, capacity_pass=False, runtime_pass=False) == audit.CAPACITY_STOP_DECISION
    assert audit.final_decision(integrity_pass=True, capacity_pass=True, runtime_pass=False) == audit.RUNTIME_STOP_DECISION
    assert audit.final_decision(integrity_pass=True, capacity_pass=True, runtime_pass=True) == audit.RETAIN_DECISION


def test_non_authorizing_mirror_cannot_replace_runner_level_future_evidence():
    pure_only = _exact_replay_receipt()
    assert not audit._replay_authorizes(
        pure_only,
        gpu_uuid="GPU-fixture",
        expected_frame_batches=1,
        expected_sources=1,
    )
    mirrored_only = _exact_replay_receipt(
        mirrored=_non_authorizing_mirror_receipt()
    )
    assert not audit._replay_authorizes(
        mirrored_only,
        gpu_uuid="GPU-fixture",
        expected_frame_batches=1,
        expected_sources=1,
    )


def test_exact_runner_level_future_gate_is_the_authorizing_evidence():
    runner_level = _runner_level_future_receipt()
    assert audit._runner_future_gate_authorizes(runner_level)

    replay = _exact_replay_receipt(
        mirrored=_non_authorizing_mirror_receipt(),
        runner_level=runner_level,
    )
    assert audit._replay_authorizes(
        replay,
        gpu_uuid="GPU-fixture",
        expected_frame_batches=1,
        expected_sources=1,
    )

    wrongly_authorizing_mirror = _non_authorizing_mirror_receipt()
    wrongly_authorizing_mirror["authorizing"] = True
    replay["mirrored_future_only_file_perturbation"] = wrongly_authorizing_mirror
    assert not audit._replay_authorizes(
        replay,
        gpu_uuid="GPU-fixture",
        expected_frame_batches=1,
        expected_sources=1,
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "warning",
        "count",
        "path_access",
        "mutation",
        "default_provider",
        "fresh_pid",
    ],
)
def test_runner_level_future_gate_fails_closed_on_any_tamper(tamper):
    receipt = _runner_level_future_receipt()
    baseline = receipt["cases"]["baseline"]
    if tamper == "warning":
        receipt["warning_policy"]["message_sha256"] = "0" * 64
    elif tamper == "count":
        receipt["authenticated_warning_count"] -= 1
    elif tamper == "path_access":
        baseline["path_access_instrumentation"][
            "future_rgb_depth_pose_or_unreferenced_access_events"
        ] = [{"event": "open", "path": "/tmp/frame_000025.jpg"}]
    elif tamper == "mutation":
        receipt["cases"]["unreferenced_future_deleted"]["mutation"][
            "operations"
        ] = ["unreferenced_future_file_added"]
    elif tamper == "default_provider":
        baseline["provider_or_frame_loader_injected"] = True
    elif tamper == "fresh_pid":
        receipt["cases"]["referenced_future_changed"]["worker_pid"] = baseline[
            "worker_pid"
        ]
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(tamper)
    assert not audit._runner_future_gate_authorizes(receipt)


def test_injected_replay_executor_can_never_authorize_a_retain_decision(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "shard.json"
    manifest.write_text("{}", encoding="ascii")
    bundle = SimpleNamespace(
        input_seals={},
        frames=[],
        gpu_uuid="GPU-fixture",
        runner_decisions=[],
        manifest_paths=[manifest.resolve()],
        source_receipt_hashes={},
        counts={},
        ledger_hashes={},
    )
    monkeypatch.setattr(audit, "load_and_validate_bundle", lambda _paths: bundle)
    monkeypatch.setattr(
        audit,
        "future_only_isolation_fixture",
        lambda: {"overall_pass": True, "authorizing": False},
    )
    monkeypatch.setattr(audit, "replay_schedule", lambda _frames: [])
    # Isolate the executor-origin gate: even a forged authorizer result cannot
    # make dependency-injected replay evidence production-authorizing.
    monkeypatch.setattr(audit, "_replay_authorizes", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(audit, "_capacity_gates", lambda _bundle: {"overall_pass": True})
    monkeypatch.setattr(audit, "_runtime_gates", lambda _bundle: {"overall_pass": True})

    receipt = audit.audit_n0a(
        manifest_paths=[manifest],
        output_path=tmp_path / "audit.json",
        replay_executor=lambda _bundle: {
            "schema": audit.WORKER_SCHEMA,
            "complete": True,
            "fresh_process": True,
            "overall_pass": True,
        },
    )

    assert receipt["integrity_and_determinism_pass"] is False
    assert receipt["decision"] == audit.DISCARD_DECISION
    assert receipt["n0b_or_gt_stage_authorized"] is False


def test_exact_replay_comparison_catches_all_iou_ulp_and_hash_fields():
    source = _source("fixture/source", 0)
    mask = np.zeros((480, 640), dtype=bool)
    all_ious = np.asarray([0.1, 0.9, 0.3], dtype="<f4")
    core = SimpleNamespace(
        source_id=source.source_id,
        result_sha256="a" * 64,
        valid=False,
        abstention_reason="fixture_abstain",
    )
    assert audit.compare_replay_source(
        source,
        selected_index=1,
        selected_iou=all_ious[1],
        all_ious=all_ious,
        selected_mask=mask,
        core_result=core,
    ) == []
    altered = all_ious.copy()
    altered[0] = np.nextafter(altered[0], np.float32(1.0), dtype=np.float32)
    failures = audit.compare_replay_source(
        source,
        selected_index=1,
        selected_iou=all_ious[1],
        all_ious=altered,
        selected_mask=mask,
        core_result=core,
    )
    assert failures == ["all_predicted_iou_bytes"]


def test_global_snapshot_rehash_detects_mutation(tmp_path):
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"before")
    seals = {str(path.resolve()): hashlib.sha256(b"before").hexdigest()}
    before = audit._snapshot_hash(seals)
    passed, changed, after = audit._rehash_snapshot(seals)
    assert passed and changed == [] and before == after
    path.write_bytes(b"after")
    passed, changed, after = audit._rehash_snapshot(seals)
    assert not passed and changed == [str(path.resolve())] and after != before


def test_core_result_receipt_self_hash_detects_tampering():
    identity = {"source_id": "fixture/source"}
    h0 = {
        "valid": True,
        "world_q02": [0.0, 0.0, 0.0],
        "world_q98": [1.0, 1.0, 1.0],
        "world_center": [0.5, 0.5, 0.5],
        "world_extent": [1.0, 1.0, 1.0],
    }
    mask_sha = "1" * 64
    points_sha = "2" * 64
    result = {
        "schema": audit.CORE_SCHEMA,
        "protocol_id": audit.PROTOCOL_ID,
        "mode": "shadow",
        "contracts": {
            "f0_source_identity_preserved": True,
            "ground_truth_access": False,
            "semantic_or_clip_access": False,
            "native_prediction_access": False,
            "history_or_state": False,
            "training": False,
            "online_learning": False,
            "birth_enabled": False,
            "native_output_mutation": False,
        },
        "f0_source_identity": identity,
        "f0_source_identity_sha256": audit._canonical_json_sha256(identity),
        "h0_input": h0,
        "h0_input_sha256": audit._canonical_json_sha256(h0),
        "hypotheses": {
            "H0": {"name": "H0", "valid": True, "q02": [0, 0, 0], "q98": [1, 1, 1], "center": [.5, .5, .5], "extent": [1, 1, 1], "abstention_reason": None},
            "HS": {"name": "HS", "valid": False, "abstention_reason": "few",},
        },
        "mask": {"shape": [480, 640], "bitorder": "little", "packed_byte_count": audit.MASK_PACKED_BYTES, "sha256": mask_sha, "tight_box_xyxy": [0, 0, 20, 20], "pixel_count": 0, "valid_depth_ratio": 0.0, "interior_pixel_count": 0, "metric_depth_pixel_count": 0, "depth_jump_pixel_count": 0, "support_pixel_count": 0},
        "points": {"voxel_size_m": .02, "voxel_representative": "centroid", "voxel_count": 0, "quantile_point_count": 0, "stored_point_count": 0, "maximum_stored_point_count": 2048, "points_and_voxel_keys_sha256": points_sha},
        "valid": False,
        "abstention_reason": "few",
        "input_sha256": "3" * 64,
    }
    result["result_sha256"] = audit._canonical_json_sha256(audit._result_payload_from_receipt(result))
    _, valid, reason = audit._validate_core_receipt(result, identity, mask_sha, points_sha, result["result_sha256"], "fixture")
    assert valid is False and reason == "few"
    result["mask"]["pixel_count"] = 1
    try:
        audit._validate_core_receipt(result, identity, mask_sha, points_sha, result["result_sha256"], "fixture")
    except audit.N0AAuditError as error:
        assert "result hash" in str(error)
    else:
        raise AssertionError("tampered core receipt was accepted")


def test_create_only_json_refuses_overwrite(tmp_path):
    path = tmp_path / "audit.json"
    audit._atomic_create_json(path, {"complete": True})
    try:
        audit._atomic_create_json(path, {"complete": False})
    except audit.N0AAuditError as error:
        assert "overwrite" in str(error)
    else:
        raise AssertionError("audit output was overwritten")


def test_fail_closed_receipt_is_a_create_only_final_discard(tmp_path):
    manifest = tmp_path / "broken.json"
    manifest.write_text("{}", encoding="ascii")
    output = tmp_path / "audit.json"
    receipt = audit._create_fail_closed_receipt(
        manifest_paths=[manifest],
        output_path=output,
        error=audit.N0AAuditError("fixture integrity failure"),
    )
    assert receipt["decision"] == audit.DISCARD_DECISION
    assert receipt["integrity_and_determinism_pass"] is False
    assert receipt["n0b_or_gt_stage_authorized"] is False
    assert receipt["auditor_is_only_final_decision_authority"] is True
    assert output.is_file()
