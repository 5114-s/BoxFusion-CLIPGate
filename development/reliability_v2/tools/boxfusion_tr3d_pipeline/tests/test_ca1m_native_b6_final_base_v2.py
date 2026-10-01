from __future__ import annotations

from argparse import Namespace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

import cv2
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
FINALIZER_PATH = ROOT / "tools" / "finalize_ca1m_native_b6_final_base_v2.py"
COLLECTOR_PATH = ROOT / "tools" / "collect_ca1m_native_b6_final_base_offline.py"
FINAL_CONFIG = ROOT / "config" / "ca1m_native_final_base_train100_v1.yaml"
OFFLINE_CONFIG = ROOT / "config" / "ca1m_native_b6_final_base_train100_v2_offline.yaml"
PAIRED_REPORT = ROOT / "reports" / "ca1m_port" / "ca1m_c4_final_base_g0_clip_topk3_fixed10_v1" / "paired_eval_report.json"
REAL_TRAIN_ROOT = Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1")
OLD_DIAGNOSTIC_ROOT = ROOT / "diagnostics" / "ca1m_native_b6_train100_v1" / "native_b6"
REAL_SCENE_LIST = ROOT / "manifests" / "ca1m_native_b6_train100_v1" / "scene_ids.txt"
sys.path.insert(0, str(ROOT / "tools"))

from build_ca1m_native_b6_dataset import (  # noqa: E402
    DEFAULT_SPLIT_NAMESPACE,
    FINAL_BASE_COLLECTION_SCHEMA,
    FINAL_BASE_COMPLETION_SCHEMA,
    build,
    load_collection_manifest,
)
from boxfusion.ca1m_native_b6_observer import FEATURE_NAMES  # noqa: E402
from train_ca1m_native_b6_quality import load_dataset  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules():
    finalizer = load_module("native_b6_final_v2_test", FINALIZER_PATH)
    collector = load_module("native_b6_offline_v2_test", COLLECTOR_PATH)
    return finalizer, collector


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box() -> np.ndarray:
    return np.asarray(
        [
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            [-1, -1, 3], [1, -1, 3], [1, 1, 3], [-1, 1, 3],
        ],
        dtype=np.float32,
    )


def write_prediction(path: Path, corners: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            [[(0, corners[index], float(scores[index])) for index in range(len(scores))]],
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )


def write_final_manifest(path: Path, finalizer, scene_to_anchor: dict[str, Path]) -> None:
    count = len(scene_to_anchor)
    path.write_text(
        json.dumps(
            {
                "schema": finalizer.FINAL_BASE_SCHEMA,
                "ok": True,
                "dataset": "CA1M",
                "split": "train100",
                "scene_count": count,
                "same_run": {
                    "byte_identity_scenes": count,
                    "semantic_identity_scenes": count,
                    "hard_link_identity_scenes": count,
                },
                "clip_appearance_gate_active": True,
                "reliable_view_top_k": 3,
                "ground_truth_access": False,
                "evaluation_invoked": False,
                "training_invoked": False,
                "scannet_learned_b6_or_gate_reused": False,
                "per_scene": {
                    scene: {
                        "active_prediction_sha256": sha(anchor),
                        "byte_identity": True,
                    }
                    for scene, anchor in scene_to_anchor.items()
                },
            },
            sort_keys=True,
        )
    )


def write_paired_report(path: Path, *, negative_threshold: str | None = None) -> None:
    active = {"AP15": 0.35, "AP25": 0.30, "AP50": 0.13}
    control = {"AP15": 0.34, "AP25": 0.29, "AP50": 0.12}
    delta = {key: active[key] - control[key] for key in active}
    if negative_threshold is not None:
        active[negative_threshold] = control[negative_threshold] - 0.01
        delta[negative_threshold] = -0.01
    path.write_text(
        json.dumps(
            {
                "schema": "boxfusion.ca1m_final_base_paired_eval.v1",
                "complete": True,
                "dataset": "CA1M",
                "split": "validation_fixed10",
                "scene_count": 10,
                "paired_official_evaluation": True,
                "positive_map_at_all_thresholds": True,
                "training_invoked": False,
                "decision": {
                    "train100_final_base_collection_authorized": True,
                    "ca1m_native_b6_retraining_required": True,
                    "canonical_active_authorized": False,
                },
                "active": {key: {"mAP": value} for key, value in active.items()},
                "control": {key: {"mAP": value} for key, value in control.items()},
                "delta": {key: {"mAP": value} for key, value in delta.items()},
            },
            sort_keys=True,
        )
    )


def make_processed_scene(root: Path, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb = np.zeros((384, 512, 3), dtype=np.uint8)
    depth = np.full((384, 512), 2000, dtype=np.uint16)
    rgb_template = root / "rgb" / "0.png"
    depth_template = root / "depth" / "0.png"
    assert cv2.imwrite(str(rgb_template), rgb)
    assert cv2.imwrite(str(depth_template), depth)
    for frame_id in range(1, frame_count):
        os.link(rgb_template, root / "rgb" / f"{frame_id}.png")
        os.link(depth_template, root / "depth" / f"{frame_id}.png")
    poses = np.broadcast_to(np.eye(4), (frame_count, 4, 4)).copy()
    intrinsics = np.broadcast_to(
        np.asarray([[300.0, 0.0, 256.0], [0.0, 300.0, 192.0], [0.0, 0.0, 1.0]]),
        (frame_count, 3, 3),
    ).copy()
    np.save(root / "all_poses.npy", poses)
    np.save(root / "K_depth_per_frame.npy", intrinsics)
    np.savetxt(root / "K_depth.txt", intrinsics[0])


def offline_fixture(tmp_path: Path) -> Namespace:
    finalizer, collector = modules()
    scene = "42000001"
    corners = np.stack((box(),))
    scores = np.asarray((0.6,), dtype=np.float32)
    data_root = tmp_path / "train"
    make_processed_scene(data_root / scene, frame_count=46)
    final_root = tmp_path / "final_base"
    final_root.mkdir()
    final_anchor = final_root / f"{scene}_boxes.pkl"
    write_prediction(final_anchor, corners, scores)
    final_manifest = tmp_path / "final_base_collection.json"
    write_final_manifest(final_manifest, finalizer, {scene: final_anchor})
    paired_report = tmp_path / "paired_eval_report.json"
    write_paired_report(paired_report)
    diagnostic_root = tmp_path / "native"
    receipt_root = tmp_path / "receipts"
    diagnostic = diagnostic_root / f"{scene}_ca1m_native_b6.npz"
    receipt = receipt_root / f"{scene}.json"
    collect_args = Namespace(
        mode="run",
        scene=scene,
        config=OFFLINE_CONFIG,
        data_root=data_root,
        final_base_root=final_root,
        final_base_manifest=final_manifest,
        diagnostics_root=diagnostic_root,
        receipt=receipt,
    )
    report = collector.collect(collect_args)
    completion_root = tmp_path / "completion"
    completion_root.mkdir()
    scene_args = Namespace(
        scene=scene,
        final_base_root=final_root,
        final_base_manifest=final_manifest,
        diagnostic=diagnostic,
        offline_receipt=receipt,
    )
    completion_value = finalizer.scene_completion(scene_args)
    completion = completion_root / f"{scene}.json"
    finalizer.create_or_verify(completion, completion_value)
    scene_list = tmp_path / "scene_ids.txt"
    scene_list.write_text(scene + "\n")
    subset = tmp_path / "subset_manifest.json"
    subset.write_text(
        json.dumps(
            {
                "schema": finalizer.SUBSET_SCHEMA,
                "selection": {"subset_size": 1, "scene_ids_sha256": sha(scene_list)},
                "safety_contract": {
                    "train_only": True,
                    "validation_ground_truth_access": False,
                    "validation_scene_overlap_count": 0,
                    "training_started": False,
                },
                "entries": [
                    {
                        "rank": 0,
                        "scene_id": scene,
                        "url": f"https://ml-site.cdn-apple.com/datasets/ca1m/train/ca1m-train-{scene}.tar",
                    }
                ],
            },
            sort_keys=True,
        )
    )
    collection_value = finalizer.collection(
        Namespace(
            subset_manifest=subset,
            expected_scenes=1,
            completion_root=completion_root,
            final_base_root=final_root,
            final_base_manifest=final_manifest,
            paired_report=paired_report,
        )
    )
    collection_manifest = tmp_path / "v2_collection.json"
    finalizer.create_or_verify(collection_manifest, collection_value)
    return Namespace(
        finalizer=finalizer,
        collector=collector,
        scene=scene,
        scene_list=scene_list,
        subset=subset,
        final_root=final_root,
        final_manifest=final_manifest,
        paired_report=paired_report,
        diagnostic=diagnostic,
        receipt=receipt,
        report=report,
        collect_args=collect_args,
        completion=completion,
        completion_root=completion_root,
        collection_manifest=collection_manifest,
        collection_value=collection_value,
    )


def test_v2_contract_is_offline_direct_and_has_no_checkpoint() -> None:
    finalizer, _ = modules()
    report = finalizer.audit_contract(FINAL_CONFIG, OFFLINE_CONFIG, PAIRED_REPORT)
    assert report["ok"] is True
    assert report["geometry_authority"] == "sealed_final_base_prediction"
    assert report["offline_direct_observer"] is True
    assert report["cross_run_boxfusion_replay_invoked"] is False
    assert report["cross_run_exact_identity_required"] is False
    assert report["fixed10_paired_report"]["sha256"] == sha(PAIRED_REPORT)
    cfg = yaml.safe_load(OFFLINE_CONFIG.read_text())
    assert cfg["source_anchor"]["required_modules"]["reliable_view_top_k"] == 3
    assert cfg["observer"]["top_k_views"] == 5
    assert "ca1m_native_b6_score" not in cfg


def test_fixed10_paired_report_rejects_any_nonpositive_map_delta(
    tmp_path: Path,
) -> None:
    finalizer, _ = modules()
    report = tmp_path / "negative_paired_report.json"
    write_paired_report(report, negative_threshold="AP25")
    with pytest.raises(ValueError, match="AP25 mAP delta is not positive"):
        finalizer.load_paired_report(report)


def test_demo_lineage_simulation_does_not_force_terminal_frame() -> None:
    _, collector = modules()
    assert collector.selected_frame_ids(1, 20) == (0,)
    assert collector.selected_frame_ids(22, 20) == (0,)
    assert collector.selected_frame_ids(46, 20) == (0, 20)
    assert collector.selected_frame_ids(326, 20) == tuple(range(0, 301, 20))


def test_demo_lineage_matches_all_100_sealed_v1_diagnostics() -> None:
    if not (REAL_TRAIN_ROOT.is_dir() and OLD_DIAGNOSTIC_ROOT.is_dir()):
        pytest.skip("sealed CA train100 lineage oracle is not installed")
    _, collector = modules()
    scenes = tuple(REAL_SCENE_LIST.read_text().splitlines())
    assert len(scenes) == 100
    for scene in scenes:
        frame_count = len(
            np.load(REAL_TRAIN_ROOT / scene / "all_poses.npy", allow_pickle=False)
        )
        expected = np.asarray(
            collector.selected_frame_ids(frame_count, 20), dtype=np.int64
        )
        with np.load(
            OLD_DIAGNOSTIC_ROOT / f"{scene}_ca1m_native_b6.npz",
            allow_pickle=False,
        ) as archive:
            assert np.array_equal(archive["used_frame_ids"], expected), scene


@pytest.mark.parametrize(
    ("scene", "frame_id", "orientation_kind"),
    (
        ("48018894", 0, "landscape"),
        ("43649774", 0, "portrait"),
        ("42445047", 640, "mixed"),
    ),
)
def test_offline_depth_k_pose_match_real_ca1m_dataset(
    scene: str, frame_id: int, orientation_kind: str
) -> None:
    if not (REAL_TRAIN_ROOT / scene).is_dir():
        pytest.skip("processed real CA train scene is not installed")
    _, collector = modules()
    from boxfusion.capture_stream import CA1MDataset

    cfg, _ = collector.load_config(OFFLINE_CONFIG)
    inputs = collector._scene_inputs(REAL_TRAIN_ROOT / scene, cfg)
    offline_depth, offline_k, offline_pose, _ = collector.load_observer_frame(
        inputs, frame_id, cfg
    )

    dataset_cfg = yaml.safe_load(FINAL_CONFIG.read_text())
    dataset_cfg["data"]["datadir"] = str(REAL_TRAIN_ROOT / scene)
    dataset = CA1MDataset(dataset_cfg)
    # Exercise the real loader implementation at the requested source frame
    # without decoding every preceding RGB image.
    dataset.img_files = [dataset.img_files[frame_id]]
    dataset.depth_paths = [dataset.depth_paths[frame_id]]
    dataset.poses = dataset.poses[[frame_id]]
    dataset.depth_intrinsics = dataset.depth_intrinsics[[frame_id]]
    dataset.frame_ids = range(1)
    dataset.num_frames = 1
    sample = next(iter(dataset))
    online_depth = np.asarray(sample["wide"]["depth"][-1].numpy(), dtype=np.float32)
    online_k = np.asarray(
        sample["sensor_info"].wide.depth.K[-1].numpy(), dtype=np.float64
    )
    online_pose = np.asarray(
        sample["sensor_info"].gt.RT[-1].numpy(), dtype=np.float64
    )
    assert np.array_equal(offline_depth, online_depth)
    assert np.array_equal(offline_k, online_k)
    assert np.array_equal(offline_pose, online_pose)

    orientation = json.loads(
        (REAL_TRAIN_ROOT / scene / "derived_train_gt_manifest.json").read_text()
    )["orientation"]
    if orientation_kind == "mixed":
        assert sum(int(value) > 0 for value in orientation["rotation_counts"].values()) > 1
    else:
        assert orientation["target_orientation"] == orientation_kind
        assert sum(int(value) > 0 for value in orientation["rotation_counts"].values()) == 1


def test_offline_scene_collection_bind_directly_to_sealed_final_base(
    tmp_path: Path,
) -> None:
    data = offline_fixture(tmp_path)
    assert data.report["frame_protocol"]["used_frame_ids"] == [0, 20]
    assert data.report["frame_protocol"]["physical_terminal_frame_policy"] == "not_forced"
    assert data.report["cross_run_boxfusion_replay_invoked"] is False
    completion = json.loads(data.completion.read_text())
    assert completion["schema"] == FINAL_BASE_COMPLETION_SCHEMA
    assert completion["phase"] == "sealed_final_base_offline_native_b6_observer"
    assert completion["artifacts"]["prediction"] == completion["artifacts"]["final_base_anchor"]
    collection = data.collection_value
    assert collection["schema"] == FINAL_BASE_COLLECTION_SCHEMA
    assert collection["geometry_authority"] == "sealed_final_base_prediction"
    assert collection["cross_run_exact_identity_required"] is False
    assert collection["fixed10_paired_report"]["sha256"] == sha(data.paired_report)
    assert collection["split_protocol"] == {
        "kind": "deterministic_scene_grouped_5fold",
        "namespace": DEFAULT_SPLIT_NAMESPACE,
        "deployable_training_folds": [1, 2, 3, 4],
        "untouched_dev_fold": 0,
    }


def test_orphan_diagnostic_is_recomputed_before_receipt_recovery(tmp_path: Path) -> None:
    data = offline_fixture(tmp_path)
    original_diagnostic_sha = sha(data.diagnostic)
    data.receipt.chmod(0o644)
    data.receipt.unlink()
    report = data.collector.collect(data.collect_args)
    assert sha(data.diagnostic) == original_diagnostic_sha
    assert report["diagnostic_recovery"] == {
        "preexisting_orphan": True,
        "semantic_recomputation_exact": True,
        "runtime_only_field_ignored": "summary_json.observer_seconds",
    }
    receipt = json.loads(data.receipt.read_text())
    assert receipt["diagnostic"]["sha256"] == original_diagnostic_sha


def test_dataset_join_rejects_drifted_sealed_source(tmp_path: Path) -> None:
    data = offline_fixture(tmp_path)
    payload, _, _, completions = load_collection_manifest(
        data.collection_manifest,
        data.completion_root,
        scenes=(data.scene,),
        subset_sha=sha(data.subset),
        scene_ids_sha=sha(data.scene_list),
    )
    assert payload["schema"] == FINAL_BASE_COLLECTION_SCHEMA
    assert completions[data.scene]["schema"] == FINAL_BASE_COMPLETION_SCHEMA
    source = data.final_root / f"{data.scene}_boxes.pkl"
    source.chmod(0o644)
    source.write_bytes(source.read_bytes() + b"x")
    with pytest.raises(ValueError, match="final-base anchor binding disagrees"):
        load_collection_manifest(
            data.collection_manifest,
            data.completion_root,
            scenes=(data.scene,),
            subset_sha=sha(data.subset),
            scene_ids_sha=sha(data.scene_list),
        )


def test_dataset_join_rejects_tampered_authoritative_paired_report(
    tmp_path: Path,
) -> None:
    data = offline_fixture(tmp_path)
    data.paired_report.write_text(data.paired_report.read_text() + "\n")
    with pytest.raises(ValueError, match="fixed10 paired report binding disagrees"):
        load_collection_manifest(
            data.collection_manifest,
            data.completion_root,
            scenes=(data.scene,),
            subset_sha=sha(data.subset),
            scene_ids_sha=sha(data.scene_list),
        )


def test_full_v2_dataset_join_and_training_preflight_preserve_folds(
    tmp_path: Path,
) -> None:
    protocol_path = ROOT / "tests" / "test_ca1m_native_b6_training_protocol.py"
    protocol = load_module("native_b6_protocol_fixture", protocol_path)
    args = protocol._fixture(tmp_path)
    scenes = tuple(args.scene_list.read_text().splitlines())
    finalizer, _ = modules()
    final_root = tmp_path / "final_base_v2"
    final_root.mkdir()
    anchors: dict[str, Path] = {}
    for scene in scenes:
        target = final_root / f"{scene}_boxes.pkl"
        shutil.copyfile(args.prediction_root / target.name, target)
        anchors[scene] = target
    final_manifest = tmp_path / "final_base_v2_manifest.json"
    write_final_manifest(final_manifest, finalizer, anchors)
    paired_report = tmp_path / "paired_eval_report_v2.json"
    write_paired_report(paired_report)
    completion_rows = []
    for scene in scenes:
        completion_path = args.observer_completion_root / f"{scene}.json"
        completion = json.loads(completion_path.read_text())
        completion.update(
            {
                "schema": FINAL_BASE_COMPLETION_SCHEMA,
                "phase": "sealed_final_base_offline_native_b6_observer",
                "validation_prediction_access": False,
                "geometry_authority": "sealed_final_base_prediction",
                "offline_direct_observer": True,
                "cross_run_boxfusion_replay_invoked": False,
                "cross_run_exact_identity_required": False,
                "rgb_pixels_accessed": False,
                "old_native_b6_diagnostics_reused": False,
                "old_native_b6_checkpoint_reused": False,
            }
        )
        final_anchor = anchors[scene]
        final_record = {"path": str(final_anchor.resolve()), "sha256": sha(final_anchor)}
        completion["artifacts"]["prediction"] = final_record
        completion["artifacts"]["final_base_anchor"] = final_record
        completion_path.write_text(json.dumps(completion, sort_keys=True))
        completion_rows.append(
            {
                "scene_id": scene,
                "observer_completion_sha256": sha(completion_path),
                "final_base_prediction_sha256": sha(final_anchor),
            }
        )
    collection = {
        "schema": FINAL_BASE_COLLECTION_SCHEMA,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "geometry_authority": "sealed_final_base_prediction",
        "offline_direct_observer": True,
        "cross_run_boxfusion_replay_invoked": False,
        "cross_run_exact_identity_required": False,
        "rgb_pixels_accessed": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
        "source_modules": {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        },
        "scene_count": len(scenes),
        "scene_ids_sha256": sha(args.scene_list),
        "subset_manifest_sha256": sha(args.subset_manifest),
        "source_final_base_collection": {
            "path": str(final_manifest.resolve()),
            "sha256": sha(final_manifest),
            "schema": finalizer.FINAL_BASE_SCHEMA,
        },
        "source_final_base_root": str(final_root.resolve()),
        "fixed10_paired_report": {
            "path": str(paired_report.resolve()),
            "sha256": sha(paired_report),
            "schema": finalizer.PAIRED_REPORT_SCHEMA,
            "role": "authoritative_fixed10_train100_and_retraining_gate",
        },
        "split_protocol": {
            "kind": "deterministic_scene_grouped_5fold",
            "namespace": DEFAULT_SPLIT_NAMESPACE,
            "deployable_training_folds": [1, 2, 3, 4],
            "untouched_dev_fold": 0,
        },
        "scenes": completion_rows,
    }
    args.collection_manifest.write_text(json.dumps(collection, sort_keys=True))
    args.prediction_root = final_root
    args.split_namespace = DEFAULT_SPLIT_NAMESPACE
    report = build(args)
    train_collection = report["train_collection"]
    assert train_collection["schema"] == FINAL_BASE_COLLECTION_SCHEMA
    assert train_collection["geometry_authority"] == "sealed_final_base_prediction"
    values, loaded = load_dataset(args.output, args.manifest_output)
    assert loaded["split"]["namespace"] == DEFAULT_SPLIT_NAMESPACE
    assert set(values["fold_ids"].tolist()) == set(range(5))


def test_v2_runners_are_cpu_offline_isolated_and_validation_safe() -> None:
    collection = (
        ROOT / "scripts" / "collect_ca1m_native_b6_final_base_train100_v2.sh"
    ).read_text()
    training = (
        ROOT / "scripts" / "train_ca1m_native_b6_final_base_quality_v2.sh"
    ).read_text()
    assert 'MODE="preflight"' in collection
    assert "BOXFUSION_CA1M_FINAL_BASE_FIXED10_ACCEPTED" in collection
    assert "collect_ca1m_native_b6_final_base_offline.py" in collection
    assert "--offline-config" in collection
    assert "paired_eval_report.json" in collection
    assert "--paired-report" in collection
    assert "demo.py" not in collection
    assert "CUDA_VISIBLE_DEVICES" not in collection
    assert "cutr_rgbd.pth" not in collection
    assert "open_clip_pytorch_model.bin" not in collection
    assert "same-run-anchor" not in collection
    assert "eval_ca1m.py" not in collection
    assert "receipt exists without a safe diagnostic" in collection
    assert "--build-dataset|--preflight|--train" in training
    assert "ca1m_native_final_base_train100_v1" in training
    assert "completion/offline_native_b6" in training
    assert "boxfusion.ca1m-native-b6.scene-folds.v1" in training
    assert "ca1m_native_b6_final_base_iou_mlp_v2.npz" in training
    assert "evaluation/eval_ca1m.py" not in training
