from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_ca1m_native_b6_dataset import (  # noqa: E402
    COLLECTION_SCHEMA,
    COMPLETION_SCHEMA,
    DATASET_SCHEMA,
    GT_SCHEMA,
    MANIFEST_SCHEMA,
    OBSERVER_SCHEMA,
    SUBSET_SCHEMA,
    TARGET_SCHEMA,
    build,
    ca1m_pairwise_iou_v2,
)
from boxfusion.ca1m_native_b6_observer import FEATURE_NAMES  # noqa: E402
from train_ca1m_native_b6_quality import (  # noqa: E402
    deployment_scores,
    detection_average_precision,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    half = np.asarray(size, dtype=np.float32) * 0.5
    signs = np.asarray(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    return center + signs * half


def _fixture(tmp_path: Path, *, validation_overlap: bool = False) -> argparse.Namespace:
    scenes = tuple(f"{42_000_001 + index:08d}" for index in range(5))
    observer_root = tmp_path / "observer"
    prediction_root = tmp_path / "predictions"
    gt_root = tmp_path / "gt"
    completion_root = tmp_path / "completion" / "g0_observer"
    for directory in (observer_root, prediction_root, gt_root, completion_root):
        directory.mkdir(parents=True)
    scene_list = tmp_path / "scene_ids.txt"
    scene_list.write_text("".join(f"{scene}\n" for scene in scenes))
    subset = {
        "schema": SUBSET_SCHEMA,
        "purpose": "unit-test train-only subset",
        "selection": {
            "namespace": "unit-test",
            "subset_size": len(scenes),
            "scene_ids_sha256": hashlib.sha256(scene_list.read_bytes()).hexdigest(),
        },
        "source": {},
        "safety_contract": {
            "train_only": True,
            "validation_scene_overlap_count": 0,
            "validation_ground_truth_access": False,
            "training_started": False,
            "automatic_download": False,
        },
        "entries": [
            {
                "rank": index,
                "scene_id": scene,
                "url": f"https://ml-site.cdn-apple.com/datasets/ca1m/train/ca1m-train-{scene}.tar",
            }
            for index, scene in enumerate(scenes)
        ],
    }
    subset_path = tmp_path / "subset_manifest.json"
    subset_path.write_text(json.dumps(subset, sort_keys=True))
    subset_sha = _sha(subset_path)
    val_scene = scenes[0] if validation_overlap else "51000000"
    val_urls = tmp_path / "val.txt"
    val_urls.write_text(
        f"https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-{val_scene}.tar\n"
    )

    for scene_index, scene in enumerate(scenes):
        corners = np.stack((_box(), _box(center=(5.0, 0.0, 0.0))))
        scores = np.asarray((0.9, 0.1), dtype=np.float32)
        features = np.full((2, len(FEATURE_NAMES)), 0.2 + 0.01 * scene_index, dtype=np.float32)
        features[:, 0] = scores
        np.savez_compressed(
            observer_root / f"{scene}_ca1m_native_b6.npz",
            schema=np.asarray(OBSERVER_SCHEMA),
            complete=np.asarray(True),
            observer_only=np.asarray(True),
            mutation_enabled=np.asarray(False),
            ground_truth_access=np.asarray(False),
            scene_id=np.asarray(scene),
            result_indices=np.arange(2, dtype=np.int64),
            stable_ids=np.asarray((100 + scene_index * 2, 101 + scene_index * 2), dtype=np.int64),
            corners=corners,
            scores=scores,
            feature_names=np.asarray(FEATURE_NAMES),
            features=features,
            valid_evidence=np.asarray((True, True)),
        )
        with (prediction_root / f"{scene}_boxes.pkl").open("wb") as handle:
            pickle.dump([[(0, corners[row], float(scores[row])) for row in range(2)]], handle)
        scene_root = gt_root / scene
        scene_root.mkdir()
        native_gt = scene_root / "derived_train_gt_boxes.npy"
        np.save(native_gt, np.stack((_box(),)))
        compatibility = scene_root / "after_filter_boxes.npy"
        compatibility.write_bytes(native_gt.read_bytes())
        gt_manifest = {
            "schema": GT_SCHEMA,
            "scene_id": scene,
            "source_split": "train",
            "train_only": True,
            "validation_scene_overlap": False,
            "validation_ground_truth_access": False,
            "derived_train_gt": True,
            "official_validation_comparable": False,
            "source_tar": {"path": f"/official/ca1m-train-{scene}.tar", "sha256": "a" * 64},
            "frozen_subset_manifest": {"path": str(subset_path), "sha256": subset_sha},
            "derived_train_gt_sha256": _sha(native_gt),
            "compat_after_filter_sha256": _sha(compatibility),
        }
        (scene_root / "derived_train_gt_manifest.json").write_text(json.dumps(gt_manifest))
        completion = {
            "schema": COMPLETION_SCHEMA,
            "phase": "g0_native_b6_observer",
            "scene_id": scene,
            "complete": True,
            "train_only": True,
            "evaluation_invoked": False,
            "validation_ground_truth_access": False,
            "output_mutation_authorized": False,
            "artifacts": {
                "prediction": {
                    "path": str((prediction_root / f"{scene}_boxes.pkl").resolve()),
                    "sha256": _sha(prediction_root / f"{scene}_boxes.pkl"),
                },
                "native_b6_diagnostic": {
                    "path": str((observer_root / f"{scene}_ca1m_native_b6.npz").resolve()),
                    "sha256": _sha(observer_root / f"{scene}_ca1m_native_b6.npz"),
                },
            },
        }
        (completion_root / f"{scene}.json").write_text(json.dumps(completion, sort_keys=True))
    collection = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "scene_count": len(scenes),
        "scene_ids_sha256": hashlib.sha256(scene_list.read_bytes()).hexdigest(),
        "subset_manifest_sha256": subset_sha,
        "completion_collection_sha256": "b" * 64,
        "scenes": [
            {
                "scene_id": scene,
                "record_completion_sha256": "c" * 64,
                "observer_completion_sha256": _sha(completion_root / f"{scene}.json"),
            }
            for scene in scenes
        ],
    }
    collection_manifest = tmp_path / "collection_manifest.json"
    collection_manifest.write_text(json.dumps(collection, sort_keys=True))
    return argparse.Namespace(
        observer_root=observer_root,
        prediction_root=prediction_root,
        gt_root=gt_root,
        scene_list=scene_list,
        subset_manifest=subset_path,
        collection_manifest=collection_manifest,
        observer_completion_root=completion_root,
        val_url_list=val_urls,
        output=tmp_path / "quality.npz",
        manifest_output=tmp_path / "quality.manifest.json",
        split_namespace="unit-test-folds-v1",
    )


def test_ca1m_evaluator_iou_target_is_world_enclosing_aabb() -> None:
    target = _box()
    translated = _box(center=(1.0, 0.0, 0.0))
    observed = ca1m_pairwise_iou_v2(np.stack((target, translated)), np.stack((target,)))
    np.testing.assert_allclose(observed[:, 0], (1.0, 1.0 / 3.0), atol=1e-12)


def test_gate_ap_uses_strict_iou_duplicate_suppression_and_voc_envelope() -> None:
    scores = np.asarray((0.9, 0.8, 0.7, 0.6), dtype=np.float64)
    targets = np.asarray((0.9, 0.9, 0.5, 0.9), dtype=np.float64)
    scenes = np.asarray(("42000001", "42000001", "42000002", "42000002"))
    matched = np.asarray((0, 0, 0, 0), dtype=np.int64)
    observed = detection_average_precision(
        scores,
        targets,
        scenes,
        matched,
        {"42000001": 1, "42000002": 1},
        np.arange(4),
        ("42000001", "42000002"),
        0.50,
    )
    # ranks: TP, duplicate FP, strict-equality FP, TP.  eval_det's VOC
    # envelope integrates 0.5*1.0 + 0.5*0.5 (modulo its npos+1e-6).
    assert observed == pytest.approx(0.75, abs=1e-6)


def test_gate_ap_matches_repository_ca1m_eval_det() -> None:
    evaluation_utils = str(ROOT / "evaluation" / "utils")
    sys.path.insert(0, evaluation_utils)
    try:
        from eval_det import eval_det_cls, get_iou_obb_v2
    finally:
        sys.path.remove(evaluation_utils)
    box = _box()
    far = _box(center=(8.0, 0.0, 0.0))
    predictions = {
        0: [(box, 0.9), (box, 0.8)],
        1: [(far, 0.7), (box, 0.6)],
    }
    ground_truth = {0: [box], 1: [box]}
    _, _, expected = eval_det_cls(
        predictions, ground_truth, ovthresh=0.50, get_iou_func=get_iou_obb_v2
    )
    corners = np.stack((box, box, far, box))
    targets = np.concatenate(
        (
            ca1m_pairwise_iou_v2(corners[:2], np.stack((box,)))[:, 0],
            ca1m_pairwise_iou_v2(corners[2:], np.stack((box,)))[:, 0],
        )
    )
    observed = detection_average_precision(
        np.asarray((0.9, 0.8, 0.7, 0.6)),
        targets,
        np.asarray(("42000001", "42000001", "42000002", "42000002")),
        np.zeros(4, dtype=np.int64),
        {"42000001": 1, "42000002": 1},
        np.arange(4),
        ("42000001", "42000002"),
        0.50,
    )
    assert observed == pytest.approx(expected, abs=1e-12)


def test_deployment_score_matches_monotonic_projection_and_detector_blend() -> None:
    detector = np.asarray((0.8,), dtype=np.float64)
    outputs = np.asarray(((0.6, 0.9, 0.2, 0.8),), dtype=np.float64)
    weights = np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float64)
    components, quality, deployed = deployment_scores(detector, outputs, weights, 0.4)
    np.testing.assert_allclose(components, ((0.6, 0.9, 0.2, 0.2),))
    np.testing.assert_allclose(quality, (0.38,))
    np.testing.assert_allclose(deployed, (0.548,))


def test_build_train_only_join_and_balanced_scene_folds(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    report = build(args)
    assert report["schema"] == MANIFEST_SCHEMA
    assert report["train_only"] and not report["validation_ground_truth_access"]
    assert report["target"]["schema"] == TARGET_SCHEMA
    assert report["split"]["scene_counts"] == {str(index): 1 for index in range(5)}
    with np.load(args.output, allow_pickle=False) as data:
        assert data["schema"].item() == DATASET_SCHEMA
        assert data["quality_features"].shape == (10, len(FEATURE_NAMES))
        np.testing.assert_allclose(data["target_iou"].reshape(5, 2), [[1.0, 0.0]] * 5)
        assert set(data["fold_ids"].tolist()) == set(range(5))
        assert np.array_equal(data["dev_mask"], data["fold_ids"] == 0)
    assert report["dataset"]["sha256"] == _sha(args.output)


def test_join_refuses_any_official_validation_scene(tmp_path: Path) -> None:
    args = _fixture(tmp_path, validation_overlap=True)
    with pytest.raises(ValueError, match="overlaps official validation"):
        build(args)
    assert not args.output.exists()


def test_join_refuses_observer_prediction_drift(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    scene = args.scene_list.read_text().splitlines()[0]
    prediction = args.prediction_root / f"{scene}_boxes.pkl"
    payload = pickle.loads(prediction.read_bytes())
    payload[0][0] = (0, payload[0][0][1] + np.float32(0.01), payload[0][0][2])
    prediction.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="collection prediction SHA256 disagrees"):
        build(args)


def test_join_refuses_stale_collection_binding(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    scene = args.scene_list.read_text().splitlines()[0]
    completion = args.observer_completion_root / f"{scene}.json"
    payload = json.loads(completion.read_text())
    payload["artifacts"]["prediction"]["sha256"] = "0" * 64
    completion.write_text(json.dumps(payload, sort_keys=True))
    collection = json.loads(args.collection_manifest.read_text())
    collection["scenes"][0]["observer_completion_sha256"] = _sha(completion)
    args.collection_manifest.write_text(json.dumps(collection, sort_keys=True))
    with pytest.raises(ValueError, match="collection prediction SHA256 disagrees"):
        build(args)


def test_dataset_join_is_create_only(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    build(args)
    before = args.output.read_bytes(), args.manifest_output.read_bytes()
    with pytest.raises(FileExistsError, match="existing dataset/manifest"):
        build(args)
    assert before == (args.output.read_bytes(), args.manifest_output.read_bytes())


def test_training_preflight_and_native_dimension_adapter(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    build(args)
    tool = ROOT / "tools" / "train_ca1m_native_b6_quality.py"
    preflight = subprocess.run(
        [
            sys.executable, str(tool), "--dataset", str(args.output),
            "--dataset-manifest", str(args.manifest_output), "--preflight",
        ],
        text=True,
        capture_output=True,
    )
    assert preflight.returncode == 0, preflight.stderr
    payload = json.loads(preflight.stdout)
    assert payload["training_started"] is False
    assert payload["feature_dim"] == len(FEATURE_NAMES)
    assert not (tmp_path / "checkpoint.npz").exists()

    trained = subprocess.run(
        [
            sys.executable, str(tool), "--dataset", str(args.output),
            "--dataset-manifest", str(args.manifest_output), "--train",
            "--output", str(tmp_path / "checkpoint.npz"),
            "--manifest-output", str(tmp_path / "checkpoint.manifest.json"),
            "--epochs", "2", "--hidden-dims", "4",
            "--max-dev-ap15-loss", "1", "--max-dev-ap25-loss", "1",
            "--max-oof-ap15-loss", "1", "--max-oof-ap25-loss", "1",
            "--min-dev-ap50-gain", "-1", "--min-oof-ap50-gain", "-1",
            "--min-positive-ap50-folds", "0",
        ],
        text=True,
        capture_output=True,
    )
    assert trained.returncode == 0, trained.stderr
    checkpoint_manifest = json.loads((tmp_path / "checkpoint.manifest.json").read_text())
    assert checkpoint_manifest["train_only"]
    assert not checkpoint_manifest["validation_ground_truth_access"]
    assert checkpoint_manifest["model"]["untouched_dev_fold"] == 0
    with np.load(tmp_path / "checkpoint.npz", allow_pickle=False) as checkpoint:
        assert checkpoint["feature_names"].shape == (len(FEATURE_NAMES),)
        assert checkpoint["activation_authorized"].item()
        assert checkpoint["detector_blend"].item() == pytest.approx(0.4)
        assert checkpoint["monotonic_probability_projection"].item()

    checkpoint_before = (tmp_path / "checkpoint.npz").read_bytes()
    repeated = subprocess.run(trained.args, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "refusing to start with an existing checkpoint/manifest" in repeated.stderr
    assert (tmp_path / "checkpoint.npz").read_bytes() == checkpoint_before
