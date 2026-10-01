from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from apply_ca1m_native_b6_counterfactual import run  # noqa: E402
from boxfusion.ca1m_native_b6_observer import FEATURE_NAMES, SCHEMA as OBSERVER_SCHEMA  # noqa: E402
from boxfusion.ca1m_native_b6_score import (  # noqa: E402
    CA1MNativeB6ScoreConfig,
    CA1MNativeB6ScoreHook,
    CHECKPOINT_MANIFEST_SCHEMA,
    CHECKPOINT_SCHEMA,
    load_ca1m_native_b6_scorer,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box() -> np.ndarray:
    return np.arange(24, dtype=np.float32).reshape(8, 3)


def _checkpoint(tmp_path: Path, *, authorized: bool) -> tuple[Path, Path]:
    checkpoint = tmp_path / "native14.npz"
    weight = np.zeros((len(FEATURE_NAMES), 4), dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)
    np.savez_compressed(
        checkpoint,
        schema=np.asarray(CHECKPOINT_SCHEMA),
        complete=np.asarray(True),
        train_only=np.asarray(True),
        validation_ground_truth_access=np.asarray(False),
        activation_authorized=np.asarray(authorized),
        feature_names=np.asarray(FEATURE_NAMES),
        output_names=np.asarray(
            ("predicted_iou", "prob_iou_015", "prob_iou_025", "prob_iou_050")
        ),
        iou_thresholds=np.asarray((0.15, 0.25, 0.50), dtype=np.float32),
        ranking_weights=np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float32),
        detector_blend=np.asarray(0.4, dtype=np.float32),
        preserve_original_floor=np.asarray(False),
        monotonic_probability_projection=np.asarray(True),
        strict_iou_thresholds=np.asarray(True),
        feature_mean=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        feature_scale=np.ones(len(FEATURE_NAMES), dtype=np.float32),
        num_layers=np.asarray(1, dtype=np.int64),
        training_folds=np.asarray((1, 2, 3, 4), dtype=np.int8),
        heldout_dev_fold=np.asarray(0, dtype=np.int8),
        weight_0=weight,
        bias_0=bias,
    )
    manifest = tmp_path / "native14.manifest.json"
    manifest.write_text(json.dumps({
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "activation_authorized": authorized,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint)},
    }))
    return checkpoint, manifest


def _diagnostic(path: Path, scene: str, corners: np.ndarray, scores: np.ndarray) -> None:
    features = np.full((len(scores), len(FEATURE_NAMES)), 0.25, dtype=np.float32)
    features[:, 0] = scores
    np.savez_compressed(
        path,
        schema=np.asarray(OBSERVER_SCHEMA), complete=np.asarray(True),
        observer_only=np.asarray(True), mutation_enabled=np.asarray(False),
        applied_count=np.asarray(0, dtype=np.int64),
        ground_truth_access=np.asarray(False), clip_access=np.asarray(False),
        scene_id=np.asarray(scene), result_indices=np.arange(len(scores), dtype=np.int64),
        corners=corners, scores=scores, feature_names=np.asarray(FEATURE_NAMES),
        features=features, valid_evidence=np.ones(len(scores), dtype=np.bool_),
    )


def _tree(tmp_path: Path, *, authorized: bool = True) -> argparse.Namespace:
    scene = "42000001"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n")
    anchor = tmp_path / "anchor"
    observer = tmp_path / "observer"
    diagnostics = tmp_path / "diagnostics"
    for root in (anchor, observer, diagnostics):
        root.mkdir()
    corners = np.stack((_box(), _box() + np.float32(10)))
    scores = np.asarray((0.8, 0.2), dtype=np.float32)
    payload = [[(0, corners[index], float(scores[index])) for index in range(2)]]
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    (anchor / f"{scene}_boxes.pkl").write_bytes(raw)
    (observer / f"{scene}_boxes.pkl").write_bytes(raw)
    _diagnostic(diagnostics / f"{scene}_ca1m_native_b6.npz", scene, corners, scores)
    checkpoint, manifest = _checkpoint(tmp_path, authorized=authorized)
    return argparse.Namespace(
        mode="preflight", scene_list=scene_list, anchor_root=anchor,
        observer_root=observer, diagnostics_root=diagnostics,
        checkpoint=checkpoint, checkpoint_manifest=manifest,
        prediction_output_root=None, output=None,
    )


def test_observer_hook_is_exact_noop_without_checkpoint(tmp_path: Path) -> None:
    hook = CA1MNativeB6ScoreHook(CA1MNativeB6ScoreConfig())
    corners = np.stack((_box(),))
    scores = np.asarray((0.7,), dtype=np.float32)
    result = hook.apply(
        scene_id="42000001", corners=corners, scores=scores,
        observer_diagnostic=tmp_path / "absent.npz",
    )
    assert np.array_equal(result.corners, corners)
    assert np.array_equal(result.scores, scores)
    assert np.array_equal(result.source_indices, np.arange(1))
    assert result.applied_count == 0 and result.mode == "observer"


def test_active_loader_refuses_unauthorized_checkpoint(tmp_path: Path) -> None:
    checkpoint, manifest = _checkpoint(tmp_path, authorized=False)
    with pytest.raises(PermissionError, match="not activation_authorized"):
        load_ca1m_native_b6_scorer(
            checkpoint, manifest, require_activation_authorized=True
        )


def test_active_hook_changes_only_scores_and_is_create_only(tmp_path: Path) -> None:
    args = _tree(tmp_path)
    hook = CA1MNativeB6ScoreHook(CA1MNativeB6ScoreConfig(
        mode="active", checkpoint=str(args.checkpoint),
        checkpoint_manifest=str(args.checkpoint_manifest),
        diagnostics_root=str(tmp_path / "score_diag"),
    ))
    rows = pickle.loads((args.anchor_root / "42000001_boxes.pkl").read_bytes())[0]
    corners = np.stack([row[1] for row in rows])
    scores = np.asarray([row[2] for row in rows], dtype=np.float32)
    result = hook.apply(
        scene_id="42000001", corners=corners, scores=scores,
        observer_diagnostic=args.diagnostics_root / "42000001_ca1m_native_b6.npz",
    )
    assert np.array_equal(result.corners, corners)
    assert np.array_equal(result.source_indices, np.arange(2))
    np.testing.assert_allclose(result.scores, 0.4 * scores + 0.6 * 0.5)
    with pytest.raises(FileExistsError):
        hook.apply(
            scene_id="42000001", corners=corners, scores=scores,
            observer_diagnostic=args.diagnostics_root / "42000001_ca1m_native_b6.npz",
        )


def test_counterfactual_preflight_writes_nothing_and_active_preserves_identity(tmp_path: Path) -> None:
    args = _tree(tmp_path)
    preflight = run(args)
    assert preflight["mode"] == "preflight" and not preflight["evaluation_invoked"]
    assert not (tmp_path / "active").exists()
    args.mode = "active"
    args.prediction_output_root = tmp_path / "active"
    args.output = tmp_path / "report.json"
    report = run(args)
    assert report["activation_authorized"] and report["score_only"]
    assert report["obb_unchanged"] and report["row_count_unchanged"]
    source = pickle.loads((args.anchor_root / "42000001_boxes.pkl").read_bytes())[0]
    target = pickle.loads((args.prediction_output_root / "42000001_boxes.pkl").read_bytes())[0]
    assert len(source) == len(target)
    assert all(np.array_equal(left[1], right[1]) for left, right in zip(source, target))
    assert [row[0] for row in source] == [row[0] for row in target]
    assert [row[2] for row in source] != [row[2] for row in target]


def test_counterfactual_active_refuses_unauthorized_checkpoint(tmp_path: Path) -> None:
    args = _tree(tmp_path, authorized=False)
    args.mode = "active"
    args.prediction_output_root = tmp_path / "active"
    args.output = tmp_path / "report.json"
    with pytest.raises(PermissionError, match="not activation_authorized"):
        run(args)
    assert not args.prediction_output_root.exists()
    assert not args.output.exists()
