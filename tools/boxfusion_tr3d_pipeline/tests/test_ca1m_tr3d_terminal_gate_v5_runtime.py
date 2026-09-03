from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.ca1m_tr3d_terminal import sha256_array
from boxfusion.ca1m_tr3d_terminal_gate_v5 import (
    ANCHOR_SCORE_SOURCE,
    COLLECTION_SCHEMA,
    FEATURE_NAMES,
    GateDatasetV5,
    LockedFoldDisabledError,
    ROLE_RECEIPT_SCHEMA,
    ROLE_SPECS,
    build_labeled_dataset_v5,
    evaluate_fold0_reused_dev_v5,
    load_candidate_collection_v5,
    load_gate_policy_v5,
    materialize_geometry_only_v5,
    pending_runtime_preflight_v5,
    run_locked_fold1_once_v5,
    seal_candidate_collection_v5,
    seal_detector_role_receipt_v5,
    seal_gate_oof_result_v5,
    seal_role_candidate_collection_v5,
    sha256_file,
    train_gate_oof_v5,
    write_candidate_evidence_v5,
    write_json_create_only,
)


ROOT = Path(__file__).resolve().parents[1]
PENDING_CONFIG = ROOT / "config/ca1m_tr3d_exploratory_gate_xfit_r2_v5_pending.json"


def _box(center: tuple[float, float, float], size: float = 1.0) -> np.ndarray:
    center_array = np.asarray(center, np.float32)
    return np.asarray([
        center_array + np.asarray((x, y, z), np.float32) * (size * 0.5)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ], np.float32)


def _seal_json(path: Path, payload: dict) -> dict:
    output = write_json_create_only(path, payload, path.name)
    return {"path": str(output), "sha256": sha256_file(output), "schema": payload["schema"]}


@pytest.fixture(scope="module")
def candidate_collection(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("v5_candidates")
    continuation_payload = {
        "schema": "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_continuation_receipt.v1",
        "complete": True,
        "fold1_access": False,
        "official_validation_access": False,
        "checkpoint_selection": False,
        "pass": True,
        "continue_inner_training_authorized": True,
        "authorized_inner_roles": ["inner_holdout2", "inner_holdout3", "inner_holdout4"],
        "continuation_gate": {
            "pass": True,
            "continue_inner_training_authorized": True,
            "authorized_inner_roles": ["inner_holdout2", "inner_holdout3", "inner_holdout4"],
        },
    }
    continuation = _seal_json(root / "outer_continuation.json", continuation_payload)
    b6_path = root / "ca1m_native_b6_all_fold_oof_v2.npz"
    b6_path.write_bytes(b"synthetic-b6-oof-only")
    b6_path.chmod(0o444)
    b6 = {
        "path": str(b6_path),
        "sha256": sha256_file(b6_path),
        "schema": "boxfusion.ca1m_native_b6_oof_row_scores.v2",
        "score_source": ANCHOR_SCORE_SOURCE,
        "each_row_model_excludes_scene": True,
        "deploy_scores": False,
    }
    role_collections: dict[str, dict] = {}
    scene_counter = 42000000
    for role, (train_folds, heldout) in ROLE_SPECS.items():
        checkpoint = root / f"{role}_iter_11268.pth"
        checkpoint.write_bytes(f"synthetic-{role}-checkpoint".encode())
        checkpoint.chmod(0o444)
        source_payload = {
            "schema": "boxfusion.synthetic_xfit_training_receipt.v1",
            "complete": True,
            "create_only": True,
            "status": "success",
            "role": role,
            "checkpoint_selection": False,
            "checkpoint": {
                "path": str(checkpoint), "sha256": sha256_file(checkpoint),
                "optimizer_updates": 11268, "checkpoint_selection": False,
            },
            "training_protocol": {
                "train_folds": list(train_folds), "heldout_fold": heldout,
                "initialization": "random_scratch_ca_only",
                "scannet_checkpoint_or_module_access": False,
            },
            "access": {
                "fold1_metadata_or_ground_truth_access": False,
                "official_validation_access": False,
                "scannet_checkpoint_or_module_access": False,
            },
        }
        source = _seal_json(root / f"{role}_source.json", source_payload)
        normalized_path = root / f"{role}_normalized.json"
        seal_detector_role_receipt_v5(
            normalized_path, role=role, source_training_receipt=source,
            outer_continuation_receipt=continuation,
        )
        normalized = {
            "path": str(normalized_path), "sha256": sha256_file(normalized_path),
            "schema": ROLE_RECEIPT_SCHEMA,
        }
        scenes = [str(scene_counter + index) for index in range(20)]
        scene_counter += 20
        evidence_paths: dict[str, Path] = {}
        for index, scene in enumerate(scenes):
            anchor = np.asarray([_box((float(index) * 2.0, 0.0, 0.0))], np.float32)
            candidate = anchor.copy()
            features = np.zeros((1, len(FEATURE_NAMES)), np.float32)
            features[0, 1] = index / 20.0
            evidence_path = root / f"{scene}_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"
            write_candidate_evidence_v5(
                evidence_path, scene_id=scene, fold_id=heldout, producer_role=role,
                producer_checkpoint_sha256=sha256_file(checkpoint),
                training_receipt_sha256=sha256_file(normalized_path),
                outer_continuation_receipt_sha256=continuation["sha256"],
                b6_oof_sidecar_sha256=b6["sha256"], candidate_corners=candidate,
                candidate_rows=np.asarray((0,), np.int64),
                candidate_scores=np.asarray((0.5,), np.float32),
                anchor_indices=np.asarray((0,), np.int64), features=features,
                anchor_corners=anchor, anchor_scores=np.asarray((0.4,), np.float32),
            )
            evidence_paths[scene] = evidence_path
        role_path = root / f"{role}_candidate_collection.json"
        seal_role_candidate_collection_v5(
            role_path, role=role, expected_scenes=scenes, role_receipt=normalized,
            evidence_paths=evidence_paths, b6_oof_sidecar=b6,
        )
        role_collections[role] = {
            "path": str(role_path), "sha256": sha256_file(role_path),
            "schema": "boxfusion.ca1m_tr3d_xfit_r2_candidate_role_collection.v1",
        }
    combined = root / "candidate_collection_exact60_plus_outer20.json"
    seal_candidate_collection_v5(combined, role_collections=role_collections)
    return combined


def test_v5_candidate_manifest_is_exact60_plus_outer20_and_double_oof(candidate_collection: Path):
    loaded = load_candidate_collection_v5(candidate_collection)
    assert loaded.payload["schema"] == COLLECTION_SCHEMA
    assert loaded.payload["fold_counts"] == {"0": 20, "2": 20, "3": 20, "4": 20}
    assert loaded.payload["fit_scene_count"] == 60
    assert loaded.payload["outer_scene_count"] == 20
    for row in loaded.scenes.values():
        assert row["fold_id"] not in row["producer_train_folds"]
        assert "v4" not in Path(row["path"]).name


def test_v5_pending_runtime_preflight_stays_fail_closed():
    report = pending_runtime_preflight_v5(PENDING_CONFIG)
    assert report["runtime_surface_implemented"] is True
    assert report["runtime_ready"] is False
    assert report["candidate_or_gt_artifact_opened"] is False
    assert report["output_created"] is False


def test_v5_fit_dataset_opens_only_fold234_gt_after_collection_seal(candidate_collection: Path):
    loaded = load_candidate_collection_v5(candidate_collection)
    opened: list[str] = []

    def gt(scene: str) -> np.ndarray:
        opened.append(scene)
        row = loaded.scenes[scene]
        with np.load(row["path"], allow_pickle=False) as archive:
            return np.asarray(archive["anchor_corners"], np.float64)

    dataset = build_labeled_dataset_v5(
        candidate_collection, purpose="fold234_oof_fit", ground_truth_loader=gt
    )
    assert len(opened) == 60
    assert set(dataset.scene_folds.tolist()) == {2, 3, 4}
    assert not any(int(loaded.scenes[scene]["fold_id"]) == 0 for scene in opened)
    assert dataset.strict_iou50_target.all()


def _synthetic_dataset(tmp_path: Path, purpose: str, folds: tuple[int, ...]) -> GateDatasetV5:
    scene_order: list[str] = []
    scene_folds: list[int] = []
    for fold in folds:
        for index in range(20):
            scene_order.append(str(50000000 + fold * 100 + index))
            scene_folds.append(fold)
    scene_count = len(scene_order)
    anchors = np.asarray([_box((float(index) * 2.0, 0.0, 0.0)) for index in range(scene_count)], np.float32)
    candidates = np.repeat(anchors, 2, axis=0)
    features = np.zeros((scene_count * 2, len(FEATURE_NAMES)), np.float32)
    features[0::2, 1] = 1.0
    features[1::2, 1] = -1.0
    features[:, 2] = np.repeat(np.linspace(-0.5, 0.5, scene_count), 2)
    candidate_iou = np.tile(np.asarray((0.80, 0.20), np.float64), scene_count)
    same_gt_gain = np.tile(np.asarray((0.40, -0.20), np.float64), scene_count)
    collection = tmp_path / f"{purpose}_source_collection.json"
    collection.write_text("{}")
    collection.chmod(0o444)
    return GateDatasetV5(
        purpose=purpose,
        source_collection_path=collection,
        source_collection_sha256=sha256_file(collection),
        scene_order=np.asarray(scene_order),
        scene_folds=np.asarray(scene_folds, np.int64),
        scene_gt_counts=np.ones(scene_count, np.int64),
        anchor_scene_ids=np.asarray(scene_order),
        anchor_fold_ids=np.asarray(scene_folds, np.int64),
        anchor_local_indices=np.zeros(scene_count, np.int64),
        anchor_corners=anchors,
        anchor_scores_oof=np.linspace(0.9, 0.1, scene_count, dtype=np.float32),
        anchor_best_iou=np.full(scene_count, 0.40, np.float64),
        anchor_best_gt=np.zeros(scene_count, np.int64),
        candidate_scene_ids=np.repeat(np.asarray(scene_order), 2),
        candidate_fold_ids=np.repeat(np.asarray(scene_folds, np.int64), 2),
        candidate_rows=np.tile(np.asarray((0, 1), np.int64), scene_count),
        candidate_anchor_positions=np.repeat(np.arange(scene_count, dtype=np.int64), 2),
        candidate_corners=candidates,
        candidate_raw_scores=np.tile(np.asarray((0.6, 0.5), np.float32), scene_count),
        features=features,
        candidate_max_gt_iou=candidate_iou,
        candidate_best_gt=np.zeros(scene_count * 2, np.int64),
        candidate_iou_on_anchor_gt=candidate_iou.copy(),
        same_gt_gain=same_gt_gain,
        target_switch=np.zeros(scene_count * 2, np.bool_),
        strict_iou50_target=candidate_iou > 0.50,
    )


def test_v5_three_head_gate_uses_fold234_oof_thresholds_and_frozen_fold0(tmp_path: Path):
    fit = _synthetic_dataset(tmp_path, "fold234_oof_fit", (2, 3, 4))
    result = train_gate_oof_v5(fit)
    assert result.threshold_receipt["searched_operating_point_count"] == 216
    assert result.threshold_receipt["safety_gate_passed"] is True
    for fold, model in result.fold_models.items():
        assert fold not in model.train_folds
        rows = fit.candidate_fold_ids == fold
        observed = set(result.predictions.scoring_train_fold_json[rows].tolist())
        assert observed == {json.dumps(list(model.train_folds), separators=(",", ":"))}
    oof, threshold, policy = seal_gate_oof_result_v5(
        fit, result, oof_path=tmp_path / "oof.npz",
        threshold_path=tmp_path / "threshold.json", policy_path=tmp_path / "policy.json",
    )
    assert not (oof.stat().st_mode & 0o222)
    assert not (threshold.stat().st_mode & 0o222)
    _, policy_payload, model = load_gate_policy_v5(policy)
    assert model.train_folds == (2, 3, 4)
    assert policy_payload["deployable"] is False

    fold0 = _synthetic_dataset(tmp_path, "fold0_reused_dev", (0,))
    report = evaluate_fold0_reused_dev_v5(
        fold0, policy_path=policy, output_path=tmp_path / "fold0_report.json"
    )
    report_payload = json.loads(report.read_text())
    assert report_payload["thresholds_frozen_before_fold0"] is True
    assert report_payload["fold0_retuning"] is False
    assert report_payload["policy_activation_authorized"] is False


def test_v5_materializer_preserves_scores_order_and_count():
    anchors = np.asarray([_box((0.0, 0.0, 0.0)), _box((3.0, 0.0, 0.0))], np.float32)
    candidates = np.asarray([_box((0.1, 0.0, 0.0))], np.float32)
    scores = np.asarray((0.7, 0.4), np.float32)
    result = materialize_geometry_only_v5(
        anchor_corners=anchors, anchor_scores=scores, candidate_corners=candidates,
        anchor_indices=np.asarray((0,), np.int64), candidate_rows=np.asarray((0,), np.int64),
    )
    assert np.array_equal(result.scores, scores)
    assert np.array_equal(result.row_indices, np.arange(2))
    assert len(result.corners) == len(anchors)
    assert np.array_equal(result.corners[0], candidates[0])


def test_v5_locked_fold1_default_fails_before_loader_or_path_access(tmp_path: Path):
    called = False

    def loader() -> GateDatasetV5:
        nonlocal called
        called = True
        raise AssertionError("locked loader must not run")

    with pytest.raises(LockedFoldDisabledError):
        run_locked_fold1_once_v5(
            dataset_loader=loader,
            authorization_path=tmp_path / "does_not_exist_authorization.json",
            policy_path=tmp_path / "does_not_exist_policy.json",
            report_path=tmp_path / "fold1_report.json",
            consumption_receipt_path=tmp_path / "fold1_consumption.json",
        )
    assert called is False
    assert not (tmp_path / "fold1_report.json").exists()
    assert not (tmp_path / "fold1_consumption.json").exists()


def test_v5_rejects_in_sample_evidence_before_output(tmp_path: Path):
    target = tmp_path / "42009999_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"
    with pytest.raises(ValueError, match="in-sample"):
        write_candidate_evidence_v5(
            target, scene_id="42009999", fold_id=3, producer_role="inner_holdout2",
            producer_checkpoint_sha256="1" * 64, training_receipt_sha256="2" * 64,
            outer_continuation_receipt_sha256="3" * 64,
            b6_oof_sidecar_sha256="4" * 64,
            candidate_corners=np.asarray([_box((0.0, 0.0, 0.0))], np.float32),
            candidate_rows=np.asarray((0,), np.int64),
            candidate_scores=np.asarray((0.5,), np.float32),
            anchor_indices=np.asarray((0,), np.int64),
            features=np.zeros((1, len(FEATURE_NAMES)), np.float32),
            anchor_corners=np.asarray([_box((0.0, 0.0, 0.0))], np.float32),
            anchor_scores=np.asarray((0.4,), np.float32),
        )
    assert not target.exists()


def test_v5_failed_outer_receipt_blocks_before_source_or_candidate_access(tmp_path: Path):
    failed = {
        "schema": "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_continuation_receipt.v1",
        "complete": True,
        "fold1_access": False,
        "official_validation_access": False,
        "checkpoint_selection": False,
        "pass": False,
        "continue_inner_training_authorized": False,
        "authorized_inner_roles": [],
        "continuation_gate": {
            "pass": False,
            "continue_inner_training_authorized": False,
            "authorized_inner_roles": [],
        },
    }
    failed_record = _seal_json(tmp_path / "failed_outer.json", failed)
    output = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="not a passing"):
        seal_detector_role_receipt_v5(
            output, role="inner_holdout2",
            source_training_receipt={
                "path": str(tmp_path / "source_must_not_be_opened.json"),
                "sha256": "0" * 64,
                "schema": "boxfusion.synthetic_xfit_training_receipt.v1",
            },
            outer_continuation_receipt=failed_record,
        )
    assert not output.exists()
