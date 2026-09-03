"""CPU tests for the P0-repeat/P1 nondeterminism audit."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.audit_p1_nondeterminism import (
    audit_nondeterminism,
    audit_p1_diagnostics,
    compare_manifests,
    compare_predictions,
    main,
    pairwise_aabb_iou,
    parse_eval_log,
)


SCENE = "scene0001_00"
CHECKPOINT_SHA = "1" * 64
B6_SHA = "2" * 64


def _cube(center=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0)):
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(extent, dtype=np.float64) / 2.0
    minimum = center - half
    maximum = center + half
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float32,
    )


def _prediction(path: Path, boxes, scores, labels=None) -> None:
    corners = np.asarray(boxes, dtype=np.float32).reshape((-1, 8, 3))
    labels = [0] * corners.shape[0] if labels is None else list(labels)
    with path.open("wb") as handle:
        pickle.dump(
            [[
                (label, corner, np.float32(score))
                for label, corner, score in zip(
                    labels, corners, scores, strict=True
                )
            ]],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _loaded(boxes, scores, labels=None, digest="a" * 64):
    corners = np.asarray(boxes, dtype=np.float64).reshape((-1, 8, 3))
    return {
        "labels": [0] * len(corners) if labels is None else labels,
        "corners": corners,
        "scores": np.asarray(scores, dtype=np.float64),
        "sha256": digest,
    }


def _eval_log(path: Path, maps=(0.4, 0.3, 0.1)) -> None:
    lines = []
    for value in maps:
        lines.extend(
            [
                f"eval mAP: {value:.6f}",
                f"eval APrec: {value + 0.1:.6f}",
                f"eval ARecall: {value + 0.2:.6f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(
    path: Path,
    *,
    stage: str,
    prediction_root: Path,
    diagnostics_root: Path,
    log_root: Path,
) -> None:
    p1 = stage == "P1"
    payload = {
        "schema": "boxfusion.p_ablation.run_manifest.v1",
        "stage": stage,
        "profile": (
            "p1_residual_proposal_observer" if p1 else "p0_frozen_b6"
        ),
        "scene_count": 1,
        "scene_list_sha256": hashlib.sha256(
            f"{SCENE}\n".encode("utf-8")
        ).hexdigest(),
        "config_sha256": "4" * 64,
        "b6_checkpoint_sha256": B6_SHA,
        "code_tree_sha256": "5" * 64,
        "parameters": {
            "minimum_extent": 0.4,
            "proposal_interval": 5,
        },
        "prediction_root": str(prediction_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        "log_root": str(log_root.resolve()),
        "evaluation_root": str((log_root / "evaluation").resolve()),
        "p1_checkpoint": "/trusted/train-only.pt" if p1 else None,
        "p1_checkpoint_sha256": CHECKPOINT_SHA if p1 else None,
        "p1_training_provenance": (
            {
                "b6_checkpoint_sha256": B6_SHA,
                "forbidden_overlap": [],
                "train_scene_count": 10,
            }
            if p1
            else None
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _diagnostic(path: Path, *, mutation=False) -> None:
    np.savez_compressed(
        path,
        scene_id=np.asarray(SCENE),
        p1_schema=np.asarray("boxfusion.p1.residual_proposal_observer.v1"),
        p1_stage=np.asarray("P1"),
        p1_profile=np.asarray("p1_residual_proposal_observer"),
        p1_enabled=np.asarray(True, dtype=bool),
        p1_observer_only=np.asarray(True, dtype=bool),
        p1_uses_ground_truth=np.asarray(False, dtype=bool),
        p1_mutation_enabled=np.asarray(mutation, dtype=bool),
        p1_applied_count=np.asarray(int(mutation), dtype=np.int64),
        p1_complete=np.asarray(True, dtype=bool),
        p1_class_agnostic=np.asarray(True, dtype=bool),
        p1_regression_dim=np.asarray(6, dtype=np.int64),
        p1_checkpoint_sha256=np.asarray(CHECKPOINT_SHA),
        p1_feature_names=np.asarray(["occupancy"], dtype=np.str_),
        p1_voxel_features=np.asarray([[1.0]], dtype=np.float32),
        p1_voxel_centers=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        p1_voxel_offsets=np.asarray([0, 1], dtype=np.int64),
        p1_candidate_boxes=np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.float32),
        p1_candidate_scores=np.asarray([0.5], dtype=np.float32),
        p1_candidate_applied=np.asarray([mutation], dtype=bool),
    )


def _fixture(tmp_path: Path):
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(SCENE + "\n", encoding="utf-8")
    roots = {}
    for name in ("p0", "repeat", "p1"):
        prediction_root = tmp_path / name / "predictions"
        diagnostics_root = tmp_path / name / "diagnostics"
        log_root = tmp_path / name / "logs"
        prediction_root.mkdir(parents=True)
        diagnostics_root.mkdir()
        log_root.mkdir()
        roots[name] = (prediction_root, diagnostics_root, log_root)
    baseline = _cube()
    _prediction(
        roots["p0"][0] / f"{SCENE}_boxes.pkl",
        [baseline],
        [0.8],
    )
    _prediction(
        roots["repeat"][0] / f"{SCENE}_boxes.pkl",
        [baseline + 0.01, _cube(center=(3, 0, 0))],
        [0.81, 0.2],
    )
    _prediction(
        roots["p1"][0] / f"{SCENE}_boxes.pkl",
        [baseline + 0.005],
        [0.805],
    )
    _diagnostic(roots["p1"][1] / f"{SCENE}_tracks.npz")
    for name, stage in (("p0", "P0"), ("repeat", "P0"), ("p1", "P1")):
        prediction_root, diagnostics_root, log_root = roots[name]
        _manifest(
            log_root / "run_manifest.json",
            stage=stage,
            prediction_root=prediction_root,
            diagnostics_root=diagnostics_root,
            log_root=log_root,
        )
        _eval_log(log_root / "eval_stdout.log")
    return scene_list, roots


def test_pairwise_aabb_iou_known_values():
    left = np.asarray([[0, 0, 0, 1, 1, 1]], dtype=float)
    right = np.asarray(
        [[0, 0, 0, 1, 1, 1], [0.5, 0, 0, 1.5, 1, 1]], dtype=float
    )
    result = pairwise_aabb_iou(left, right)
    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(1.0 / 3.0)


def test_hungarian_matching_separates_extra_from_numeric_drift():
    baseline = _loaded([_cube(), _cube(center=(3, 0, 0))], [0.9, 0.8])
    candidate = _loaded(
        [
            _cube(center=(9, 0, 0)),
            _cube() + 0.01,
            _cube(center=(3, 0, 0)),
        ],
        [0.1, 0.89, 0.8],
        digest="b" * 64,
    )
    report = compare_predictions(baseline, candidate, match_iou=0.25)
    assert report["structure"]["matched_count"] == 2
    assert report["structure"]["candidate_extra_indices"] == [0]
    assert report["structure"]["baseline_missing_count"] == 0
    assert report["numeric"]["changed_box_count"] == 1
    assert report["numeric"]["corner_abs"]["max"] == pytest.approx(0.01)


def test_eval_parser_requires_ordered_complete_triplets(tmp_path):
    log = tmp_path / "eval.log"
    _eval_log(log)
    parsed = parse_eval_log(log, (0.15, 0.25, 0.50))
    assert parsed["0.15"]["mAP"] == pytest.approx(0.4)
    log.write_text("eval mAP: 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 9 ordered"):
        parse_eval_log(log, (0.15, 0.25, 0.50))


def test_diagnostic_audit_fails_closed_on_mutation(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    _diagnostic(diagnostics / f"{SCENE}_tracks.npz", mutation=True)
    report = audit_p1_diagnostics(
        scenes=[SCENE],
        diagnostics_root=diagnostics,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert not report["ok"]
    assert any("p1_mutation_enabled" in issue for issue in report["issues"])


def test_manifest_comparison_detects_frozen_parameter_change(tmp_path):
    scene_list, roots = _fixture(tmp_path)
    del scene_list
    p1_manifest = roots["p1"][2] / "run_manifest.json"
    payload = json.loads(p1_manifest.read_text(encoding="utf-8"))
    payload["parameters"]["minimum_extent"] = 0.3
    p1_manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = compare_manifests(
        p0_manifest_path=roots["p0"][2] / "run_manifest.json",
        repeat_manifest_paths=[
            roots["repeat"][2] / "run_manifest.json"
        ],
        p1_manifest_path=p1_manifest,
        p0_root=roots["p0"][0],
        repeat_roots=[roots["repeat"][0]],
        p1_root=roots["p1"][0],
        p1_diagnostics_root=roots["p1"][1],
    )
    assert not report["ok"]
    assert any("minimum_extent" in issue for issue in report["issues"])


def test_full_audit_reports_metric_identity_inside_repeat_envelope(tmp_path):
    scene_list, roots = _fixture(tmp_path)
    report = audit_nondeterminism(
        scene_list=scene_list,
        p0_root=roots["p0"][0],
        p0_manifest=roots["p0"][2] / "run_manifest.json",
        p0_eval_log=roots["p0"][2] / "eval_stdout.log",
        p0_repeats=[
            (
                roots["repeat"][0],
                roots["repeat"][2] / "run_manifest.json",
                roots["repeat"][2] / "eval_stdout.log",
            )
        ],
        p1_root=roots["p1"][0],
        p1_diagnostics_root=roots["p1"][1],
        p1_manifest=roots["p1"][2] / "run_manifest.json",
        p1_eval_log=roots["p1"][2] / "eval_stdout.log",
    )
    assert report["ok"]
    assert (
        report["verdict"]
        == "metric_identical_consistent_with_observed_p0_drift"
    )
    assert report["conclusions"]["structure_identical"]
    assert report["conclusions"]["metric_identical"]
    assert not report["conclusions"]["bit_exact"]
    assert report["conclusions"]["within_observed_aggregate_p0_drift"]
    assert report["evidence_limits"]["single_repeat_warning"]
    # The report must remain strict JSON (no NaN/Infinity).
    json.dumps(report, allow_nan=False)


def test_cli_requires_explicit_pickle_trust(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--scene-list",
                str(tmp_path / "missing.txt"),
                "--p0-root",
                str(tmp_path),
                "--p0-manifest",
                str(tmp_path / "p0.json"),
                "--p0-eval-log",
                str(tmp_path / "p0.log"),
                "--p0-repeat",
                str(tmp_path),
                str(tmp_path / "repeat.json"),
                str(tmp_path / "repeat.log"),
                "--p1-root",
                str(tmp_path),
                "--p1-diagnostics-root",
                str(tmp_path),
                "--p1-manifest",
                str(tmp_path / "p1.json"),
                "--p1-eval-log",
                str(tmp_path / "p1.log"),
            ]
        )
    assert error.value.code == 2


def test_sha_fixture_is_well_formed():
    # Guard against accidentally weakening the synthetic manifest setup.
    assert len(CHECKPOINT_SHA) == hashlib.sha256().digest_size * 2
