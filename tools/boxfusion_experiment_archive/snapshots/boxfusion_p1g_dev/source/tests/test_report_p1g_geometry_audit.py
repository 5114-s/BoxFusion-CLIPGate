"""Contracts for the read-only P1G geometry audit of P1S artifacts."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.report_p1_residual_recall import center_size_to_corners
from tools.report_p1g_geometry_audit import (
    MATCHING_CONTRACT,
    P1_DIAGNOSTIC_SCHEMA,
    RECALL_SCHEMA,
    SCHEMA,
    build_report,
    main,
    world_aabb_representation_ceiling,
    world_aabb_target_roundtrip_iou,
)


def _center_size(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> np.ndarray:
    return np.asarray((*center, *size), dtype=np.float64)


def _minmax(box: np.ndarray) -> np.ndarray:
    value = np.asarray(box, dtype=np.float64)
    return np.concatenate(
        (value[:3] - 0.5 * value[3:], value[:3] + 0.5 * value[3:])
    )


def _write_diagnostic(
    path: Path,
    scene: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    mutation_enabled: bool = False,
) -> None:
    count = len(boxes)
    np.savez_compressed(
        path,
        scene_id=np.asarray(scene),
        p1_schema=np.asarray(P1_DIAGNOSTIC_SCHEMA),
        p1_stage=np.asarray("P1S"),
        p1_profile=np.asarray("p1s_native_sparse_context_observer"),
        p1_enabled=np.asarray(True, dtype=bool),
        p1_observer_only=np.asarray(True, dtype=bool),
        p1_uses_ground_truth=np.asarray(False, dtype=bool),
        p1_reads_semantic_labels=np.asarray(False, dtype=bool),
        p1_mutation_enabled=np.asarray(mutation_enabled, dtype=bool),
        p1_applied_count=np.asarray(0, dtype=np.int64),
        p1_complete=np.asarray(True, dtype=bool),
        p1_class_agnostic=np.asarray(True, dtype=bool),
        p1_regression_dim=np.asarray(6, dtype=np.int64),
        p1_head_architecture=np.asarray("native_sparse_context_v1"),
        p1_target_assignment_scope=np.asarray("snapshot_inside_only"),
        p1_step_failed=np.asarray([False], dtype=bool),
        p1_candidate_boxes=np.asarray(boxes, dtype=np.float32),
        p1_candidate_corners=np.asarray(
            center_size_to_corners(boxes), dtype=np.float32
        ),
        p1_candidate_scores=np.asarray(scores, dtype=np.float32),
        p1_candidate_objectness=np.asarray(scores, dtype=np.float32),
        p1_candidate_ids=np.asarray(
            [f"{scene}:candidate:{index}" for index in range(count)],
            dtype=np.str_,
        ),
        p1_candidate_frame_ids=np.arange(count, dtype=np.int64),
    )


def _recall_payload(scene: str) -> dict:
    threshold = {
        "ground_truth_count": 2,
        "b6_true_positives": 1,
        "p1_true_positives": 1,
        "union_true_positives": 1,
    }
    return {
        "schema": RECALL_SCHEMA,
        "stage": "P1S",
        "matching_contract": MATCHING_CONTRACT,
        "observer_only": True,
        "unsafe_scenes": [],
        "scene_count": 1,
        "ground_truth_count": 2,
        "baseline_prediction_count": 1,
        "p1_candidate_count": 3,
        "p1": {"candidate_count": 3},
        "thresholds": {"0.50": dict(threshold)},
        "per_scene": {
            scene: {
                "ground_truth_count": 2,
                "baseline_predictions": 1,
                "p1_candidates": 3,
                "p1_mutation_enabled": False,
                "p1_applied_count": 0,
                "thresholds": {"0.50": dict(threshold)},
            }
        },
    }


def _write_fixture(
    root: Path, *, mutation_enabled: bool = False
) -> tuple[Path, Path, Path, Path, Path]:
    scene = "scene0001_00"
    diagnostics = root / "diagnostics"
    predictions = root / "predictions"
    gt_root = root / "gt"
    scans = root / "scans"
    for path in (diagnostics, predictions, gt_root, scans / scene):
        path.mkdir(parents=True, exist_ok=True)

    gt0 = _center_size((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    gt1 = _center_size((5.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    np.save(gt_root / f"{scene}_bbox.npy", np.stack((gt0, gt1)))

    baseline_corners = center_size_to_corners(gt0[None])[0]
    with (predictions / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[("object", baseline_corners, 0.95)]], handle)

    candidates = np.stack(
        (
            gt0,
            _center_size((5.0, 0.0, 0.0), (4.2, 2.0, 2.0)),
            _center_size((20.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
    )
    _write_diagnostic(
        diagnostics / f"{scene}_tracks.npz",
        scene,
        candidates,
        np.asarray([0.90, 0.80, 0.10], dtype=np.float32),
        mutation_enabled=mutation_enabled,
    )
    (scans / scene / f"{scene}.txt").write_text(
        "axisAlignment = "
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n",
        encoding="utf-8",
    )
    recall = root / "recall.json"
    recall.write_text(
        json.dumps(_recall_payload(scene), indent=2) + "\n",
        encoding="utf-8",
    )
    return recall, diagnostics, predictions, gt_root, scans


def _bands(distribution: dict) -> dict[str, int]:
    return {row["name"]: row["count"] for row in distribution["bands"]}


def test_build_report_measures_bands_missed_gt_and_oracles(tmp_path):
    recall, diagnostics, predictions, gt_root, scans = _write_fixture(
        tmp_path
    )
    report = build_report(
        recall_report=recall,
        diagnostics_root=diagnostics,
        prediction_root=predictions,
        gt_root=gt_root,
        scans_root=scans,
    )

    assert report["schema"] == SCHEMA
    assert report["observer_only"] is True
    assert report["candidate_count"] == 3
    candidate_bands = _bands(report["candidate_best_iou"])
    assert candidate_bands["0p00_to_0p05"] == 1
    assert candidate_bands["0p475_to_0p49"] == 1
    assert candidate_bands["strict_gt_0p50"] == 1

    missed = report["b6_missed_best_iou"]
    assert missed["count"] == 1
    assert missed["quantiles"]["q50"] == pytest.approx(2.0 / 4.2)
    assert len(missed["records"]) == 1
    assert missed["records"][0]["gt_index"] == 1

    near = report["center_size_oracle"][
        "candidate_best_iou_0p45_to_0p50_inclusive"
    ]
    assert near["count"] == 1
    assert near["center_oracle_iou"]["q50"] == pytest.approx(2.0 / 4.2)
    assert near["size_oracle_iou"]["q50"] == pytest.approx(1.0)
    assert near["center_oracle_strict_gt_0p50_count"] == 0
    assert near["size_oracle_strict_gt_0p50_count"] == 1

    world = report["world_aabb_representation_ceiling"]
    assert world["all_ground_truth"]["quantiles"]["q50"] == pytest.approx(1.0)
    assert world["enclosing_target_roundtrip"]["all_ground_truth"][
        "quantiles"
    ]["q50"] == pytest.approx(1.0)


def test_world_aabb_ceiling_distinguishes_theory_from_target_roundtrip():
    gt = _minmax(
        _center_size((0.0, 0.0, 0.0), (4.0, 1.0, 1.0))
    )[None]
    cosine = math.sqrt(0.5)
    alignment = np.asarray(
        [
            [cosine, -cosine, 0.0, 0.0],
            [cosine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ceiling = world_aabb_representation_ceiling(gt, alignment)
    roundtrip = world_aabb_target_roundtrip_iou(gt, alignment)
    assert ceiling[0] == pytest.approx(0.25)
    assert roundtrip[0] == pytest.approx(0.16)
    assert roundtrip[0] < ceiling[0]


def test_report_rejects_wrong_recall_stage_and_unsafe_diagnostic(tmp_path):
    recall, diagnostics, predictions, gt_root, scans = _write_fixture(
        tmp_path
    )
    payload = json.loads(recall.read_text(encoding="utf-8"))
    payload["stage"] = "P1R"
    recall.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="recall stage"):
        build_report(
            recall_report=recall,
            diagnostics_root=diagnostics,
            prediction_root=predictions,
            gt_root=gt_root,
            scans_root=scans,
        )

    unsafe_root = tmp_path / "unsafe"
    (
        unsafe_recall,
        unsafe_diagnostics,
        unsafe_predictions,
        unsafe_gt,
        unsafe_scans,
    ) = _write_fixture(unsafe_root, mutation_enabled=True)
    with pytest.raises(ValueError, match="unsafe p1_mutation_enabled"):
        build_report(
            recall_report=unsafe_recall,
            diagnostics_root=unsafe_diagnostics,
            prediction_root=unsafe_predictions,
            gt_root=unsafe_gt,
            scans_root=unsafe_scans,
        )


def test_report_rejects_recall_artifact_count_mismatch(tmp_path):
    recall, diagnostics, predictions, gt_root, scans = _write_fixture(
        tmp_path
    )
    payload = json.loads(recall.read_text(encoding="utf-8"))
    payload["p1_candidate_count"] = 4
    recall.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="p1_candidate_count disagrees"):
        build_report(
            recall_report=recall,
            diagnostics_root=diagnostics,
            prediction_root=predictions,
            gt_root=gt_root,
            scans_root=scans,
        )


def test_cli_writes_machine_readable_output(tmp_path, capsys):
    recall, diagnostics, predictions, gt_root, scans = _write_fixture(
        tmp_path
    )
    output = tmp_path / "reports" / "geometry.json"
    status = main(
        [
            "--recall-report",
            str(recall),
            "--diagnostics-root",
            str(diagnostics),
            "--prediction-root",
            str(predictions),
            "--gt-root",
            str(gt_root),
            "--scans-root",
            str(scans),
            "--output",
            str(output),
        ]
    )
    assert status == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["schema"] == SCHEMA
