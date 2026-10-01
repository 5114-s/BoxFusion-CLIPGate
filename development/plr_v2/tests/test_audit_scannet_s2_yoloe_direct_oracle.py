from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.audit_scannet_s2_yoloe_direct_oracle import (
    S2OracleError,
    _array_content_sha256,
    _row_payload_sha256,
    audit_scannet_s2_yoloe_direct_oracle,
    main,
)


SCENE = "scene0001_00"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corners(center, extent) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    extent = np.asarray(extent, dtype=np.float32)
    lower = center - extent / 2.0
    upper = center + extent / 2.0
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float32,
    )


def _write_pickle(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([list(rows)], handle, protocol=pickle.HIGHEST_PROTOCOL)


def _candidate_public(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "scene_id": SCENE,
        "scene_index": 0,
        "terminal_rank": 0,
        "diagnostic_row": 4,
        "result_index": 8,
        "track_id": -7,
        "box_center_extent": arrays["candidate_box_center_extent"][0].tolist(),
        "corners_world": arrays["candidate_corners_world"][0].tolist(),
        "raw_score_provenance": float(
            arrays["candidate_raw_score_provenance"][0]
        ),
        "stored_appended_score_diagnostic_only": float(
            arrays["candidate_stored_appended_score_diagnostic_only"][0]
        ),
        "formal_evaluation_score": 1.0,
        "max_native_aabb_iou": 0.0,
        "valid_point_count": 128,
    }


def _make_tree(root: Path, *, candidate_matches_gt: bool = True) -> dict[str, Path]:
    baseline_root = root / "baseline"
    output_root = root / "counterfactual"
    gt_root = root / "gt"
    scan_root = root / "scans"
    seal_root = root / "seal"
    for directory in (baseline_root, output_root, gt_root, seal_root):
        directory.mkdir(parents=True, exist_ok=True)
    scene_scan = scan_root / SCENE
    scene_scan.mkdir(parents=True)
    identity = " ".join(str(value) for value in np.eye(4).reshape(-1))
    (scene_scan / f"{SCENE}.txt").write_text(
        f"axisAlignment = {identity}\n", encoding="utf-8"
    )

    extent = np.asarray([0.8, 0.8, 0.8], dtype=np.float32)
    native_center = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    candidate_center = np.asarray(
        [2.0, 0.0, 1.0] if candidate_matches_gt else [20.0, 0.0, 1.0],
        dtype=np.float32,
    )
    second_gt_center = np.asarray([2.0, 0.0, 1.0], dtype=np.float32)
    native_corners = _corners(native_center, extent)
    candidate_corners = _corners(candidate_center, extent)
    native_rows = [(0, native_corners, 0.70)]
    candidate_row = (0, candidate_corners, 0.60)
    native_path = baseline_root / f"{SCENE}_boxes.pkl"
    output_path = output_root / f"{SCENE}_boxes.pkl"
    _write_pickle(native_path, native_rows)
    _write_pickle(output_path, native_rows + [candidate_row])
    np.save(
        gt_root / f"{SCENE}_bbox.npy",
        np.asarray(
            [
                [*native_center, *extent, 1.0],
                [*second_gt_center, *extent, 2.0],
            ],
            dtype=np.float64,
        ),
    )

    arrays = {
        "scene_ids": np.asarray([SCENE], dtype="<U12"),
        "candidate_scene_index": np.asarray([0], dtype=np.int16),
        "candidate_terminal_rank": np.asarray([0], dtype=np.int16),
        "candidate_diagnostic_row": np.asarray([4], dtype=np.int32),
        "candidate_result_index": np.asarray([8], dtype=np.int32),
        "candidate_track_id": np.asarray([-7], dtype=np.int64),
        "candidate_box_center_extent": np.asarray(
            [[*candidate_center, *extent]], dtype=np.float32
        ),
        "candidate_corners_world": candidate_corners.reshape(1, 8, 3),
        "candidate_raw_score_provenance": np.asarray([0.80], dtype=np.float32),
        "candidate_stored_appended_score_diagnostic_only": np.asarray(
            [0.60], dtype=np.float32
        ),
        "candidate_formal_evaluation_score": np.ones(1, dtype=np.float32),
        "candidate_max_native_aabb_iou": np.zeros(1, dtype=np.float32),
        "candidate_valid_point_count": np.asarray([128], dtype=np.int16),
        "counterfactual_scene_offsets": np.asarray([0, 2], dtype=np.int32),
        "counterfactual_corners_world": np.stack(
            [native_corners, candidate_corners]
        ).astype(np.float32),
        "counterfactual_stored_score_provenance": np.asarray(
            [0.70, 0.60], dtype=np.float32
        ),
        "counterfactual_formal_evaluation_score": np.ones(2, dtype=np.float32),
        "counterfactual_is_native_prefix": np.asarray([True, False], dtype=bool),
    }
    npz_path = seal_root / "s2_yoloe_direct_shadow.npz"
    np.savez(npz_path, **arrays)
    native_hash = _sha256(native_path)
    output_hash = _sha256(output_path)
    prefix_hash = _row_payload_sha256(native_rows)
    fake_hash = "a" * 64
    manifest = {
        "schema": "boxfusion.s2_yoloe_direct_shadow.v1",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "native_mutation_applied": False,
        "gt_access": False,
        "oracle_access": False,
        "training_free": True,
        "online_learning": False,
        "past_current_only": True,
        "future_frames_used": False,
        "detector_semantics_used_for_gate": False,
        "native_clip_access": False,
        "native_clip_unchanged": True,
        "coordinate_frame": "scannet_world",
        "score_mode_for_formal_evaluation": "constant_1.0",
        "stored_scores_are_diagnostic_only": True,
        "scene_count": 1,
        "scene_order": [SCENE],
        "candidate_count": 1,
        "npz_file": npz_path.name,
        "npz_sha256": _sha256(npz_path),
        "candidate_content_sha256": _array_content_sha256(arrays),
        "counterfactual_prediction_root": str(output_root),
        "counterfactual_prediction_sha256": {SCENE: output_hash},
        "input": {
            "candidate_root": str(root / "diagnostics"),
            "baseline_root": str(baseline_root),
            "preregistration": str(root / "prereg.md"),
            "preregistration_expected_sha256": fake_hash,
            "preregistration_sha256": fake_hash,
            "materializer_source": str(root / "materializer.py"),
            "materializer_source_sha256": fake_hash,
            "frozen_inputs": {},
        },
        "frozen_policy": {
            "candidate_source_index": -1,
            "diagnostic_order_preserved": True,
            "native_novelty_aabb_iou_strict_less_than": 0.10,
            "candidate_self_nms_aabb_iou_strict_less_than": 0.25,
            "maximum_appended_candidates_per_scene": 6,
            "terminal_candidate_labels_ignored": True,
            "terminal_clip_access": False,
            "native_prefix_rows_exact": True,
            "formal_evaluation_score": 1.0,
        },
        "input_hash_identity": {
            "candidate_diagnostics_before_after": True,
            "native_predictions_before_after": True,
            "frozen_sources_before_after": True,
            "preregistration_before_after": True,
            "materializer_before_after": True,
        },
        "scenes": {
            SCENE: {
                "scene_index": 0,
                "native_prediction_sha256_before": native_hash,
                "native_prediction_sha256_after": native_hash,
                "native_input_unchanged": True,
                "native_prefix_row_count": 1,
                "native_prefix_payload_sha256_input": prefix_hash,
                "native_prefix_payload_sha256_output": prefix_hash,
                "native_prefix_exact": True,
                "accepted_candidate_count": 1,
                "counterfactual_row_count": 2,
                "counterfactual_prediction_sha256": output_hash,
                "accepted_candidates": [_candidate_public(arrays)],
            }
        },
    }
    json_path = seal_root / "s2_yoloe_direct_shadow.json"
    json_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return {
        "baseline_root": baseline_root,
        "output_root": output_root,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "json_path": json_path,
        "npz_path": npz_path,
        "native_path": native_path,
    }


def _run(paths: dict[str, Path]):
    return audit_scannet_s2_yoloe_direct_oracle(
        shadow_json=paths["json_path"],
        shadow_npz=paths["npz_path"],
        baseline_root=paths["baseline_root"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
    )


def test_fixed_suffix_passes_strict_gate_and_never_changes_native(tmp_path):
    paths = _make_tree(tmp_path)
    before = paths["native_path"].read_bytes()
    report = _run(paths)
    assert report["candidate_selection_used_gt"] is False
    assert report["candidate_suppression_used_gt"] is False
    assert report["candidate_ranking_used_gt"] is False
    assert report["score_mode"] == "constant_1.0"
    assert report["stored_scores_used_for_evaluation"] is False
    assert report["promotion"]["passes_sealed_dev3_promotion_gate"] is True
    assert report["promotion"]["decision"] == "promote_to_h10_shadow"
    for threshold in ("0.15", "0.25", "0.50"):
        row = report["per_threshold"][threshold]
        assert row["candidate_maximum_matching_count"] == 1
        assert row["candidate_tp_precision_maximum_matching"] == 1.0
        assert row["additional_union_matching_over_native"] == 1
        assert row["fixed_suffix_delta_ap_points"] > 0.0
    assert paths["native_path"].read_bytes() == before
    assert report["native_prediction_sha256_before"] == report[
        "native_prediction_sha256_after"
    ]


def test_fixed_false_positive_is_not_gt_filtered_and_fails_gate(tmp_path):
    paths = _make_tree(tmp_path, candidate_matches_gt=False)
    report = _run(paths)
    assert report["totals"]["fixed_candidate_count"] == 1
    assert report["promotion"]["passes_sealed_dev3_promotion_gate"] is False
    assert report["promotion"]["decision"] == "reject_s2_active_birth"
    for threshold in ("0.15", "0.25", "0.50"):
        row = report["per_threshold"][threshold]
        assert row["candidate_maximum_matching_count"] == 0
        assert row["candidate_tp_precision_maximum_matching"] == 0.0
        assert row["additional_union_matching_over_native"] == 0


def test_npz_tamper_is_rejected_before_missing_gt_is_touched(tmp_path):
    paths = _make_tree(tmp_path)
    with paths["npz_path"].open("ab") as handle:
        handle.write(b"tamper")
    paths["gt_root"].rename(tmp_path / "gt-hidden")
    with pytest.raises(S2OracleError, match="NPZ SHA-256 mismatch"):
        _run(paths)


def test_native_tamper_is_rejected_before_missing_gt_is_touched(tmp_path):
    paths = _make_tree(tmp_path)
    rows = [(0, _corners([0.0, 0.0, 1.0], [0.8, 0.8, 0.8]), 0.99)]
    _write_pickle(paths["native_path"], rows)
    paths["gt_root"].rename(tmp_path / "gt-hidden")
    with pytest.raises(S2OracleError, match="differs from seal"):
        _run(paths)


def test_manifest_cannot_enable_semantic_gate(tmp_path):
    paths = _make_tree(tmp_path)
    manifest = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    manifest["detector_semantics_used_for_gate"] = True
    paths["json_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(S2OracleError, match="detector_semantics_used_for_gate"):
        _run(paths)


def test_cli_writes_once_outside_protected_roots(tmp_path):
    paths = _make_tree(tmp_path)
    out = tmp_path / "report" / "oracle.json"
    argv = [
        "--shadow-json",
        str(paths["json_path"]),
        "--shadow-npz",
        str(paths["npz_path"]),
        "--baseline-root",
        str(paths["baseline_root"]),
        "--gt-root",
        str(paths["gt_root"]),
        "--scan-root",
        str(paths["scan_root"]),
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["schema"].endswith(
        "s2_yoloe_direct_oracle.v1"
    )
    with pytest.raises(S2OracleError, match="overwrite"):
        main(argv)


def test_cli_refuses_output_inside_native_root(tmp_path):
    paths = _make_tree(tmp_path)
    with pytest.raises(S2OracleError, match="protected input root"):
        main(
            [
                "--shadow-json",
                str(paths["json_path"]),
                "--shadow-npz",
                str(paths["npz_path"]),
                "--baseline-root",
                str(paths["baseline_root"]),
                "--gt-root",
                str(paths["gt_root"]),
                "--scan-root",
                str(paths["scan_root"]),
                "--out",
                str(paths["baseline_root"] / "oracle.json"),
            ]
        )
