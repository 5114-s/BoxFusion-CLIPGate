import argparse
import json
import pickle

import numpy as np

from tools.audit_sgcdet_sparse_identity import (
    Prediction,
    _cross_run_report,
    _strict_control_scene,
    audit,
)


def _corners(center, size):
    center = np.asarray(center, dtype=np.float32)
    half = 0.5 * np.asarray(size, dtype=np.float32)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return center + signs * half


def _write_control(
    path,
    *,
    mutate_full_post=False,
    profile="sgcdet_sparse_observer",
    sparse_model_enabled=False,
):
    boxes = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.4, 0.5, 0.6],
            [1.0, 0.0, 1.0, 0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )
    corners = np.stack(
        [_corners(box[:3], box[3:6]) for box in boxes]
    ).astype(np.float32)
    post_corners = corners.copy()
    if mutate_full_post:
        post_corners[1, 0, 0] += np.float32(0.01)
    scores = np.asarray([0.7, 0.8], dtype=np.float32)
    indices = np.arange(2, dtype=np.int64)
    flags = {
        "mutation_refit_enabled": False,
        "mutation_box_refiner_enabled": False,
        "mutation_quality_enabled": True,
        "mutation_joint_local_head_enabled": False,
        "mutation_joint_geometry_enabled": False,
        "mutation_joint_scores_enabled": False,
        "mutation_sparse_refiner_enabled": sparse_model_enabled,
        "mutation_sparse_geometry_enabled": False,
        "sparse_collect_diagnostics": True,
        "mutation_supplemental_output_enabled": False,
        "mutation_soft_nms_enabled": False,
    }
    summary = {
        "online_ablation_profile": profile,
        **flags,
        "sparse_accepted": 0,
        "sparse_instances": 2,
        "sparse_inputs_valid": 2,
        "sparse_invalid_identity": 0,
        "sparse_unobserved_identity": 0,
    }
    np.savez_compressed(
        path,
        online_ablation_profile=np.asarray(profile),
        **{key: np.asarray(value, dtype=bool) for key, value in flags.items()},
        source_indices=indices,
        result_indices=indices,
        sparse_pair_source_indices=indices,
        sparse_pair_stable_ids=np.asarray([10, 11], dtype=np.int64),
        track_ids=np.asarray([10, 11], dtype=np.int64),
        boxes=boxes,
        scores=scores,
        sparse_original_boxes=boxes,
        sparse_active_boxes=boxes,
        sparse_original_corners=corners,
        sparse_active_corners=corners,
        sparse_final_b6_scores=scores,
        refit_original_boxes=boxes,
        refit_candidate_boxes=boxes,
        refit_original_corners=corners,
        refit_candidate_corners=corners,
        refit_applied=np.zeros(2, dtype=bool),
        refit_boundary_delta=np.zeros((2, 6), dtype=np.float32),
        sparse_input_valid=np.ones(2, dtype=bool),
        sparse_output_valid=np.zeros(2, dtype=bool),
        sparse_accepted=np.zeros(2, dtype=bool),
        sparse_center_residual=np.full((2, 3), np.nan, dtype=np.float32),
        sparse_center_residual_fraction=np.full(
            (2, 3), np.nan, dtype=np.float32
        ),
        sparse_log_dimension_residual=np.full(
            (2, 3), np.nan, dtype=np.float32
        ),
        summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
        output_geometry_schema=np.asarray(
            "boxfusion.full_output_geometry_prepost.v1"
        ),
        output_pre_geometry_boxes=boxes,
        output_pre_geometry_corners=corners,
        output_post_geometry_boxes=boxes,
        output_post_geometry_corners=post_corners,
        output_source_indices=indices,
        output_stable_ids=np.asarray([10, 11], dtype=np.int64),
        output_refit_applied=np.zeros(2, dtype=bool),
    )
    return Prediction(
        labels=np.asarray([0, 1], dtype=np.int64),
        corners=post_corners,
        scores=scores,
    )


def _write_prediction(path, prediction):
    rows = [
        (int(label), corners.copy(), float(score))
        for label, corners, score in zip(
            prediction.labels, prediction.corners, prediction.scores
        )
    ]
    with path.open("wb") as handle:
        pickle.dump([rows], handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_full_output_sparse_identity_audit_has_complete_coverage(tmp_path):
    diagnostic = tmp_path / "scene0000_00_tracks.npz"
    prediction = _write_control(diagnostic)
    report = _strict_control_scene(
        label="S1 observer",
        expected_profile="sgcdet_sparse_observer",
        prediction=prediction,
        diagnostic_path=diagnostic,
        sparse_model_enabled=False,
    )
    assert report["ok"] is True
    assert report["full_output_prepost_available"] is True
    assert report["strict_geometry_rows"] == 2
    assert report["exact_row_coverage"] == 1.0


def test_full_output_sparse_identity_audit_rejects_geometry_mutation(tmp_path):
    diagnostic = tmp_path / "scene0000_00_tracks.npz"
    prediction = _write_control(diagnostic, mutate_full_post=True)
    report = _strict_control_scene(
        label="S1 observer",
        expected_profile="sgcdet_sparse_observer",
        prediction=prediction,
        diagnostic_path=diagnostic,
        sparse_model_enabled=False,
    )
    assert report["ok"] is False
    assert any(
        "output_pre_geometry_corners/output_post_geometry_corners" in issue
        for issue in report["issues"]
    )


def test_cross_run_count_and_label_drift_are_report_only(tmp_path):
    scene_names = ("scene0000_00", "scene0001_00")
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scene_names) + "\n", encoding="utf-8")
    roots = {
        name: tmp_path / name
        for name in ("baseline", "observer", "identity", "observer_diag", "identity_diag")
    }
    for root in roots.values():
        root.mkdir()

    for scene in scene_names:
        observer = _write_control(
            roots["observer_diag"] / f"{scene}_tracks.npz"
        )
        identity = _write_control(
            roots["identity_diag"] / f"{scene}_tracks.npz",
            profile="sgcdet_sparse_identity",
            sparse_model_enabled=True,
        )
        _write_prediction(
            roots["observer"] / f"{scene}_boxes.pkl", observer
        )
        _write_prediction(
            roots["identity"] / f"{scene}_boxes.pkl", identity
        )

        if scene == "scene0000_00":
            baseline = Prediction(
                labels=observer.labels[:1],
                corners=observer.corners[:1],
                scores=observer.scores[:1],
            )
        else:
            baseline = Prediction(
                labels=observer.labels[::-1],
                corners=observer.corners,
                scores=observer.scores,
            )
        _write_prediction(
            roots["baseline"] / f"{scene}_boxes.pkl", baseline
        )

    report = audit(
        argparse.Namespace(
            baseline_root=roots["baseline"],
            observer_root=roots["observer"],
            identity_root=roots["identity"],
            observer_diagnostics_root=roots["observer_diag"],
            identity_diagnostics_root=roots["identity_diag"],
            scene_list=scene_list,
        )
    )

    assert report["ok"] is True
    assert report["issues"] == []
    assert report["strict_within_run"]["S1 observer"]["ok"] is True
    assert report["strict_within_run"]["S2 identity"]["ok"] is True
    assert report["cross_run_report"]["S0_vs_S1"]["structural_ok"] is False
    assert any("prediction count differs" in item for item in report["warnings"])
    assert any("label sequence differs" in item for item in report["warnings"])


def test_cross_run_iou_matching_does_not_misalign_after_insert(tmp_path):
    baseline_root = tmp_path / "baseline"
    control_root = tmp_path / "control"
    baseline_root.mkdir()
    control_root.mkdir()
    scene = "scene0000_00"
    common_corners = np.stack(
        [
            _corners([0.0, 0.0, 1.0], [0.4, 0.5, 0.6]),
            _corners([2.0, 0.0, 1.0], [0.7, 0.8, 0.9]),
        ]
    ).astype(np.float32)
    baseline = Prediction(
        labels=np.asarray([0, 0], dtype=np.int64),
        corners=common_corners,
        scores=np.asarray([0.9, 0.8], dtype=np.float32),
    )
    control = Prediction(
        labels=np.asarray([0, 0, 0], dtype=np.int64),
        corners=np.concatenate(
            [
                _corners([20.0, 20.0, 20.0], [1.0, 1.0, 1.0])[None],
                common_corners,
            ],
            axis=0,
        ).astype(np.float32),
        scores=np.asarray([0.1, 0.9, 0.8], dtype=np.float32),
    )
    baseline_path = baseline_root / f"{scene}_boxes.pkl"
    control_path = control_root / f"{scene}_boxes.pkl"
    _write_prediction(baseline_path, baseline)
    _write_prediction(control_path, control)

    report = _cross_run_report(
        baseline_paths={scene: baseline_path},
        control_paths={scene: control_path},
        scenes=(scene,),
    )

    assert report["structural_ok"] is False
    assert report["prediction_rows_compared"] == 2
    assert report["unmatched_baseline_rows"] == 0
    assert report["unmatched_control_rows"] == 1
    assert report["corner_abs_drift"]["max"] == 0.0
    assert report["score_abs_drift"]["max"] == 0.0
    assert report["matched_aabb_iou"]["min"] == 1.0
    assert report["score_rank"]["mismatch_positions"] == 0
    scene_report = report["per_scene"][scene]
    assert scene_report["matched_baseline_indices"] == [0, 1]
    assert scene_report["matched_control_indices"] == [1, 2]
    assert scene_report["unmatched_control_indices"] == [0]


def test_cross_run_iou_matching_leaves_replaced_object_unmatched(tmp_path):
    baseline_root = tmp_path / "baseline"
    control_root = tmp_path / "control"
    baseline_root.mkdir()
    control_root.mkdir()
    scene = "scene0000_00"
    shared = _corners([0.0, 0.0, 1.0], [0.4, 0.5, 0.6])
    replaced = _corners([2.0, 0.0, 1.0], [0.7, 0.8, 0.9])
    replacement = _corners([20.0, 20.0, 20.0], [0.7, 0.8, 0.9])
    baseline = Prediction(
        labels=np.asarray([0, 0], dtype=np.int64),
        corners=np.stack([shared, replaced]).astype(np.float32),
        scores=np.asarray([0.9, 0.8], dtype=np.float32),
    )
    control = Prediction(
        labels=np.asarray([0, 0], dtype=np.int64),
        corners=np.stack([shared, replacement]).astype(np.float32),
        scores=np.asarray([0.9, 0.7], dtype=np.float32),
    )
    baseline_path = baseline_root / f"{scene}_boxes.pkl"
    control_path = control_root / f"{scene}_boxes.pkl"
    _write_prediction(baseline_path, baseline)
    _write_prediction(control_path, control)

    report = _cross_run_report(
        baseline_paths={scene: baseline_path},
        control_paths={scene: control_path},
        scenes=(scene,),
    )

    assert report["matching_min_aabb_iou"] == 0.5
    assert report["prediction_rows_compared"] == 1
    assert report["unmatched_baseline_rows"] == 1
    assert report["unmatched_control_rows"] == 1
    assert report["corner_abs_drift"]["max"] == 0.0
    assert report["matched_aabb_iou"]["min"] == 1.0
    scene_report = report["per_scene"][scene]
    assert scene_report["matched_baseline_indices"] == [0]
    assert scene_report["matched_control_indices"] == [0]
    assert scene_report["unmatched_baseline_indices"] == [1]
    assert scene_report["unmatched_control_indices"] == [1]
