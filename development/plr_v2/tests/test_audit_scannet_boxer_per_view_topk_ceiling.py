import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools.audit_scannet_boxer_per_view_topk_ceiling import (
    SCHEMA,
    BoxerTopKCeilingError,
    _array_content_sha256,
    _load_sealed_sidecar,
    _select_per_frame_topk,
    audit_scannet_boxer_per_view_topk_ceiling,
    main,
)


SCENE = "scene0000_00"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corners(center, extent=(1.0, 1.0, 1.0)):
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


def _make_tree(tmp_path: Path):
    baseline_root = tmp_path / "baseline"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    sidecar_root = tmp_path / "sealed"
    for root in (baseline_root, gt_root, sidecar_root):
        root.mkdir(parents=True)
    (scan_root / SCENE).mkdir(parents=True)

    # All three proposals have equal score in one frame.  Source rows 2 and 3
    # match GT while source row 5 is a false positive.  K=2 therefore proves
    # that ascending source_row, not NPZ row order, is the frozen tie-break.
    arrays = {
        "scene_ids": np.asarray([SCENE], dtype="<U12"),
        "per_view_scene_index": np.zeros(3, dtype=np.int16),
        "per_view_frame_id": np.zeros(3, dtype=np.int64),
        "per_view_source_row": np.asarray([5, 2, 3], dtype=np.int32),
        "per_view_source_instance_id": np.asarray([15, 12, 13], dtype=np.int32),
        "per_view_center_world": np.asarray(
            [[10.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "per_view_quaternion_wxyz": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1)
        ),
        "per_view_extent_xyz": np.ones((3, 3), dtype=np.float32),
        "per_view_source_score": np.full(3, 0.9, dtype=np.float32),
        "tracked_scene_index": np.empty(0, dtype=np.int16),
        "tracked_source_row": np.empty(0, dtype=np.int32),
        "tracked_instance_id": np.empty(0, dtype=np.int32),
        "tracked_center_world": np.empty((0, 3), dtype=np.float32),
        "tracked_quaternion_wxyz": np.empty((0, 4), dtype=np.float32),
        "tracked_extent_xyz": np.empty((0, 3), dtype=np.float32),
        "tracked_source_score": np.empty(0, dtype=np.float32),
    }
    npz_path = sidecar_root / "boxer_shadow_candidates.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "schema": "boxfusion.owl_boxer_shadow_candidates.v1",
        "profile": "clean_in2",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "gt_access": False,
        "gt_access_guard_verified": True,
        "semantic_source_exported": False,
        "native_clip_unchanged": True,
        "native_before_after_identity": True,
        "coordinate_frame": "scannet_world",
        "scene_count": 1,
        "per_view_candidate_count": 3,
        "tracked_candidate_count": 0,
        "npz_file": npz_path.name,
        "npz_sha256": _sha256(npz_path),
        "candidate_content_sha256": _array_content_sha256(arrays),
        "assets_and_protocol": {
            "profile": "clean_in2",
            "detector": "owl",
            "threshold_2d": 0.25,
            "threshold_3d": 0.5,
            "nms_iou_2d": 0.5,
            "start_n": 1,
            "skip_n": 25,
        },
        "scenes": [
            {
                "scene_id": SCENE,
                "scene_index": 0,
                "gt_access_guard_verified": True,
                "per_view_extra_schedule_rows_excluded": 0,
                "per_view_kept_rows": 3,
            }
        ],
    }
    json_path = sidecar_root / "boxer_shadow_candidates.json"
    json_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    baseline_path = baseline_root / f"{SCENE}_boxes.pkl"
    with baseline_path.open("wb") as handle:
        pickle.dump(
            [[(0, _corners([0.0, 0.0, 0.0]), 0.5)]],
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    gt_path = gt_root / f"{SCENE}_bbox.npy"
    np.save(
        gt_path,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [6.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [9.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    axis_path = scan_root / SCENE / f"{SCENE}.txt"
    axis_path.write_text(
        "axisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n",
        encoding="utf-8",
    )
    return {
        "baseline_root": baseline_root,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "json_path": json_path,
        "npz_path": npz_path,
        "baseline_path": baseline_path,
        "gt_path": gt_path,
        "axis_path": axis_path,
    }


def _run(paths):
    return audit_scannet_boxer_per_view_topk_ceiling(
        shadow_json=paths["json_path"],
        shadow_npz=paths["npz_path"],
        baseline_root=paths["baseline_root"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
    )


def test_score_only_topk_uses_source_row_tie_break_and_reports_ceiling(tmp_path):
    paths = _make_tree(tmp_path)
    originals = {
        key: value.read_bytes() for key, value in paths.items() if key.endswith("path")
    }
    manifest, arrays, scenes, _ = _load_sealed_sidecar(
        paths["json_path"], paths["npz_path"]
    )
    assert manifest["gt_access"] is False
    selected = _select_per_frame_topk(arrays, scenes)
    assert selected[2][0].tolist() == [1, 2]

    report = _run(paths)
    assert report["schema"] == SCHEMA
    assert report["posthoc_dev_diagnostic"] is True
    assert report["not_deployable"] is True
    assert report["before_mobilesam"] is True
    assert report["MobileSAM_used"] is False
    assert report["H10_not_authorized"] is True
    assert report["full100_not_authorized"] is True
    assert report["threshold_tuning_performed"] is False
    assert report["selection_used_gt"] is False
    assert report["selection_used_only_frozen_source_score"] is True
    assert report["selection_completed_before_gt_access"] is True
    assert report["gt_count"] == 4
    assert report["plus_10_required_additional_matches"] == 1
    assert report["budgets"]["2"]["candidate_count"] == 2
    for budget in (4, 6, 8):
        assert report["budgets"][str(budget)]["candidate_count"] == 3
    for threshold in ("0.15", "0.25", "0.50"):
        row = report["budgets"]["2"]["per_threshold"][threshold]
        assert row["candidate_maximum_matching_count"] == 2
        assert row["additional_union_matching_over_native"] == 2
        assert row["incremental_recall_headroom_points"] == 50.0
        assert row["supports_plus_10_recall_headroom"] is True
    assert report["input_hash_identity"] is True
    for key, content in originals.items():
        assert paths[key].read_bytes() == content


def test_npz_tamper_fails_before_gt_path_is_touched(tmp_path):
    paths = _make_tree(tmp_path)
    with paths["npz_path"].open("ab") as handle:
        handle.write(b"tamper")
    paths["gt_root"].rename(tmp_path / "hidden-gt")
    with pytest.raises(BoxerTopKCeilingError, match="NPZ SHA-256 mismatch"):
        _run(paths)


def test_content_hash_and_scene_ledger_fail_closed(tmp_path):
    paths = _make_tree(tmp_path)
    manifest = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    manifest["candidate_content_sha256"] = "0" * 64
    paths["json_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BoxerTopKCeilingError, match="candidate content hash"):
        _run(paths)

    paths = _make_tree(tmp_path / "ledger")
    manifest = json.loads(paths["json_path"].read_text(encoding="utf-8"))
    manifest["scenes"][0]["per_view_kept_rows"] = 2
    paths["json_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BoxerTopKCeilingError, match="scene count mismatch"):
        _run(paths)


def test_cli_is_create_only_and_protects_inputs(tmp_path):
    paths = _make_tree(tmp_path)
    out = tmp_path / "reports" / "topk.json"
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
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["schema"] == SCHEMA
    before = out.read_bytes()
    with pytest.raises(BoxerTopKCeilingError, match="refusing to overwrite"):
        main(argv)
    assert out.read_bytes() == before

    protected_argv = list(argv)
    protected_argv[-1] = str(paths["baseline_root"] / "forbidden.json")
    with pytest.raises(BoxerTopKCeilingError, match="outside all protected inputs"):
        main(protected_argv)
