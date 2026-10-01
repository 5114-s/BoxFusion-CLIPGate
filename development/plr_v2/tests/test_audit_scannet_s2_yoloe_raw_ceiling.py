import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools.audit_scannet_s2_yoloe_raw_ceiling import (
    GEOMETRY_QUANTILES,
    SCHEMA,
    S2RawCeilingError,
    audit_scannet_s2_yoloe_raw_ceiling,
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


def _point_box(center, extent=(1.0, 1.0, 1.0)):
    corners = _corners(center, extent)
    return np.repeat(corners, 64, axis=0).astype(np.float32)


def _write_prediction(path: Path):
    rows = [(0, _corners([0.0, 0.0, 0.0]), 0.5)]
    with path.open("wb") as handle:
        pickle.dump([rows], handle, protocol=pickle.HIGHEST_PROTOCOL)


def _make_tree(tmp_path: Path):
    candidate_root = tmp_path / "candidate"
    baseline_root = tmp_path / "baseline"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    for root in (candidate_root, baseline_root, gt_root):
        root.mkdir(parents=True)
    (scan_root / SCENE).mkdir(parents=True)

    # GT0 is already covered by native T05.  Reported row 0 covers GT1 but the
    # frozen terminal gate rejected it.  Row 2 only covers GT1 under points.
    boxes = np.asarray(
        [
            [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [10.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [20.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    points = np.stack(
        [
            _point_box([3.0, 0.0, 0.0]),
            _point_box([10.0, 0.0, 0.0]),
            _point_box([3.0, 0.0, 0.0]),
        ]
    )
    diagnostic_path = candidate_root / f"{SCENE}_tracks.npz"
    np.savez_compressed(
        diagnostic_path,
        scene_id=np.asarray(SCENE),
        boxes=boxes,
        scores=np.asarray([0.91, 0.72, 0.51], dtype=np.float32),
        points=points,
        point_mask=np.ones((3, 512), dtype=bool),
        source_indices=-np.ones(3, dtype=np.int64),
        track_ids=np.asarray([-11, -12, -13], dtype=np.int64),
        result_indices=np.arange(1, 4, dtype=np.int64),
        labels=np.asarray(
            ["SECRET_LABEL_ALPHA", "SECRET_LABEL_BETA", "SECRET_LABEL_GAMMA"]
        ),
    )

    prediction_path = baseline_root / f"{SCENE}_boxes.pkl"
    _write_prediction(prediction_path)
    np.save(
        gt_root / f"{SCENE}_bbox.npy",
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    alignment_path = scan_root / SCENE / f"{SCENE}.txt"
    alignment_path.write_text(
        "axisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n",
        encoding="utf-8",
    )

    diagnostic_hash = _sha256(diagnostic_path)
    native_hash = _sha256(prediction_path)
    manifest = {
        "schema": "boxfusion.s2_yoloe_direct_shadow.v1",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "gt_access": False,
        "oracle_access": False,
        "scene_count": 1,
        "scene_order": [SCENE],
        "input": {
            "candidate_root": str(candidate_root.resolve()),
            "baseline_root": str(baseline_root.resolve()),
        },
        "scenes": {
            SCENE: {
                "diagnostic_row_count": 3,
                "supplemental_rows_read_source_index_minus_one": 3,
                "diagnostic_sha256_before": diagnostic_hash,
                "diagnostic_sha256_after": diagnostic_hash,
                "native_prediction_sha256_before": native_hash,
                "native_prediction_sha256_after": native_hash,
                "native_prefix_row_count": 1,
                "accepted_candidates": [{"diagnostic_row": 1}],
                "terminal_rejections": {
                    "native_overlap_rejected_diagnostic_rows": [0],
                    "self_nms_rejected_diagnostic_rows": [],
                    "output_cap_rejected_diagnostic_rows": [2],
                },
            }
        },
    }
    manifest_path = tmp_path / "sealed.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return {
        "candidate_root": candidate_root,
        "baseline_root": baseline_root,
        "gt_root": gt_root,
        "scan_root": scan_root,
        "diagnostic_path": diagnostic_path,
        "prediction_path": prediction_path,
        "gt_path": gt_root / f"{SCENE}_bbox.npy",
        "alignment_path": alignment_path,
        "manifest_path": manifest_path,
    }


def _run(paths):
    return audit_scannet_s2_yoloe_raw_ceiling(
        candidate_root=paths["candidate_root"],
        sealed_manifest=paths["manifest_path"],
        baseline_root=paths["baseline_root"],
        gt_root=paths["gt_root"],
        scan_root=paths["scan_root"],
    )


def test_all_rows_and_all_geometries_are_audited_without_label_leak(tmp_path):
    paths = _make_tree(tmp_path)
    original = {key: path.read_bytes() for key, path in paths.items() if key.endswith("path")}
    report = _run(paths)

    assert report["schema"] == SCHEMA
    assert report["posthoc_dev_diagnostic"] is True
    assert report["not_deployable"] is True
    assert report["H10_not_authorized"] is True
    assert report["h10_gt_accessed"] is False
    assert report["candidate_selection_applied"] is False
    assert report["candidate_suppression_applied"] is False
    assert report["candidate_geometry_mutated"] is False
    assert report["labels_read"] is False
    assert report["labels_used"] is False
    assert report["labels_output"] is False
    assert report["candidate_count"] == 3
    assert report["candidate_count_by_scene"] == {SCENE: 3}
    assert report["geometry_order"] == list(GEOMETRY_QUANTILES)
    assert set(report["geometries"]) == set(GEOMETRY_QUANTILES)
    assert all(value["candidate_count"] == 3 for value in report["geometries"].values())
    assert report["terminal_disposition_counts"] == {
        "accepted": 1,
        "native_overlap": 1,
        "self_nms": 0,
        "output_cap": 1,
    }

    reported = report["geometries"]["reported"]["per_threshold"]["0.50"]
    assert reported["candidate_maximum_matching_count"] == 1
    assert reported["additional_union_matching_over_native"] == 1
    q00 = report["geometries"]["points_q00"]["per_threshold"]["0.50"]
    assert q00["candidate_maximum_matching_count"] == 1
    assert q00["additional_union_matching_over_native"] == 1

    recovered = report["reported_geometry_official_unmatched_recoveries"]["0.50"]
    assert recovered["rows_with_any_strict_overlap_count"] == 1
    row = recovered["rows_with_any_strict_overlap"][0]
    assert row["diagnostic_row"] == 0
    assert row["score"] == pytest.approx(0.91)
    assert row["terminal_disposition"] == "rejected"
    assert row["terminal_reason"] == "native_overlap"
    assert row["baseline_official_unmatched_gt_index"] == 1
    serialized = json.dumps(report, sort_keys=True)
    assert "SECRET_LABEL" not in serialized
    assert '"label"' not in serialized
    assert report["input_hash_identity"] is True

    for key, content in original.items():
        assert paths[key].read_bytes() == content


def test_diagnostic_tamper_fails_before_gt_is_opened(tmp_path):
    paths = _make_tree(tmp_path)
    with paths["diagnostic_path"].open("ab") as handle:
        handle.write(b"tamper")
    hidden = tmp_path / "hidden-gt"
    paths["gt_root"].rename(hidden)
    with pytest.raises(S2RawCeilingError, match="diagnostic SHA-256 differs from seal"):
        _run(paths)


def test_terminal_mapping_must_cover_each_row_exactly(tmp_path):
    paths = _make_tree(tmp_path)
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    manifest["scenes"][SCENE]["terminal_rejections"][
        "self_nms_rejected_diagnostic_rows"
    ] = [0]
    paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(S2RawCeilingError, match="duplicate terminal mapping"):
        _run(paths)


def test_cli_output_is_create_only_and_outside_protected_roots(tmp_path):
    paths = _make_tree(tmp_path)
    output = tmp_path / "reports" / "raw.json"
    argv = [
        "--candidate-root",
        str(paths["candidate_root"]),
        "--sealed-manifest",
        str(paths["manifest_path"]),
        "--baseline-root",
        str(paths["baseline_root"]),
        "--gt-root",
        str(paths["gt_root"]),
        "--scan-root",
        str(paths["scan_root"]),
        "--out",
        str(output),
    ]
    assert main(argv) == 0
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["schema"] == SCHEMA
    before = output.read_bytes()
    with pytest.raises(S2RawCeilingError, match="refusing to overwrite"):
        main(argv)
    assert output.read_bytes() == before

    protected_output = paths["candidate_root"] / "forbidden.json"
    protected_argv = list(argv)
    protected_argv[-1] = str(protected_output)
    with pytest.raises(S2RawCeilingError, match="outside every protected input root"):
        main(protected_argv)


def test_invalid_valid_point_is_rejected(tmp_path):
    paths = _make_tree(tmp_path)
    with np.load(paths["diagnostic_path"], allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    arrays["points"][0, 0, 0] = np.nan
    np.savez_compressed(paths["diagnostic_path"], **arrays)
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    digest = _sha256(paths["diagnostic_path"])
    manifest["scenes"][SCENE]["diagnostic_sha256_before"] = digest
    manifest["scenes"][SCENE]["diagnostic_sha256_after"] = digest
    paths["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(S2RawCeilingError, match="valid points are non-finite"):
        _run(paths)
