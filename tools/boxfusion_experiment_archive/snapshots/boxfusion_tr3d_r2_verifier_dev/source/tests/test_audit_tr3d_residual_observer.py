from __future__ import annotations

import hashlib
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

from boxfusion.frozen_b6_manifest import (
    build_frozen_b6_manifest,
    write_frozen_b6_manifest,
)
from boxfusion.tr3d_residual_cache import (
    TR3DResidualCache,
    tr3d_residual_cache_path,
    write_tr3d_residual_cache,
    transform_sha256,
)


_TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "audit_tr3d_residual_observer.py"
)
_SPEC = importlib.util.spec_from_file_location("audit_tr3d", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _corners(center, size):
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
    return np.asarray(center, dtype=np.float32) + signs * (
        np.asarray(size, dtype=np.float32) / 2
    )


def test_union_oracle_adds_missing_gt_and_verifies_zero_write(
    tmp_path: Path,
) -> None:
    scene = "scene0001_00"
    b6_root = tmp_path / "b6"
    b6_root.mkdir()
    b6_path = b6_root / f"{scene}_boxes.pkl"
    with b6_path.open("wb") as handle:
        pickle.dump(
            [[(0, _corners([0, 0, 0], [1, 1, 1]), 0.9)]],
            handle,
        )
    checkpoint = tmp_path / "b6.npz"
    checkpoint.write_bytes(b"b6")
    scene_list = tmp_path / "val.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    manifest_payload = build_frozen_b6_manifest(
        reference_root=b6_root,
        checkpoint=checkpoint,
        scene_list=scene_list,
        required_scene_count=1,
    )
    manifest = tmp_path / "manifest.json"
    write_frozen_b6_manifest(manifest, manifest_payload)
    before = b6_path.read_bytes()

    checkpoint_sha = _sha(b"tr3d")
    config_sha = _sha(b"config")
    candidate_corners = _corners([3, 0, 0], [1, 1, 1])[None]
    cache_inverse = np.eye(4, dtype=np.float64)
    # A valid inverse can differ by one ULP across NumPy/LAPACK builds while
    # still satisfying the exact geometric provenance contract.
    cache_inverse[0, 0] = np.nextafter(cache_inverse[0, 0], 2.0)
    candidate = TR3DResidualCache(
        scene_id=scene,
        sample_idx=f"{scene}:full",
        prefix_id="full",
        prefix_fraction=1.0,
        boxes_world=np.asarray(
            [[3, 0, 0, 1, 1, 1, 0]], dtype=np.float32
        ),
        corners_world=candidate_corners,
        aligned_to_unaligned=cache_inverse,
        axis_alignment_sha256=transform_sha256(cache_inverse),
        scores_3d=np.asarray([0.8], dtype=np.float32),
        labels_3d=np.asarray([0], dtype=np.int64),
        proposal_ids=np.asarray([0], dtype=np.int64),
        point_count=np.asarray([50], dtype=np.int32),
        voxel_size=0.02,
        runtime_s=0.2,
        num_input_points=100,
        checkpoint_sha256=checkpoint_sha,
        config_sha256=config_sha,
        source_scene_sha256=_sha(b"source"),
    )
    cache_root = tmp_path / "cache"
    write_tr3d_residual_cache(
        tr3d_residual_cache_path(cache_root, scene), candidate
    )
    gt_root = tmp_path / "gt"
    gt_root.mkdir()
    np.save(
        gt_root / f"{scene}_bbox.npy",
        np.asarray(
            [
                [0, 0, 0, 1, 1, 1, 3],
                [3, 0, 0, 1, 1, 1, 3],
            ],
            dtype=np.float32,
        ),
    )
    scans = tmp_path / "scans" / scene
    scans.mkdir(parents=True)
    (scans / f"{scene}.txt").write_text(
        "axisAlignment = "
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n",
        encoding="utf-8",
    )

    report = _MODULE.audit(
        manifest_path=manifest,
        cache_root=cache_root,
        gt_root=gt_root,
        scans_root=tmp_path / "scans",
        scene_list=None,
        prefix_id="full",
        checkpoint_sha256=checkpoint_sha,
        config_sha256=config_sha,
    )
    row = report["thresholds"]["0.50"]
    assert row["b6_oracle_tp"] == 1
    assert row["union_oracle_tp"] == 2
    assert row["novel_oracle_tp"] == 1
    assert report["observer_contract"][
        "frozen_b6_verified_before_and_after"
    ]
    assert report["anchor"]["name"] == "B6"
    score_row = report["score_frontier"]["rows"]["0.5000"]
    assert score_row["candidate_count"] == 1
    assert score_row["thresholds"]["0.50"]["novel_oracle_tp"] == 1
    assert b6_path.read_bytes() == before

    cli_report = tmp_path / "reports" / "union_oracle.json"
    assert (
        _MODULE.main(
            [
                "--manifest",
                str(manifest),
                "--cache-root",
                str(cache_root),
                "--gt-root",
                str(gt_root),
                "--scans-root",
                str(tmp_path / "scans"),
                "--scene-list",
                str(scene_list),
                "--checkpoint-sha256",
                checkpoint_sha,
                "--config-sha256",
                config_sha,
                "--report",
                str(cli_report),
            ]
        )
        == 0
    )
    assert cli_report.is_file()
    assert b6_path.read_bytes() == before


def test_alignment_provenance_fails_closed_on_material_drift() -> None:
    transform = np.asarray(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(transform)
    ulp_equivalent = inverse.copy()
    ulp_equivalent[0, 1] = np.nextafter(
        ulp_equivalent[0, 1],
        2.0,
    )
    _MODULE._validate_alignment_provenance(
        "scene0001_00",
        transform,
        ulp_equivalent,
    )

    corrupted = inverse.copy()
    corrupted[0, 3] += 1e-4
    with pytest.raises(ValueError, match="provenance mismatch"):
        _MODULE._validate_alignment_provenance(
            "scene0001_00",
            transform,
            corrupted,
        )
