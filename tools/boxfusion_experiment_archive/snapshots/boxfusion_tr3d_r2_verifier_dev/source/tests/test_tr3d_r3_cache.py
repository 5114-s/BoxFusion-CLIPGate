from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from boxfusion.frozen_anchor_manifest import (
    build_frozen_anchor_manifest,
    write_frozen_anchor_manifest,
)
from boxfusion.tr3d_r3_cache import (
    load_tr3d_r3_cache,
    make_tr3d_r3_cache,
    tr3d_r3_cache_path,
    validate_tr3d_r3_payload,
    write_tr3d_r3_cache,
)
from boxfusion.tr3d_residual_cache import (
    TR3DResidualCache,
    transform_sha256,
    write_tr3d_residual_cache,
)


_SIGNS = np.asarray(
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


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _corners(center, size):
    return np.asarray(center, dtype=np.float32) + _SIGNS * (
        np.asarray(size, dtype=np.float32) / 2
    )


def _fixture(tmp_path: Path):
    scene = "scene0001_00"
    prefix = "p100"
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    anchor_corners = np.stack(
        (_corners([0, 0, 0], [2, 2, 2]), _corners([4, 0, 0], [2, 2, 2]))
    )
    anchor_scores = np.asarray([0.6, 0.7], dtype=np.float64)
    prediction = prediction_root / f"{scene}_boxes.pkl"
    with prediction.open("wb") as handle:
        pickle.dump(
            [[(0, anchor_corners[0], anchor_scores[0]), (0, anchor_corners[1], anchor_scores[1])]],
            handle,
        )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("frozen", encoding="utf-8")
    manifest_payload = build_frozen_anchor_manifest(
        anchor_name="test-g0",
        reference_root=prediction_root,
        scene_list=scene_list,
        artifacts={"artifact": artifact},
        anchor_metrics_percent={"AP15": 1, "AP25": 1, "AP50": 1},
        required_scene_count=1,
    )
    manifest = tmp_path / "frozen.json"
    write_frozen_anchor_manifest(manifest, manifest_payload)

    transform = np.eye(4, dtype=np.float64)
    metadata = tmp_path / f"{scene}.txt"
    metadata.write_text(
        "axisAlignment = " + " ".join(str(value) for value in transform.reshape(-1)) + "\n",
        encoding="utf-8",
    )
    proposal_corners = np.stack(
        (
            _corners([0.25, 0, 0], [2, 2, 2]),
            _corners([20, 0, 0], [1, 1, 1]),
        )
    )
    boxes = np.asarray(
        [[0.25, 0, 0, 2, 2, 2, 0], [20, 0, 0, 1, 1, 1, 0]],
        dtype=np.float32,
    )
    parent = TR3DResidualCache(
        scene_id=scene,
        sample_idx=f"{scene}:{prefix}",
        prefix_id=prefix,
        prefix_fraction=1.0,
        boxes_world=boxes,
        corners_world=proposal_corners,
        aligned_to_unaligned=transform,
        axis_alignment_sha256=transform_sha256(transform),
        scores_3d=np.asarray([0.9, 0.8], dtype=np.float32),
        labels_3d=np.zeros(2, dtype=np.int64),
        proposal_ids=np.asarray([10, 20], dtype=np.int64),
        point_count=np.asarray([80, 10], dtype=np.int32),
        voxel_size=0.02,
        runtime_s=0.1,
        num_input_points=100,
        checkpoint_sha256=_sha("checkpoint"),
        config_sha256=_sha("config"),
        source_scene_sha256=_sha("source"),
    )
    parent_path = tmp_path / "parent.npz"
    write_tr3d_residual_cache(parent_path, parent)
    return {
        "scene": scene,
        "prefix": prefix,
        "prediction": prediction,
        "manifest": manifest,
        "anchor_corners": anchor_corners,
        "anchor_scores": anchor_scores,
        "metadata": metadata,
        "transform": transform,
        "parent": parent_path,
    }


def _contract(fixture):
    return {
        "parent_tr3d_cache_path": fixture["parent"],
        "frozen_anchor_manifest_path": fixture["manifest"],
        "anchor_prediction_path": fixture["prediction"],
        "anchor_corners_world": fixture["anchor_corners"],
        "anchor_scores": fixture["anchor_scores"],
        "axis_alignment_metadata_path": fixture["metadata"],
        "expected_checkpoint_sha256": _sha("checkpoint"),
        "expected_config_sha256": _sha("config"),
        "expected_r3_config_sha256": _sha("r3-config"),
        "expected_r3_code_sha256": _sha("r3-code"),
        "expected_scene_id": fixture["scene"],
        "expected_prefix_id": fixture["prefix"],
    }


def _cache(fixture):
    return make_tr3d_r3_cache(
        parent_tr3d_cache_path=fixture["parent"],
        frozen_anchor_manifest_path=fixture["manifest"],
        anchor_prediction_path=fixture["prediction"],
        anchor_corners_world=fixture["anchor_corners"],
        anchor_scores=fixture["anchor_scores"],
        axis_alignment_metadata_path=fixture["metadata"],
        axis_alignment=fixture["transform"],
        expected_checkpoint_sha256=_sha("checkpoint"),
        expected_config_sha256=_sha("config"),
        r3_config_sha256=_sha("r3-config"),
        r3_code_sha256=_sha("r3-code"),
    )


def test_optional_evidence_roundtrip_is_immutable_and_rederived(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cache = _cache(fixture)
    assert not cache.parent_r2a_available
    assert not cache.parent_r2b_available
    assert cache.parent_r2a_cache_sha256 == ""
    assert cache.parent_r2b_cache_sha256 == ""
    assert cache.proposal_ids.tolist() == [10]
    assert cache.anchor_index.tolist() == [0]
    assert not cache.r2a_evidence_available.any()
    assert not cache.r2b_feature_available.any()
    assert np.all(cache.r2a_depth_evidence == 0)
    path = tr3d_r3_cache_path(tmp_path / "r3", fixture["scene"], fixture["prefix"])
    contract = _contract(fixture)
    write_tr3d_r3_cache(path, cache, **contract)
    loaded = load_tr3d_r3_cache(path, **contract)
    np.testing.assert_array_equal(loaded.proposal_corners_world, cache.proposal_corners_world)
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="immutable R3"):
        write_tr3d_r3_cache(path, cache, **contract)


def test_derived_geometry_and_parent_hashes_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cache = _cache(fixture)
    payload = cache.as_npz_payload()
    changed = payload["anchor_iou"].copy()
    changed[0] -= np.float32(0.1)
    payload["anchor_iou"] = changed
    with pytest.raises(ValueError, match="anchor_iou disagrees"):
        validate_tr3d_r3_payload(payload, **_contract(fixture))

    payload = cache.as_npz_payload()
    payload["parent_tr3d_cache_sha256"] = np.asarray(_sha("wrong"))
    with pytest.raises(ValueError, match="TR3D parent bytes"):
        validate_tr3d_r3_payload(payload, **_contract(fixture))


def test_path_validation() -> None:
    assert tr3d_r3_cache_path("root", "scene0001_00", "p100") == Path(
        "root/scene0001_00/p100.npz"
    )
    with pytest.raises(ValueError, match="scene"):
        tr3d_r3_cache_path("root", "bad", "p100")
