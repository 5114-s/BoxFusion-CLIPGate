from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_residual_cache import (
    TR3DResidualCache,
    load_tr3d_residual_cache,
    make_tr3d_residual_cache_from_aligned,
    tr3d_residual_cache_path,
    write_tr3d_residual_cache,
    transform_sha256,
)
from boxfusion.tr3d_residual_observer import TR3DResidualObserver


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cache() -> TR3DResidualCache:
    boxes = np.asarray([[1, 2, 3, 2, 4, 6, 0]], dtype=np.float32)
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
    corners = boxes[:, None, :3] + signs[None] * boxes[:, None, 3:6] / 2
    return TR3DResidualCache(
        scene_id="scene0001_00",
        sample_idx="scene0001_00:full",
        prefix_id="full",
        prefix_fraction=1.0,
        boxes_world=boxes,
        corners_world=corners,
        aligned_to_unaligned=np.eye(4, dtype=np.float64),
        axis_alignment_sha256=transform_sha256(np.eye(4)),
        scores_3d=np.asarray([0.8], dtype=np.float32),
        labels_3d=np.asarray([0], dtype=np.int64),
        proposal_ids=np.asarray([10], dtype=np.int64),
        point_count=np.asarray([42], dtype=np.int32),
        voxel_size=0.02,
        runtime_s=0.1,
        num_input_points=100,
        checkpoint_sha256=_hash(b"checkpoint"),
        config_sha256=_hash(b"config"),
        source_scene_sha256=_hash(b"scene"),
    )


def test_cache_roundtrip_is_readonly_and_immutable(tmp_path: Path) -> None:
    cache = _cache()
    path = tr3d_residual_cache_path(tmp_path, cache.scene_id)
    write_tr3d_residual_cache(path, cache)
    loaded = load_tr3d_residual_cache(
        path,
        expected_scene_id=cache.scene_id,
        expected_prefix_id="full",
        expected_checkpoint_sha256=cache.checkpoint_sha256,
        expected_config_sha256=cache.config_sha256,
    )
    assert loaded.proposal_count == 1
    assert np.array_equal(loaded.corners_world, cache.corners_world)
    assert not loaded.corners_world.flags.writeable
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError):
        loaded.corners_world[0, 0, 0] = 0
    with pytest.raises(FileExistsError, match="immutable"):
        write_tr3d_residual_cache(path, cache)


def test_cache_fails_closed_on_nonzero_labels_and_unknown_field() -> None:
    payload = _cache().as_npz_payload()
    payload["labels_3d"] = np.asarray([1], dtype=np.int64)
    from boxfusion.tr3d_residual_cache import validate_tr3d_residual_payload

    with pytest.raises(ValueError, match="labels_3d"):
        validate_tr3d_residual_payload(payload)
    payload = _cache().as_npz_payload()
    payload["legacy_p1_field"] = np.asarray(1)
    with pytest.raises(ValueError, match="unknown"):
        validate_tr3d_residual_payload(payload)


def test_observer_returns_identical_prediction_object_and_never_writes_b6(
    tmp_path: Path,
) -> None:
    cache = _cache()
    cache_path = tr3d_residual_cache_path(
        tmp_path / "cache", cache.scene_id
    )
    write_tr3d_residual_cache(cache_path, cache)
    frozen = tmp_path / "frozen" / f"{cache.scene_id}_boxes.pkl"
    frozen.parent.mkdir()
    frozen.write_bytes(b"frozen-b6-prediction")
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    predictions = [{"frozen": True}]
    observer = TR3DResidualObserver(
        tmp_path / "cache",
        checkpoint_sha256=cache.checkpoint_sha256,
        config_sha256=cache.config_sha256,
    )
    result = observer.observe(cache.scene_id, predictions)
    assert result.predictions is predictions
    assert result.mutation_enabled is False
    assert result.applied_count == 0
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before


def test_official_aligned_factory_preserves_authoritative_corners() -> None:
    angle = np.pi / 2
    alignment = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0, 2],
            [np.sin(angle), np.cos(angle), 0, -1],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    cache = make_tr3d_residual_cache_from_aligned(
        scene_id="scene0001_00",
        boxes_aligned=np.asarray([[2, 3, 1, 1, 2, 3]], dtype=np.float32),
        scores_3d=np.asarray([0.7], dtype=np.float32),
        unaligned_to_aligned=alignment,
        checkpoint_sha256=_hash(b"checkpoint"),
        config_sha256=_hash(b"config"),
        source_scene_sha256=_hash(b"scene"),
    )
    expected_center = (
        np.asarray([2, 3, 1]) @ np.linalg.inv(alignment)[:3, :3].T
        + np.linalg.inv(alignment)[:3, 3]
    )
    assert np.allclose(cache.corners_world.mean(1)[0], expected_center)
    assert np.allclose(
        cache.aligned_to_unaligned, np.linalg.inv(alignment)
    )
