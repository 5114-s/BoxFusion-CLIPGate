from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_r2_provenance import sha256_file
from boxfusion.tr3d_residual_cache import (
    TR3DResidualCache,
    transform_sha256,
    write_tr3d_residual_cache,
)
from tools.run_tr3d_r2_observer import (
    _load_bound_parent,
    _prefix_point_contract,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_points(path: Path, count: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.arange(count * 6, dtype=np.float32).reshape(count, 6).tofile(path)
    return path


def _write_parent(
    path: Path,
    point_path: Path,
    *,
    num_input_points: int = 2,
) -> Path:
    box = np.asarray([[0, 0, 0, 2, 2, 2, 0]], dtype=np.float32)
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
    transform = np.eye(4, dtype=np.float64)
    cache = TR3DResidualCache(
        scene_id="scene0001_00",
        sample_idx="scene0001_00:p100",
        prefix_id="p100",
        prefix_fraction=1.0,
        boxes_world=box,
        corners_world=box[:, None, :3] + signs[None],
        aligned_to_unaligned=transform,
        axis_alignment_sha256=transform_sha256(transform),
        scores_3d=np.asarray([0.9], dtype=np.float32),
        labels_3d=np.asarray([0], dtype=np.int64),
        proposal_ids=np.asarray([0], dtype=np.int64),
        point_count=np.asarray([2], dtype=np.int32),
        voxel_size=0.01,
        runtime_s=0.0,
        num_input_points=num_input_points,
        checkpoint_sha256=_sha("checkpoint"),
        config_sha256=_sha("config"),
        source_scene_sha256=sha256_file(point_path),
    )
    return write_tr3d_residual_cache(path, cache)


def _row(point_path: Path, point_count: int = 2) -> dict:
    return {
        "point_path": str(point_path),
        "point_count": point_count,
        "fraction": 1.0,
    }


def _load(parent: Path, row: dict, manifest: Path):
    return _load_bound_parent(
        parent,
        row,
        manifest,
        expected_scene_id="scene0001_00",
        expected_prefix_id="p100",
        expected_checkpoint_sha256=_sha("checkpoint"),
        expected_config_sha256=_sha("config"),
    )


def test_parent_is_bound_to_exact_prefix_point_bytes_and_count(
    tmp_path: Path,
) -> None:
    point_path = _write_points(tmp_path / "points" / "prefix.bin")
    parent = _write_parent(tmp_path / "parent.npz", point_path)
    manifest = tmp_path / "manifest.jsonl"
    loaded = _load(parent, _row(point_path), manifest)
    assert loaded.num_input_points == 2
    assert loaded.source_scene_sha256 == sha256_file(point_path)

    np.full((2, 6), 7, dtype=np.float32).tofile(point_path)
    with pytest.raises(ValueError, match="source-scene SHA256 mismatch"):
        _load(parent, _row(point_path), manifest)


def test_point_layout_and_parent_point_count_fail_closed(
    tmp_path: Path,
) -> None:
    point_path = _write_points(tmp_path / "points" / "prefix.bin")
    parent = _write_parent(
        tmp_path / "parent.npz", point_path, num_input_points=1
    )
    manifest = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="num_input_points"):
        _load(parent, _row(point_path), manifest)
    with pytest.raises(ValueError, match="file size"):
        _prefix_point_contract(_row(point_path, 1), manifest)


def test_checkpoint_and_config_locks_fail_closed(tmp_path: Path) -> None:
    point_path = _write_points(tmp_path / "prefix.bin")
    parent = _write_parent(tmp_path / "parent.npz", point_path)
    row = _row(point_path)
    manifest = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        _load_bound_parent(
            parent,
            row,
            manifest,
            expected_scene_id="scene0001_00",
            expected_prefix_id="p100",
            expected_checkpoint_sha256=_sha("wrong"),
            expected_config_sha256=_sha("config"),
        )
    with pytest.raises(ValueError, match="config SHA256 mismatch"):
        _load_bound_parent(
            parent,
            row,
            manifest,
            expected_scene_id="scene0001_00",
            expected_prefix_id="p100",
            expected_checkpoint_sha256=_sha("checkpoint"),
            expected_config_sha256=_sha("wrong"),
        )
