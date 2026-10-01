from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_inference import (
    AlignedTR3DOutput,
    SyntheticTR3DAdapter,
    export_inference_inputs,
    load_axis_alignment_file,
    load_inference_manifest,
)
from boxfusion.tr3d_residual_cache import (
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _points(path: Path) -> np.ndarray:
    points = np.asarray(
        [
            [-0.4, -0.4, -0.4, 10, 20, 30],
            [0.4, 0.4, 0.4, 40, 50, 60],
            [3.0, 3.0, 3.0, 70, 80, 90],
        ],
        dtype=np.float32,
    )
    points.tofile(path)
    return points


class RecordingAdapter:
    def __init__(self) -> None:
        self.points = None
        self.axis = None

    def infer(self, points_unaligned, axis_align_matrix):
        self.points = np.array(points_unaligned, copy=True)
        self.axis = np.array(axis_align_matrix, copy=True)
        return AlignedTR3DOutput(
            boxes_aligned=np.asarray(
                [
                    [1, 2, 3, 1, 1, 1],
                    [8, 8, 8, 2, 2, 2],
                ],
                dtype=np.float32,
            ),
            scores_3d=np.asarray([0.9, 0.05], dtype=np.float32),
            labels_3d=np.asarray([0, 0], dtype=np.int64),
            runtime_s=0.25,
        )


def test_manifest_to_official_adapter_cache_is_immutable_and_resumable(
    tmp_path: Path,
) -> None:
    point_path = tmp_path / "scene0001_00.bin"
    points = _points(point_path)
    axis = np.asarray(
        [
            [0, -1, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "scene_id": "scene0001_00",
                "tag": "p050",
                "fraction": 0.5,
                "point_path": str(point_path),
                "coordinate_frame": "world_unaligned",
                "axis_align_matrix": axis.tolist(),
                "point_count": len(points),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_inference_manifest(manifest, prefix_ids=["p050"])
    adapter = RecordingAdapter()
    checkpoint_sha = _sha(b"checkpoint")
    config_sha = _sha(b"config")
    cache_root = tmp_path / "cache"
    report = export_inference_inputs(
        inputs=items,
        adapter=adapter,
        cache_root=cache_root,
        checkpoint_sha256=checkpoint_sha,
        config_sha256=config_sha,
        score_threshold=0.1,
        max_proposals=10,
        resume=False,
    )
    assert report["sample_count"] == 1
    assert report["proposal_count"] == 1
    assert np.array_equal(adapter.points, points)
    assert np.array_equal(adapter.axis, axis)
    cache_path = tr3d_residual_cache_path(
        cache_root, "scene0001_00", "p050"
    )
    cache = load_tr3d_residual_cache(
        cache_path,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_config_sha256=config_sha,
        expected_source_scene_sha256=hashlib.sha256(
            point_path.read_bytes()
        ).hexdigest(),
    )
    assert cache.proposal_count == 1
    assert cache.prefix_fraction == 0.5
    assert cache.runtime_s == 0.25
    assert cache.num_input_points == 3
    frozen_bytes = cache_path.read_bytes()

    second = export_inference_inputs(
        inputs=items,
        adapter=RecordingAdapter(),
        cache_root=cache_root,
        checkpoint_sha256=checkpoint_sha,
        config_sha256=config_sha,
        score_threshold=0.1,
        max_proposals=10,
        resume=True,
    )
    assert second["rows"][0]["status"] == "verified_existing"
    assert cache_path.read_bytes() == frozen_bytes
    with pytest.raises(FileExistsError, match="immutable"):
        export_inference_inputs(
            inputs=items,
            adapter=RecordingAdapter(),
            cache_root=cache_root,
            checkpoint_sha256=checkpoint_sha,
            config_sha256=config_sha,
            resume=False,
        )


def test_synthetic_dry_run_does_not_create_cache(tmp_path: Path) -> None:
    point_path = tmp_path / "scene0002_00.bin"
    _points(point_path)
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "scene_id": "scene0002_00",
                "point_path": str(point_path),
                "coordinate_frame": "world_unaligned",
                "axis_align_matrix": np.eye(4).tolist(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = export_inference_inputs(
        inputs=load_inference_manifest(manifest),
        adapter=SyntheticTR3DAdapter(),
        cache_root=tmp_path / "cache",
        checkpoint_sha256=_sha(b"checkpoint"),
        config_sha256=_sha(b"config"),
        write_cache=False,
    )
    assert report["rows"][0]["status"] == "dry_run_not_written"
    assert not (tmp_path / "cache").exists()


def test_manifest_sharding_is_deterministic(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        point_path = tmp_path / f"scene000{index}_00.bin"
        _points(point_path)
        rows.append(
            {
                "scene_id": f"scene000{index}_00",
                "point_path": str(point_path),
                "axis_align_matrix": np.eye(4).tolist(),
            }
        )
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )
    shard0 = load_inference_manifest(
        manifest, shard_index=0, num_shards=2
    )
    shard1 = load_inference_manifest(
        manifest, shard_index=1, num_shards=2
    )
    assert [item.scene_id for item in shard0] == [
        "scene0000_00",
        "scene0002_00",
    ]
    assert [item.scene_id for item in shard1] == [
        "scene0001_00",
        "scene0003_00",
    ]


def test_scannet_metadata_axis_alignment_is_accepted(tmp_path: Path) -> None:
    metadata = tmp_path / "scene0001_00.txt"
    metadata.write_text(
        "sceneType = office\n"
        "axisAlignment = 1 0 0 2 0 1 0 3 0 0 1 4 0 0 0 1\n",
        encoding="utf-8",
    )
    matrix = load_axis_alignment_file(metadata)
    assert np.array_equal(matrix[:3, 3], [2, 3, 4])
