from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import merge_scannet_fastsam_f2_paper100 as f2_merger
import run_scannet_fastsam_f2_paper100 as f2_runner
import run_scannet_fastsam_f3_paper100 as runner
import test_run_scannet_fastsam_f0_full200 as f0_test


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_f3(
    tmp_path: Path,
    *,
    frame_ids=(0, 25, 50),
    pose_valid: tuple[bool, ...] | None = None,
) -> tuple[dict, dict]:
    input_root = tmp_path / "inputs"
    input_root.mkdir(parents=True)
    scene = "scene0000_00"
    data = f0_test._fixture(input_root, scenes=(scene,))
    f0_test._write_scene(
        data["scene_root"],
        scene,
        frames=tuple(frame_ids),
        pose_valid=(
            tuple(True for _ in frame_ids) if pose_valid is None else pose_valid
        ),
    )
    f0_test._write_schedule(
        data["roots"][0],
        scene,
        frames=tuple(frame_ids),
        scene_root=data["scene_root"],
    )
    f0_calls = []
    f0_manifest = f0_test._run(data, f0_calls)
    f0_row = f0_manifest["scenes"][0]
    f0_receipt = tmp_path / "F0_TEST.json"
    f0_receipt.write_text(
        json.dumps(
            {
                "schema": f2_runner.EXPECTED_F0_MERGE_SCHEMA,
                "protocol_id": f2_runner.EXPECTED_F0_PROTOCOL_ID,
                "complete": True,
                "overall_pass": True,
                "run_signature_sha256": f0_manifest["run_signature_sha256"],
                "coverage": {"scene_order": [scene]},
                "scenes": [
                    {
                        "scene_id": scene,
                        "scene_index": 0,
                        "sidecar": {
                            "path": f0_row["sidecar_path"],
                            "sha256": f0_row["sidecar_sha256"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    f2_calls = []
    f2_runner.run_f2(
        schedule_roots=data["roots"],
        scene_root=data["scene_root"],
        scene_list_path=data["scene_list"],
        f0_receipt_path=f0_receipt,
        output_root=tmp_path / "f2",
        device="cpu",
        shard_index=0,
        num_shards=1,
        provider_factory=f0_test._factory(data["checkpoint"], f2_calls),
        _expected_scene_count=1,
    )
    f2_receipt = f2_merger.merge_f2(
        shard_paths=(tmp_path / "f2/shards/shard-000-of-001.json",),
        scene_list_path=data["scene_list"],
        output_dir=tmp_path / "f2/final",
        _expected_scene_count=1,
    )
    f2_receipt_path = tmp_path / "f2/final/F2_FASTSAM_PAPER100.json"
    assert f2_receipt["overall_pass"]
    oracle_path = tmp_path / "F2_ORACLE_TEST.json"
    oracle_path.write_text(
        json.dumps(
            {
                "schema": runner.EXPECTED_F2_ORACLE_SCHEMA,
                "decision": {
                    "authorize_f3_projection_self_validation_shadow": True,
                    "f3_shadow_geometry_input": "F1_H0_only",
                    "retain_f2_geometry_for_f3": False,
                    "authorize_active_birth": False,
                },
            }
        ),
        encoding="utf-8",
    )
    calls_before_f3 = len(f2_calls)
    manifest = runner.run_f3(
        f2_receipt_path=f2_receipt_path,
        f2_oracle_path=oracle_path,
        output_root=tmp_path / "f3",
        shard_index=0,
        num_shards=1,
        _expected_scene_count=1,
    )
    assert len(f2_calls) == calls_before_f3
    return manifest, {
        "data": data,
        "f2_receipt": f2_receipt_path,
        "f2_oracle": oracle_path,
    }


def test_f3_replays_sealed_h0_without_fastsam_and_seals_causal_tracks(
    tmp_path: Path,
) -> None:
    manifest, _inputs = _prepare_f3(tmp_path)
    assert manifest["complete"]
    assert manifest["contracts"]["fastsam_rerun"] is False
    assert manifest["contracts"]["birth_enabled"] is False
    assert manifest["totals"]["keyframe_count"] == 3
    assert manifest["totals"]["source_count"] == 3

    row = manifest["scenes"][0]
    sidecar_path = Path(row["sidecar_path"])
    assert _hash(sidecar_path) == row["sidecar_sha256"]
    scene = json.loads(sidecar_path.read_text())
    assert scene["causality"]["overall_pass"]
    assert scene["counts"]["source_count"] == 3
    assert scene["counts"]["identity_verified_source_count"] == 3
    assert [frame["ordinal"] for frame in scene["frames"]] == [0, 1, 2]
    assert all(
        frame["max_logical_accessed_ordinal"] <= frame["ordinal"]
        for frame in scene["frames"]
    )
    expected_sources = [
        source
        for frame in scene["frames"]
        for source in frame["source_ids"]
    ]
    assigned_sources = [
        assignment["source_id"]
        for frame in scene["frames"]
        for assignment in frame["assignments"]
    ]
    track_sources = [
        source for track in scene["tracks"] for source in track["source_ids"]
    ]
    assert assigned_sources == expected_sources
    assert sorted(track_sources) == sorted(expected_sources)
    assert len(track_sources) == len(set(track_sources))
    confirmed = [track for track in scene["tracks"] if track["confirmed"]]
    assert confirmed
    for track in scene["tracks"]:
        assert tuple(track["hypotheses"]) == ("B", "C")
        chosen = track["selector"]["chosen"]
        if chosen is not None:
            for key in ("q02", "q98", "center", "extent", "score"):
                assert track["selector"][key] == track["hypotheses"][chosen][key]


def test_fast_bounded_5cm_voxels_are_bit_exact_to_public_core_normalization() -> None:
    rng = np.random.default_rng(20260829)
    points = rng.normal(0.0, 2.0, size=(4096, 3)).astype(np.float64)
    boundaries = np.asarray(
        [
            [-0.10000000000000002, -0.05, -0.0],
            [0.0, 0.049999999999999996, 0.05],
            [0.09999999999999999, 0.1, 0.15000000000000002],
        ],
        dtype=np.float64,
    )
    points[: len(boundaries)] = boundaries
    raw_keys = np.floor(points / runner.F3_VOXEL_SIZE_M).astype(np.int64)
    fast_keys = runner._bounded_f3_voxel_keys(points)
    packed = np.zeros(runner.MASK_PACKED_BYTES, dtype=np.uint8)
    common = {
        "source_id": "scene0000_00/frame_000000/raw_000",
        "frame_id": 0,
        "frame_ordinal": 0,
        "confidence": 0.5,
        "world_q02": [-1.0, -1.0, 1.0],
        "world_q98": [1.0, 1.0, 2.0],
        "camera_to_world": np.eye(4, dtype=np.float64),
        "intrinsics": np.asarray(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "mask_packbits": packed,
    }
    reference = runner.f3_core.make_observation(voxel_keys=raw_keys, **common)
    optimized = runner.f3_core.make_observation(voxel_keys=fast_keys, **common)
    assert np.array_equal(optimized.voxel_keys, reference.voxel_keys)
    assert optimized.observation_sha256 == reference.observation_sha256


def test_f3_fails_closed_when_a_sealed_pose_changes(tmp_path: Path) -> None:
    manifest, inputs = _prepare_f3(tmp_path)
    sidecar = json.loads(Path(manifest["scenes"][0]["sidecar_path"]).read_text())
    f0_scene = json.loads(Path(sidecar["inputs"]["f0_sidecar"]["path"]).read_text())
    pose = Path(f0_scene["frames"][0]["inputs"]["pose_path"])
    pose.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(runner.F3RunnerError, match="pose rehash differs"):
        runner.run_f3(
            f2_receipt_path=inputs["f2_receipt"],
            f2_oracle_path=inputs["f2_oracle"],
            output_root=tmp_path / "changed-f3",
            shard_index=0,
            num_shards=1,
            _expected_scene_count=1,
        )


def test_f3_failed_pose_frame_advances_causal_tracker_without_sources(
    tmp_path: Path,
) -> None:
    manifest, _inputs = _prepare_f3(
        tmp_path, pose_valid=(True, True, False)
    )
    scene = json.loads(Path(manifest["scenes"][0]["sidecar_path"]).read_text())
    failed = scene["frames"][-1]
    assert failed["successful"] is False
    assert failed["source_ids"] == []
    assert failed["assignments"] == []
    assert failed["max_logical_accessed_ordinal"] == failed["ordinal"]
    assert scene["counts"]["keyframe_count"] == 3
    assert scene["counts"]["successful_frame_count"] == 2
    assert scene["counts"]["source_count"] == 2
    assert scene["causality"]["overall_pass"]


def test_f3_output_is_create_only_and_resume_is_exact(tmp_path: Path) -> None:
    _manifest, inputs = _prepare_f3(tmp_path)
    kwargs = {
        "f2_receipt_path": inputs["f2_receipt"],
        "f2_oracle_path": inputs["f2_oracle"],
        "output_root": tmp_path / "f3",
        "shard_index": 0,
        "num_shards": 1,
        "_expected_scene_count": 1,
    }
    with pytest.raises(runner.F3RunnerError, match="refusing to overwrite"):
        runner.run_f3(**kwargs)
    resumed = runner.run_f3(**kwargs, resume=True)
    assert resumed["complete"]

def test_production_shard_census_keys_match_sealed_scene_counts_schema() -> None:
    expected_keys = {
        "keyframe_count",
        "successful_frame_count",
        "source_count",
    }
    assert set(runner.EXPECTED_SHARD_COUNTS) == {0, 1}
    assert all(
        set(counts) == expected_keys
        for counts in runner.EXPECTED_SHARD_COUNTS.values()
    )

