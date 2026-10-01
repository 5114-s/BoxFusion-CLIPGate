from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_scannet_fastsam_f2_paper100 as runner
import test_run_scannet_fastsam_f0_full200 as f0_test


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_f2(tmp_path: Path) -> tuple[dict, Path, list[np.ndarray]]:
    input_root = tmp_path / "inputs"
    input_root.mkdir(parents=True)
    data = f0_test._fixture(input_root, scenes=("scene0000_00",))
    f0_calls: list[np.ndarray] = []
    f0_manifest = f0_test._run(data, f0_calls)
    f0_row = f0_manifest["scenes"][0]
    merged = {
        "schema": runner.EXPECTED_F0_MERGE_SCHEMA,
        "protocol_id": runner.EXPECTED_F0_PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
        "run_signature_sha256": f0_manifest["run_signature_sha256"],
        "coverage": {"scene_order": ["scene0000_00"]},
        "scenes": [
            {
                "scene_id": "scene0000_00",
                "scene_index": 0,
                "sidecar": {
                    "path": f0_row["sidecar_path"],
                    "sha256": f0_row["sidecar_sha256"],
                },
            }
        ],
    }
    f0_receipt = tmp_path / "F0_TEST.json"
    f0_receipt.write_text(json.dumps(merged), encoding="utf-8")
    f2_calls: list[np.ndarray] = []
    manifest = runner.run_f2(
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
    return manifest, f0_receipt, f2_calls


def test_f2_end_to_end_replays_f0_and_seals_raw_points_masks_and_indices(
    tmp_path: Path,
) -> None:
    manifest, _f0_receipt, calls = _prepare_f2(tmp_path)
    assert manifest["complete"]
    assert manifest["contracts"]["shadow_only"]
    assert not manifest["contracts"]["birth_enabled"]
    assert manifest["totals"]["keyframes"] == 2
    assert manifest["totals"]["successful_frames"] == 2
    assert manifest["totals"]["sources"] == 2
    assert manifest["totals"]["identity_verified_sources"] == 2
    assert len(calls) == 2

    row = manifest["scenes"][0]
    sidecar_path = Path(row["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_text())
    assert _hash(sidecar_path) == row["sidecar_sha256"]
    for frame in sidecar["frames"]:
        assert frame["identity"]["exact_equal"]
        runtime = frame["runtime"]
        assert runtime["complete_ms"] == pytest.approx(
            runtime["provider_ms"] + runtime["f0_core_ms"] + runtime["f2_core_ms"],
            abs=1e-12,
        )
        assert runtime["evidence_prepare_ms"] >= 0.0
        source = frame["sources"][0]
        assert source["candidate_index"] == source["rank"] == 0
        assert source["hypotheses"]["H0"]["q02"] == source["f0_world_q02"]
        assert source["hypotheses"]["H0"]["q98"] == source["f0_world_q98"]
        assert set(source["hypotheses"]) == {"H0", "HL", "HLG"}
        assert source["f2_receipt"]["schema"] == runner.f2_core.SCHEMA

    evidence_path = Path(row["evidence_npz_path"])
    assert _hash(evidence_path) == row["evidence_npz_sha256"]
    with np.load(evidence_path, allow_pickle=False) as evidence:
        assert evidence["masks_packbits"].shape == (2, runner.MASK_PACKED_BYTES)
        assert evidence["point_offsets"].shape == (3,)
        assert evidence["points_world"].shape == evidence["voxel_keys"].shape
        assert evidence["hl_index_offsets"].shape == (3,)
        assert evidence["hlg_index_offsets"].shape == (3,)
        for index, source in enumerate(
            [frame["sources"][0] for frame in sidecar["frames"]]
        ):
            assert hashlib.sha256(
                evidence["masks_packbits"][index].tobytes()
            ).hexdigest() == source["mask_sha256"]


def test_real_core_serialization_preserves_original_index_space_and_is_json_safe(
    tmp_path: Path,
) -> None:
    # Dense cluster plus deterministic outliers exercises accepted filtering;
    # retained indices must still address the original input rows.
    grid = np.asarray(
        [(x, y, z) for x in range(5) for y in range(5) for z in range(2)],
        dtype=np.int64,
    )
    outliers = np.asarray([[30, 30, 30], [40, 40, 40]], dtype=np.int64)
    keys = np.concatenate((grid, outliers), axis=0)
    points = keys.astype(np.float64) * 0.02 + 0.001
    raw = np.quantile(points, [0.02, 0.98], axis=0)
    center = raw.mean(axis=0)
    extent = np.maximum(raw[1] - raw[0], 0.02)
    candidate = SimpleNamespace(
        points_world=points,
        voxel_keys=keys,
        world_q02=center - extent * 0.5,
        world_q98=center + extent * 0.5,
        stored_point_count=len(points),
    )
    result = runner._refine_candidate(candidate)
    hypotheses, receipt = runner._serialize_refined_candidate(candidate, result)
    assert np.array_equal(
        points[result.hlg.retained_indices],
        candidate.points_world[result.hlg.retained_indices],
    )
    assert hypotheses["HLG"]["stored_point_count"] == len(
        result.hlg.retained_indices
    )
    output = tmp_path / "roundtrip.json"
    runner._atomic_create_json(
        output, {"hypotheses": hypotheses, "f2_receipt": receipt}
    )
    loaded = json.loads(output.read_text())
    assert loaded["f2_receipt"]["result_sha256"] == result.result_sha256


def test_replay_fails_closed_when_f0_funnel_is_tampered(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir(parents=True)
    data = f0_test._fixture(input_root, scenes=("scene0000_00",))
    f0_manifest = f0_test._run(data, [])
    sidecar_path = Path(f0_manifest["scenes"][0]["sidecar_path"])
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["frames"][0]["funnel"]["masks"][0]["decision"] = "tampered"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    f0_receipt = tmp_path / "F0_TEST.json"
    f0_receipt.write_text(
        json.dumps(
            {
                "schema": runner.EXPECTED_F0_MERGE_SCHEMA,
                "protocol_id": runner.EXPECTED_F0_PROTOCOL_ID,
                "complete": True,
                "overall_pass": True,
                "run_signature_sha256": f0_manifest["run_signature_sha256"],
                "coverage": {"scene_order": ["scene0000_00"]},
                "scenes": [
                    {
                        "scene_id": "scene0000_00",
                        "scene_index": 0,
                        "sidecar": {
                            "path": str(sidecar_path),
                            "sha256": _hash(sidecar_path),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.F2RunnerError, match="replay differs"):
        runner.run_f2(
            schedule_roots=data["roots"],
            scene_root=data["scene_root"],
            scene_list_path=data["scene_list"],
            f0_receipt_path=f0_receipt,
            output_root=tmp_path / "f2",
            device="cpu",
            shard_index=0,
            num_shards=1,
            provider_factory=f0_test._factory(data["checkpoint"], []),
            _expected_scene_count=1,
        )


def test_production_frozen_source_hash_mismatch_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(runner, "EXPECTED_F0_CORE_SHA256", "0" * 64)
    with pytest.raises(runner.F2RunnerError, match="frozen F0 source SHA-256 differs"):
        runner._validate_production_frozen_sources()
