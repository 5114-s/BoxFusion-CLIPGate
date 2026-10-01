from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import tools.run_scannet_fastsam_f5_selector_paper100 as runner


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _hash(path)}


def _seal(path: Path, data: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path.resolve()), "sha256": _hash(path)}


def _hypotheses() -> dict:
    h0 = {
        "valid": True,
        "q02": [0.0, 0.0, 2.0],
        "q98": [1.0, 1.0, 3.0],
        "center": [0.5, 0.5, 2.5],
        "extent": [1.0, 1.0, 1.0],
        "stored_point_count": 16,
        "diagnostics": {
            "applied": True,
            "fallback": False,
            "retained_point_count": 16,
            "source_point_count": 16,
        },
    }
    result = {"H0": h0}
    for name in ("HL", "HLG"):
        result[name] = json.loads(json.dumps(h0))
    return result


def _prepare(tmp_path: Path, frame_count: int = 4) -> dict[str, Path]:
    scene = "scene0000_00"
    intrinsic_path = tmp_path / "sealed/intrinsic.txt"
    intrinsic_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsic = np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])
    np.savetxt(intrinsic_path, intrinsic)
    intrinsic_ref = {"path": str(intrinsic_path.resolve()), "sha256": _hash(intrinsic_path)}

    f2_frames = []
    f4_frames = []
    all_points = []
    all_keys = []
    offsets = [0]
    source_ids = []
    frame_ids = []
    raw_indices = []
    ranks = []
    candidate_indices = []
    for ordinal in range(frame_count):
        frame_id = ordinal * 25
        raw_index = 3
        source_id = f"{scene}/frame_{frame_id:06d}/raw_{raw_index:03d}"
        pose_path = tmp_path / f"sealed/pose/{frame_id}.txt"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(pose_path, np.eye(4))
        pose_ref = {"path": str(pose_path.resolve()), "sha256": _hash(pose_path)}
        points = np.asarray(
            [[0.1 + 0.05 * (index % 4), 0.1 + 0.05 * ((index // 4) % 4), 2.2] for index in range(16)],
            dtype="<f8",
        )
        keys = np.arange(48, dtype="<i8").reshape(16, 3) + ordinal * 100
        digest = hashlib.sha256(points.tobytes() + keys.tobytes()).hexdigest()
        base = _hypotheses()
        for row in base.values():
            row["points_and_voxel_keys_sha256"] = digest
        f2_source = {
            "source_id": source_id,
            "candidate_index": 0,
            "rank": 0,
            "raw_index": raw_index,
            "mask_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
            "points_and_voxel_keys_sha256": digest,
            "stored_point_count": 16,
            "hypotheses": base,
        }
        f2_frames.append(
            {
                "frame_ordinal": ordinal,
                "frame_id": frame_id,
                "successful": True,
                "runtime": {"complete_ms": 10.0},
                "sources": [f2_source],
            }
        )
        f4_source = {
            "source_id": source_id,
            "scene_index": 0,
            "frame_ordinal": ordinal,
            "frame_id": frame_id,
            "candidate_index": 0,
            "rank": 0,
            "raw_index": raw_index,
            "mask_sha256": f2_source["mask_sha256"],
            "points_and_voxel_keys_sha256": digest,
            "tight_box_xyxy": [300.0, 220.0, 360.0, 280.0],
            "hypotheses": {
                **json.loads(json.dumps(base)),
                "HB": {"valid": False, "abstention_reason": "provider_invalid", "confidence": 0.9},
            },
            "source_lineage_sha256": hashlib.sha256((source_id + "lineage").encode()).hexdigest(),
        }
        f4_frames.append(
            {
                "frame_ordinal": ordinal,
                "frame_id": frame_id,
                "successful": True,
                "abstention": None,
                "input": {"pose": pose_ref},
                "sources": [f4_source],
                "runtime": {"replay_composed_ms": 20.0},
            }
        )
        all_points.append(points)
        all_keys.append(keys)
        offsets.append(offsets[-1] + len(points))
        source_ids.append(source_id)
        frame_ids.append(frame_id)
        raw_indices.append(raw_index)
        ranks.append(0)
        candidate_indices.append(0)

    evidence_path = tmp_path / "sealed/evidence.npz"
    np.savez_compressed(
        evidence_path,
        schema=np.asarray(runner.EXPECTED_F2_EVIDENCE_SCHEMA),
        scene_id=np.asarray(scene),
        mask_shape=np.asarray([480, 640], dtype=np.int64),
        mask_bitorder=np.asarray("little"),
        source_ids=np.asarray(source_ids),
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        raw_indices=np.asarray(raw_indices, dtype=np.int64),
        ranks=np.asarray(ranks, dtype=np.int64),
        candidate_indices=np.asarray(candidate_indices, dtype=np.int64),
        masks_packbits=np.zeros((frame_count, runner.MASK_PACKED_BYTES), dtype=np.uint8),
        point_offsets=np.asarray(offsets, dtype=np.int64),
        points_world=np.concatenate(all_points).astype("<f8"),
        voxel_keys=np.concatenate(all_keys).astype("<i8"),
        hl_index_offsets=np.asarray(offsets, dtype=np.int64),
        hl_retained_indices=np.arange(offsets[-1], dtype=np.int64),
        hlg_index_offsets=np.asarray(offsets, dtype=np.int64),
        hlg_retained_indices=np.arange(offsets[-1], dtype=np.int64),
    )
    evidence_ref = {"path": str(evidence_path.resolve()), "sha256": _hash(evidence_path)}
    f2 = {
        "schema": runner.EXPECTED_F2_SCENE_SCHEMA,
        "complete": True,
        "scene_id": scene,
        "scene_index": 0,
        "frames": f2_frames,
    }
    f2_ref = _write_json(tmp_path / "sealed/f2.json", f2)
    f4 = {
        "schema": runner.EXPECTED_F4_SCENE_SCHEMA,
        "protocol_id": runner.EXPECTED_F4_PROTOCOL,
        "complete": True,
        "scene_id": scene,
        "scene_index": 0,
        "run_signature_sha256": "a" * 64,
        "inputs": {"f2_sidecar": f2_ref, "f2_evidence": evidence_ref, "intrinsic": intrinsic_ref},
        "frames": f4_frames,
        "counts": {"keyframe_count": frame_count, "successful_frame_count": frame_count, "source_count": frame_count},
        "runtime": {"cuda_peak_memory_bytes": 1024},
        "native_output_mutation_count": 0,
    }
    f4["content_sha256"] = runner._canonical_json_sha256(f4)
    f4_ref = _write_json(tmp_path / "sealed/f4.json", f4)
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    merge = {
        "schema": runner.EXPECTED_F4_MERGE_SCHEMA,
        "protocol_id": runner.EXPECTED_F4_PROTOCOL,
        "complete": True,
        "overall_pass": True,
        "run_signature_sha256": "a" * 64,
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "native_output_mutation": False,
            "gt_access": False,
            "prediction_access": False,
            "evaluator_access": False,
            "future_frame_access": False,
            "training": False,
            "online_learning": False,
        },
        "coverage": {"scene_count": 1, "scene_order": [scene]},
        "totals": {"keyframe_count": frame_count, "successful_frame_count": frame_count, "source_count": frame_count},
        "scenes": [{"scene_id": scene, "scene_index": 0, "sidecar": f4_ref}],
        "native_output_mutation_count": 0,
    }
    merge["content_sha256"] = runner._canonical_json_sha256(merge)
    merge_ref = _write_json(tmp_path / "sealed/f4-merge.json", merge)
    return {
        "receipt": Path(merge_ref["path"]),
        "scene_list": scene_list,
        "output": tmp_path / "f5",
        "evidence": evidence_path,
    }


def test_runner_current_offsets_query_commit_and_replays(tmp_path: Path) -> None:
    inputs = _prepare(tmp_path)
    manifest = runner.run_f5(
        f4_receipt_path=inputs["receipt"],
        scene_list_path=inputs["scene_list"],
        output_root=inputs["output"],
        shard_index=0,
        expected_scene_count=1,
        expected_keyframes=4,
        expected_successful_frames=4,
        expected_sources=4,
    )
    assert manifest["totals"]["source_count"] == 4
    assert manifest["determinism"]["overall_pass"]
    stored_manifest = json.loads(Path(manifest["manifest_path"]).read_text())
    assert stored_manifest["content_sha256"] == runner._content_hash_without(
        stored_manifest, "content_sha256"
    )
    scene = json.loads(Path(manifest["scenes"][0]["sidecar"]["path"]).read_text())
    assert scene["prefix_replay"]["passed"]
    assert scene["prefix_replay"]["successful_frame_count"] == 2
    assert scene["determinism"]["passed"]
    assert all(frame["query"]["query_before_commit"] for frame in scene["frames"])
    assert all(frame["query"]["token"] == frame["commit"]["token"] for frame in scene["frames"])
    assert all(frame["runtime"]["f5_cuda_allocated_bytes"] == 0 for frame in scene["frames"])
    assert all(frame["sources"][0]["selected_hypothesis"] == "HLG" for frame in scene["frames"])


def test_runner_is_create_only_and_evidence_offsets_fail_closed(tmp_path: Path) -> None:
    inputs = _prepare(tmp_path)
    kwargs = dict(
        f4_receipt_path=inputs["receipt"],
        scene_list_path=inputs["scene_list"],
        output_root=inputs["output"],
        shard_index=0,
        expected_scene_count=1,
        expected_keyframes=4,
        expected_successful_frames=4,
        expected_sources=4,
    )
    runner.run_f5(**kwargs)
    with pytest.raises(runner.F5RunnerError, match="overwrite"):
        runner.run_f5(**kwargs)

    accessor = runner._EvidenceAccessor(inputs["evidence"], _hash(inputs["evidence"]), "scene0000_00", 4)
    try:
        with pytest.raises(runner.F5RunnerError, match="offset differs"):
            accessor.expose_current(
                scene="scene0000_00",
                frame_id=999,
                frame_ordinal=0,
                sources=[{"source_id": "scene0000_00/frame_000999/raw_003", "raw_index": 3, "rank": 0, "candidate_index": 0}],
            )
    finally:
        accessor.close()


def test_cli_has_no_forbidden_data_inputs() -> None:
    options = {option for action in runner._parser()._actions for option in action.option_strings}
    assert not any(
        token in option
        for option in options
        for token in ("gt", "annotation", "oracle", "evaluator", "native", "prediction")
    )
