from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import tools.merge_scannet_fastsam_f5_selector_paper100 as merger
import tools.run_scannet_fastsam_f5_selector_paper100 as runner
from boxfusion import fastsam_f5_selector as core


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _hypotheses(point_hash: str) -> dict[str, Any]:
    h0 = {
        "valid": True,
        "q02": [0.0, 0.0, 2.0],
        "q98": [1.0, 1.0, 3.0],
        "center": [0.5, 0.5, 2.5],
        "extent": [1.0, 1.0, 1.0],
        "stored_point_count": 16,
        "points_and_voxel_keys_sha256": point_hash,
        "diagnostics": {
            "applied": True,
            "fallback": False,
            "retained_point_count": 16,
            "source_point_count": 16,
        },
    }
    return {
        "H0": json.loads(json.dumps(h0)),
        "HL": json.loads(json.dumps(h0)),
        "HLG": json.loads(json.dumps(h0)),
    }


def _make_scene(tmp_path: Path, scene_id: str, scene_index: int, frame_count: int = 4) -> dict[str, Any]:
    root = tmp_path / "sealed" / scene_id
    intrinsic_path = root / "intrinsic.txt"
    intrinsic_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        intrinsic_path,
        np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]),
    )
    intrinsic = {"path": str(intrinsic_path.resolve()), "sha256": _sha(intrinsic_path)}
    f2_frames: list[dict[str, Any]] = []
    f4_frames: list[dict[str, Any]] = []
    all_points: list[np.ndarray] = []
    all_keys: list[np.ndarray] = []
    offsets = [0]
    source_ids: list[str] = []
    lineages: list[str] = []
    frame_ids: list[int] = []
    raw_indices: list[int] = []
    ranks: list[int] = []
    candidate_indices: list[int] = []
    for ordinal in range(frame_count):
        frame_id = ordinal * 25
        raw_index = 3
        source_id = f"{scene_id}/frame_{frame_id:06d}/raw_{raw_index:03d}"
        pose_path = root / "pose" / f"{frame_id}.txt"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(pose_path, np.eye(4, dtype=np.float64))
        pose = {"path": str(pose_path.resolve()), "sha256": _sha(pose_path)}
        points = np.asarray(
            [
                [0.10 + 0.05 * (index % 4), 0.10 + 0.05 * ((index // 4) % 4), 2.20]
                for index in range(16)
            ],
            dtype="<f8",
        )
        keys = np.arange(48, dtype="<i8").reshape(16, 3) + (scene_index * 10_000 + ordinal * 100)
        point_hash = hashlib.sha256(points.tobytes() + keys.tobytes()).hexdigest()
        f2_hypotheses = _hypotheses(point_hash)
        mask_hash = hashlib.sha256(source_id.encode("ascii")).hexdigest()
        f2_source = {
            "source_id": source_id,
            "candidate_index": 0,
            "rank": 0,
            "raw_index": raw_index,
            "mask_sha256": mask_hash,
            "points_and_voxel_keys_sha256": point_hash,
            "stored_point_count": 16,
            "hypotheses": f2_hypotheses,
        }
        lineage = hashlib.sha256((source_id + "/f4-lineage").encode("ascii")).hexdigest()
        f4_source = {
            "source_id": source_id,
            "scene_index": scene_index,
            "frame_ordinal": ordinal,
            "frame_id": frame_id,
            "candidate_index": 0,
            "rank": 0,
            "raw_index": raw_index,
            "mask_sha256": mask_hash,
            "points_and_voxel_keys_sha256": point_hash,
            "tight_box_xyxy": [300.0, 220.0, 360.0, 280.0],
            "hypotheses": {
                **json.loads(json.dumps(f2_hypotheses)),
                "HB": {
                    "valid": False,
                    "abstention_reason": "provider_invalid:test",
                    "confidence": 0.90,
                },
            },
            "source_lineage_sha256": lineage,
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
        f4_frames.append(
            {
                "frame_ordinal": ordinal,
                "frame_id": frame_id,
                "successful": True,
                "abstention": None,
                "input": {"pose": pose},
                "sources": [f4_source],
                "runtime": {"replay_composed_ms": 20.0},
            }
        )
        all_points.append(points)
        all_keys.append(keys)
        offsets.append(offsets[-1] + len(points))
        source_ids.append(source_id)
        lineages.append(lineage)
        frame_ids.append(frame_id)
        raw_indices.append(raw_index)
        ranks.append(0)
        candidate_indices.append(0)

    evidence_path = root / "evidence.npz"
    np.savez_compressed(
        evidence_path,
        schema=np.asarray(runner.EXPECTED_F2_EVIDENCE_SCHEMA),
        scene_id=np.asarray(scene_id),
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
    evidence = {"path": str(evidence_path.resolve()), "sha256": _sha(evidence_path)}
    f2 = {
        "schema": runner.EXPECTED_F2_SCENE_SCHEMA,
        "complete": True,
        "scene_id": scene_id,
        "scene_index": scene_index,
        "frames": f2_frames,
    }
    f2_sidecar = _write_json(tmp_path / "f2" / f"{scene_id}.json", f2)
    counts = {
        "keyframe_count": frame_count,
        "successful_frame_count": frame_count,
        "source_count": frame_count,
    }
    f4 = {
        "schema": runner.EXPECTED_F4_SCENE_SCHEMA,
        "protocol_id": runner.EXPECTED_F4_PROTOCOL,
        "complete": True,
        "scene_id": scene_id,
        "scene_index": scene_index,
        "run_signature_sha256": "f" * 64,
        "inputs": {
            "f2_sidecar": f2_sidecar,
            "f2_evidence": evidence,
            "intrinsic": intrinsic,
        },
        "frames": f4_frames,
        "counts": counts,
        "runtime": {"cuda_peak_memory_bytes": 1024},
        "native_output_mutation_count": 0,
    }
    f4["content_sha256"] = runner._canonical_json_sha256(f4)
    f4_sidecar = _write_json(tmp_path / "f4" / f"{scene_id}.json", f4)
    return {
        "scene_id": scene_id,
        "scene_index": scene_index,
        "sidecar": f4_sidecar,
        "counts": counts,
        "source_ids_sha256": runner._canonical_json_sha256(source_ids),
        "source_lineage_sha256": runner._canonical_json_sha256(lineages),
    }


def _prepare_pair(tmp_path: Path) -> dict[str, Any]:
    scenes = ("scene0000_00", "scene0001_00")
    rows = [_make_scene(tmp_path, scene_id, index) for index, scene_id in enumerate(scenes)]
    scene_list = tmp_path / "scene-list.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    f4_receipt = {
        "schema": runner.EXPECTED_F4_MERGE_SCHEMA,
        "protocol_id": runner.EXPECTED_F4_PROTOCOL,
        "complete": True,
        "overall_pass": True,
        "run_signature_sha256": "f" * 64,
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
        "coverage": {"scene_count": 2, "scene_order": list(scenes)},
        "totals": {"keyframe_count": 8, "successful_frame_count": 8, "source_count": 8},
        "scenes": rows,
        "native_output_mutation_count": 0,
    }
    f4_receipt["content_sha256"] = runner._canonical_json_sha256(f4_receipt)
    f4_ref = _write_json(tmp_path / "F4.json", f4_receipt)
    output = tmp_path / "f5"
    manifests = []
    for shard_index in range(2):
        manifests.append(
            runner.run_f5(
                f4_receipt_path=Path(f4_ref["path"]),
                scene_list_path=scene_list,
                output_root=output,
                shard_index=shard_index,
                expected_scene_count=2,
                expected_keyframes=8,
                expected_successful_frames=8,
                expected_sources=8,
            )
        )
    return {
        "shards": tuple(Path(row["manifest_path"]) for row in manifests),
        "output": output,
    }


def _merge(inputs: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    return merger.merge_f5(
        shard_paths=inputs["shards"],
        output_dir=output_dir,
        expected_scene_count=2,
        expected_keyframes=8,
        expected_successful_frames=8,
        expected_sources=8,
        min_selected_hb_sources=0,
        min_selected_hb_scenes=0,
        max_selected_hb_fraction=1.0,
    )


def _reseal_scene_and_shard(shard_path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    scene_ref = shard["scenes"][0]["sidecar"]
    scene_path = Path(scene_ref["path"])
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    mutate(scene)
    scene["content_sha256"] = merger._content_hash_without(scene, "content_sha256")
    scene_path.write_text(json.dumps(scene, sort_keys=True, allow_nan=False), encoding="utf-8")
    shard["scenes"][0]["sidecar"]["sha256"] = _sha(scene_path)
    shard["content_sha256"] = merger._content_hash_without(shard, "content_sha256")
    shard_path.write_text(json.dumps(shard, sort_keys=True, allow_nan=False), encoding="utf-8")


def test_merge_replays_every_source_and_is_create_only(tmp_path: Path) -> None:
    inputs = _prepare_pair(tmp_path)
    receipt = _merge(inputs, tmp_path / "final")
    assert receipt["overall_pass"] is True
    assert receipt["decision"] == "retain_f5_for_one_separately_sealed_evaluation_only"
    assert receipt["totals"]["source_count"] == 8
    assert receipt["selection"]["formal_score"] == 1.0
    assert receipt["selection"]["selected_hlg_count"] == 8
    assert receipt["causality"]["maximum_buffered_frames"] == 3
    assert receipt["causality"]["maximum_sources_per_buffered_frame"] == 1
    assert receipt["runtime"]["f5_cuda_allocated_bytes"] == 0
    assert receipt["inputs"]["merge_source"]["sha256"] == _sha(Path(merger.__file__))
    with pytest.raises(merger.F5MergeError, match="overwrite"):
        _merge(inputs, tmp_path / "final")


def test_merge_rejects_resealed_geometry_and_formal_score_mutations(tmp_path: Path) -> None:
    geometry_inputs = _prepare_pair(tmp_path / "geometry")

    def mutate_geometry(scene: dict[str, Any]) -> None:
        scene["frames"][0]["sources"][0]["selected_geometry"]["q02"][0] += 0.25

    _reseal_scene_and_shard(geometry_inputs["shards"][0], mutate_geometry)
    with pytest.raises(merger.F5MergeError, match="selector row|geometry"):
        _merge(geometry_inputs, tmp_path / "geometry-final")

    score_inputs = _prepare_pair(tmp_path / "score")

    def mutate_score(scene: dict[str, Any]) -> None:
        scene["frames"][0]["sources"][0]["formal_score"] = True

    _reseal_scene_and_shard(score_inputs["shards"][0], mutate_score)
    with pytest.raises(merger.F5MergeError, match="selector row|formal score"):
        _merge(score_inputs, tmp_path / "score-final")


def test_merge_rejects_resealed_unbounded_buffer_and_runtime_mutation(tmp_path: Path) -> None:
    buffer_inputs = _prepare_pair(tmp_path / "buffer")

    def mutate_buffer(scene: dict[str, Any]) -> None:
        fake = {
            "frame_id": 0,
            "frame_ordinal": 0,
            "source_ids": [],
            "result_sha256": [],
        }
        scene["frames"][3]["buffer_before"] = [dict(fake) for _ in range(4)]

    _reseal_scene_and_shard(buffer_inputs["shards"][0], mutate_buffer)
    with pytest.raises(merger.F5MergeError, match="three-frame bound|buffer"):
        _merge(buffer_inputs, tmp_path / "buffer-final")

    runtime_inputs = _prepare_pair(tmp_path / "runtime")

    def mutate_runtime(scene: dict[str, Any]) -> None:
        scene["frames"][0]["runtime"]["f5_cuda_allocated_bytes"] = 1

    _reseal_scene_and_shard(runtime_inputs["shards"][0], mutate_runtime)
    with pytest.raises(merger.F5MergeError, match="CUDA"):
        _merge(runtime_inputs, tmp_path / "runtime-final")


def test_merge_cli_has_no_forbidden_data_argument() -> None:
    options = {option for action in merger._parser()._actions for option in action.option_strings}
    assert not any(
        token in option
        for option in options
        for token in ("gt", "annotation", "oracle", "evaluator", "native", "prediction", "training")
    )


def test_selected_hb_proof_uses_frozen_hb_consistency_evaluated_field() -> None:
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    center = np.asarray([0.5, 0.5, 2.5], dtype=np.float64)
    extent = np.ones(3, dtype=np.float64)
    corners = center[None, :] + signs * 0.5
    h0 = {
        "valid": True,
        "q02": [0.0, 0.0, 2.0],
        "q98": [1.0, 1.0, 3.0],
        "center": center.tolist(),
        "extent": extent.tolist(),
        "stored_point_count": 16,
    }
    hlg = json.loads(json.dumps(h0))
    hlg["diagnostics"] = {
        "applied": True,
        "fallback": False,
        "retained_point_count": 16,
    }
    hb = {
        "valid": True,
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": np.eye(3).tolist(),
        "world_corners": corners.tolist(),
        "camera_depth": 2.5,
        "confidence": 0.8,
    }
    hypotheses = {"H0": h0, "HL": None, "HLG": hlg, "HB": hb}
    state = core.F5SelectorState()
    last_source = None
    last_query = None
    for ordinal, frame_id in enumerate((0, 25, 50)):
        source_id = f"scene0000_00/frame_{frame_id:06d}/raw_003"
        source = core.F5SourceEvidence(
            source_id=source_id,
            frame_id=frame_id,
            frame_ordinal=ordinal,
            rank=0,
            hypotheses=hypotheses,
            points_world=np.tile(center, (16, 1)),
            tight_box_xyxy=np.asarray([320.0, 240.0, 370.0, 290.0]),
            camera_to_world=np.eye(4),
            intrinsic=np.asarray(
                [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]
            ),
            source_lineage_sha256="a" * 64,
        )
        query = state.query_frame(frame_id=frame_id, frame_ordinal=ordinal, sources=(source,))
        last_source, last_query = source, query
        if ordinal < 2:
            state.commit_frame(query)
    assert last_source is not None and last_query is not None
    row = last_query.rows[0]
    assert row["selected_hypothesis"] == "HB"
    assert all(proof["hb_consistency_evaluated"] is True for proof in row["matched_past"])
    f4_source = {
        "source_id": last_source.source_id,
        "source_lineage_sha256": last_source.source_lineage_sha256,
        "hypotheses": hypotheses,
    }
    selected, proof = merger._verify_selected_row(
        row,
        row,
        f4_source,
        buffer_before=[dict(value) for value in last_query.buffer_before],
    )
    assert selected == "HB" and proof is True
