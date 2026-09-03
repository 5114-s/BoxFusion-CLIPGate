from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

import boxfusion
import tools.run_scannet_fastsam_f6_mvdc_paper100 as runner


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _hash(path)}


def _hypotheses(point_count: int) -> dict:
    h0 = {
        "valid": True,
        "q02": [0.0, 0.0, 2.0],
        "q98": [1.0, 1.0, 3.0],
        "center": [0.5, 0.5, 2.5],
        "extent": [1.0, 1.0, 1.0],
        "stored_point_count": point_count,
        "diagnostics": {
            "applied": True,
            "fallback": False,
            "retained_point_count": point_count,
            "source_point_count": point_count,
        },
    }
    return {name: json.loads(json.dumps(h0)) for name in ("H0", "HL", "HLG")}


def _prepare(tmp_path: Path, frame_count: int = 4) -> dict[str, Path]:
    scene = "scene0000_00"
    intrinsic_path = tmp_path / "sealed/intrinsic.txt"
    intrinsic_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        intrinsic_path,
        np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]]),
    )
    intrinsic_ref = {"path": str(intrinsic_path.resolve()), "sha256": _hash(intrinsic_path)}

    mask_bool = np.zeros((480, 640), dtype=np.uint8)
    mask_bool[220:280, 300:360] = 1
    packed_mask = np.packbits(mask_bool.reshape(-1), bitorder="little")
    mask_sha = hashlib.sha256(packed_mask.tobytes()).hexdigest()
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
            [
                [0.1 + 0.05 * (index % 4), 0.1 + 0.05 * ((index // 4) % 4), 2.2]
                for index in range(16)
            ],
            dtype="<f8",
        )
        keys = np.arange(48, dtype="<i8").reshape(16, 3) + ordinal * 100
        point_sha = hashlib.sha256(points.tobytes() + keys.tobytes()).hexdigest()
        base = _hypotheses(len(points))
        f2_source = {
            "source_id": source_id,
            "candidate_index": 0,
            "rank": 0,
            "raw_index": raw_index,
            "mask_sha256": mask_sha,
            "points_and_voxel_keys_sha256": point_sha,
            "stored_point_count": len(points),
            "hypotheses": base,
        }
        f2_frames.append({
            "frame_ordinal": ordinal,
            "frame_id": frame_id,
            "successful": True,
            "runtime": {"complete_ms": 10.0},
            "sources": [f2_source],
        })
        lineage = hashlib.sha256((source_id + "lineage").encode()).hexdigest()
        f4_source = {
            "source_id": source_id,
            "scene_index": 0,
            "frame_ordinal": ordinal,
            "frame_id": frame_id,
            "candidate_index": 0,
            "rank": 0,
            "raw_index": raw_index,
            "mask_sha256": mask_sha,
            "points_and_voxel_keys_sha256": point_sha,
            "tight_box_xyxy": [300.0, 220.0, 360.0, 280.0],
            "hypotheses": {
                **json.loads(json.dumps(base)),
                "HB": {"valid": False, "abstention_reason": "provider_invalid", "confidence": 0.9},
            },
            "source_lineage_sha256": lineage,
        }
        f4_frames.append({
            "frame_ordinal": ordinal,
            "frame_id": frame_id,
            "successful": True,
            "abstention": None,
            "input": {"pose": pose_ref},
            "sources": [f4_source],
            "runtime": {"replay_composed_ms": 20.0},
        })
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
        masks_packbits=np.repeat(packed_mask[None, :], frame_count, axis=0),
        point_offsets=np.asarray(offsets, dtype=np.int64),
        points_world=np.concatenate(all_points).astype("<f8"),
        voxel_keys=np.concatenate(all_keys).astype("<i8"),
        hl_index_offsets=np.asarray(offsets, dtype=np.int64),
        hl_retained_indices=np.arange(offsets[-1], dtype=np.int64),
        hlg_index_offsets=np.asarray(offsets, dtype=np.int64),
        hlg_retained_indices=np.arange(offsets[-1], dtype=np.int64),
    )
    evidence_ref = {"path": str(evidence_path.resolve()), "sha256": _hash(evidence_path)}
    f2_ref = _write_json(
        tmp_path / "sealed/f2.json",
        {
            "schema": runner.EXPECTED_F2_SCENE_SCHEMA,
            "complete": True,
            "scene_id": scene,
            "scene_index": 0,
            "frames": f2_frames,
        },
    )
    f4 = {
        "schema": runner.EXPECTED_F4_SCENE_SCHEMA,
        "protocol_id": runner.EXPECTED_F4_PROTOCOL,
        "complete": True,
        "scene_id": scene,
        "scene_index": 0,
        "run_signature_sha256": "a" * 64,
        "inputs": {"f2_sidecar": f2_ref, "f2_evidence": evidence_ref, "intrinsic": intrinsic_ref},
        "frames": f4_frames,
        "counts": {
            "keyframe_count": frame_count,
            "successful_frame_count": frame_count,
            "source_count": frame_count,
        },
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
        "totals": {
            "keyframe_count": frame_count,
            "successful_frame_count": frame_count,
            "source_count": frame_count,
        },
        "scenes": [{"scene_id": scene, "scene_index": 0, "sidecar": f4_ref}],
        "native_output_mutation_count": 0,
    }
    merge["content_sha256"] = runner._canonical_json_sha256(merge)
    merge_ref = _write_json(tmp_path / "sealed/f4-merge.json", merge)
    dummy_core = tmp_path / "sources/fake_core.py"
    dummy_core.parent.mkdir(parents=True, exist_ok=True)
    dummy_core.write_text("# sealed fake core source for runner test\n", encoding="ascii")
    return {
        "receipt": Path(merge_ref["path"]),
        "scene_list": scene_list,
        "output": tmp_path / "f6",
        "evidence": evidence_path,
        "dummy_core": dummy_core,
    }


@dataclass(frozen=True)
class _FakeEvidence:
    source_id: str
    frame_id: int
    frame_ordinal: int
    rank: int
    hypotheses: dict
    points_world: np.ndarray
    mask_packbits: np.ndarray
    tight_box_xyxy: object
    camera_to_world: np.ndarray
    intrinsic: np.ndarray
    source_lineage_sha256: str


def _result_hash(row: dict) -> str:
    payload = dict(row)
    payload.pop("result_sha256", None)
    return runner._canonical_json_sha256(payload)


class _FakeState:
    def __init__(self) -> None:
        self._buffer = []
        self._pending = None
        self.raw_array_payload_bytes = 0

    def query_frame(self, *, frame_id: int, frame_ordinal: int, sources: tuple) -> SimpleNamespace:
        before = tuple(self._buffer[-3:])
        rows = []
        for source in sources:
            row = {
                "schema": runner.CORE_SCHEMA,
                "protocol_id": runner.PROTOCOL_ID,
                "source_id": source.source_id,
                "source_lineage_sha256": source.source_lineage_sha256,
                "frame_id": frame_id,
                "frame_ordinal": frame_ordinal,
                "rank": source.rank,
                "base_hypothesis": "HLG",
                "selected_hypothesis": "HLG",
                "switched_from_base": False,
                "selected_geometry_sha256": "b" * 64,
                "formal_score": 1.0,
                "matched_past_frame_count": min(2, len(before)),
                "maximum_lookahead_frames": 0,
                "observer_only": True,
                "birth_applied": False,
                "native_output_mutation_applied": False,
            }
            row["result_sha256"] = _result_hash(row)
            rows.append(MappingProxyType(row))
        token = runner._canonical_json_sha256(
            {"frame_id": frame_id, "frame_ordinal": frame_ordinal, "rows": [row["result_sha256"] for row in rows]}
        )
        self._pending = SimpleNamespace(
            frame_id=frame_id,
            frame_ordinal=frame_ordinal,
            rows=tuple(rows),
            buffer_before=before,
            maximum_accessed_frame_ordinal=max(
                (int(row["frame_ordinal"]) for row in before), default=-1
            ),
            state_raw_array_payload_bytes=self.raw_array_payload_bytes,
            audit_hash_ns=1_000,
            audit_serialization_ns=2_000,
            token=token,
        )
        return self._pending

    def commit_frame(self, query: SimpleNamespace) -> SimpleNamespace:
        assert query is self._pending
        source_ids = [row["source_id"] for row in query.rows]
        self._buffer = (self._buffer + [{
            "frame_id": query.frame_id,
            "frame_ordinal": query.frame_ordinal,
            "source_ids": source_ids,
            "result_sha256": [row["result_sha256"] for row in query.rows],
        }])[-3:]
        self.raw_array_payload_bytes = len(source_ids) * 100
        self._pending = None
        return SimpleNamespace(
            source_count=len(source_ids),
            buffer_after=tuple(self._buffer),
            state_raw_array_payload_bytes=self.raw_array_payload_bytes,
            audit_hash_ns=3_000,
            audit_serialization_ns=4_000,
            token=query.token,
        )


def _install_fake_core(monkeypatch: pytest.MonkeyPatch, inputs: dict[str, Path]) -> None:
    module = SimpleNamespace(
        PROTOCOL_ID=runner.PROTOCOL_ID,
        SCHEMA=runner.CORE_SCHEMA,
        POLICY={"maximum_lookahead_frames": 0},
        F6SourceEvidence=_FakeEvidence,
        F6SelectorState=_FakeState,
        canonical_result_sha256=_result_hash,
    )
    monkeypatch.setitem(sys.modules, "boxfusion.fastsam_f6_mvdc_selector", module)
    monkeypatch.setattr(boxfusion, "fastsam_f6_mvdc_selector", module, raising=False)
    receipts = {
        "runner": {"path": str(Path(runner.__file__).resolve()), "sha256": _hash(Path(runner.__file__))},
        "core": {"path": str(inputs["dummy_core"].resolve()), "sha256": _hash(inputs["dummy_core"])},
        "protocol": {"path": str(runner.PROTOCOL_PATH.resolve()), "sha256": _hash(runner.PROTOCOL_PATH)},
    }
    monkeypatch.setattr(runner, "_source_receipts", lambda: receipts)


def test_runner_join_replays_memory_and_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _prepare(tmp_path)
    _install_fake_core(monkeypatch, inputs)
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
    manifest = runner.run_f6(**kwargs)
    assert manifest["totals"]["source_count"] == 4
    assert manifest["totals"]["switch_count"] == 0
    assert manifest["totals"]["multiview_evaluated_source_count"] == 2
    assert manifest["determinism"]["overall_pass"]
    assert manifest["bounded_state"]["overall_pass"]
    stored_manifest = json.loads(Path(manifest["manifest_path"]).read_text())
    assert stored_manifest["content_sha256"] == runner._content_hash_without(
        stored_manifest, "content_sha256"
    )
    scene = json.loads(Path(manifest["scenes"][0]["sidecar"]["path"]).read_text())
    assert scene["prefix_replay"]["successful_frame_count"] == 2
    assert scene["determinism"]["passed"]
    assert all(frame["query"]["token"] == frame["commit"]["token"] for frame in scene["frames"])
    assert all(frame["runtime"]["f6_cuda_allocated_bytes"] == 0 for frame in scene["frames"])
    for frame in scene["frames"]:
        runtime = frame["runtime"]
        assert runtime["f6_audit_hash_excluded_ms"] == pytest.approx(0.004)
        assert runtime["f6_audit_serialization_excluded_ms"] == pytest.approx(0.006)
        assert runtime["f6_audit_total_excluded_ms"] == pytest.approx(0.010)
        assert runtime["f6_incremental_gross_ms"] == pytest.approx(
            runtime["f6_incremental_formal_ms"]
            + runtime["f6_audit_total_excluded_ms"]
        )
        assert runtime["f6_incremental_ms"] == runtime["f6_incremental_formal_ms"]
        assert runtime["replay_composed_ms"] == pytest.approx(
            20.0 + runtime["f6_incremental_formal_ms"]
        )
    with pytest.raises(runner.F6RunnerError, match="overwrite"):
        runner.run_f6(**kwargs)


def test_accessor_authenticates_mask_and_current_offsets(tmp_path: Path) -> None:
    inputs = _prepare(tmp_path)
    accessor = runner._EvidenceAccessor(
        inputs["evidence"], _hash(inputs["evidence"]), "scene0000_00", 4
    )
    try:
        with pytest.raises(runner.F6RunnerError, match="offset differs"):
            accessor.expose_current(
                scene="scene0000_00",
                frame_id=999,
                frame_ordinal=0,
                sources=[{
                    "source_id": "scene0000_00/frame_000999/raw_003",
                    "raw_index": 3,
                    "rank": 0,
                    "candidate_index": 0,
                }],
            )
    finally:
        accessor.close()


def test_runner_real_core_api_smoke(tmp_path: Path) -> None:
    inputs = _prepare(tmp_path)
    manifest = runner.run_f6(
        f4_receipt_path=inputs["receipt"],
        scene_list_path=inputs["scene_list"],
        output_root=tmp_path / "f6-real",
        shard_index=0,
        expected_scene_count=1,
        expected_keyframes=4,
        expected_successful_frames=4,
        expected_sources=4,
    )
    assert manifest["totals"]["source_count"] == 4
    assert manifest["bounded_state"]["overall_pass"]
    assert manifest["determinism"]["overall_pass"]


def test_frozen_protocol_and_cli_have_no_forbidden_inputs() -> None:
    assert _hash(runner.PROTOCOL_PATH) == runner.EXPECTED_PROTOCOL_SHA256
    options = {option for action in runner._parser()._actions for option in action.option_strings}
    assert not any(
        token in option
        for option in options
        for token in ("gt", "annotation", "oracle", "evaluator", "native", "prediction")
    )


def test_direct_script_cli_imports_repository_from_arbitrary_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--plan-only" in completed.stdout
