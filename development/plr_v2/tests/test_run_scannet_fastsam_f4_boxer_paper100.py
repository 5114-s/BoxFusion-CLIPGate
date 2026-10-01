from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import run_scannet_fastsam_f4_boxer_paper100 as runner


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": _hash(path)}


def _sealed_file(path: Path, payload: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": _hash(path)}


def _hypotheses(points_sha: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, delta in (("H0", 0.0), ("HL", 0.01), ("HLG", 0.02)):
        q02 = [delta, 0.0, 1.0]
        q98 = [1.0 + delta, 1.0, 2.0]
        result[name] = {
            "valid": True,
            "q02": q02,
            "q98": q98,
            "center": [(a + b) / 2.0 for a, b in zip(q02, q98)],
            "extent": [b - a for a, b in zip(q02, q98)],
            "stored_point_count": 4,
            "points_and_voxel_keys_sha256": points_sha,
            "diagnostics": {"applied": True, "fallback": False, "reason": name},
        }
    return result


def _prepare_inputs(tmp_path: Path) -> dict[str, Path]:
    scenes = ("scene0000_00", "scene0001_00")
    f0_rows = []
    f2_rows = []
    for scene_index, scene_id in enumerate(scenes):
        root = tmp_path / "sealed" / scene_id
        rgb = _sealed_file(root / "color/0.jpg", b"sealed-rgb")
        depth = _sealed_file(root / "depth/0.png", b"sealed-depth")
        pose = _sealed_file(root / "pose/0.txt", b"sealed-pose")
        intrinsic_path = root / "intrinsic/intrinsic_depth.txt"
        intrinsic_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(intrinsic_path, np.eye(4))
        intrinsic = {"path": str(intrinsic_path.resolve()), "sha256": _hash(intrinsic_path)}
        schedule = _write_json(root / "schedule.json", {"scene_id": scene_id, "frames": [0]})
        evidence = _sealed_file(root / "evidence.npz", b"sealed-evidence")
        mask_sha = hashlib.sha256(f"mask-{scene_id}".encode()).hexdigest()
        points_sha = hashlib.sha256(f"points-{scene_id}".encode()).hexdigest()
        source_id = f"{scene_id}/frame_000000/raw_003"
        candidate = {
            "rank": 0,
            "raw_index": 3,
            "confidence": 0.75,
            "mask_sha256": mask_sha,
            "points_and_voxel_keys_sha256": points_sha,
            "stored_point_count": 4,
            "world_q02": [0.0, 0.0, 1.0],
            "world_q98": [1.0, 1.0, 2.0],
            "tight_box_xyxy": [1, 2, 20, 30],
        }
        mask = {
            "rank": 0,
            "raw_index": 3,
            "confidence": 0.75,
            "mask_sha256": mask_sha,
            "selected": True,
            "decision": "selected",
            "tight_box_xyxy": [1, 2, 20, 30],
            "provider_box_xyxy": [100.0, 100.0, 200.0, 200.0],
        }
        f0_frame = {
            "frame_ordinal": 0,
            "frame_id": 0,
            "successful": True,
            "abstention": None,
            "inputs": {
                "current_pose_valid": True,
                "f0_pose_forward_filled": False,
                "producer_orientation": 0,
                "producer_rotation_k": 0,
                "producer_depth_shape": [480, 640],
                "producer_image_shape": [480, 640, 3],
                "rgb_path": rgb["path"],
                "rgb_sha256": rgb["sha256"],
                "depth_path": depth["path"],
                "depth_sha256": depth["sha256"],
                "pose_path": pose["path"],
                "pose_sha256": pose["sha256"],
            },
            "funnel": {"candidates": [candidate], "masks": [mask]},
        }
        f0_scene = {
            "schema": runner.EXPECTED_F0_SCENE_SCHEMA,
            "protocol_id": runner.EXPECTED_F0_PROTOCOL,
            "complete": True,
            "scene_id": scene_id,
            "scene_index": scene_index,
            "frames": [f0_frame],
        }
        f0_sidecar = _write_json(tmp_path / "f0/scenes" / f"{scene_id}.json", f0_scene)
        f0_rows.append({"scene_id": scene_id, "scene_index": scene_index, "sidecar": f0_sidecar})

        source = {
            "candidate_index": 0,
            "confidence": 0.75,
            "f0_world_q02": [0.0, 0.0, 1.0],
            "f0_world_q98": [1.0, 1.0, 2.0],
            "f2_receipt": {"result_sha256": hashlib.sha256(source_id.encode()).hexdigest()},
            "hypotheses": _hypotheses(points_sha),
            "mask_sha256": mask_sha,
            "points_and_voxel_keys_sha256": points_sha,
            "rank": 0,
            "raw_index": 3,
            "source_id": source_id,
            "stored_point_count": 4,
        }
        f2_frame = {
            "frame_ordinal": 0,
            "frame_id": 0,
            "successful": True,
            "abstention": None,
            "runtime": {"complete_ms": 10.0},
            "sources": [source],
        }
        f2_scene = {
            "schema": runner.EXPECTED_F2_SCENE_SCHEMA,
            "protocol_id": runner.EXPECTED_F2_PROTOCOL,
            "complete": True,
            "scene_id": scene_id,
            "scene_index": scene_index,
            "frames": [f2_frame],
            "f0_sidecar": f0_sidecar,
            "evidence_npz": evidence,
            "schedule": schedule,
            "intrinsic": intrinsic,
        }
        f2_sidecar = _write_json(tmp_path / "f2/scenes" / f"{scene_id}.json", f2_scene)
        f2_rows.append({"scene_id": scene_id, "scene_index": scene_index, "sidecar": f2_sidecar})

    f0_receipt = _write_json(
        tmp_path / "F0.json",
        {
            "schema": runner.EXPECTED_F0_SCHEMA,
            "protocol_id": runner.EXPECTED_F0_PROTOCOL,
            "complete": True,
            "run_signature_sha256": "f0-test-signature",
            "scenes": f0_rows,
        },
    )
    f2_receipt = _write_json(
        tmp_path / "F2.json",
        {
            "schema": runner.EXPECTED_F2_SCHEMA,
            "protocol_id": runner.EXPECTED_F2_PROTOCOL,
            "complete": True,
            "overall_pass": True,
            "run_signature_sha256": "f2-test-signature",
            "coverage": {
                "scene_count": 2,
                "scene_order": list(scenes),
                "keyframe_count": 2,
                "successful_frame_count": 2,
                "source_count": 2,
                "identity_verified_source_count": 2,
            },
            "scenes": f2_rows,
        },
    )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    return {
        "f0": Path(f0_receipt["path"]),
        "f2": Path(f2_receipt["path"]),
        "scene_list": scene_list,
        "output": tmp_path / "f4",
    }


class _FakeProvider:
    frozen_receipts = {"injected_test_provider": True, "frozen": True}
    model_load_ms = 1.0

    def __init__(self, calls: list[dict[str, Any]], wrong_count: bool = False):
        self.calls = calls
        self.wrong_count = wrong_count

    def infer_batch(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        rows = []
        signs = np.asarray(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
            dtype=np.float64,
        )
        for index, source_id in enumerate(kwargs["source_ids"]):
            rows.append(
                {
                    "source_id": source_id,
                    "row_index": index,
                    "input_tight_box_xyxy": kwargs["boxes_xyxy"][index].tolist(),
                    "world_corners": (np.asarray([0.0, 0.0, 2.0]) + signs * 0.5).tolist(),
                    "world_center": [0.0, 0.0, 2.0],
                    "local_extent": [1.0, 1.0, 1.0],
                    "world_rotation": np.eye(3).tolist(),
                    "camera_depth": 2.0,
                    "confidence": 0.5,
                    "logvar": [0.0] * 6,
                    "raw_params": [0.0] * 13,
                    "valid": True,
                    "validity": {"right_handed_orthonormal": True},
                    "result_sha256": hashlib.sha256(f"provider-{source_id}".encode()).hexdigest(),
                }
            )
        if self.wrong_count:
            rows = rows[:-1]
        return {
            "rows": rows,
            "diagnostics": {
                "source_count": len(kwargs["source_ids"]),
                "valid_count": len(kwargs["source_ids"]),
                "invalid_count": 0,
                "total_ms": 1.0,
                "cuda_max_memory_allocated_bytes": 1024,
                "cuda_synchronized": True,
                "model_eval": True,
                "model_parameters_frozen": True,
                "model_forward_calls": 1,
            },
        }


def _loader(*_args: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640), dtype=np.float32),
        np.eye(4, dtype=np.float64),
    )


def _run_shard(inputs: dict[str, Path], shard_index: int, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return runner.run_f4(
        f2_receipt_path=inputs["f2"],
        f0_receipt_path=inputs["f0"],
        scene_list_path=inputs["scene_list"],
        output_root=inputs["output"],
        shard_index=shard_index,
        provider_factory=lambda **_kwargs: _FakeProvider(calls),
        frame_loader=_loader,
        expected_scene_count=2,
        expected_keyframes=2,
        expected_successful_frames=2,
        expected_sources=2,
    )


def test_runner_exact_join_uses_tight_box_and_preserves_f2_hypotheses(tmp_path: Path) -> None:
    inputs = _prepare_inputs(tmp_path)
    calls: list[dict[str, Any]] = []
    manifest = _run_shard(inputs, 0, calls)
    assert manifest["totals"]["source_count"] == 1
    assert len(calls) == 1
    assert calls[0]["boxes_xyxy"].tolist() == [[1.0, 2.0, 20.0, 30.0]]
    scene = json.loads(Path(manifest["scenes"][0]["sidecar"]["path"]).read_text())
    source = scene["frames"][0]["sources"][0]
    # JSON object key order is intentionally irrelevant (writer sorts keys).
    assert set(source["hypotheses"]) == {"H0", "HL", "HLG", "HB"}
    assert source["hypotheses"]["HB"]["valid"]
    f2 = json.loads(Path(scene["inputs"]["f2_sidecar"]["path"]).read_text())
    assert {key: source["hypotheses"][key] for key in ("H0", "HL", "HLG")} == f2["frames"][0]["sources"][0]["hypotheses"]
    assert all(value is False for key, value in manifest["contracts"].items() if key != "shadow_only")
    assert manifest["contracts"]["shadow_only"] is True


def test_runner_is_create_only_and_rejects_provider_row_count(tmp_path: Path) -> None:
    inputs = _prepare_inputs(tmp_path)
    _run_shard(inputs, 0, [])
    with pytest.raises(runner.F4RunnerError, match="refusing to overwrite"):
        _run_shard(inputs, 0, [])

    bad_inputs = _prepare_inputs(tmp_path / "bad")
    with pytest.raises(runner.F4RunnerError, match="output row count differs"):
        runner.run_f4(
            f2_receipt_path=bad_inputs["f2"],
            f0_receipt_path=bad_inputs["f0"],
            scene_list_path=bad_inputs["scene_list"],
            output_root=bad_inputs["output"],
            shard_index=0,
            provider_factory=lambda **_kwargs: _FakeProvider([], wrong_count=True),
            frame_loader=_loader,
            expected_scene_count=2,
        )


def test_cli_exposes_no_gt_prediction_native_or_evaluator_input() -> None:
    options = {option for action in runner._parser()._actions for option in action.option_strings}
    assert not any(token in option for option in options for token in ("gt", "oracle", "prediction", "native", "evaluator"))


def test_invalid_nan_provider_row_abstains_and_remains_json_safe() -> None:
    hb = runner._normalize_hb(
        {
            "source_id": "scene0000_00/frame_000000/raw_003",
            "row_index": 0,
            "input_tight_box_xyxy": [1.0, 2.0, 20.0, 30.0],
            "world_corners": [[float("nan")] * 3] * 8,
            "world_center": [float("nan"), 0.0, 2.0],
            "local_extent": [1.0, 1.0, 1.0],
            "world_rotation": np.eye(3),
            "camera_depth": 2.0,
            "confidence": float("nan"),
            "logvar": [float("nan")],
            "raw_params": [float("inf")],
            "valid": False,
            "validity": {"finite_center": False, "reasons": ["nonfinite_center"], "orthogonality_error": float("nan")},
            "result_sha256": hashlib.sha256(b"provider-invalid-diagnostic").hexdigest(),
        },
        source_id="scene0000_00/frame_000000/raw_003",
        row_index=0,
        tight_box_xyxy=[1.0, 2.0, 20.0, 30.0],
    )
    assert hb["valid"] is False
    assert hb["abstention_reason"] == "provider_invalid:nonfinite_center"
    for key in ("world_corners", "world_center", "local_extent", "world_rotation", "camera_depth", "confidence", "logvar", "raw_params"):
        assert hb[key] is None
    json.dumps(hb, allow_nan=False)


def test_invalid_geometry_preserves_only_finite_nonfiltering_diagnostics() -> None:
    hb = runner._normalize_hb(
        {
            "source_id": "scene0000_00/frame_000000/raw_003",
            "row_index": 0,
            "input_tight_box_xyxy": [1.0, 2.0, 20.0, 30.0],
            "world_corners": None,
            "world_center": None,
            "local_extent": None,
            "world_rotation": None,
            "camera_depth": None,
            "confidence": 0.125,
            "logvar": [0.25],
            "raw_params": [1.0, 2.0],
            "valid": False,
            "validity": {"finite_center": False, "reasons": ["nonfinite_center"]},
            "result_sha256": hashlib.sha256(b"provider-invalid-diagnostic").hexdigest(),
        },
        source_id="scene0000_00/frame_000000/raw_003",
        row_index=0,
        tight_box_xyxy=[1.0, 2.0, 20.0, 30.0],
    )
    assert hb["valid"] is False
    assert hb["world_center"] is None and hb["camera_depth"] is None
    assert hb["confidence"] == 0.125
    assert hb["logvar"] == [0.25]
    assert hb["raw_params"] == [1.0, 2.0]
