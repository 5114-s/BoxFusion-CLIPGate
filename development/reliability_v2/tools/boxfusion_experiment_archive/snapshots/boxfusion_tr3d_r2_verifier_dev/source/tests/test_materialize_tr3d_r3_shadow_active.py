from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from tools import materialize_tr3d_r3_shadow_active as materializer


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


def _corners(center: float) -> np.ndarray:
    return np.asarray([center, 0, 0], dtype=np.float32) + _SIGNS * 0.5


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_invariant_validator_preserves_label_score_order_and_count() -> None:
    source = [[(0, _corners(0), 0.75), (0, _corners(3), 0.25)]]
    active = [[(0, _corners(0.2), 0.75), (0, _corners(3), 0.25)]]
    row = materializer._validate_invariants(source, active)
    assert row == {
        "rows": 2,
        "changed_rows": 1,
        "count_unchanged": True,
        "order_unchanged": True,
        "labels_unchanged": True,
        "scores_unchanged": True,
    }

    with pytest.raises(ValueError, match="changed score bytes"):
        materializer._validate_invariants(
            source, [[(0, _corners(0.2), 0.70), (0, _corners(3), 0.25)]]
        )
    with pytest.raises(ValueError, match="prediction count"):
        materializer._validate_invariants(source, [[source[0][0]]])


def test_active_dataclass_summary_is_recursively_json_safe() -> None:
    source = [[(0, _corners(0), 0.5)]]
    cache = SimpleNamespace(
        anchor_count=1,
        proposal_ids=np.asarray([17], dtype=np.int64),
        proposal_corners_world=np.stack([_corners(0.2)]),
        anchor_index=np.asarray([0], dtype=np.int64),
        tr3d_score=np.asarray([0.8], dtype=np.float32),
        anchor_score=np.asarray([0.5], dtype=np.float32),
    )
    active, summary = materializer._materialize_payload(source, cache)
    assert summary["selected_count"] == 1
    assert summary["changed_count"] == 1
    assert summary["selections"] == [
        {
            "anchor_index": 0,
            "proposal_row": 0,
            "proposal_id": 17,
            "tr3d_score": float(np.float32(0.8)),
            "anchor_score": 0.5,
            "geometry_changed": True,
        }
    ]
    json.dumps(summary, sort_keys=True)
    assert _sha_payload(active) != _sha_payload(source)


def _sha_payload(payload) -> str:
    return hashlib.sha256(
        pickle.dumps(payload, protocol=materializer.PICKLE_PROTOCOL)
    ).hexdigest()


def test_prediction_writer_is_atomic_create_only_and_readonly(tmp_path: Path) -> None:
    target = tmp_path / "scene0001_00_boxes.pkl"
    materializer._write_bytes_create_only(target, b"first")
    assert target.read_bytes() == b"first"
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="immutable shadow-active"):
        materializer._write_bytes_create_only(target, b"second")


def _fake_materialization_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scenes = [f"scene{index:04d}_00" for index in range(10)]
    frozen_root = tmp_path / "frozen"
    r3_root = tmp_path / "r3"
    scans_root = tmp_path / "scans"
    frozen_root.mkdir()
    r3_root.mkdir()
    scans_root.mkdir()
    source_hashes: dict[str, str] = {}
    sidecar_hashes: dict[str, str] = {}
    for index, scene in enumerate(scenes):
        prediction = frozen_root / f"{scene}_boxes.pkl"
        with prediction.open("wb") as handle:
            pickle.dump(
                [[(0, _corners(float(index)), 0.5)]],
                handle,
                protocol=materializer.PICKLE_PROTOCOL,
            )
        source_hashes[prediction.name] = _sha(prediction)
        sidecar = r3_root / scene / "p100.npz"
        sidecar.parent.mkdir()
        sidecar.write_bytes(f"sidecar:{scene}".encode())
        sidecar_hashes[scene] = _sha(sidecar)

    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("".join(f"{scene}\n" for scene in scenes), encoding="utf-8")
    frozen_manifest = tmp_path / "frozen.json"
    frozen_manifest.write_text("{}\n", encoding="utf-8")
    export_path = tmp_path / "export.json"
    export_path.write_text("{}\n", encoding="utf-8")
    parent_root = tmp_path / "parent"
    prefix_manifest = tmp_path / "prefix.jsonl"
    parent_root.mkdir()
    prefix_manifest.write_text("", encoding="utf-8")
    snapshot = {
        "anchor_name": "test-g0",
        "prediction_tree_sha256": "1" * 64,
        "artifact_tree_sha256": "2" * 64,
        "scene_list_sha256": _sha(scene_list),
    }
    verified = {
        **snapshot,
        "scene_ids": scenes,
        "reference_result_root": str(frozen_root),
        "prediction_files": source_hashes,
    }
    export = {
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": _sha(frozen_manifest),
        "frozen_prediction_tree_sha256": snapshot["prediction_tree_sha256"],
        "parent_cache_root": str(parent_root),
        "prefix_manifest": str(prefix_manifest),
        "r3_cache_root": str(r3_root),
        "scene_list": str(scene_list),
        "scans_root": str(scans_root),
        "prefix_id": "p100",
        "expected_parent_checkpoint_sha256": "3" * 64,
        "expected_parent_config_sha256": "4" * 64,
        "r3_config": {"r2a_enabled": False, "r2b_enabled": False},
        "r3_config_sha256": "5" * 64,
        "r3_code_sha256": "6" * 64,
        "parent_evidence_hashes": {},
        "scenes": [
            {"scene_id": scene, "r3_sidecar_sha256": sidecar_hashes[scene]}
            for scene in scenes
        ],
    }

    monkeypatch.setattr(materializer, "verify_frozen_anchor_manifest", lambda _: dict(verified))
    monkeypatch.setattr(materializer, "_load_export", lambda _path, _scenes: dict(export))
    monkeypatch.setattr(materializer, "_code_hash", lambda: "7" * 64)
    monkeypatch.setattr(
        materializer,
        "_load_bound_cache",
        lambda **kwargs: SimpleNamespace(
            proposal_corners_world=np.stack([_corners(100.0)]),
            scene_id=kwargs["scene_id"],
        ),
    )

    def fake_active(source, cache):
        output = [[(row[0], np.array(row[1], copy=True), row[2]) for row in source[0]]]
        output[0][0] = (
            output[0][0][0],
            np.array(cache.proposal_corners_world[0], copy=True),
            output[0][0][2],
        )
        return output, {"applied_count": 1, "scene_id": cache.scene_id}

    monkeypatch.setattr(materializer, "_materialize_payload", fake_active)
    args = argparse.Namespace(
        frozen_manifest=frozen_manifest,
        r3_export_report=export_path,
        r3_cache_root=r3_root,
        scene_list=scene_list,
        scans_root=scans_root,
        output_root=tmp_path / "active",
        manifest=tmp_path / "active_manifest.json",
        prefix_id="p100",
        resume=False,
    )
    return args, scenes


def test_materialization_is_isolated_create_only_and_resume_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, scenes = _fake_materialization_fixture(tmp_path, monkeypatch)
    first = materializer.materialize(args)
    assert first["complete"] is True
    assert first["shadow_only"] is True
    assert first["formal_active_authorized"] is False
    assert first["ground_truth_access"] is False
    assert first["counterfactual_report_access"] is False
    assert first["clip_access"] is False
    assert first["counts"] == {
        "scenes": 10,
        "rows": 10,
        "applied_replacements": 10,
        "byte_changed_rows": 10,
        "resumed_scenes": 0,
    }
    assert {path.name for path in args.output_root.iterdir()} == {
        f"{scene}_boxes.pkl" for scene in scenes
    }
    assert all(path.stat().st_mode & 0o222 == 0 for path in args.output_root.iterdir())
    before_hashes = {path.name: _sha(path) for path in args.output_root.iterdir()}

    with pytest.raises(FileExistsError, match="already exists"):
        materializer.materialize(args)

    args.resume = True
    resumed = materializer.materialize(args)
    assert resumed == first
    assert {path.name: _sha(path) for path in args.output_root.iterdir()} == before_hashes

    # A mismatching existing prediction is never overwritten or silently
    # repaired.  Resume fails closed and leaves its bytes untouched.
    corrupt = args.output_root / f"{scenes[0]}_boxes.pkl"
    corrupt.chmod(0o644)
    corrupt.write_bytes(b"not a pickle")
    corrupt_bytes = corrupt.read_bytes()
    with pytest.raises(ValueError, match="regular immutable"):
        materializer.materialize(args)
    assert corrupt.read_bytes() == corrupt_bytes


def test_partial_resume_only_creates_missing_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, scenes = _fake_materialization_fixture(tmp_path, monkeypatch)
    materializer.materialize(args)
    missing = args.output_root / f"{scenes[-1]}_boxes.pkl"
    args.manifest.unlink()
    missing.unlink()
    preserved = {
        path.name: _sha(path) for path in args.output_root.iterdir()
    }

    args.resume = True
    resumed = materializer.materialize(args)
    assert resumed["counts"]["resumed_scenes"] == 9
    assert missing.is_file()
    assert missing.stat().st_mode & 0o222 == 0
    assert {
        path.name: _sha(path)
        for path in args.output_root.iterdir()
        if path.name in preserved
    } == preserved


def test_resume_refuses_absent_namespace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="resume output root is absent"):
        materializer._claim_namespace(
            tmp_path / "absent",
            tmp_path / "manifest.json",
            resume=True,
        )


def test_manifest_cannot_pollute_prediction_namespace(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    with pytest.raises(ValueError, match="outside"):
        materializer._claim_namespace(
            root,
            root / "metadata" / "manifest.json",
            resume=False,
        )
    assert not root.exists()
