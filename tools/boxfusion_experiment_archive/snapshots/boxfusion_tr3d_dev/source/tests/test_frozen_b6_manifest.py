from __future__ import annotations

from pathlib import Path

import pytest

from boxfusion.frozen_b6_manifest import (
    build_frozen_b6_manifest,
    verify_frozen_b6_manifest,
    write_frozen_b6_manifest,
)


def _manifest(tmp_path: Path):
    root = tmp_path / "b6"
    root.mkdir()
    (root / "scene0001_00_boxes.pkl").write_bytes(b"prediction")
    checkpoint = tmp_path / "b6.npz"
    checkpoint.write_bytes(b"checkpoint")
    scene_list = tmp_path / "val.txt"
    scene_list.write_text("scene0001_00\n", encoding="utf-8")
    payload = build_frozen_b6_manifest(
        reference_root=root,
        checkpoint=checkpoint,
        scene_list=scene_list,
        required_scene_count=1,
    )
    path = tmp_path / "manifest.json"
    assert write_frozen_b6_manifest(path, payload) == "created"
    return path, root


def test_manifest_verifies_exact_prediction_bytes(tmp_path: Path) -> None:
    path, root = _manifest(tmp_path)
    verified = verify_frozen_b6_manifest(path, required_scene_count=1)
    assert verified["scene_count"] == 1
    assert len(verified["prediction_files"]) == 1
    (root / "scene0001_00_boxes.pkl").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        verify_frozen_b6_manifest(path, required_scene_count=1)


def test_manifest_rejects_extra_prediction(tmp_path: Path) -> None:
    path, root = _manifest(tmp_path)
    (root / "scene0002_00_boxes.pkl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="prediction set"):
        verify_frozen_b6_manifest(path, required_scene_count=1)


def test_repository_static_anchor_contains_100_verified_predictions() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "frozen_b6_full100.json"
    )
    verified = verify_frozen_b6_manifest(path, required_scene_count=100)
    assert verified["checkpoint_sha256"] == (
        "d60abf798edbfa3d7902b42651be7d6053727948f740e05795de6feed60a7071"
    )
    assert len(verified["prediction_files"]) == 100
