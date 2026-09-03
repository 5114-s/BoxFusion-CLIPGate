from __future__ import annotations

from pathlib import Path

import pytest

from boxfusion.frozen_anchor_manifest import (
    build_frozen_anchor_manifest,
    verify_frozen_anchor_manifest,
    write_frozen_anchor_manifest,
)


def _anchor(tmp_path: Path):
    root = tmp_path / "predictions"
    root.mkdir()
    prediction = root / "scene0001_00_boxes.pkl"
    prediction.write_bytes(b"prediction")
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("scene0001_00\n", encoding="utf-8")
    payload = build_frozen_anchor_manifest(
        anchor_name="test-anchor",
        reference_root=root,
        scene_list=scene_list,
        artifacts={"model": artifact},
        anchor_metrics_percent={"AP15": 1, "AP25": 2, "AP50": 3},
        metadata={"score_threshold": 0.4},
        required_scene_count=1,
    )
    path = tmp_path / "anchor.json"
    assert write_frozen_anchor_manifest(path, payload) == "created"
    return path, prediction, artifact


def test_generic_anchor_verifies_prediction_and_artifact_bytes(
    tmp_path: Path,
) -> None:
    path, prediction, artifact = _anchor(tmp_path)
    verified = verify_frozen_anchor_manifest(path, required_scene_count=1)
    assert verified["anchor_name"] == "test-anchor"
    assert verified["metadata"]["score_threshold"] == 0.4
    prediction.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_frozen_anchor_manifest(path, required_scene_count=1)
    prediction.write_bytes(b"prediction")
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_frozen_anchor_manifest(path, required_scene_count=1)


def test_legacy_b6_is_normalized_as_generic_anchor() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "frozen_b6_full100.json"
    )
    verified = verify_frozen_anchor_manifest(path, required_scene_count=100)
    assert verified["anchor_name"] == "B6"
    assert "artifact_tree_sha256" in verified
