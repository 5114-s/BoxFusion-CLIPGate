from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import build_scannet_s3r_h10_exact_schedule as schedule_builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_manifest(path: Path, scene: str, frame_ids: list[int]) -> None:
    path.parent.mkdir(parents=True)
    value = {
        "schema": "boxfusion.cutr_postfilter_cache.v2",
        "namespace": "scannet-score05-gap25-postfilter-v2",
        "scene_id": scene,
        "record_count": len(frame_ids),
        "recorded_frame_ids": frame_ids,
        "records": [{"frame_id": frame_id} for frame_id in frame_ids],
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _configure_tiny_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    scene_root = repository / "scenes"
    source_root = tmp_path / "source"
    formal_root = repository / "formal"
    holdout = repository / "holdout.txt"
    scenes = ("scene0001_00", "scene0002_00")
    holdout.parent.mkdir(parents=True)
    holdout.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    raw_ids = {scenes[0]: [0, 25], scenes[1]: [0, 25]}
    source_hashes = {}
    formal_hashes = {}
    for scene in scenes:
        manifest = source_root / scene / "manifest.json"
        _write_source_manifest(manifest, scene, raw_ids[scene])
        source_hashes[scene] = _sha(manifest)
        formal = formal_root / f"{scene}_boxes.pkl"
        formal.parent.mkdir(parents=True, exist_ok=True)
        formal.write_bytes(scene.encode("ascii"))
        formal_hashes[scene] = _sha(formal)
        base = scene_root / scene / "frames"
        (base / "intrinsic").mkdir(parents=True)
        np.savetxt(base / "intrinsic" / "intrinsic_color.txt", np.eye(4))
        for frame_id in raw_ids[scene]:
            (base / "color").mkdir(exist_ok=True)
            (base / "depth").mkdir(exist_ok=True)
            (base / "pose").mkdir(exist_ok=True)
            (base / "color" / f"{frame_id}.jpg").write_bytes(
                f"color-{scene}-{frame_id}".encode("ascii")
            )
            (base / "depth" / f"{frame_id}.png").write_bytes(
                f"depth-{scene}-{frame_id}".encode("ascii")
            )
            pose = np.eye(4)
            if scene == scenes[1] and frame_id == 25:
                pose[:] = np.inf
            np.savetxt(base / "pose" / f"{frame_id}.txt", pose)

    monkeypatch.setattr(schedule_builder, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(schedule_builder, "HOLDOUT_LIST", holdout)
    monkeypatch.setattr(schedule_builder, "SCENE_ROOT", scene_root)
    monkeypatch.setattr(schedule_builder, "SOURCE_SCHEDULE_ROOT", source_root)
    monkeypatch.setattr(schedule_builder, "FORMAL_T05_ROOT", formal_root)
    monkeypatch.setattr(schedule_builder, "SCENE_ORDER", scenes)
    monkeypatch.setattr(schedule_builder, "EXPECTED_HOLDOUT_SHA256", _sha(holdout))
    monkeypatch.setattr(
        schedule_builder, "EXPECTED_SOURCE_MANIFEST_SHA256", source_hashes
    )
    monkeypatch.setattr(schedule_builder, "EXPECTED_FORMAL_T05_SHA256", formal_hashes)
    monkeypatch.setattr(schedule_builder, "EXPECTED_RAW_COUNTS", dict.fromkeys(scenes, 2))
    monkeypatch.setattr(
        schedule_builder, "EXPECTED_EXCLUSIONS", {scenes[1]: (25,)}
    )
    monkeypatch.setattr(schedule_builder, "EXPECTED_RAW_TOTAL", 4)
    monkeypatch.setattr(schedule_builder, "EXPECTED_VALID_TOTAL", 3)


def test_builds_self_contained_exact_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tiny_bundle(monkeypatch, tmp_path)
    value = schedule_builder.build_schedule()
    assert value["raw_frame_count"] == 4
    assert value["valid_frame_count"] == 3
    assert value["provider"] == {
        "annotation_path": None,
        "track": False,
        "directory_enumeration": False,
        "prefetch": False,
        "persist_before_advance": True,
    }
    assert value["scenes"][0]["valid_frame_ids"] == [0, 25]
    assert value["scenes"][1]["valid_frame_ids"] == [0]
    assert value["scenes"][1]["excluded_frames"][0]["frame_id"] == 25
    assert len(value["scenes"][0]["frames"]) == 2
    assert value["scenes"][0]["source_schedule_manifest_relpath"] == (
        "scene0001_00/manifest.json"
    )
    assert set(value["scenes"][0]["frames"][0]) == {
        "frame_id",
        "color_relpath",
        "color_sha256",
        "depth_relpath",
        "depth_sha256",
        "pose_relpath",
        "pose_sha256",
    }


def test_manifest_order_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tiny_bundle(monkeypatch, tmp_path)
    scene = schedule_builder.SCENE_ORDER[0]
    manifest = schedule_builder.SOURCE_SCHEDULE_ROOT / scene / "manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["records"].reverse()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setitem(
        schedule_builder.EXPECTED_SOURCE_MANIFEST_SHA256, scene, _sha(manifest)
    )
    with pytest.raises(schedule_builder.ScheduleBuildError, match="record order"):
        schedule_builder.build_schedule()


def test_unregistered_nonfinite_pose_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_tiny_bundle(monkeypatch, tmp_path)
    monkeypatch.setattr(schedule_builder, "EXPECTED_EXCLUSIONS", {})
    with pytest.raises(schedule_builder.ScheduleBuildError, match="exclusions differ"):
        schedule_builder.build_schedule()


def test_publish_is_create_only_and_durable(tmp_path: Path) -> None:
    output = tmp_path / "schedule.json"
    payload = b'{"frozen":true}\n'
    schedule_builder._publish_create_only(output, payload)
    assert output.read_bytes() == payload
    with pytest.raises(schedule_builder.ScheduleBuildError, match="already exists"):
        schedule_builder._publish_create_only(output, payload)


def test_builder_source_has_no_frame_directory_enumeration() -> None:
    source = Path(schedule_builder.__file__).read_text(encoding="utf-8")
    forbidden = ("os.listdir(", ".iterdir(", ".glob(", ".rglob(")
    assert all(token not in source for token in forbidden)
