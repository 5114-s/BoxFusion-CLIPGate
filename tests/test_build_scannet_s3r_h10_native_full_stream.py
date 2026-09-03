from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from boxfusion.s3r_h10_provider_core import (
    ExcludedFrame,
    ExactScheduleBundle,
    SceneSchedule,
    ScheduledFrame,
)
from tools import build_scannet_s3r_h10_native_full_stream as native_builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scheduled_frame(scene_directory: Path, frame_id: int) -> ScheduledFrame:
    return ScheduledFrame(
        frame_id=frame_id,
        color_relpath=f"frames/color/{frame_id}.jpg",
        color_sha256=_sha(scene_directory / f"frames/color/{frame_id}.jpg"),
        depth_relpath=f"frames/depth/{frame_id}.png",
        depth_sha256=_sha(scene_directory / f"frames/depth/{frame_id}.png"),
        pose_relpath=f"frames/pose/{frame_id}.txt",
        pose_sha256=_sha(scene_directory / f"frames/pose/{frame_id}.txt"),
    )


def _make_fixture(tmp_path: Path) -> tuple[Path, ExactScheduleBundle]:
    scene_root = tmp_path / "scenes"
    scene_ids = ("scene0001_00", "scene0002_00")
    for scene_id in scene_ids:
        base = scene_root / scene_id / "frames"
        for role in ("color", "depth", "pose", "intrinsic"):
            (base / role).mkdir(parents=True, exist_ok=True)
        np.savetxt(base / "intrinsic" / "intrinsic_color.txt", np.eye(4))
        depth_intrinsic = np.eye(4)
        depth_intrinsic[0, 0] = 500.0
        depth_intrinsic[1, 1] = 500.0
        np.savetxt(base / "intrinsic" / "intrinsic_depth.txt", depth_intrinsic)
        for frame_id in range(4):
            (base / "color" / f"{frame_id}.jpg").write_bytes(
                f"rgb:{scene_id}:{frame_id}".encode("ascii")
            )
            (base / "depth" / f"{frame_id}.png").write_bytes(
                f"depth:{scene_id}:{frame_id}".encode("ascii")
            )
            pose = np.eye(4)
            pose[0, 3] = frame_id
            if scene_id == scene_ids[1] and frame_id == 2:
                pose[0, 0] = np.inf
            np.savetxt(base / "pose" / f"{frame_id}.txt", pose)

    first_directory = scene_root / scene_ids[0]
    second_directory = scene_root / scene_ids[1]
    first_intrinsic = _sha(first_directory / "frames/intrinsic/intrinsic_color.txt")
    second_intrinsic = _sha(second_directory / "frames/intrinsic/intrinsic_color.txt")
    first = SceneSchedule(
        scene_id=scene_ids[0],
        source_schedule_manifest_relpath=f"{scene_ids[0]}/manifest.json",
        source_schedule_manifest_sha256="1" * 64,
        formal_t05_relpath=f"formal/{scene_ids[0]}.bin",
        formal_t05_sha256="2" * 64,
        intrinsic_color_relpath="frames/intrinsic/intrinsic_color.txt",
        intrinsic_color_sha256=first_intrinsic,
        raw_frame_ids=(0, 2),
        valid_frame_ids=(0, 2),
        excluded_frames=(),
        frames=(
            _scheduled_frame(first_directory, 0),
            _scheduled_frame(first_directory, 2),
        ),
    )
    excluded_path = second_directory / "frames/pose/2.txt"
    second = SceneSchedule(
        scene_id=scene_ids[1],
        source_schedule_manifest_relpath=f"{scene_ids[1]}/manifest.json",
        source_schedule_manifest_sha256="3" * 64,
        formal_t05_relpath=f"formal/{scene_ids[1]}.bin",
        formal_t05_sha256="4" * 64,
        intrinsic_color_relpath="frames/intrinsic/intrinsic_color.txt",
        intrinsic_color_sha256=second_intrinsic,
        raw_frame_ids=(0, 2, 3),
        valid_frame_ids=(0, 3),
        excluded_frames=(
            ExcludedFrame(
                frame_id=2,
                reason="nonfinite_pose",
                pose_relpath="frames/pose/2.txt",
                pose_sha256=_sha(excluded_path),
            ),
        ),
        frames=(
            _scheduled_frame(second_directory, 0),
            _scheduled_frame(second_directory, 3),
        ),
    )
    bundle = ExactScheduleBundle(
        schema="boxfusion.s3r_h10_exact_schedule.v1",
        scene_order=scene_ids,
        raw_frame_count=5,
        valid_frame_count=4,
        holdout_list_sha256="5" * 64,
        scenes=(first, second),
        sha256="a" * 64,
    )
    return scene_root, bundle


def _build(scene_root: Path, bundle: ExactScheduleBundle) -> dict:
    return native_builder.build_native_manifest(
        scene_root=scene_root,
        provider_bundle=bundle,
        require_frozen_provider=False,
    )


def _assert_no_absolute_paths(value, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _assert_no_absolute_paths(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_paths(child, key)
    elif key.endswith("relpath"):
        assert isinstance(value, str)
        assert not Path(value).is_absolute()
        assert ".." not in Path(value).parts


def test_builds_full_native_manifest_and_causal_provider_abstention(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)

    assert value["scene_order"] == list(bundle.scene_order)
    assert value["native_frame_count"] == 8
    assert value["native_finite_pose_frame_count"] == 7
    assert value["native_nonfinite_pose_frame_count"] == 1
    assert value["per_scene_native_frame_count"] == {
        "scene0001_00": 4,
        "scene0002_00": 4,
    }
    assert value["provider_schedule"]["valid_frame_count"] == 4
    assert value["policy"]["native_color_producer_glob"] == "frames/color/*.jpg"
    second = value["scenes"][1]
    assert second["frame_ids"] == [0, 1, 2, 3]
    assert second["nonfinite_pose_frame_ids"] == [2]
    nonfinite = second["frames"][2]
    assert nonfinite["raw_pose_finite"] is False
    assert nonfinite["effective_pose_frame_id"] == 1
    assert nonfinite["effective_pose_relpath"] == "frames/pose/1.txt"
    assert nonfinite["pose_resolution"] == "past_most_recent_valid"
    assert nonfinite["provider_status"] == "provider_abstain_nonfinite_pose"
    assert second["frames"][3]["provider_status"] == "provider_member"
    assert second["intrinsic_depth_relpath"] == (
        "frames/intrinsic/intrinsic_depth.txt"
    )
    assert set(second["role_mounts"]) == set(native_builder._MOUNT_ROLES)
    assert all(
        set(record) == native_builder._MOUNT_KEYS
        for record in second["role_mounts"].values()
    )
    assert all(
        set(frame) == native_builder._FRAME_KEYS
        for scene in value["scenes"]
        for frame in scene["frames"]
    )
    _assert_no_absolute_paths(value)
    assert (
        native_builder.validate_native_manifest(
            value,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )
        == value
    )


def test_provider_v2_subset_path_and_hash_are_exact(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)
    broken = deepcopy(value)
    broken["scenes"][0]["frames"][0]["color_sha256"] = "f" * 64
    with pytest.raises(native_builder.NativeManifestError, match="subset identity"):
        native_builder.validate_native_manifest(
            broken,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )

    bad_frame = replace(bundle.scenes[0].frames[0], color_sha256="e" * 64)
    bad_scene = replace(
        bundle.scenes[0], frames=(bad_frame, bundle.scenes[0].frames[1])
    )
    bad_bundle = replace(bundle, scenes=(bad_scene, bundle.scenes[1]))
    with pytest.raises(native_builder.NativeManifestError, match="subset identity"):
        _build(scene_root, bad_bundle)


def test_forged_bundle_cannot_reuse_frozen_v2_self_reported_sha():
    genuine = native_builder._load_frozen_provider_bundle(
        native_builder.DEFAULT_PROVIDER_SCHEDULE
    )
    forged_frame = replace(genuine.scenes[0].frames[0], color_sha256="0" * 64)
    forged_scene = replace(
        genuine.scenes[0],
        frames=(forged_frame, *genuine.scenes[0].frames[1:]),
    )
    forged = replace(genuine, scenes=(forged_scene, *genuine.scenes[1:]))
    assert forged.sha256 == genuine.sha256
    with pytest.raises(native_builder.NativeManifestError, match="object differs"):
        native_builder._require_exact_frozen_provider_bundle(forged)


def test_future_or_non_most_recent_pose_substitution_is_rejected(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)
    broken = deepcopy(value)
    future = broken["scenes"][1]["frames"][3]
    nonfinite = broken["scenes"][1]["frames"][2]
    nonfinite["effective_pose_frame_id"] = 3
    nonfinite["effective_pose_relpath"] = future["pose_relpath"]
    nonfinite["effective_pose_sha256"] = future["pose_sha256"]
    with pytest.raises(native_builder.NativeManifestError, match="past-most-recent"):
        native_builder.validate_native_manifest(
            broken,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )


def test_infinite_first_nan_and_malformed_pose_fail_closed(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    pose = scene_root / "scene0002_00/frames/pose/0.txt"
    matrix = np.eye(4)
    matrix[0, 0] = np.inf
    np.savetxt(pose, matrix)
    first = bundle.scenes[1].frames[0]
    updated_first = replace(first, pose_sha256=_sha(pose))
    updated_scene = replace(
        bundle.scenes[1], frames=(updated_first, bundle.scenes[1].frames[1])
    )
    updated_bundle = replace(bundle, scenes=(bundle.scenes[0], updated_scene))
    with pytest.raises(native_builder.NativeManifestError, match="no past finite pose"):
        _build(scene_root, updated_bundle)

    scene_root, bundle = _make_fixture(tmp_path / "nan")
    pose = scene_root / "scene0001_00/frames/pose/1.txt"
    matrix = np.eye(4)
    matrix[0, 0] = np.nan
    np.savetxt(pose, matrix)
    with pytest.raises(native_builder.NativeManifestError, match="contains NaN"):
        _build(scene_root, bundle)

    scene_root, bundle = _make_fixture(tmp_path / "malformed")
    pose = scene_root / "scene0001_00/frames/pose/1.txt"
    np.savetxt(pose, np.eye(3))
    with pytest.raises(native_builder.NativeManifestError, match="shape 4x4"):
        _build(scene_root, bundle)


def test_frame_symlink_nonjpg_and_role_set_mismatch_fail_closed(
    tmp_path: Path,
):
    scene_root, bundle = _make_fixture(tmp_path / "symlink")
    color = scene_root / "scene0001_00/frames/color/1.jpg"
    target = tmp_path / "symlink-target.jpg"
    target.write_bytes(color.read_bytes())
    color.unlink()
    color.symlink_to(target)
    with pytest.raises(native_builder.NativeManifestError, match="symlink"):
        _build(scene_root, bundle)

    for suffix in (".jpeg", ".png"):
        scene_root, bundle = _make_fixture(tmp_path / f"nonjpg-{suffix[1:]}")
        nonjpg = scene_root / f"scene0001_00/frames/color/1{suffix}"
        nonjpg.write_bytes(b"native-producer-will-not-consume-this")
        with pytest.raises(
            native_builder.NativeManifestError, match="noncanonical frame filename"
        ):
            _build(scene_root, bundle)

    scene_root, bundle = _make_fixture(tmp_path / "mismatch")
    (scene_root / "scene0001_00/frames/depth/1.png").unlink()
    with pytest.raises(native_builder.NativeManifestError, match="ID sets differ"):
        _build(scene_root, bundle)

    scene_root, bundle = _make_fixture(tmp_path / "same-gap")
    for role, suffix in (("color", ".jpg"), ("depth", ".png"), ("pose", ".txt")):
        (scene_root / f"scene0001_00/frames/{role}/1{suffix}").unlink()
    with pytest.raises(native_builder.NativeManifestError, match="runtime ordinals"):
        _build(scene_root, bundle)


def test_strict_frame_order_counts_and_identity_hashes_are_recomputed(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)

    broken = deepcopy(value)
    broken["scenes"][0]["frame_ids"][1:3] = [2, 1]
    broken["scenes"][0]["frames"][1:3] = list(
        reversed(broken["scenes"][0]["frames"][1:3])
    )
    with pytest.raises(native_builder.NativeManifestError, match="strictly increasing"):
        native_builder.validate_native_manifest(
            broken,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )

    broken = deepcopy(value)
    broken["native_frame_count"] += 1
    with pytest.raises(native_builder.NativeManifestError, match="totals differ"):
        native_builder.validate_native_manifest(
            broken,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )

    broken = deepcopy(value)
    broken["native_input_identity_sha256"] = "0" * 64
    with pytest.raises(
        native_builder.NativeManifestError, match="identity hash differs"
    ):
        native_builder.validate_native_manifest(
            broken,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )


def test_duplicate_json_key_and_symlink_manifest_are_rejected(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"first","schema":"second"}\n')
    with pytest.raises(native_builder.NativeManifestError, match="duplicate JSON key"):
        native_builder.load_and_validate_manifest(
            duplicate,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )

    value = _build(scene_root, bundle)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(native_builder._canonical_bytes(value))
    link = tmp_path / "manifest-link.json"
    link.symlink_to(manifest.name)
    with pytest.raises(native_builder.NativeManifestError, match="non-symlink"):
        native_builder.load_and_validate_manifest(
            link,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )


def test_file_reverification_detects_post_build_change(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)
    native_builder.verify_manifest_files(value, scene_root=scene_root)
    changed = scene_root / "scene0001_00/frames/color/1.jpg"
    changed.write_bytes(b"changed-after-build")
    with pytest.raises(native_builder.NativeManifestError, match="hash changed"):
        native_builder.verify_manifest_files(value, scene_root=scene_root)

    scene_root, bundle = _make_fixture(tmp_path / "intrinsic-change")
    value = _build(scene_root, bundle)
    intrinsic = scene_root / "scene0001_00/frames/intrinsic/intrinsic_depth.txt"
    changed_intrinsic = np.eye(4)
    changed_intrinsic[0, 0] = 777.0
    np.savetxt(intrinsic, changed_intrinsic)
    with pytest.raises(native_builder.NativeManifestError, match="intrinsic hash changed"):
        native_builder.verify_manifest_files(value, scene_root=scene_root)


def test_depth_intrinsic_missing_symlink_and_nan_fail_closed(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path / "missing")
    intrinsic = scene_root / "scene0001_00/frames/intrinsic/intrinsic_depth.txt"
    intrinsic.unlink()
    with pytest.raises(native_builder.NativeManifestError, match="depth intrinsic"):
        _build(scene_root, bundle)

    scene_root, bundle = _make_fixture(tmp_path / "symlink")
    intrinsic = scene_root / "scene0001_00/frames/intrinsic/intrinsic_depth.txt"
    target = tmp_path / "depth-intrinsic-target.txt"
    target.write_bytes(intrinsic.read_bytes())
    intrinsic.unlink()
    intrinsic.symlink_to(target)
    with pytest.raises(native_builder.NativeManifestError, match="non-symlink"):
        _build(scene_root, bundle)

    scene_root, bundle = _make_fixture(tmp_path / "nan")
    intrinsic = scene_root / "scene0001_00/frames/intrinsic/intrinsic_depth.txt"
    matrix = np.eye(4)
    matrix[0, 0] = np.nan
    np.savetxt(intrinsic, matrix)
    with pytest.raises(native_builder.NativeManifestError, match="must be finite"):
        _build(scene_root, bundle)

    scene_root, bundle = _make_fixture(tmp_path / "infinite")
    intrinsic = scene_root / "scene0001_00/frames/intrinsic/intrinsic_depth.txt"
    matrix = np.eye(4)
    matrix[0, 0] = np.inf
    np.savetxt(intrinsic, matrix)
    with pytest.raises(native_builder.NativeManifestError, match="must be finite"):
        _build(scene_root, bundle)

    scene_root, bundle = _make_fixture(tmp_path / "wrong-shape")
    intrinsic = scene_root / "scene0001_00/frames/intrinsic/intrinsic_depth.txt"
    np.savetxt(intrinsic, np.eye(3))
    with pytest.raises(native_builder.NativeManifestError, match="shape 4x4"):
        _build(scene_root, bundle)


def test_role_directory_mount_identity_is_bound_and_reverified(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    role_path = scene_root / "scene0001_00/frames/color"
    target = tmp_path / "mounted-color"
    role_path.rename(target)
    role_path.symlink_to(target, target_is_directory=True)

    value = _build(scene_root, bundle)
    mount = value["scenes"][0]["role_mounts"]["color"]
    assert mount["entry_type"] == "symlink_directory_mount"
    assert len(mount["link_target_sha256"]) == 64
    assert os.fspath(target) not in json.dumps(value, sort_keys=True)

    replacement = tmp_path / "replacement-color"
    shutil.copytree(target, replacement)
    role_path.unlink()
    role_path.symlink_to(replacement, target_is_directory=True)
    with pytest.raises(native_builder.NativeManifestError, match="mount identity"):
        native_builder.verify_manifest_files(value, scene_root=scene_root)


def test_mount_records_and_global_digest_are_strictly_recomputed(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)

    broken = deepcopy(value)
    broken["scenes"][0]["role_mounts"]["color"]["identity_sha256"] = "0" * 64
    with pytest.raises(native_builder.NativeManifestError, match="identity hash differs"):
        native_builder.validate_native_manifest(
            broken, provider_bundle=bundle, require_frozen_provider=False
        )

    impossible = deepcopy(value)
    record = impossible["scenes"][0]["role_mounts"]["color"]
    record["target_inode"] += 1
    base = {key: record[key] for key in record if key != "identity_sha256"}
    record["identity_sha256"] = native_builder._hash_bytes(
        native_builder._canonical_bytes(base)
    )
    with pytest.raises(native_builder.NativeManifestError, match="direct directory"):
        native_builder.validate_native_manifest(
            impossible, provider_bundle=bundle, require_frozen_provider=False
        )


def test_json_booleans_cannot_smuggle_integer_counts_or_frame_ids(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)

    mutations = []
    top = deepcopy(value)
    top["native_nonfinite_pose_frame_count"] = True
    mutations.append(top)
    provider = deepcopy(value)
    provider["provider_schedule"]["excluded_frame_count"] = True
    mutations.append(provider)
    scene = deepcopy(value)
    scene["scenes"][1]["provider_abstention_frame_count"] = True
    mutations.append(scene)
    frame = deepcopy(value)
    frame["scenes"][0]["frames"][1]["frame_id"] = True
    mutations.append(frame)
    policy = deepcopy(value)
    policy["policy"]["future_pose_substitution"] = 0
    mutations.append(policy)

    for index, broken in enumerate(mutations):
        expected = "policy" if index == len(mutations) - 1 else "integer"
        with pytest.raises(native_builder.NativeManifestError, match=expected):
            native_builder.validate_native_manifest(
                broken, provider_bundle=bundle, require_frozen_provider=False
            )

    scene_root, bundle = _make_fixture(tmp_path / "nonfinite-one")
    pose = scene_root / "scene0002_00/frames/pose/1.txt"
    matrix = np.eye(4)
    matrix[0, 0] = np.inf
    np.savetxt(pose, matrix)
    value = _build(scene_root, bundle)
    assert value["scenes"][1]["nonfinite_pose_frame_ids"] == [1, 2]
    broken = deepcopy(value)
    broken["scenes"][1]["nonfinite_pose_frame_ids"][0] = True
    with pytest.raises(native_builder.NativeManifestError, match="integer"):
        native_builder.validate_native_manifest(
            broken, provider_bundle=bundle, require_frozen_provider=False
        )

    broken = deepcopy(value)
    broken["role_mount_identity_sha256"] = "0" * 64
    with pytest.raises(native_builder.NativeManifestError, match="mount identity hash"):
        native_builder.validate_native_manifest(
            broken, provider_bundle=bundle, require_frozen_provider=False
        )

    broken = deepcopy(value)
    broken["scenes"][0]["role_mounts"]["color"]["target_inode"] = True
    with pytest.raises(native_builder.NativeManifestError, match="must be an integer"):
        native_builder.validate_native_manifest(
            broken, provider_bundle=bundle, require_frozen_provider=False
        )


def test_create_only_publish_and_canonical_round_trip(tmp_path: Path):
    scene_root, bundle = _make_fixture(tmp_path)
    value = _build(scene_root, bundle)
    payload = native_builder._canonical_bytes(value)
    output = tmp_path / "candidate.json"
    native_builder._publish_create_only(output, payload)
    assert output.read_bytes() == payload
    assert (
        native_builder.load_and_validate_manifest(
            output,
            provider_bundle=bundle,
            require_frozen_provider=False,
        )
        == value
    )
    with pytest.raises(native_builder.NativeManifestError, match="output exists"):
        native_builder._publish_create_only(output, payload)


def test_output_scene_root_overlap_fails_before_any_build_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scene_root, _ = _make_fixture(tmp_path)
    inside = scene_root / "scene0001_00/frames/color/native_manifest.json"
    assert not inside.exists()

    def forbidden_provider_load(_path):
        pytest.fail("provider load must not run after output/input overlap")

    monkeypatch.setattr(
        native_builder, "_load_frozen_provider_bundle", forbidden_provider_load
    )
    with pytest.raises(native_builder.NativeManifestError, match="disjoint"):
        native_builder.main(
            [
                "--output",
                os.fspath(inside),
                "--scene-root",
                os.fspath(scene_root),
            ]
        )
    assert not inside.exists()

    nonexistent_output = tmp_path / "outer"
    nonexistent_root = nonexistent_output / "nested-scenes"
    with pytest.raises(native_builder.NativeManifestError, match="disjoint"):
        native_builder._check_output_location(nonexistent_output, nonexistent_root)
    assert not nonexistent_output.exists()
    assert not nonexistent_root.exists()

    alias = tmp_path / "scene-alias"
    alias.symlink_to(scene_root, target_is_directory=True)
    with pytest.raises(native_builder.NativeManifestError, match="disjoint"):
        native_builder._check_output_location(alias / "manifest.json", scene_root)


def test_publish_rejects_payload_cap_and_parent_name_swap_without_named_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "candidate.json"
    monkeypatch.setattr(native_builder, "MAX_MANIFEST_BYTES", 3)
    with pytest.raises(native_builder.NativeManifestError, match="byte cap"):
        native_builder._publish_create_only(output, b"four")
    assert not output.exists()
    monkeypatch.setattr(native_builder, "MAX_MANIFEST_BYTES", 1024)

    real_link = os.link
    observed_source: list[str] = []

    def inspect_anonymous_link(src, dst, *args, **kwargs):
        observed_source.append(src)
        assert src.startswith("/proc/self/fd/")
        assert not any(name.endswith(".tmp") for name in os.listdir(kwargs["dst_dir_fd"]))
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(native_builder.os, "link", inspect_anonymous_link)
    native_builder._publish_create_only(output, b"payload")
    assert output.read_bytes() == b"payload"
    assert len(observed_source) == 1
    output.unlink()
    monkeypatch.setattr(native_builder.os, "link", real_link)

    public_parent = tmp_path / "public"
    public_parent.mkdir()
    moved_parent = tmp_path / "public-moved"
    public_output = public_parent / "candidate.json"

    def swap_parent_after_link(src, dst, *args, **kwargs):
        result = real_link(src, dst, *args, **kwargs)
        public_parent.rename(moved_parent)
        public_parent.mkdir()
        return result

    monkeypatch.setattr(native_builder.os, "link", swap_parent_after_link)
    with pytest.raises(native_builder.NativeManifestError, match="parent public path"):
        native_builder._publish_create_only(public_output, b"payload")
    assert not public_output.exists()


def test_publish_name_swap_never_deletes_attacker_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_link = os.link
    output = tmp_path / "candidate.json"
    displaced = tmp_path / "our-published-link"

    def swap_output_after_link(src, dst, *args, **kwargs):
        result = real_link(src, dst, *args, **kwargs)
        os.rename(
            dst,
            displaced.name,
            src_dir_fd=kwargs["dst_dir_fd"],
            dst_dir_fd=kwargs["dst_dir_fd"],
        )
        attacker = os.open(
            dst,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        try:
            os.write(attacker, b"attacker-output")
        finally:
            os.close(attacker)
        return result

    monkeypatch.setattr(native_builder.os, "link", swap_output_after_link)
    with pytest.raises(native_builder.NativeManifestError, match="published manifest"):
        native_builder._publish_create_only(output, b"payload")
    assert output.read_bytes() == b"attacker-output"
    assert displaced.read_bytes() == b"payload"
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))


def test_builder_has_no_forbidden_data_or_inference_interface():
    source = Path(native_builder.__file__).read_text(encoding="utf-8")
    forbidden = (
        "full_annotations",
        "axisAlignment",
        "bbox.npy",
        "pickle.load",
        "import pickle",
        "import torch",
        "evaluation.data_util",
        "results/scannet",
        "CUDA",
    )
    assert all(token not in source for token in forbidden)
    assert source.count("os.scandir(") == 1
    assert native_builder._POLICY["enumerated_directories"] == [
        "frames/color",
        "frames/depth",
        "frames/pose",
    ]


def test_builder_enumerates_only_three_native_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    scene_root, bundle = _make_fixture(tmp_path)
    real_scandir = os.scandir
    enumerated: list[str] = []

    def tracked_scandir(directory_fd):
        enumerated.append(os.path.basename(os.readlink(f"/proc/self/fd/{directory_fd}")))
        return real_scandir(directory_fd)

    monkeypatch.setattr(native_builder.os, "scandir", tracked_scandir)
    _build(scene_root, bundle)
    assert len(enumerated) == 12
    assert set(enumerated) == {"color", "depth", "pose"}
    assert "intrinsic" not in enumerated
