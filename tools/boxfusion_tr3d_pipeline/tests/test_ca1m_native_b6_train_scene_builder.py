from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_ca1m_native_b6_train_scene.py"
AUDITOR = ROOT / "tools" / "audit_ca1m_native_b6_train_scene.py"
SCENE = "42000001"
HOST = "https://ml-site.cdn-apple.com/datasets/ca1m"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png(value: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", value)
    assert ok
    return encoded.tobytes()


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o444
    archive.addfile(info, io.BytesIO(payload))


def _json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _corners(center_z: float) -> list[list[float]]:
    return [
        [x, y, center_z + z]
        for z in (-0.02, 0.02)
        for x, y in ((-0.02, -0.02), (0.02, -0.02), (0.02, 0.02), (-0.02, 0.02))
    ]


def _make_tar(root: Path, scene: str = SCENE) -> Path:
    tar_path = root / f"ca1m-train-{scene}.tar"
    depth = np.full((32, 48), 2000, dtype=np.uint16)
    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    depth_k = [[40.0, 0.0, 24.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]
    rgb_k = [[80.0, 0.0, 48.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]]
    pose = np.eye(4).tolist()
    with tarfile.open(tar_path, "w") as archive:
        for index in range(3):
            frame = str(1000 + index)
            prefix = f"{scene}/{frame}"
            _add(archive, prefix + ".gt/RT.json", _json(pose))
            _add(archive, prefix + ".gt/depth.png", _png(depth))
            _add(archive, prefix + ".gt/depth/K.json", _json(depth_k))
            _add(archive, prefix + ".gt/image/K.json", _json(rgb_k))
            _add(archive, prefix + ".wide/T_gravity.json", _json(pose))
            _add(archive, prefix + ".wide/image.png", _png(rgb))
        rows = [
            {
                "id": "surface-box",
                "category": "chair",
                "position": [0.0, 0.0, 2.0],
                "scale": [0.04, 0.04, 0.04],
                "R": np.eye(3).tolist(),
                "corners": _corners(2.0),
                "caption": "train only",
            },
            {
                "id": "floating-box",
                "category": "chair",
                "position": [0.0, 0.0, 3.0],
                "scale": [0.04, 0.04, 0.04],
                "R": np.eye(3).tolist(),
                "corners": _corners(3.0),
                "caption": "filtered",
            },
        ]
        _add(archive, f"{scene}/world.gt/instances.json", _json(rows))
    return tar_path


def _contract(root: Path, scene: str = SCENE, *, overlap: bool = False) -> tuple[Path, Path]:
    val = root / "val.txt"
    val_scene = scene if overlap else "51000001"
    val.write_text(f"{HOST}/val/ca1m-val-{val_scene}.tar\n")
    manifest = root / "subset_manifest.json"
    payload = {
        "schema": "boxfusion.ca1m_native_b6_train_subset.v1",
        "source": {
            "train_val_overlap": [],
            "val_url_list_sha256": _sha(val),
        },
        "safety_contract": {
            "train_only": True,
            "validation_ground_truth_access": False,
            "validation_scene_overlap_count": 0,
        },
        "entries": [
            {
                "rank": 0,
                "scene_id": scene,
                "selection_key_sha256": "a" * 64,
                "tar_name": f"ca1m-train-{scene}.tar",
                "url": f"{HOST}/train/ca1m-train-{scene}.tar",
            }
        ],
    }
    manifest.write_text(json.dumps(payload))
    return manifest, val


def _run_builder(
    tar_path: Path, manifest: Path, val: Path, output: Path, mode: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--tar",
            str(tar_path),
            "--scene-id",
            SCENE,
            "--subset-manifest",
            str(manifest),
            "--val-url-list",
            str(val),
            "--output-root",
            str(output),
            "--mode",
            mode,
        ],
        text=True,
        capture_output=True,
    )


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def test_preflight_is_read_only_and_build_is_train_only(tmp_path: Path) -> None:
    tar_path = _make_tar(tmp_path)
    manifest, val = _contract(tmp_path)
    output = tmp_path / "processed"
    preflight = _run_builder(tar_path, manifest, val, output, "preflight")
    assert preflight.returncode == 0, preflight.stderr
    report = json.loads(preflight.stdout)
    assert report["output_created"] is False
    assert report["train_only"] is True
    assert report["validation_ground_truth_access"] is False
    assert not output.exists()

    built = _run_builder(tar_path, manifest, val, output, "build")
    assert built.returncode == 0, built.stderr
    scene = output / SCENE
    try:
        scene_manifest = json.loads((scene / "derived_train_gt_manifest.json").read_text())
        assert scene_manifest["schema"] == "boxfusion.ca1m_native_b6_train_scene.v1"
        assert scene_manifest["counts"] == {
            "derived_train_gt_boxes": 1,
            "frames": 3,
            "frustum_boxes": 2,
            "raw_world_boxes": 2,
            "surface_proxy_points": 96,
        }
        derived = np.load(scene / "derived_train_gt_boxes.npy")
        compatibility = np.load(scene / "after_filter_boxes.npy")
        assert derived.shape == (1, 8, 3)
        assert np.array_equal(derived, compatibility)
        assert (scene / "derived_train_gt_boxes.npy").stat().st_ino == (
            scene / "after_filter_boxes.npy"
        ).stat().st_ino
        with np.load(scene / "per_frame_intrinsics.npz", allow_pickle=False) as data:
            assert data["depth_intrinsics_raw"].shape == (3, 3, 3)
            assert data["raw_frame_ids"].tolist() == ["1000", "1001", "1002"]
            assert np.array_equal(
                data["depth_intrinsics_processed"],
                np.load(scene / "K_depth_per_frame.npy"),
            )

        audit_output = tmp_path / "audit.json"
        audited = subprocess.run(
            [
                sys.executable,
                str(AUDITOR),
                "--scene-dir",
                str(scene),
                "--geometry-check",
                "full",
                "--pixel-check",
                "all",
                "--output",
                str(audit_output),
            ],
            text=True,
            capture_output=True,
        )
        assert audited.returncode == 0, audited.stderr
        audit = json.loads(audited.stdout)
        assert audit["ok"] is True
        assert audit["per_frame_intrinsics_preserved"] is True
        assert audit["storage_filesystem_policy"]["posix_mode_enforceable"] is True

        repeated = _run_builder(tar_path, manifest, val, output, "build")
        assert repeated.returncode != 0
        assert "refusing to overwrite train scene" in repeated.stderr
    finally:
        _make_writable(output)


def test_dynamic_validation_overlap_fails_closed(tmp_path: Path) -> None:
    tar_path = _make_tar(tmp_path)
    manifest, val = _contract(tmp_path, overlap=True)
    output = tmp_path / "processed"
    result = _run_builder(tar_path, manifest, val, output, "preflight")
    assert result.returncode != 0
    assert "overlaps validation split" in result.stderr
    assert not output.exists()


def test_tar_member_outside_exact_scene_prefix_is_rejected(tmp_path: Path) -> None:
    tar_path = _make_tar(tmp_path)
    # Append a regular member belonging to another scene; no extraction occurs,
    # but the strict source contract still rejects the archive.
    with tarfile.open(tar_path, "a") as archive:
        _add(archive, "51000001/payload.txt", b"unexpected")
    manifest, val = _contract(tmp_path)
    result = _run_builder(tar_path, manifest, val, tmp_path / "processed", "preflight")
    assert result.returncode != 0
    assert "unsafe or unsupported tar member" in result.stderr


def test_processed_intrinsics_cardinal_rotation_policy() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import build_ca1m_native_b6_train_scene as native

    raw = np.asarray(
        [
            [[420.0, 0.0, 190.0], [0.0, 421.0, 250.0], [0.0, 0.0, 1.0]],
            [[422.0, 0.0, 255.0], [0.0, 423.0, 191.0], [0.0, 0.0, 1.0]],
            [[424.0, 0.0, 256.0], [0.0, 425.0, 192.0], [0.0, 0.0, 1.0]],
            [[426.0, 0.0, 193.0], [0.0, 427.0, 252.0], [0.0, 0.0, 1.0]],
        ]
    )
    shapes = ((512, 384), (384, 512), (384, 512), (512, 384))
    observed = native.processed_intrinsics(
        raw, shapes, np.asarray([0, 1, 2, 3], dtype=np.int8)
    )
    expected = np.asarray(
        [
            raw[0],
            [[423.0, 0.0, 193.0], [0.0, 422.0, 257.0], [0.0, 0.0, 1.0]],
            [[424.0, 0.0, 256.0], [0.0, 425.0, 192.0], [0.0, 0.0, 1.0]],
            [[427.0, 0.0, 260.0], [0.0, 426.0, 191.0], [0.0, 0.0, 1.0]],
        ]
    )
    assert np.array_equal(observed, expected)


def test_aspect_clockwise_to_majority_and_v2_evidence_binding(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import convert_ca1m_apple_tar as apple

    shapes = ((32, 48), (32, 48), (48, 32), (32, 48), (48, 32))
    rotations, orientation = apple.infer_rot90_aspect_clockwise_to_majority(shapes)
    assert rotations.tolist() == [0, 0, 3, 0, 3]
    assert orientation["target_orientation"] == "landscape"

    tar_path = tmp_path / "source.tar"
    tar_path.write_bytes(b"frozen source bytes")
    rotation_sha = hashlib.sha256(rotations.tobytes()).hexdigest()
    evidence_path = tmp_path / "evidence.json"
    evidence = {
        "schema": apple.ORIENTATION_EVIDENCE_SCHEMA,
        "scene_id": SCENE,
        "apple_tar_sha256": _sha(tar_path),
        "method": "aspect_clockwise_to_majority",
        "rotation_vector_int8_sha256": rotation_sha,
        "approved_for_train_only_conversion": True,
        "checked_raw_rgb_frames": [{"frame": 2}, {"frame": 4}],
    }
    evidence_path.write_text(json.dumps(evidence))
    policy_path = tmp_path / "policy.json"
    policy = {
        "schema": apple.ORIENTATION_POLICY_SCHEMA_V2,
        "default": {"method": "pose_continuity", "min_margin_degrees": 30.0},
        "scene_overrides": {
            SCENE: {
                "method": "aspect_clockwise_to_majority",
                "apple_tar_sha256": _sha(tar_path),
                "evidence_report": evidence_path.name,
                "evidence_report_sha256": _sha(evidence_path),
                "rotation_vector_int8_sha256": rotation_sha,
                "reason": "synthetic evidence",
            }
        },
    }
    policy_path.write_text(json.dumps(policy))
    rule, provenance = apple.load_orientation_policy(policy_path, SCENE, tar_path)
    assert rule["method"] == "aspect_clockwise_to_majority"
    assert provenance["override_applied"] is True
    apple.infer_orientation(np.repeat(np.eye(4)[None], len(shapes), axis=0), shapes, rule)

    evidence_path.write_text(json.dumps({**evidence, "approved_for_train_only_conversion": False}))
    try:
        apple.load_orientation_policy(policy_path, SCENE, tar_path)
    except ValueError as error:
        assert "evidence report hash mismatch" in str(error)
    else:
        raise AssertionError("tampered orientation evidence was accepted")


def test_frustum_and_surface_accept_true_per_frame_intrinsics(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    import build_ca1m_native_b6_train_scene as native

    corners = np.asarray([_corners(2.0)], dtype=np.float64)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    # The second K deliberately projects outside. A static first-frame K would
    # incorrectly make both frames equivalent; the API must retain cardinality.
    values = np.asarray(
        [
            [[40.0, 0.0, 24.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]],
            [[40.0, 0.0, 240.0], [0.0, 40.0, 160.0], [0.0, 0.0, 1.0]],
        ]
    )
    assert native.frustum_indices(corners, values, poses, (32, 48), 0.1, 10.0).tolist() == [0]
    try:
        native.frustum_indices(corners, values[:1], poses, (32, 48), 0.1, 10.0)
    except ValueError as error:
        assert "cardinality" in str(error)
    else:
        raise AssertionError("per-frame K cardinality mismatch was accepted")
