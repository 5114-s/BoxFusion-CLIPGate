#!/usr/bin/env python3
"""Convert one official Apple CA-1M tar into BoxFusion's processed layout.

The converter is deliberately fail-closed.  RGB/depth are reconstructed from
the official Apple tar, while the small evaluation/calibration files are copied
byte-for-byte from a frozen Hugging Face checkout.  Output is first built in an
isolated directory and atomically renamed into the staging root.  Promotion to
the live data root is optional, same-filesystem only, and never overwrites an
existing scene.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np


SCHEMA = "boxfusion.ca1m_apple_conversion.v2"
ORIENTATION_POLICY_SCHEMA = "boxfusion.ca1m_orientation_policy.v1"
ORIENTATION_POLICY_SCHEMA_V2 = "boxfusion.ca1m_orientation_policy.v2"
ORIENTATION_EVIDENCE_SCHEMA = "boxfusion.ca1m_orientation_evidence.v1"
HF_REQUIRED = (
    "K_depth.txt",
    "K_rgb.txt",
    "all_poses.npy",
    "T_gravity.npy",
    "after_filter_boxes.npy",
)
FRAME_SUFFIXES = (
    ".gt/RT.json",
    ".gt/depth.png",
    ".gt/depth/K.json",
    ".gt/image/K.json",
    ".wide/T_gravity.json",
    ".wide/image.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_orientation_policy(
    path: Path | None, scene_id: str, tar_path: Path
) -> tuple[dict, dict]:
    if path is None:
        return (
            {"method": "pose_continuity", "min_margin_degrees": 30.0},
            {"path": None, "sha256": None, "override_applied": False},
        )
    resolved = path.resolve()
    payload = json.loads(resolved.read_text())
    if set(payload) != {"schema", "default", "scene_overrides"}:
        raise ValueError("orientation policy has unknown/missing top-level fields")
    schema = payload["schema"]
    if schema not in {ORIENTATION_POLICY_SCHEMA, ORIENTATION_POLICY_SCHEMA_V2}:
        raise ValueError("unsupported orientation policy schema")
    default = payload["default"]
    if set(default) != {"method", "min_margin_degrees"}:
        raise ValueError("invalid orientation policy default")
    margin = float(default["min_margin_degrees"])
    if default["method"] != "pose_continuity" or not math.isfinite(margin) or not 0 <= margin <= 30:
        raise ValueError("invalid default orientation policy")
    overrides = payload["scene_overrides"]
    if not isinstance(overrides, dict) or any(not re.fullmatch(r"\d{8}", key) for key in overrides):
        raise ValueError("invalid orientation policy scene overrides")
    selected = dict(default)
    override = overrides.get(scene_id)
    if override is not None and schema == ORIENTATION_POLICY_SCHEMA:
        required = {
            "method",
            "apple_tar_sha256",
            "evidence_report_sha256",
            "evidence_checked_image_rows",
            "evidence_match_fraction",
            "reason",
        }
        if set(override) != required:
            raise ValueError(f"{scene_id}: invalid orientation override fields")
        if override["method"] != "aspect_clockwise_to_portrait":
            raise ValueError(f"{scene_id}: unsupported orientation override method")
        if override["apple_tar_sha256"] != sha256(tar_path):
            raise ValueError(f"{scene_id}: orientation override Apple tar hash mismatch")
        if int(override["evidence_checked_image_rows"]) <= 0 or float(override["evidence_match_fraction"]) != 1.0:
            raise ValueError(f"{scene_id}: orientation override lacks exact pixel evidence")
        selected = dict(override)
    elif override is not None:
        required = {
            "method",
            "apple_tar_sha256",
            "evidence_report",
            "evidence_report_sha256",
            "rotation_vector_int8_sha256",
            "reason",
        }
        if set(override) != required:
            raise ValueError(f"{scene_id}: invalid v2 orientation override fields")
        if override["method"] != "aspect_clockwise_to_majority":
            raise ValueError(f"{scene_id}: unsupported v2 orientation override method")
        if override["apple_tar_sha256"] != sha256(tar_path):
            raise ValueError(f"{scene_id}: orientation override Apple tar hash mismatch")
        for label in ("evidence_report_sha256", "rotation_vector_int8_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(override[label])) is None:
                raise ValueError(f"{scene_id}: invalid {label}")
        evidence_path = (resolved.parent / str(override["evidence_report"])).resolve()
        if evidence_path.is_symlink() or not evidence_path.is_file() or evidence_path.stat().st_size <= 0:
            raise ValueError(f"{scene_id}: missing regular orientation evidence report")
        if sha256(evidence_path) != override["evidence_report_sha256"]:
            raise ValueError(f"{scene_id}: orientation evidence report hash mismatch")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            evidence.get("schema") != ORIENTATION_EVIDENCE_SCHEMA
            or evidence.get("scene_id") != scene_id
            or evidence.get("apple_tar_sha256") != override["apple_tar_sha256"]
            or evidence.get("method") != override["method"]
            or evidence.get("rotation_vector_int8_sha256")
            != override["rotation_vector_int8_sha256"]
            or evidence.get("approved_for_train_only_conversion") is not True
            or len(evidence.get("checked_raw_rgb_frames", ())) < 2
        ):
            raise ValueError(f"{scene_id}: orientation evidence contract failed")
        selected = dict(override)
        selected["evidence_report"] = str(evidence_path)
    return selected, {
        "schema": schema,
        "path": str(resolved),
        "sha256": sha256(resolved),
        "override_applied": override is not None,
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"tar member is not a regular file: {name}")
    handle: BinaryIO | None = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"cannot extract tar member: {name}")
    return handle.read()


def parse_json_member(archive: tarfile.TarFile, name: str) -> np.ndarray:
    return np.asarray(json.loads(member_bytes(archive, name)), dtype=np.float64)


def frame_ids(archive: tarfile.TarFile, scene_id: str) -> tuple[str, ...]:
    pattern = re.compile(rf"^{re.escape(scene_id)}/(\d+)\.gt/RT\.json$")
    ids = sorted(
        match.group(1)
        for name in archive.getnames()
        if (match := pattern.fullmatch(name)) is not None
    )
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{scene_id}: empty or duplicate raw frame IDs")
    for frame_id in ids:
        for suffix in FRAME_SUFFIXES:
            name = f"{scene_id}/{frame_id}{suffix}"
            try:
                member = archive.getmember(name)
            except KeyError as exc:
                raise ValueError(f"{scene_id}: missing tar member {name}") from exc
            if not member.isfile():
                raise ValueError(f"{scene_id}: non-regular tar member {name}")
    return tuple(ids)


def decode_png(payload: bytes, label: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to decode {label}")
    return image


def depth_shape(
    archive: tarfile.TarFile, scene_id: str, frame_id: str
) -> tuple[int, int]:
    size_name = f"{scene_id}/{frame_id}._gt/depth/size"
    try:
        text = member_bytes(archive, size_name).decode("utf-8")
        values = json.loads(text)
        if len(values) != 2:
            raise ValueError
        width, height = (int(values[0]), int(values[1]))
        if width <= 0 or height <= 0:
            raise ValueError
        return height, width
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        depth = decode_png(
            member_bytes(archive, f"{scene_id}/{frame_id}.gt/depth.png"),
            f"{scene_id}/{frame_id} depth",
        )
        if depth.ndim != 2:
            raise ValueError(f"{scene_id}/{frame_id}: depth is not single-channel")
        return int(depth.shape[0]), int(depth.shape[1])


def rz_quarter(k: int) -> np.ndarray:
    angle = k * math.pi / 2.0
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = (float(np.trace(left.T @ right)) - 1.0) / 2.0
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def validate_poses(poses: np.ndarray, scene_id: str) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{scene_id}: invalid pose array {poses.shape}")
    if not np.allclose(poses[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{scene_id}: poses are not homogeneous")
    rotations = poses[:, :3, :3]
    orthogonal = rotations.transpose(0, 2, 1) @ rotations
    if np.max(np.abs(orthogonal - np.eye(3))) > 1e-4:
        raise ValueError(f"{scene_id}: pose rotations are not orthogonal")
    if np.max(np.abs(np.linalg.det(rotations) - 1.0)) > 1e-4:
        raise ValueError(f"{scene_id}: pose rotations are not proper")


def infer_rot90(
    poses: np.ndarray,
    shapes: tuple[tuple[int, int], ...],
    min_margin_degrees: float,
) -> tuple[np.ndarray, dict]:
    """Infer the author's per-frame cardinal image orientation from raw RT.

    CA-1M contains sequences whose encoded image orientation changes while the
    physical camera motion remains smooth.  For a candidate image rotation k,
    ``R_raw @ Rz(k*pi/2)`` is the corrected camera rotation.  We select the
    parity-compatible k that makes adjacent corrected rotations continuous,
    then fix the unavoidable global 180-degree gauge by assigning k=0 to the
    modal state among frames already in the sequence's majority aspect ratio.
    """

    portrait = np.asarray([height > width for height, width in shapes], dtype=bool)
    portrait_count = int(portrait.sum())
    landscape_count = len(shapes) - portrait_count
    if portrait_count == landscape_count:
        raise ValueError("raw orientation vote is tied; refusing an arbitrary target")
    target_portrait = portrait_count > landscape_count
    target_mask = portrait == target_portrait
    allowed = [
        (0, 2) if bool(is_portrait) == target_portrait else (1, 3)
        for is_portrait in portrait
    ]
    rotations = poses[:, :3, :3]
    chosen = np.empty(len(shapes), dtype=np.int8)
    chosen[0] = 0 if 0 in allowed[0] else 1
    margins: list[float] = []
    corrected_steps: list[float] = []
    for index in range(1, len(shapes)):
        previous = rotations[index - 1] @ rz_quarter(int(chosen[index - 1]))
        costs = [
            rotation_distance(previous, rotations[index] @ rz_quarter(k))
            for k in allowed[index]
        ]
        order = np.argsort(costs)
        chosen[index] = allowed[index][int(order[0])]
        corrected_steps.append(float(costs[int(order[0])]))
        margins.append(float(costs[int(order[1])] - costs[int(order[0])]))

    target_states = chosen[target_mask]
    counts = collections.Counter(int(value) for value in target_states)
    count0, count2 = counts.get(0, 0), counts.get(2, 0)
    if count0 == count2:
        raise ValueError("global 0/180 orientation gauge is tied")
    modal = 0 if count0 > count2 else 2
    chosen = ((chosen.astype(np.int16) - modal) % 4).astype(np.int8)

    min_margin = min(margins, default=math.pi)
    if math.degrees(min_margin) < min_margin_degrees:
        raise ValueError(
            "cardinal orientation is ambiguous: "
            f"minimum temporal margin={math.degrees(min_margin):.3f} deg "
            f"< {min_margin_degrees:.3f} deg"
        )
    runs = []
    start = 0
    for index in range(1, len(chosen) + 1):
        if index == len(chosen) or chosen[index] != chosen[start]:
            runs.append(
                {"start": start, "end": index - 1, "k_ccw": int(chosen[start])}
            )
            start = index
    return chosen, {
        "target_orientation": "portrait" if target_portrait else "landscape",
        "raw_portrait_frames": portrait_count,
        "raw_landscape_frames": landscape_count,
        "rotation_counts": {
            str(k): int(np.count_nonzero(chosen == k)) for k in range(4)
        },
        "rotation_runs": runs,
        "minimum_choice_margin_degrees": math.degrees(min_margin),
        "corrected_step_degrees_q50_p95_max": [
            math.degrees(float(value))
            for value in np.quantile(corrected_steps or [0.0], (0.5, 0.95, 1.0))
        ],
    }


def infer_rot90_aspect_clockwise_to_portrait(
    shapes: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, dict]:
    portrait = np.asarray([height > width for height, width in shapes], dtype=bool)
    portrait_count = int(portrait.sum())
    landscape_count = len(shapes) - portrait_count
    if portrait_count <= landscape_count:
        raise ValueError("aspect-clockwise policy requires a portrait-majority scene")
    chosen = np.where(portrait, 0, 3).astype(np.int8)
    runs = []
    start = 0
    for index in range(1, len(chosen) + 1):
        if index == len(chosen) or chosen[index] != chosen[start]:
            runs.append({"start": start, "end": index - 1, "k_ccw": int(chosen[start])})
            start = index
    return chosen, {
        "method": "aspect_clockwise_to_portrait",
        "target_orientation": "portrait",
        "raw_portrait_frames": portrait_count,
        "raw_landscape_frames": landscape_count,
        "rotation_counts": {str(k): int(np.count_nonzero(chosen == k)) for k in range(4)},
        "rotation_runs": runs,
    }


def infer_rot90_aspect_clockwise_to_majority(
    shapes: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, dict]:
    """Normalize every frame to the majority aspect using clockwise turns.

    This rule is intentionally available only through a hash-bound orientation
    policy override.  Majority-aspect frames remain unchanged and minority-
    aspect frames receive ``k=3`` (90 degrees clockwise).
    """

    portrait = np.asarray([height > width for height, width in shapes], dtype=bool)
    portrait_count = int(portrait.sum())
    landscape_count = len(shapes) - portrait_count
    if portrait_count == landscape_count:
        raise ValueError("raw orientation vote is tied; refusing an arbitrary target")
    target_portrait = portrait_count > landscape_count
    target_mask = portrait == target_portrait
    chosen = np.where(target_mask, 0, 3).astype(np.int8)
    runs = []
    start = 0
    for index in range(1, len(chosen) + 1):
        if index == len(chosen) or chosen[index] != chosen[start]:
            runs.append({"start": start, "end": index - 1, "k_ccw": int(chosen[start])})
            start = index
    return chosen, {
        "method": "aspect_clockwise_to_majority",
        "target_orientation": "portrait" if target_portrait else "landscape",
        "raw_portrait_frames": portrait_count,
        "raw_landscape_frames": landscape_count,
        "rotation_counts": {
            str(k): int(np.count_nonzero(chosen == k)) for k in range(4)
        },
        "rotation_runs": runs,
    }


def infer_orientation(
    poses: np.ndarray,
    shapes: tuple[tuple[int, int], ...],
    rule: dict,
) -> tuple[np.ndarray, dict]:
    """Apply one validated orientation rule and verify any frozen vector hash."""

    method = rule.get("method")
    if method == "pose_continuity":
        chosen, orientation = infer_rot90(
            poses, shapes, float(rule["min_margin_degrees"])
        )
        orientation["method"] = method
    elif method == "aspect_clockwise_to_portrait":
        chosen, orientation = infer_rot90_aspect_clockwise_to_portrait(shapes)
    elif method == "aspect_clockwise_to_majority":
        chosen, orientation = infer_rot90_aspect_clockwise_to_majority(shapes)
    else:
        raise ValueError(f"unsupported orientation method: {method!r}")
    expected_hash = rule.get("rotation_vector_int8_sha256")
    if expected_hash is not None:
        observed_hash = hashlib.sha256(
            np.ascontiguousarray(chosen, dtype=np.int8).tobytes()
        ).hexdigest()
        if observed_hash != expected_hash:
            raise ValueError("orientation rotation-vector hash mismatch")
    return chosen, orientation


def expected_raw_metadata(
    archive: tarfile.TarFile,
    scene_id: str,
    ids: tuple[str, ...],
    shapes: tuple[tuple[int, int], ...],
    target_orientation: str,
) -> dict[str, np.ndarray]:
    poses = np.stack(
        [parse_json_member(archive, f"{scene_id}/{fid}.gt/RT.json") for fid in ids]
    )
    gravity = np.stack(
        [
            parse_json_member(archive, f"{scene_id}/{fid}.wide/T_gravity.json")
            for fid in ids
        ]
    )
    rgb_k = np.stack(
        [parse_json_member(archive, f"{scene_id}/{fid}.gt/image/K.json") for fid in ids]
    )
    depth_k = np.stack(
        [parse_json_member(archive, f"{scene_id}/{fid}.gt/depth/K.json") for fid in ids]
    )
    target_portrait = target_orientation == "portrait"
    selected = np.asarray(
        [(height > width) == target_portrait for height, width in shapes]
    )
    if not selected.any():
        raise ValueError(f"{scene_id}: no target-orientation intrinsics")
    return {
        "all_poses.npy": poses,
        "T_gravity.npy": gravity,
        "K_rgb.txt": rgb_k[selected].mean(axis=0),
        "K_depth.txt": depth_k[selected].mean(axis=0),
    }


def validate_hf_metadata(
    metadata_scene: Path,
    expected: dict[str, np.ndarray],
    scene_id: str,
) -> dict[str, str]:
    if not metadata_scene.is_dir() or metadata_scene.is_symlink():
        raise ValueError(f"{scene_id}: invalid Hugging Face metadata directory")
    hashes: dict[str, str] = {}
    for name in HF_REQUIRED:
        path = metadata_scene / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{scene_id}: missing/non-regular HF metadata {path}")
        hashes[name] = sha256(path)
    hf_poses = np.load(metadata_scene / "all_poses.npy")
    hf_gravity = np.load(metadata_scene / "T_gravity.npy")
    hf_rgb_k = np.loadtxt(metadata_scene / "K_rgb.txt")
    hf_depth_k = np.loadtxt(metadata_scene / "K_depth.txt")
    comparisons = {
        "all_poses.npy": hf_poses,
        "T_gravity.npy": hf_gravity,
        "K_rgb.txt": hf_rgb_k,
        "K_depth.txt": hf_depth_k,
    }
    for name, observed in comparisons.items():
        reference = expected[name]
        if observed.shape != reference.shape or not np.array_equal(observed, reference):
            maximum = (
                float(np.max(np.abs(observed - reference)))
                if observed.shape == reference.shape
                else float("inf")
            )
            raise ValueError(
                f"{scene_id}: raw Apple {name} disagrees with frozen HF metadata "
                f"(shape {reference.shape} vs {observed.shape}, max_abs={maximum})"
            )
    gt = np.load(metadata_scene / "after_filter_boxes.npy")
    if gt.size == 0 or gt.shape[-2:] != (8, 3) or not np.isfinite(gt).all():
        raise ValueError(f"{scene_id}: invalid after_filter_boxes.npy {gt.shape}")
    return hashes


def encoded_rotated_png(raw: bytes, k: int, label: str) -> tuple[bytes, tuple[int, ...]]:
    if k == 0:
        image = decode_png(raw, label)
        return raw, tuple(int(value) for value in image.shape)
    image = decode_png(raw, label)
    rotated = np.ascontiguousarray(np.rot90(image, k))
    success, encoded = cv2.imencode(".png", rotated)
    if not success:
        raise ValueError(f"failed to encode rotated {label}")
    return encoded.tobytes(), tuple(int(value) for value in rotated.shape)


def convert(args: argparse.Namespace) -> dict:
    tar_path = args.tar.resolve()
    metadata_scene = args.metadata_scene.resolve()
    staging_root = args.staging_root.resolve()
    promote_root = args.promote_root.resolve() if args.promote_root else None
    match = re.search(r"ca1m-val-(\d+)\.tar$", tar_path.name)
    if args.scene_id is None and match is None:
        raise ValueError("cannot infer scene ID from tar name; pass --scene-id")
    scene_id = args.scene_id or match.group(1)
    if not re.fullmatch(r"\d{8}", scene_id):
        raise ValueError(f"invalid CA-1M scene ID: {scene_id}")
    if not tar_path.is_file() or tar_path.is_symlink():
        raise ValueError(f"missing/non-regular Apple tar: {tar_path}")
    if metadata_scene.name != scene_id:
        raise ValueError(
            f"metadata scene ID {metadata_scene.name} does not match {scene_id}"
        )
    orientation_rule, orientation_policy = load_orientation_policy(
        args.orientation_policy, scene_id, tar_path
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = staging_root / scene_id
    building = staging_root / f".{scene_id}.building.{os.getpid()}"
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(f"staging scene already exists: {stage}")
    if building.exists() or building.is_symlink():
        raise FileExistsError(building)
    if args.promote:
        if promote_root is None:
            raise ValueError("--promote requires --promote-root")
        promote_root.mkdir(parents=True, exist_ok=True)
        live = promote_root / scene_id
        if live.exists() or live.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite existing live scene (including partial): {live}"
            )
        if staging_root.stat().st_dev != promote_root.stat().st_dev:
            raise ValueError("atomic promotion requires staging/live roots on one filesystem")

    building.mkdir(mode=0o755)
    (building / "rgb").mkdir()
    (building / "depth").mkdir()
    try:
        with tarfile.open(tar_path, mode="r:") as archive:
            ids = frame_ids(archive, scene_id)
            shapes = tuple(depth_shape(archive, scene_id, fid) for fid in ids)
            raw_poses = np.stack(
                [
                    parse_json_member(archive, f"{scene_id}/{fid}.gt/RT.json")
                    for fid in ids
                ]
            )
            validate_poses(raw_poses, scene_id)
            rotations, orientation = infer_orientation(
                raw_poses, shapes, orientation_rule
            )
            expected = expected_raw_metadata(
                archive,
                scene_id,
                ids,
                shapes,
                orientation["target_orientation"],
            )
            metadata_hashes = validate_hf_metadata(
                metadata_scene, expected, scene_id
            )

            target_depth_shape: tuple[int, int] | None = None
            target_rgb_shape: tuple[int, int] | None = None
            if len(ids) != len(rotations):
                raise ValueError("frame/rotation count mismatch")
            for index, (frame_id, k) in enumerate(zip(ids, rotations)):
                rgb_raw = member_bytes(
                    archive, f"{scene_id}/{frame_id}.wide/image.png"
                )
                depth_raw = member_bytes(
                    archive, f"{scene_id}/{frame_id}.gt/depth.png"
                )
                rgb, rgb_shape = encoded_rotated_png(
                    rgb_raw, int(k), f"{scene_id}/{frame_id} RGB"
                )
                depth, depth_shape_value = encoded_rotated_png(
                    depth_raw, int(k), f"{scene_id}/{frame_id} depth"
                )
                if len(depth_shape_value) != 2 or len(rgb_shape) != 3:
                    raise ValueError(f"{scene_id}/{frame_id}: invalid RGB-D channels")
                depth_hw = depth_shape_value[:2]
                rgb_hw = rgb_shape[:2]
                if target_depth_shape is None:
                    target_depth_shape = depth_hw
                    target_rgb_shape = rgb_hw
                if depth_hw != target_depth_shape or rgb_hw != target_rgb_shape:
                    raise ValueError(
                        f"{scene_id}/{frame_id}: output dimensions are inconsistent"
                    )
                if rgb_hw != (2 * depth_hw[0], 2 * depth_hw[1]):
                    raise ValueError(
                        f"{scene_id}/{frame_id}: RGB is not 2x depth resolution"
                    )
                atomic_bytes(building / "rgb" / f"{index}.png", rgb)
                atomic_bytes(building / "depth" / f"{index}.png", depth)

        for name in HF_REQUIRED:
            shutil.copyfile(metadata_scene / name, building / name)
        for optional_name in ("instances.json", "mesh.ply"):
            optional = metadata_scene / optional_name
            if optional.is_file() and not optional.is_symlink():
                shutil.copyfile(optional, building / optional.name)

        manifest = {
            "schema": SCHEMA,
            "scene_id": scene_id,
            "frame_count": len(ids),
            "frame_id_first_last": [ids[0], ids[-1]],
            "apple_tar": str(tar_path),
            "apple_tar_sha256": sha256(tar_path),
            "hf_metadata_scene": str(metadata_scene),
            "hf_metadata_sha256": metadata_hashes,
            "rgb_shape": list(target_rgb_shape or ()),
            "depth_shape": list(target_depth_shape or ()),
            "orientation": orientation,
            "orientation_policy": orientation_policy,
            "orientation_rule": orientation_rule,
            "image_policy": {
                "rgb_member": ".wide/image.png",
                "depth_member": ".gt/depth.png",
                "k0": "byte_copy",
                "rotated": "cv2_decode_np_rot90_cv2_default_png_encode",
            },
            "metadata_policy": "byte_copy_from_frozen_huggingface_checkout",
            "promotion_policy": "no_overwrite_same_filesystem_atomic_rename",
        }
        atomic_json(building / "conversion_manifest.json", manifest)
        os.replace(building, stage)
        output = stage
        if args.promote:
            assert promote_root is not None
            output = promote_root / scene_id
            os.replace(stage, output)
        result = dict(manifest)
        result.update({"output_scene": str(output), "promoted": bool(args.promote)})
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception:
        # Keep a failed build isolated for diagnosis; it can never be mistaken
        # for a completed numeric scene directory.
        if building.exists():
            failed = staging_root / f".{scene_id}.failed.{os.getpid()}"
            if not failed.exists():
                os.replace(building, failed)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--tar", type=Path, required=True)
    result.add_argument("--scene-id")
    result.add_argument("--metadata-scene", type=Path, required=True)
    result.add_argument("--staging-root", type=Path, required=True)
    result.add_argument("--min-orientation-margin-degrees", type=float, default=30.0)
    result.add_argument("--orientation-policy", type=Path)
    result.add_argument("--promote", action="store_true")
    result.add_argument("--promote-root", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if not math.isfinite(args.min_orientation_margin_degrees) or args.min_orientation_margin_degrees < 0:
        raise ValueError("orientation margin must be non-negative")
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
