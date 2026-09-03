#!/usr/bin/env python3
"""Build the frozen, no-GT exact-frame schedule for the S3R H10 gate.

The legacy cache manifests are used only as an already-sealed frame clock.
This builder never enumerates a ScanNet frame directory.  It resolves only
the exact paths named by those manifests, records their byte hashes, and
excludes the one preregistered non-finite pose without replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_LIST = (
    REPOSITORY_ROOT
    / "evaluation"
    / "data_util"
    / "meta_data"
    / "scannetv2_boxer_past3_s1_holdout10.txt"
)
SCENE_ROOT = REPOSITORY_ROOT / "upstream_clean" / "scannet_readme_frames"
SOURCE_SCHEDULE_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/"
    "scannet-score05-gap25-postfilter-v2"
)
FORMAL_T05_ROOT = REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"

SCHEMA = "boxfusion.s3r_h10_exact_schedule.v1"
EXPECTED_HOLDOUT_SHA256 = (
    "8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90"
)
SCENE_ORDER = (
    "scene0304_00",
    "scene0412_00",
    "scene0019_00",
    "scene0575_00",
    "scene0426_00",
    "scene0426_03",
    "scene0578_00",
    "scene0665_00",
    "scene0050_01",
    "scene0025_00",
)
EXPECTED_SOURCE_MANIFEST_SHA256 = {
    "scene0304_00": "eb4f86b14376a32bc34977185f9845caafb3fe7e888de7775902428e9db4a9d7",
    "scene0412_00": "c5cd532f63c14708fee04a5a8ac21e3d61b24f342fe2eed19642302ff92ad900",
    "scene0019_00": "b10a9eabbe56e532ddfd316286a06e248aa3896c09593378c3a9a5d56fd3b49a",
    "scene0575_00": "a9aa923fd30b02b63c9e5c48909f7abb9e4ae4628511c067d5fbe187ad38939a",
    "scene0426_00": "fdfa46d8c4398b3a8138cd8b8578c357da2d354681c165262db4f92bc18a3bf9",
    "scene0426_03": "b4c3a6b35690ced5cb659e307e59441a3e8acfceafa144abaa0897621573371d",
    "scene0578_00": "c302e6aed65ceb5ee84217052553be1a608c364c2acfac6629f446abbb970a6c",
    "scene0665_00": "c2b1bcb36b2cdd61feb22bd5a088ee08410a62845f4403468ba253309dc9c5ae",
    "scene0050_01": "0fae3f64e16e8b5ac5816d839149c4c45c14f6f397962d01bb1b3056e4733d81",
    "scene0025_00": "9c11a6b2d30c244c7d16907e17b24b8676e05cda9962e9fdd507c42566a871ad",
}
EXPECTED_FORMAL_T05_SHA256 = {
    "scene0304_00": "8f118eee95237037a4f7f9e27ce1acd713054fd25c05812bcdabb7affd581b48",
    "scene0412_00": "504af4e0dcbc7663af72dc85d7d8aafe2031174ee58729dbe6a5cbf63c5fa0cb",
    "scene0019_00": "9a595dc17968aa45f83b283e6c2e48b7907acebfd20aa212f2ce8e44ce30faa8",
    "scene0575_00": "f0949f9e83829f97aebff64a7df3776c3dc8bd7b24e201e336da024ed642043b",
    "scene0426_00": "96890e541d1d5515f966ec4cfdb0040ff21b90f60a5733769b4703b106430d04",
    "scene0426_03": "029f7222bb658ecdd1cc53f414a6bbfb535b2c325f138346f70b9f5ebf1f287d",
    "scene0578_00": "ceca961e81edc33336df7774a9dac9f3801357ce94e7d1f23f7dbec1377720d0",
    "scene0665_00": "377fa1b9da5ecc9f2097eb781fcccbd953ae5c443adf2e28df339f64ccaa4d5d",
    "scene0050_01": "09ec901dfd47618e0357ab2aa07fdff0848518b6864ae5c8dcf3e8f3431592ef",
    "scene0025_00": "ea34fedf8abe64f17366b22e4a1ab188a7d2cf99d8680954c4a29af4e4f7eeb3",
}
EXPECTED_RAW_COUNTS = {
    "scene0304_00": 70,
    "scene0412_00": 94,
    "scene0019_00": 23,
    "scene0575_00": 117,
    "scene0426_00": 92,
    "scene0426_03": 49,
    "scene0578_00": 59,
    "scene0665_00": 41,
    "scene0050_01": 148,
    "scene0025_00": 77,
}
EXPECTED_EXCLUSIONS = {"scene0412_00": (2325,)}
EXPECTED_RAW_TOTAL = 770
EXPECTED_VALID_TOTAL = 769


class ScheduleBuildError(ValueError):
    """Raised when a frozen schedule/input invariant is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ScheduleBuildError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ScheduleBuildError(f"missing {label}: {path}") from error
    if not resolved.is_file():
        raise ScheduleBuildError(f"{label} must be a regular file: {path}")
    # Keep the logical workspace path.  ScanNet frame subdirectories are
    # intentionally symlink-mounted; resolving a parent would erase the scene
    # boundary needed by the exact relative-path contract.
    return path.absolute()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScheduleBuildError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ScheduleBuildError(f"{label} must be a JSON object")
    return value


def _resolve_color(scene_dir: Path, frame_id: int) -> Path:
    candidates = (
        scene_dir / "frames" / "color" / f"{frame_id}.png",
        scene_dir / "frames" / "color" / f"{frame_id}.jpg",
        scene_dir / "frames" / "color" / f"{frame_id}.jpeg",
    )
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise ScheduleBuildError(
            f"frame {frame_id} must resolve to exactly one non-symlink color file"
        )
    return present[0].absolute()


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _publish_create_only(output_path: Path, payload: bytes) -> None:
    if output_path.is_symlink() or output_path.exists():
        raise ScheduleBuildError(f"output already exists: {output_path}")
    parent = output_path.parent
    if parent.is_symlink():
        raise ScheduleBuildError(f"output parent must not be a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise ScheduleBuildError(f"output race: {output_path}") from error
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_schedule() -> dict[str, Any]:
    holdout = _regular_file(HOLDOUT_LIST, "H10 holdout list")
    if _sha256(holdout) != EXPECTED_HOLDOUT_SHA256:
        raise ScheduleBuildError("H10 holdout list hash mismatch")
    scenes = tuple(
        line.strip()
        for line in holdout.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if scenes != SCENE_ORDER:
        raise ScheduleBuildError(f"unexpected H10 scene order: {scenes}")

    scene_records: list[dict[str, Any]] = []
    raw_total = 0
    valid_total = 0
    for scene_id in scenes:
        scene_dir = (SCENE_ROOT / scene_id).absolute()
        intrinsic = _regular_file(
            scene_dir / "frames" / "intrinsic" / "intrinsic_color.txt",
            f"{scene_id} intrinsic",
        )
        source_manifest = SOURCE_SCHEDULE_ROOT / scene_id / "manifest.json"
        source = _read_json(source_manifest, f"{scene_id} source schedule")
        if _sha256(source_manifest) != EXPECTED_SOURCE_MANIFEST_SHA256[scene_id]:
            raise ScheduleBuildError(f"{scene_id} source schedule hash mismatch")
        expected_source_header = {
            "schema": "boxfusion.cutr_postfilter_cache.v2",
            "namespace": "scannet-score05-gap25-postfilter-v2",
            "scene_id": scene_id,
        }
        for key, expected in expected_source_header.items():
            if source.get(key) != expected:
                raise ScheduleBuildError(f"{scene_id} source {key} mismatch")
        raw_ids = [int(value) for value in source.get("recorded_frame_ids", [])]
        record_ids = [int(record.get("frame_id", -1)) for record in source.get("records", [])]
        if raw_ids != record_ids:
            raise ScheduleBuildError(f"{scene_id} manifest record order mismatch")
        if len(raw_ids) != EXPECTED_RAW_COUNTS[scene_id]:
            raise ScheduleBuildError(f"{scene_id} raw frame count mismatch")
        if raw_ids != list(range(0, 25 * len(raw_ids), 25)):
            raise ScheduleBuildError(f"{scene_id} is not the exact gap-25 clock")

        formal_t05 = _regular_file(
            FORMAL_T05_ROOT / f"{scene_id}_boxes.pkl", f"{scene_id} formal T05"
        )
        if _sha256(formal_t05) != EXPECTED_FORMAL_T05_SHA256[scene_id]:
            raise ScheduleBuildError(f"{scene_id} formal T05 hash mismatch")

        frames: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        valid_ids: list[int] = []
        for frame_id in raw_ids:
            pose = _regular_file(
                scene_dir / "frames" / "pose" / f"{frame_id}.txt",
                f"{scene_id}/{frame_id} pose",
            )
            try:
                pose_values = np.loadtxt(pose)
            except (OSError, ValueError) as error:
                raise ScheduleBuildError(f"invalid pose file: {pose}") from error
            finite_pose = pose_values.shape == (4, 4) and bool(np.isfinite(pose_values).all())
            if not finite_pose:
                excluded.append(
                    {
                        "frame_id": frame_id,
                        "reason": "nonfinite_pose",
                        "pose_relpath": pose.relative_to(scene_dir).as_posix(),
                        "pose_sha256": _sha256(pose),
                    }
                )
                continue
            color = _resolve_color(scene_dir, frame_id)
            depth = _regular_file(
                scene_dir / "frames" / "depth" / f"{frame_id}.png",
                f"{scene_id}/{frame_id} depth",
            )
            valid_ids.append(frame_id)
            frames.append(
                {
                    "frame_id": frame_id,
                    "color_relpath": color.relative_to(scene_dir).as_posix(),
                    "color_sha256": _sha256(color),
                    "depth_relpath": depth.relative_to(scene_dir).as_posix(),
                    "depth_sha256": _sha256(depth),
                    "pose_relpath": pose.relative_to(scene_dir).as_posix(),
                    "pose_sha256": _sha256(pose),
                }
            )

        expected_excluded = list(EXPECTED_EXCLUSIONS.get(scene_id, ()))
        if [item["frame_id"] for item in excluded] != expected_excluded:
            raise ScheduleBuildError(
                f"{scene_id} exclusions differ: {excluded} != {expected_excluded}"
            )
        raw_total += len(raw_ids)
        valid_total += len(valid_ids)
        scene_records.append(
            {
                "scene_id": scene_id,
                "source_schedule_manifest_relpath": source_manifest.relative_to(
                    SOURCE_SCHEDULE_ROOT
                ).as_posix(),
                "source_schedule_manifest_sha256": _sha256(source_manifest),
                "formal_t05_relpath": formal_t05.relative_to(REPOSITORY_ROOT).as_posix(),
                "formal_t05_sha256": _sha256(formal_t05),
                "intrinsic_color_relpath": intrinsic.relative_to(scene_dir).as_posix(),
                "intrinsic_color_sha256": _sha256(intrinsic),
                "raw_frame_ids": raw_ids,
                "valid_frame_ids": valid_ids,
                "excluded_frames": excluded,
                "frames": frames,
            }
        )

    if raw_total != EXPECTED_RAW_TOTAL or valid_total != EXPECTED_VALID_TOTAL:
        raise ScheduleBuildError(
            f"unexpected H10 totals: raw={raw_total}, valid={valid_total}"
        )
    return {
        "schema": SCHEMA,
        "scene_order": list(SCENE_ORDER),
        "raw_frame_count": raw_total,
        "valid_frame_count": valid_total,
        "holdout_list_sha256": EXPECTED_HOLDOUT_SHA256,
        "provider": {
            "annotation_path": None,
            "track": False,
            "directory_enumeration": False,
            "prefetch": False,
            "persist_before_advance": True,
        },
        "scenes": scene_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = build_schedule()
    payload = _canonical_bytes(schedule)
    _publish_create_only(args.output, payload)
    print(f"wrote {args.output}")
    print(f"sha256={hashlib.sha256(payload).hexdigest()}")
    print(
        f"raw_frames={EXPECTED_RAW_TOTAL} valid_frames={EXPECTED_VALID_TOTAL} "
        "excluded=scene0412_00/2325"
    )


if __name__ == "__main__":
    main()
