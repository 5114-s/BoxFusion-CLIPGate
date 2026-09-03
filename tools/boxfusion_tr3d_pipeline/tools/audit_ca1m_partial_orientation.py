#!/usr/bin/env python3
"""Validate inferred CA-1M cardinal rotations against released partial frames."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

from convert_ca1m_apple_tar import (
    decode_png,
    depth_shape,
    frame_ids,
    infer_rot90,
    member_bytes,
    parse_json_member,
    validate_poses,
)


def numeric_pngs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    rows = [item for item in path.iterdir() if item.is_file() and item.suffix == ".png"]
    if any(not item.stem.isdigit() for item in rows):
        raise ValueError(f"non-numeric PNG in {path}")
    return sorted(rows, key=lambda item: int(item.stem))


def audit(args: argparse.Namespace) -> dict:
    scene = args.scene
    tar_path = args.tar.resolve()
    released = args.released_scene.resolve()
    if released.name != scene:
        raise ValueError("released-scene directory does not match --scene")

    results: dict[str, object] = {"scene": scene, "required_policy": args.required_policy}
    with tarfile.open(tar_path, "r:") as archive:
        ids = frame_ids(archive, scene)
        shapes = tuple(depth_shape(archive, scene, frame_id) for frame_id in ids)
        poses = np.stack(
            [parse_json_member(archive, f"{scene}/{frame_id}.gt/RT.json") for frame_id in ids]
        )
        validate_poses(poses, scene)
        rotations, orientation = infer_rot90(poses, shapes, 0.0)
        target_portrait = orientation["target_orientation"] == "portrait"
        aspect_rotations = np.asarray(
            [
                0 if (height > width) == target_portrait else (3 if target_portrait else 1)
                for height, width in shapes
            ],
            dtype=np.int8,
        )
        results["frame_count"] = len(ids)
        results["orientation"] = orientation

        checked = 0
        matched = 0
        aspect_matched = 0
        mismatches: list[dict[str, object]] = []
        unique_rotation_counts = {str(k): 0 for k in range(4)}
        ambiguous_reference_rows = 0
        for kind, suffix in (("rgb", ".wide/image.png"), ("depth", ".gt/depth.png")):
            paths = numeric_pngs(released / kind)
            results[f"released_{kind}_frames"] = len(paths)
            for output in paths:
                index = int(output.stem)
                if not 0 <= index < len(ids):
                    raise ValueError(f"released frame index is out of range: {output}")
                raw = member_bytes(archive, f"{scene}/{ids[index]}{suffix}")
                raw_image = decode_png(raw, str(output))
                observed = cv2.imread(
                    str(output), cv2.IMREAD_COLOR if kind == "rgb" else cv2.IMREAD_UNCHANGED
                )
                if observed is None:
                    raise ValueError(f"cannot decode released frame: {output}")
                matching_k = [
                    k
                    for k in range(4)
                    if np.rot90(raw_image, k).shape == observed.shape
                    and np.array_equal(np.rot90(raw_image, k), observed)
                ]
                predicted = int(rotations[index])
                checked += 1
                if predicted in matching_k:
                    matched += 1
                elif len(mismatches) < args.max_mismatch_rows:
                    mismatches.append(
                        {
                            "kind": kind,
                            "index": index,
                            "frame_id": ids[index],
                            "predicted_k": predicted,
                            "matching_k": matching_k,
                        }
                    )
                if len(matching_k) == 1:
                    unique_rotation_counts[str(matching_k[0])] += 1
                else:
                    ambiguous_reference_rows += 1
                if int(aspect_rotations[index]) in matching_k:
                    aspect_matched += 1

    pose_ok = checked > 0 and matched == checked
    aspect_ok = checked > 0 and aspect_matched == checked
    selected_ok = pose_ok if args.required_policy == "pose_continuity" else aspect_ok
    results.update(
        {
            "checked_image_rows": checked,
            "matched_image_rows": matched,
            "match_fraction": matched / checked if checked else 0.0,
            "aspect_policy_matched_image_rows": aspect_matched,
            "aspect_policy_match_fraction": aspect_matched / checked if checked else 0.0,
            "aspect_policy_rotation_counts": {
                str(k): int(np.count_nonzero(aspect_rotations == k)) for k in range(4)
            },
            "unique_reference_rotation_counts": unique_rotation_counts,
            "ambiguous_reference_rows": ambiguous_reference_rows,
            "mismatches": mismatches,
            "pose_continuity_ok": pose_ok,
            "aspect_policy_ok": aspect_ok,
            "ok": selected_ok,
        }
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene", required=True)
    result.add_argument("--tar", type=Path, required=True)
    result.add_argument("--released-scene", type=Path, required=True)
    result.add_argument("--output", type=Path)
    result.add_argument("--max-mismatch-rows", type=int, default=20)
    result.add_argument(
        "--required-policy",
        choices=("pose_continuity", "aspect_clockwise_to_portrait"),
        default="pose_continuity",
    )
    return result


def main() -> int:
    result = audit(parser().parse_args())
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
