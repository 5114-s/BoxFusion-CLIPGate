#!/usr/bin/env python3
"""Run the frozen F0-F3 -> N0 -> RGB-D/MV3DIS -> dual-OBB shadow.

This runner is deliberately blind to ScanNet annotations, native prediction
pickles, evaluator code, classes, and CLIP.  It consumes only the already
sealed F2 masks, F3 causal track identities, F4 Boxer hypotheses, and current
RGB-D camera evidence.  No detector output is created or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from boxfusion.sam2_tsdf_mv3dis_shadow import (  # noqa: E402
    PROTOCOL_ID as GEOMETRY_PROTOCOL_ID,
    build_track_geometry,
    lift_mask_view,
    policy_receipt,
)
from boxfusion.sam2_video_track_provider import (  # noqa: E402
    PROTOCOL_ID as SAM2_PROTOCOL_ID,
    FrozenSAM2VideoTrackProvider,
)


SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100.shard.v1"
PROTOCOL_ID = "F0-F3-N0-SAM2-TSDF-MV3DIS-DUAL-OBB-SHADOW-PAPER100-V1"
EXPECTED_F3_SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.scene.v1"
EXPECTED_F4_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
EXPECTED_F2_ARRAY_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"
MASK_SHAPE = (480, 640)
MASK_PACKED_BYTES = MASK_SHAPE[0] * MASK_SHAPE[1] // 8
SOURCE_RE = re.compile(
    r"^(?P<scene>scene[0-9]{4}_[0-9]{2})/frame_(?P<frame>[0-9]{6})/raw_(?P<raw>[0-9]{3})$"
)

DEFAULT_SCENE_LIST = REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
DEFAULT_F2_ARRAY_ROOT = REPOSITORY_ROOT / "logs/scannet_fastsam_f2_paper100_score05/arrays"
DEFAULT_F3_ROOT = REPOSITORY_ROOT / "logs/scannet_fastsam_f3_openbox_paper100_score05/scenes"
DEFAULT_F4_ROOT = REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/scenes"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "logs/scannet_sam2_n0_fullroute_paper100_score05"


class N0RunnerError(RuntimeError):
    """A frozen input or output invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise N0RunnerError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise N0RunnerError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise N0RunnerError(f"{label} must be one JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise N0RunnerError(f"refusing to overwrite output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise N0RunnerError(f"refusing to overwrite output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scene_list(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scenes) != len(set(scenes)) or any(re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", item) is None for item in scenes):
        raise N0RunnerError("scene list is invalid")
    return scenes


def _decode_mask(packed: np.ndarray) -> np.ndarray:
    if packed.shape != (MASK_PACKED_BYTES,) or packed.dtype != np.uint8:
        raise N0RunnerError("sealed mask packbits differ")
    return np.unpackbits(packed, bitorder="little").reshape(MASK_SHAPE).astype(np.bool_)


class _FrameCache:
    def __init__(self, frame_rows: Mapping[int, Mapping[str, Any]]) -> None:
        self._rows = frame_rows
        self._values: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _verified_path(row: Mapping[str, Any], label: str) -> Path:
        path = Path(str(row.get("path", "")))
        expected = row.get("sha256")
        if path.is_symlink() or not path.is_file() or not isinstance(expected, str):
            raise N0RunnerError(f"sealed {label} receipt is invalid")
        if _sha256(path) != expected:
            raise N0RunnerError(f"sealed {label} SHA-256 differs: {path}")
        return path

    def get(self, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._values.get(frame_id)
        if cached is not None:
            return cached
        row = self._rows.get(frame_id)
        if not isinstance(row, Mapping):
            raise N0RunnerError(f"missing sealed frame input: {frame_id}")
        inputs = row.get("input")
        if not isinstance(inputs, Mapping):
            raise N0RunnerError(f"missing sealed frame input receipt: {frame_id}")
        rgb_path = self._verified_path(inputs.get("rgb", {}), "RGB")
        depth_path = self._verified_path(inputs.get("depth", {}), "depth")
        pose_path = self._verified_path(inputs.get("pose", {}), "pose")
        intrinsic_path = self._verified_path(inputs.get("intrinsic", {}), "intrinsic")
        bgr = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise N0RunnerError(f"sealed RGB cannot be decoded: {rgb_path}")
        if depth_raw is None or depth_raw.shape != (480, 640) or depth_raw.dtype != np.uint16:
            raise N0RunnerError(f"sealed depth cannot be decoded: {depth_path}")
        pose = np.loadtxt(pose_path, dtype=np.float64)
        intrinsic_full = np.loadtxt(intrinsic_path, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise N0RunnerError(f"sealed pose is invalid: {pose_path}")
        if intrinsic_full.shape not in ((3, 3), (4, 4)) or not np.isfinite(intrinsic_full).all():
            raise N0RunnerError(f"sealed intrinsic is invalid: {intrinsic_path}")
        if bgr.shape[:2] != (480, 640):
            bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depth = np.ascontiguousarray(depth_raw.astype(np.float64) / 1000.0)
        intrinsic = np.ascontiguousarray(intrinsic_full[:3, :3])
        value = (rgb, depth, np.ascontiguousarray(pose), intrinsic)
        self._values[frame_id] = value
        return value


def _load_f2_masks(path: Path, scene: str) -> tuple[dict[str, np.ndarray], str]:
    if path.is_symlink() or not path.is_file():
        raise N0RunnerError(f"missing sealed F2 arrays: {scene}")
    digest = _sha256(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                str(archive["schema"].item()) != EXPECTED_F2_ARRAY_SCHEMA
                or str(archive["scene_id"].item()) != scene
                or archive["mask_shape"].tolist() != list(MASK_SHAPE)
                or str(archive["mask_bitorder"].item()) != "little"
            ):
                raise N0RunnerError(f"sealed F2 array metadata differs: {scene}")
            source_ids = np.array(archive["source_ids"], copy=True)
            masks = np.array(archive["masks_packbits"], copy=True)
    except (KeyError, OSError, ValueError) as error:
        raise N0RunnerError(f"could not decode sealed F2 arrays: {scene}") from error
    if source_ids.ndim != 1 or masks.shape != (len(source_ids), MASK_PACKED_BYTES) or masks.dtype != np.uint8:
        raise N0RunnerError(f"sealed F2 mask arrays differ: {scene}")
    result: dict[str, np.ndarray] = {}
    for source, packed in zip(source_ids, masks):
        source_id = str(source)
        if source_id in result:
            raise N0RunnerError(f"duplicate sealed F2 source: {source_id}")
        result[source_id] = np.ascontiguousarray(packed)
    return result, digest


def _f4_indices(
    payload: Mapping[str, Any], scene: str
) -> tuple[dict[int, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if payload.get("complete") is not True or payload.get("contracts", {}).get("gt_access") is not False:
        raise N0RunnerError(f"F4 sidecar contract differs: {scene}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise N0RunnerError(f"F4 frames are absent: {scene}")
    frame_rows: dict[int, Mapping[str, Any]] = {}
    boxer: dict[str, Mapping[str, Any]] = {}
    for frame in frames:
        if not isinstance(frame, Mapping) or type(frame.get("frame_id")) is not int:
            raise N0RunnerError(f"invalid F4 frame: {scene}")
        frame_id = int(frame["frame_id"])
        frame_rows[frame_id] = frame
        sources = frame.get("sources")
        if not isinstance(sources, list):
            raise N0RunnerError(f"invalid F4 sources: {scene}:{frame_id}")
        for source in sources:
            if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
                raise N0RunnerError(f"invalid F4 source: {scene}:{frame_id}")
            hypotheses = source.get("hypotheses")
            if not isinstance(hypotheses, Mapping) or not isinstance(hypotheses.get("HB"), Mapping):
                raise N0RunnerError(f"missing F4 HB: {source.get('source_id')}")
            boxer[str(source["source_id"])] = hypotheses["HB"]
    return frame_rows, boxer


def _track_rows(payload: Mapping[str, Any], scene: str) -> list[Mapping[str, Any]]:
    if payload.get("complete") is not True or payload.get("contracts", {}).get("ground_truth_access") is not False:
        raise N0RunnerError(f"F3 sidecar contract differs: {scene}")
    rows = payload.get("tracks")
    if not isinstance(rows, list):
        raise N0RunnerError(f"F3 track ledger is absent: {scene}")
    confirmed = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("confirmed") is not True:
            continue
        sources = row.get("retained_source_ids")
        frames = row.get("retained_frame_ids")
        if (
            not isinstance(sources, list)
            or not isinstance(frames, list)
            or len(sources) != len(frames)
            or not 3 <= len(sources) <= 5
            or frames != sorted(frames)
            or len(frames) != len(set(frames))
        ):
            raise N0RunnerError(f"confirmed F3 track differs: {scene}:{row.get('track_id')}")
        for source_id, frame_id in zip(sources, frames):
            match = SOURCE_RE.fullmatch(str(source_id))
            if match is None or match["scene"] != scene or int(match["frame"]) != frame_id:
                raise N0RunnerError(f"F3 retained source identity differs: {source_id}")
        confirmed.append(row)
    return confirmed


def run_scene(
    *,
    scene: str,
    f2_array_path: Path,
    f3_path: Path,
    f4_path: Path,
    provider: FrozenSAM2VideoTrackProvider,
    output_root: Path,
    overwrite: bool,
    track_limit: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    f3_payload = _read_json(f3_path, f"F3 sidecar {scene}")
    f4_payload = _read_json(f4_path, f"F4 sidecar {scene}")
    if f3_payload.get("schema") != EXPECTED_F3_SCHEMA or f4_payload.get("schema") != EXPECTED_F4_SCHEMA:
        raise N0RunnerError(f"sealed upstream schema differs: {scene}")
    f2_masks, f2_sha = _load_f2_masks(f2_array_path, scene)
    frame_rows, boxer_by_source = _f4_indices(f4_payload, scene)
    tracks = _track_rows(f3_payload, scene)
    if track_limit is not None:
        tracks = tracks[:track_limit]
    frame_cache = _FrameCache(frame_rows)

    packed_outputs: list[np.ndarray] = []
    output_sources: list[str] = []
    output_track_ids: list[int] = []
    output_ordinals: list[int] = []
    output_predicted: list[bool] = []
    track_results: list[dict[str, Any]] = []
    valid_geometry_count = 0
    invalid_lift_count = 0
    sam2_complete_ms = 0.0

    for track_position, track in enumerate(tracks):
        track_id = int(track["track_id"])
        source_ids = [str(item) for item in track["retained_source_ids"]]
        frame_ids = [int(item) for item in track["retained_frame_ids"]]
        images: list[np.ndarray] = []
        corrections: list[np.ndarray] = []
        frame_values = []
        for source_id, frame_id in zip(source_ids, frame_ids):
            packed = f2_masks.get(source_id)
            if packed is None or source_id not in boxer_by_source:
                raise N0RunnerError(f"upstream retained source is missing: {source_id}")
            images.append(frame_cache.get(frame_id)[0])
            frame_values.append(frame_cache.get(frame_id))
            corrections.append(_decode_mask(packed))
        video_result = provider.predict_track(
            images_rgb=np.stack(images, axis=0),
            frozen_masks=np.stack(corrections, axis=0),
        )
        sam2_complete_ms += sum(item.complete_ms for item in video_result.timings)
        observations: list[dict[str, Any]] = []
        lifted_views = []
        for ordinal, (source_id, frame_id, mask, timing, predicted_flag, values) in enumerate(
            zip(
                source_ids,
                frame_ids,
                video_result.masks,
                video_result.timings,
                video_result.predicted_flags,
                frame_values,
            )
        ):
            rgb, depth, pose, intrinsic = values
            del rgb
            mask_index = len(packed_outputs)
            packed_outputs.append(np.packbits(mask.reshape(-1), bitorder="little"))
            output_sources.append(source_id)
            output_track_ids.append(track_id)
            output_ordinals.append(ordinal)
            output_predicted.append(predicted_flag)
            lifted = lift_mask_view(
                source_id=source_id,
                frame_id=frame_id,
                mask=mask,
                depth_m=depth,
                intrinsic=intrinsic,
                camera_to_world=pose,
            )
            if lifted is None:
                invalid_lift_count += 1
            else:
                lifted_views.append(lifted)
            observations.append(
                {
                    "source_id": source_id,
                    "frame_id": frame_id,
                    "observation_ordinal": ordinal,
                    "n0_mask_index": mask_index,
                    "n0_mask_pixel_count": int(np.count_nonzero(mask)),
                    "f2_correction_mask_pixel_count": int(np.count_nonzero(corrections[ordinal])),
                    "predicted_before_current_commit": bool(predicted_flag),
                    "current_correction_committed_after_output": True,
                    "lift_valid": lifted is not None,
                    "lift_support_pixel_count": None if lifted is None else lifted.support_pixel_count,
                    "lift_uncapped_voxel_count": None if lifted is None else lifted.uncapped_voxel_count,
                    "lift_stored_voxel_count": None if lifted is None else len(lifted.voxel_keys),
                    "timing_ms": {
                        "add_frame": timing.add_frame_ms,
                        "infer": timing.infer_ms,
                        "commit": timing.commit_ms,
                        "complete": timing.complete_ms,
                    },
                }
            )
        if len(lifted_views) == len(source_ids):
            geometry = build_track_geometry(
                views=lifted_views,
                boxer_by_source=boxer_by_source,
            )
        else:
            geometry = {
                "valid": False,
                "reason": "one_or_more_n0_views_failed_lifting",
                "valid_lifted_view_count": len(lifted_views),
            }
        valid_geometry_count += int(geometry.get("valid") is True)
        track_results.append(
            {
                "track_id": track_id,
                "track_position": track_position,
                "source_ids": source_ids,
                "frame_ids": frame_ids,
                "query_before_commit": True,
                "maximum_lookahead_observations": 0,
                "state_observation_bound": len(source_ids),
                "observations": observations,
                "geometry": geometry,
            }
        )

    arrays_path = output_root / "arrays" / f"{scene}.npz"
    scene_path = output_root / "scenes" / f"{scene}.json"
    arrays = {
        "schema": np.asarray("boxfusion.scannet_sam2_n0_fullroute_paper100.arrays.v1"),
        "scene_id": np.asarray(scene),
        "mask_shape": np.asarray(MASK_SHAPE, dtype=np.int64),
        "mask_bitorder": np.asarray("little"),
        "masks_packbits": np.stack(packed_outputs, axis=0).astype(np.uint8) if packed_outputs else np.empty((0, MASK_PACKED_BYTES), dtype=np.uint8),
        "source_ids": np.asarray(output_sources),
        "track_ids": np.asarray(output_track_ids, dtype=np.int64),
        "observation_ordinals": np.asarray(output_ordinals, dtype=np.int64),
        "predicted_before_commit": np.asarray(output_predicted, dtype=np.bool_),
    }
    _atomic_npz(arrays_path, arrays, overwrite=overwrite)
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "scene_id": scene,
        "complete": True,
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "native_output_mutation": False,
            "ground_truth_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "native_prediction_access": False,
            "semantic_or_clip_access": False,
            "training": False,
            "online_learning": False,
            "current_and_past_only": True,
            "query_before_commit": True,
            "maximum_lookahead_observations": 0,
        },
        "upstream": {
            "f2_arrays": {"path": os.fspath(f2_array_path.resolve()), "sha256": f2_sha},
            "f3_sidecar": {"path": os.fspath(f3_path.resolve()), "sha256": _sha256(f3_path)},
            "f4_sidecar": {"path": os.fspath(f4_path.resolve()), "sha256": _sha256(f4_path)},
        },
        "sam2": provider.production_receipt(),
        "geometry_policy": policy_receipt(),
        "arrays": {
            "path": os.fspath(arrays_path.resolve()),
            "sha256": _sha256(arrays_path),
            "mask_count": len(packed_outputs),
        },
        "counts": {
            "confirmed_track_count": len(tracks),
            "observation_count": len(packed_outputs),
            "valid_geometry_track_count": valid_geometry_count,
            "invalid_geometry_track_count": len(tracks) - valid_geometry_count,
            "invalid_lift_observation_count": invalid_lift_count,
        },
        "runtime": {
            "wall_seconds": elapsed,
            "sam2_synchronized_complete_ms": sam2_complete_ms,
            "mean_wall_ms_per_track": 1000.0 * elapsed / max(len(tracks), 1),
        },
        "tracks": track_results,
        "conclusion_guardrail": "Shadow/oracle evidence only; this file has no AP and cannot authorize birth.",
    }
    _atomic_json(scene_path, receipt, overwrite=overwrite)
    return {
        "scene_id": scene,
        "scene_path": os.fspath(scene_path.resolve()),
        "scene_sha256": _sha256(scene_path),
        "arrays_path": os.fspath(arrays_path.resolve()),
        "arrays_sha256": _sha256(arrays_path),
        "track_count": len(tracks),
        "observation_count": len(packed_outputs),
        "valid_geometry_track_count": valid_geometry_count,
        "wall_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--f2-array-root", type=Path, default=DEFAULT_F2_ARRAY_ROOT)
    parser.add_argument("--f3-root", type=Path, default=DEFAULT_F3_ROOT)
    parser.add_argument("--f4-root", type=Path, default=DEFAULT_F4_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--track-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise N0RunnerError("invalid shard index/count")
    if args.max_scenes is not None and args.max_scenes < 1:
        raise N0RunnerError("max-scenes must be positive")
    if args.track_limit is not None and args.track_limit < 1:
        raise N0RunnerError("track-limit must be positive")
    scenes = _scene_list(args.scene_list)
    if args.scene:
        requested = set(args.scene)
        missing = requested - set(scenes)
        if missing:
            raise N0RunnerError(f"requested scenes are outside paper100: {sorted(missing)}")
        scenes = [scene for scene in scenes if scene in requested]
    scenes = [scene for index, scene in enumerate(scenes) if index % args.shard_count == args.shard_index]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise N0RunnerError("shard has no scenes")

    provider = FrozenSAM2VideoTrackProvider()
    rows = []
    started = time.perf_counter()
    for position, scene in enumerate(scenes, start=1):
        row = run_scene(
            scene=scene,
            f2_array_path=args.f2_array_root / f"{scene}.npz",
            f3_path=args.f3_root / f"{scene}.json",
            f4_path=args.f4_root / f"{scene}.json",
            provider=provider,
            output_root=args.output_root,
            overwrite=args.overwrite,
            track_limit=args.track_limit,
        )
        rows.append(row)
        print(
            f"[{position}/{len(scenes)}] {scene}: tracks={row['track_count']} "
            f"valid_geometry={row['valid_geometry_track_count']} "
            f"wall={row['wall_seconds']:.1f}s",
            flush=True,
        )
    shard_receipt = {
        "schema": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "scene_count": len(rows),
        "track_count": sum(int(row["track_count"]) for row in rows),
        "observation_count": sum(int(row["observation_count"]) for row in rows),
        "valid_geometry_track_count": sum(int(row["valid_geometry_track_count"]) for row in rows),
        "wall_seconds": time.perf_counter() - started,
        "sam2_protocol_id": SAM2_PROTOCOL_ID,
        "geometry_protocol_id": GEOMETRY_PROTOCOL_ID,
        "scenes": rows,
    }
    shard_path = args.output_root / "shards" / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json"
    _atomic_json(shard_path, shard_receipt, overwrite=args.overwrite)
    print(f"Saved shard receipt: {shard_path}", flush=True)


if __name__ == "__main__":
    main()
