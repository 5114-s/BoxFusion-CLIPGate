#!/usr/bin/env python3
"""Export and seal the complete confirmed YOLOE-direct track universe.

This is an isolated runtime wrapper around the frozen S2 producer.  It does
not edit that producer: before ``demo.py`` is imported, its controller factory
is replaced in memory by a thin observer which records the exact source-frame
schedule and, after normal finalization, reads every active or archived
confirmed candidate track.  Labels and CLIP state are never read or exported.

The normal terminal rows from the replay must be array-identical to the frozen
S2 diagnostics before the larger pre-terminal universe can be published.
Ground truth is intentionally absent from every API in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import MethodType
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPOSITORY_ROOT / "tools" / "boxfusion_tr3d_pipeline"
DEMO_PATH = PIPELINE_ROOT / "demo.py"

SCENE_SCHEMA = "boxfusion.s3_yoloe_confirmed_universe_scene.v1"
SEAL_SCHEMA = "boxfusion.s3_yoloe_confirmed_universe_dev3_seal.v1"
SCENE_SUFFIX = "_s3_confirmed_universe"
DEV3_SCENES = ("scene0568_00", "scene0606_01", "scene0377_02")
EXPECTED_CONFIRMED_COUNTS = {
    "scene0568_00": 88,
    "scene0606_01": 155,
    "scene0377_02": 41,
}

CONFIG_PATH = REPOSITORY_ROOT / "config" / "scannet_s2_yoloe_direct_shadow_score05.yaml"
CHECKPOINT_PATH = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/"
    "yoloe-11s-seg-pf.pt"
)
STREAM_SEAL_PATH = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_s2_yoloe_direct_shadow_score05"
    / "dev3_stream_input_seal.json"
)
S2_SEALED_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_s2_yoloe_direct_shadow_score05"
    / "sealed_dev3_v2_frozen"
    / "s2_yoloe_direct_shadow.json"
)
S2_DIAGNOSTIC_ROOT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_s2_yoloe_direct_shadow_score05"
    / "direct_candidates_v2_frozen"
)
PREREGISTRATION_PATH = REPOSITORY_ROOT / "docs" / "S3_YOLOE_CONFIRMED_UNIVERSE_PREREGISTRATION.md"

STREAM_SEAL_SHA256 = "cf363f9d92bd5b0c1aaa51ee6c200744fbf60404d671320c75df96ae20128655"
S2_MANIFEST_SHA256 = "0f15ee414003139a6b59e2092d8a0d73897acecba06132213fd7394f93cd5017"

FROZEN_INPUTS: Mapping[str, tuple[Path, str]] = {
    "config": (
        CONFIG_PATH,
        "4f3e9739b296197d41c0d322c0a1e30230385ccb8c1384a36615ffa413e83441",
    ),
    "yoloe_checkpoint": (
        CHECKPOINT_PATH,
        "292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d",
    ),
    "demo_source": (
        DEMO_PATH,
        "57fb58596401324785ee9696d16ebc15eed082df00dd6afede9e6d440b217423",
    ),
    "online_refinement_source": (
        PIPELINE_ROOT / "boxfusion" / "online_refinement.py",
        "0faf3d7d6242facdd9300a942fe1e2bf2364f5f9ebc17e8f8f278382a0102f61",
    ),
    "object_memory_source": (
        PIPELINE_ROOT / "boxfusion" / "object_memory.py",
        "c2f3f0e0753a34430f0d9d03c65039aa6eee80114a1337676ec4b5f1eaa60938",
    ),
    "supplemental_proposals_source": (
        PIPELINE_ROOT / "boxfusion" / "supplemental_proposals.py",
        "dcab601eb7bd70328be882e8944619e4dffd6d366214dd74eb6c2d5a3cfc001d",
    ),
    "tr3d_c2_observer_source": (
        PIPELINE_ROOT / "boxfusion" / "tr3d_c2_maskrgbd_observer.py",
        "108e4c1684a6f5e3b352b31a9d6e026e393bc1872540653312e7bdfb0d1e4778",
    ),
}

AUXILIARY_RUNTIME_INPUTS: Mapping[str, Path] = {
    "cutr_checkpoint": REPOSITORY_ROOT / "models" / "cutr_rgbd.pth",
    "clip_checkpoint": REPOSITORY_ROOT / "models" / "open_clip_pytorch_model.bin",
    "class_text": REPOSITORY_ROOT / "data" / "panoptic_categories_nomerge.txt",
    "class_features": REPOSITORY_ROOT / "data" / "class_features.pt",
    "pst": REPOSITORY_ROOT / "data" / "pst_1024_0.tiff",
    "pipeline_capture_stream": PIPELINE_ROOT / "boxfusion" / "capture_stream.py",
}

_FROZEN_DIAGNOSTIC_ARRAYS = {
    "boxes",
    "point_mask",
    "points",
    "quality_feature_names",
    "quality_features",
    "result_indices",
    "scene_id",
    "scores",
    "source_indices",
    "summary_json",
    "track_ids",
}


class S3ExportError(ValueError):
    """Raised when replay provenance or the no-GT export contract fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise S3ExportError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S3ExportError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise S3ExportError(f"{label} must contain a JSON object")
    return value


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as raw:
        with zipfile.ZipFile(
            raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload, np.ascontiguousarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue(), compresslevel=9)
        raw.flush()
        os.fsync(raw.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _hash_records(
    records: Mapping[str, Path], *, expected: Mapping[str, tuple[Path, str]] | None = None
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for name, raw_path in records.items():
        path = _regular_file(raw_path, name)
        digest = _sha256(path)
        if expected is not None:
            wanted = expected[name][1]
            if digest != wanted:
                raise S3ExportError(
                    f"frozen {name} SHA-256 mismatch: expected={wanted}, actual={digest}"
                )
        output[name] = {"path": os.fspath(path), "sha256": digest}
    return output


def _validate_frozen_inputs() -> dict[str, dict[str, str]]:
    records = {name: path for name, (path, _) in FROZEN_INPUTS.items()}
    result = _hash_records(records, expected=FROZEN_INPUTS)
    for name, (_, expected_hash) in FROZEN_INPUTS.items():
        result[name]["expected_sha256"] = expected_hash
    return result


def _load_stream_contract(scene: str) -> tuple[tuple[int, ...], dict[str, Any]]:
    if _sha256(_regular_file(STREAM_SEAL_PATH, "S2 stream seal")) != STREAM_SEAL_SHA256:
        raise S3ExportError("S2 stream seal SHA-256 mismatch")
    manifest = _read_json(STREAM_SEAL_PATH, "S2 stream seal")
    required = {
        "mode": "no_gt_stream_input_seal",
        "gt_access": False,
        "oracle_access": False,
        "output_mutation": False,
        "scene_count": 3,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise S3ExportError(f"stream seal contract mismatch for {key}")
    rows = manifest.get("scenes")
    if not isinstance(rows, list) or [row.get("scene_id") for row in rows] != list(
        DEV3_SCENES
    ):
        raise S3ExportError("stream seal dev3 order changed")
    row = rows[DEV3_SCENES.index(scene)]
    schedule = row.get("schedule")
    if not isinstance(schedule, dict):
        raise S3ExportError(f"stream seal schedule missing for {scene}")
    frames = schedule.get("recorded_frame_ids")
    if (
        not isinstance(frames, list)
        or not frames
        or any(isinstance(value, bool) or not isinstance(value, int) for value in frames)
        or frames != sorted(set(frames))
        or schedule.get("record_count") != len(frames)
    ):
        raise S3ExportError(f"invalid sealed frame schedule for {scene}")
    return tuple(frames), row


def _load_s2_anchor(scene: str) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    if _sha256(_regular_file(S2_SEALED_MANIFEST_PATH, "sealed S2 manifest")) != S2_MANIFEST_SHA256:
        raise S3ExportError("sealed S2 manifest SHA-256 mismatch")
    manifest = _read_json(S2_SEALED_MANIFEST_PATH, "sealed S2 manifest")
    if manifest.get("schema") != "boxfusion.s2_yoloe_direct_shadow.v1":
        raise S3ExportError("unexpected sealed S2 schema")
    if manifest.get("scene_order") != list(DEV3_SCENES):
        raise S3ExportError("sealed S2 scene order changed")
    ledger = manifest.get("scenes", {}).get(scene)
    if not isinstance(ledger, dict):
        raise S3ExportError(f"sealed S2 ledger missing for {scene}")
    path = _regular_file(
        S2_DIAGNOSTIC_ROOT / f"{scene}_tracks.npz", f"frozen S2 diagnostic for {scene}"
    )
    expected_hash = ledger.get("diagnostic_sha256_before")
    if (
        not isinstance(expected_hash, str)
        or expected_hash != ledger.get("diagnostic_sha256_after")
        or _sha256(path) != expected_hash
    ):
        raise S3ExportError(f"frozen S2 diagnostic hash differs from seal for {scene}")
    try:
        with np.load(path, allow_pickle=False) as source:
            if not _FROZEN_DIAGNOSTIC_ARRAYS.issubset(source.files):
                raise S3ExportError(f"frozen S2 diagnostic schema incomplete for {scene}")
            arrays = {
                name: np.array(source[name], copy=True)
                for name in sorted(_FROZEN_DIAGNOSTIC_ARRAYS)
            }
    except (OSError, ValueError) as error:
        if isinstance(error, S3ExportError):
            raise
        raise S3ExportError(f"invalid frozen S2 diagnostic for {scene}") from error
    # Labels are deliberately absent from the loaded set.
    if arrays["scene_id"].shape != () or str(arrays["scene_id"].item()) != scene:
        raise S3ExportError(f"frozen diagnostic scene mismatch for {scene}")
    return path, ledger, arrays


def _summary_without_runtime(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"provider_seconds", "geometry_seconds"}
    }


def _assert_terminal_identity(
    *,
    scene: str,
    result: Any,
    controller: Any,
    frozen: Mapping[str, np.ndarray],
    deterministic_bounded_sample: Any,
) -> dict[str, Any]:
    positions = np.flatnonzero(np.asarray(result.source_indices) == -1)
    comparisons = {
        "boxes": np.asarray(result.boxes)[positions],
        "scores": np.asarray(result.scores)[positions],
        "track_ids": np.asarray(result.stable_ids)[positions],
        "result_indices": positions.astype(np.int64),
        "quality_features": np.asarray(result.quality_features)[positions],
        "source_indices": np.asarray(result.source_indices)[positions],
    }
    for name, actual in comparisons.items():
        expected = np.asarray(frozen[name])
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise S3ExportError(f"S3 replay terminal {name} differs from frozen S2 for {scene}")
    manager = controller.track_manager
    if manager is None:
        raise S3ExportError("enabled S3 controller has no track manager")
    track_by_id = {track.track_id: track for track in manager.confirmed_tracks(True)}
    frozen_points = np.asarray(frozen["points"])
    frozen_mask = np.asarray(frozen["point_mask"])
    for row, stable_id in enumerate(comparisons["track_ids"]):
        track_id = -(int(stable_id) + 1)
        if track_id not in track_by_id:
            raise S3ExportError(f"terminal track {track_id} missing from confirmed universe")
        sampled = deterministic_bounded_sample(
            track_by_id[track_id].memory.points, frozen_points.shape[1]
        ).astype(frozen_points.dtype, copy=False)
        valid_count = int(frozen_mask[row].sum())
        if (
            valid_count != len(sampled)
            or not np.array_equal(frozen_points[row, :valid_count], sampled)
            or np.any(frozen_mask[row, valid_count:])
        ):
            raise S3ExportError(
                f"S3 replay terminal point sample differs from frozen S2 for {scene} row {row}"
            )
    try:
        frozen_summary = json.loads(str(np.asarray(frozen["summary_json"]).item()))
    except (ValueError, json.JSONDecodeError) as error:
        raise S3ExportError(f"invalid frozen S2 summary for {scene}") from error
    replay_summary = dict(result.summary)
    if _summary_without_runtime(replay_summary) != _summary_without_runtime(frozen_summary):
        raise S3ExportError(f"S3 replay summary counters differ from frozen S2 for {scene}")
    return {
        "array_identity": sorted(comparisons),
        "point_sample_identity": True,
        "summary_counter_identity_excluding_runtime_seconds": True,
        "terminal_output_count": len(positions),
        "frozen_summary": frozen_summary,
        "replay_summary": replay_summary,
    }


def _export_controller_universe(
    *,
    scene: str,
    controller: Any,
    result: Any,
    processed_source_frames: Sequence[int],
    expected_source_frames: Sequence[int],
    frozen_diagnostic: Mapping[str, np.ndarray],
    deterministic_bounded_sample: Any,
    output_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(processed_source_frames) != tuple(expected_source_frames):
        raise S3ExportError(f"S3 replay did not consume the exact sealed stream for {scene}")
    manager = controller.track_manager
    if manager is None:
        raise S3ExportError("enabled S3 controller has no track manager")
    tracks = manager.confirmed_tracks(include_archived=True)
    expected_count = EXPECTED_CONFIRMED_COUNTS[scene]
    if len(tracks) != expected_count or result.summary.get("confirmed_supplemental_tracks") != len(
        tracks
    ):
        raise S3ExportError(
            f"confirmed universe count mismatch for {scene}: {len(tracks)} != {expected_count}"
        )
    terminal_identity = _assert_terminal_identity(
        scene=scene,
        result=result,
        controller=controller,
        frozen=frozen_diagnostic,
        deterministic_bounded_sample=deterministic_bounded_sample,
    )
    archived_ids = set(manager.archived_tracks)
    terminal_by_track = {
        -(int(stable_id) + 1): int(position)
        for position, stable_id in enumerate(np.asarray(result.stable_ids))
        if int(np.asarray(result.source_indices)[position]) == -1
    }

    rows: list[dict[str, Any]] = []
    all_points: list[np.ndarray] = []
    point_offsets = [0]
    all_scores: list[float] = []
    score_offsets = [0]
    record_scores: list[float] = []
    record_keyframes: list[int] = []
    record_source_frames: list[int] = []
    record_boxes: list[np.ndarray] = []
    record_offsets = [0]
    for track in tracks:
        metadata = controller.supplemental_metadata.get(track.track_id)
        if metadata is None:
            raise S3ExportError(f"confirmed track {track.track_id} lacks frozen metadata")
        aabb = track.memory.aabb
        if aabb is None:
            raise S3ExportError(f"confirmed track {track.track_id} lacks a robust AABB")
        center, extent = (np.asarray(value, dtype=np.float32) for value in aabb)
        points = np.asarray(track.memory.points, dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) == 0
            or not np.isfinite(points).all()
            or np.any(extent <= 0.0)
        ):
            raise S3ExportError(f"invalid confirmed geometry for track {track.track_id}")
        if not track.confirmed or track.view_count < 3:
            raise S3ExportError(f"unconfirmed track leaked into S3 universe: {track.track_id}")
        scores = [float(value) for value in metadata.stats.scores]
        if not scores or not np.isfinite(scores).all():
            raise S3ExportError(f"invalid score provenance for track {track.track_id}")
        box_records = list(metadata.stats.box_records)
        for score, keyframe, box in box_records:
            if keyframe < 0 or keyframe >= len(expected_source_frames):
                raise S3ExportError(f"box-record frame is outside sealed stream: {track.track_id}")
            record_scores.append(float(score))
            record_keyframes.append(int(keyframe))
            record_source_frames.append(int(expected_source_frames[keyframe]))
            record_boxes.append(np.asarray(box, dtype=np.float32))
        record_offsets.append(len(record_scores))
        all_points.append(points)
        point_offsets.append(point_offsets[-1] + len(points))
        all_scores.extend(scores)
        score_offsets.append(len(all_scores))
        created_keyframe = int(track.created_frame)
        last_keyframe = int(track.last_frame)
        if not (
            0 <= created_keyframe < len(expected_source_frames)
            and 0 <= last_keyframe < len(expected_source_frames)
        ):
            raise S3ExportError(f"track lifecycle is outside sealed stream: {track.track_id}")
        rows.append(
            {
                "track_id": int(track.track_id),
                "archived": track.track_id in archived_ids,
                "created_keyframe_index": created_keyframe,
                "last_keyframe_index": last_keyframe,
                "created_source_frame_id": int(expected_source_frames[created_keyframe]),
                "last_source_frame_id": int(expected_source_frames[last_keyframe]),
                "created_lifecycle_step": int(track.created_lifecycle_step),
                "last_lifecycle_step": int(track.last_lifecycle_step),
                "hit_count": int(track.hit_count),
                "view_count": int(track.view_count),
                "memory_observation_count": int(track.memory.observation_count),
                "memory_unique_view_count": int(track.memory.unique_view_count),
                "center_extent": np.concatenate((center, extent)),
                "point_count": len(points),
                "mean_score": float(np.mean(scores)),
                "score_count": len(scores),
                "box_record_count": len(box_records),
                "terminal_output": track.track_id in terminal_by_track,
                "terminal_result_index": terminal_by_track.get(track.track_id, -1),
            }
        )
    if [row["track_id"] for row in rows] != sorted(row["track_id"] for row in rows):
        raise S3ExportError("confirmed track universe is not deterministically ordered")

    arrays: dict[str, np.ndarray] = {
        "scene_id": np.asarray(scene),
        "processed_source_frame_ids": np.asarray(processed_source_frames, dtype=np.int64),
        "track_id": np.asarray([row["track_id"] for row in rows], dtype=np.int64),
        "archived": np.asarray([row["archived"] for row in rows], dtype=bool),
        "created_keyframe_index": np.asarray(
            [row["created_keyframe_index"] for row in rows], dtype=np.int32
        ),
        "last_keyframe_index": np.asarray(
            [row["last_keyframe_index"] for row in rows], dtype=np.int32
        ),
        "created_source_frame_id": np.asarray(
            [row["created_source_frame_id"] for row in rows], dtype=np.int64
        ),
        "last_source_frame_id": np.asarray(
            [row["last_source_frame_id"] for row in rows], dtype=np.int64
        ),
        "created_lifecycle_step": np.asarray(
            [row["created_lifecycle_step"] for row in rows], dtype=np.int32
        ),
        "last_lifecycle_step": np.asarray(
            [row["last_lifecycle_step"] for row in rows], dtype=np.int32
        ),
        "hit_count": np.asarray([row["hit_count"] for row in rows], dtype=np.int32),
        "view_count": np.asarray([row["view_count"] for row in rows], dtype=np.int32),
        "memory_observation_count": np.asarray(
            [row["memory_observation_count"] for row in rows], dtype=np.int32
        ),
        "memory_unique_view_count": np.asarray(
            [row["memory_unique_view_count"] for row in rows], dtype=np.int32
        ),
        "box_center_extent": np.asarray(
            [row["center_extent"] for row in rows], dtype=np.float32
        ).reshape((-1, 6)),
        "point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "points_world": np.concatenate(all_points, axis=0).astype(np.float32, copy=False),
        "score_offsets": np.asarray(score_offsets, dtype=np.int32),
        "source_scores": np.asarray(all_scores, dtype=np.float32),
        "mean_score": np.asarray([row["mean_score"] for row in rows], dtype=np.float32),
        "box_record_offsets": np.asarray(record_offsets, dtype=np.int32),
        "box_record_score": np.asarray(record_scores, dtype=np.float32),
        "box_record_keyframe_index": np.asarray(record_keyframes, dtype=np.int32),
        "box_record_source_frame_id": np.asarray(record_source_frames, dtype=np.int64),
        "box_record_center_extent": np.asarray(record_boxes, dtype=np.float32).reshape((-1, 6)),
        "terminal_output": np.asarray(
            [row["terminal_output"] for row in rows], dtype=bool
        ),
        "terminal_result_index": np.asarray(
            [row["terminal_result_index"] for row in rows], dtype=np.int32
        ),
    }
    for value in arrays.values():
        value.setflags(write=False)

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{scene}{SCENE_SUFFIX}.json"
    npz_path = output_root / f"{scene}{SCENE_SUFFIX}.npz"
    if json_path.exists() or json_path.is_symlink() or npz_path.exists() or npz_path.is_symlink():
        raise S3ExportError(f"refusing to overwrite S3 scene artifact for {scene}")
    _write_deterministic_npz(npz_path, arrays)
    manifest = {
        "schema": SCENE_SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "gt_access": False,
        "oracle_access": False,
        "training_free": True,
        "online_learning": False,
        "past_current_only": True,
        "future_frames_used": False,
        "labels_read": False,
        "labels_exported": False,
        "clip_access_for_gate": False,
        "scene_id": scene,
        "confirmed_track_count": len(rows),
        "expected_confirmed_track_count": expected_count,
        "terminal_output_count": int(sum(row["terminal_output"] for row in rows)),
        "preterminal_rejected_track_count": int(sum(not row["terminal_output"] for row in rows)),
        "processed_source_frame_count": len(processed_source_frames),
        "processed_source_frames_exactly_match_stream_seal": True,
        "terminal_identity_to_frozen_s2": terminal_identity,
        "npz_file": npz_path.name,
        "npz_sha256": _sha256(npz_path),
        "array_content_sha256": _array_content_sha256(arrays),
        "provenance": dict(provenance),
    }
    try:
        _write_json_exclusive(json_path, manifest)
    except Exception:
        npz_path.unlink(missing_ok=True)
        raise
    return manifest


def _run_scene(args: argparse.Namespace) -> dict[str, Any]:
    scene = args.scene
    expected_frames, stream_ledger = _load_stream_contract(scene)
    frozen_path, s2_ledger, frozen_diagnostic = _load_s2_anchor(scene)
    frozen_before = _validate_frozen_inputs()
    auxiliary_before = _hash_records(AUXILIARY_RUNTIME_INPUTS)
    protected_before = {
        "stream_seal": _sha256(STREAM_SEAL_PATH),
        "s2_manifest": _sha256(S2_SEALED_MANIFEST_PATH),
        "s2_diagnostic": _sha256(frozen_path),
    }
    exporter_source = _regular_file(Path(__file__), "S3 exporter source")
    exporter_hash = _sha256(exporter_source)

    runtime_output = args.runtime_output_root.resolve()
    runtime_diagnostics = args.runtime_diagnostics_root.resolve()
    if runtime_output.exists() or runtime_diagnostics.exists():
        raise S3ExportError("runtime output roots must be create-only")
    runtime_output.mkdir(parents=True)
    runtime_diagnostics.mkdir(parents=True)

    # Do not import the repository-level ``tools``/``boxfusion`` packages
    # before selecting the frozen pipeline root.
    if os.fspath(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, os.fspath(PIPELINE_ROOT))
    import boxfusion.online_refinement as online_module  # type: ignore
    from boxfusion.object_memory import deterministic_bounded_sample  # type: ignore

    original_factory = online_module.build_online_refinement_controller
    state: dict[str, Any] = {"controller_count": 0, "exported": False}

    def observer_factory(*factory_args: Any, **factory_kwargs: Any) -> Any:
        controller = original_factory(*factory_args, **factory_kwargs)
        state["controller_count"] += 1
        if state["controller_count"] != 1:
            raise S3ExportError("demo constructed more than one online controller")
        processed: list[int] = []
        original_process = controller.process_keyframe
        original_finalize = controller.finalize

        def observed_process(_self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            if call_args:
                raise S3ExportError("unexpected positional process_keyframe call")
            cache_id = call_kwargs.get("cache_frame_id")
            if not isinstance(cache_id, str) or not cache_id.isdigit():
                raise S3ExportError("S3 replay lacks a numeric source-frame ID")
            processed.append(int(cache_id))
            return original_process(**call_kwargs)

        def observed_finalize(_self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            if call_args:
                raise S3ExportError("unexpected positional finalize call")
            result = original_finalize(**call_kwargs)
            provenance = {
                "exporter_source": os.fspath(exporter_source),
                "exporter_source_sha256": exporter_hash,
                "frozen_inputs": frozen_before,
                "auxiliary_runtime_inputs": auxiliary_before,
                "stream_seal": os.fspath(STREAM_SEAL_PATH),
                "stream_seal_sha256": STREAM_SEAL_SHA256,
                "stream_scene_schedule_sha256": stream_ledger["schedule"]["sha256"],
                "sealed_s2_manifest": os.fspath(S2_SEALED_MANIFEST_PATH),
                "sealed_s2_manifest_sha256": S2_MANIFEST_SHA256,
                "frozen_s2_diagnostic": os.fspath(frozen_path),
                "frozen_s2_diagnostic_sha256": s2_ledger["diagnostic_sha256_before"],
            }
            state["manifest"] = _export_controller_universe(
                scene=scene,
                controller=controller,
                result=result,
                processed_source_frames=processed,
                expected_source_frames=expected_frames,
                frozen_diagnostic=frozen_diagnostic,
                deterministic_bounded_sample=deterministic_bounded_sample,
                output_root=args.output_root,
                provenance=provenance,
            )
            state["exported"] = True
            return result

        controller.process_keyframe = MethodType(observed_process, controller)
        controller.finalize = MethodType(observed_finalize, controller)
        return controller

    online_module.build_online_refinement_controller = observer_factory
    original_argv = sys.argv[:]
    demo_argv = [
        os.fspath(DEMO_PATH),
        "scannet",
        "--model-path",
        os.fspath(AUXILIARY_RUNTIME_INPUTS["cutr_checkpoint"]),
        "--config",
        os.fspath(CONFIG_PATH),
        "--clip_path",
        os.fspath(AUXILIARY_RUNTIME_INPUTS["clip_checkpoint"]),
        "--seq",
        scene,
        "--class_txt",
        os.fspath(AUXILIARY_RUNTIME_INPUTS["class_text"]),
        "--class-features",
        os.fspath(AUXILIARY_RUNTIME_INPUTS["class_features"]),
        "--output-dir",
        os.fspath(runtime_output),
        "--diagnostics-root",
        os.fspath(runtime_diagnostics),
        "--online-proposal-checkpoint",
        os.fspath(CHECKPOINT_PATH),
        "--online-proposal-every-keyframes",
        "1",
        "--device",
        "cuda",
    ]
    try:
        sys.argv = demo_argv
        try:
            runpy.run_path(os.fspath(DEMO_PATH), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        sys.argv = original_argv
        online_module.build_online_refinement_controller = original_factory
    if state["controller_count"] != 1 or not state["exported"]:
        raise S3ExportError("S3 observer did not export exactly one finalized scene")

    protected_after = {
        "stream_seal": _sha256(STREAM_SEAL_PATH),
        "s2_manifest": _sha256(S2_SEALED_MANIFEST_PATH),
        "s2_diagnostic": _sha256(frozen_path),
    }
    if protected_after != protected_before:
        raise S3ExportError("a frozen S2 input changed during S3 replay")
    if _validate_frozen_inputs() != frozen_before:
        raise S3ExportError("a frozen model/config/source changed during S3 replay")
    if _hash_records(AUXILIARY_RUNTIME_INPUTS) != auxiliary_before:
        raise S3ExportError("an auxiliary runtime input changed during S3 replay")
    if _sha256(exporter_source) != exporter_hash:
        raise S3ExportError("S3 exporter source changed during replay")
    return state["manifest"]


def seal_dev3(*, scene_root: Path, output_manifest: Path) -> dict[str, Any]:
    """Create an aggregate no-GT seal after all three scene exports exist."""

    scene_root = scene_root.resolve()
    output_manifest = output_manifest.resolve()
    preregistration = _regular_file(PREREGISTRATION_PATH, "S3 preregistration")
    if output_manifest.exists() or output_manifest.is_symlink():
        raise S3ExportError(f"refusing to overwrite S3 dev3 seal: {output_manifest}")
    scene_records: dict[str, Any] = {}
    exporter_hashes = set()
    total = 0
    for scene in DEV3_SCENES:
        json_path = _regular_file(
            scene_root / f"{scene}{SCENE_SUFFIX}.json", f"S3 scene manifest for {scene}"
        )
        manifest = _read_json(json_path, f"S3 scene manifest for {scene}")
        required = {
            "schema": SCENE_SCHEMA,
            "mode": "shadow",
            "output_inert": True,
            "birth": False,
            "active_authorized": False,
            "gt_access": False,
            "oracle_access": False,
            "labels_read": False,
            "labels_exported": False,
            "scene_id": scene,
            "expected_confirmed_track_count": EXPECTED_CONFIRMED_COUNTS[scene],
            "processed_source_frames_exactly_match_stream_seal": True,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise S3ExportError(f"S3 scene seal mismatch for {scene}.{key}")
        count = manifest.get("confirmed_track_count")
        if count != EXPECTED_CONFIRMED_COUNTS[scene]:
            raise S3ExportError(f"S3 confirmed count mismatch for {scene}")
        npz_path = _regular_file(scene_root / str(manifest.get("npz_file")), "S3 scene NPZ")
        if _sha256(npz_path) != manifest.get("npz_sha256"):
            raise S3ExportError(f"S3 scene NPZ hash mismatch for {scene}")
        provenance = manifest.get("provenance")
        if not isinstance(provenance, dict):
            raise S3ExportError(f"S3 provenance missing for {scene}")
        exporter_hashes.add(provenance.get("exporter_source_sha256"))
        scene_records[scene] = {
            "manifest": os.fspath(json_path),
            "manifest_sha256": _sha256(json_path),
            "npz": os.fspath(npz_path),
            "npz_sha256": _sha256(npz_path),
            "array_content_sha256": manifest.get("array_content_sha256"),
            "confirmed_track_count": count,
            "terminal_output_count": manifest.get("terminal_output_count"),
            "preterminal_rejected_track_count": manifest.get(
                "preterminal_rejected_track_count"
            ),
        }
        total += int(count)
    if len(exporter_hashes) != 1 or not all(
        isinstance(value, str) and len(value) == 64 for value in exporter_hashes
    ):
        raise S3ExportError("S3 scenes were not produced by one frozen exporter")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    seal = {
        "schema": SEAL_SCHEMA,
        "mode": "sealed_no_gt_candidate_universe",
        "gt_access": False,
        "oracle_access": False,
        "H10_accessed": False,
        "full100_accessed": False,
        "active_birth_authorized": False,
        "scene_order": list(DEV3_SCENES),
        "scene_count": len(DEV3_SCENES),
        "confirmed_track_count_by_scene": dict(EXPECTED_CONFIRMED_COUNTS),
        "confirmed_track_count": total,
        "exporter_source_sha256": next(iter(exporter_hashes)),
        "stream_seal": {
            "path": os.fspath(STREAM_SEAL_PATH),
            "sha256": STREAM_SEAL_SHA256,
        },
        "sealed_s2_manifest": {
            "path": os.fspath(S2_SEALED_MANIFEST_PATH),
            "sha256": S2_MANIFEST_SHA256,
        },
        "preregistration": {
            "path": os.fspath(preregistration),
            "sha256": _sha256(preregistration),
        },
        "scenes": scene_records,
    }
    _write_json_exclusive(output_manifest, seal)
    return seal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export/seal no-GT S3 confirmed universe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-scene")
    run.add_argument("--scene", required=True, choices=DEV3_SCENES)
    run.add_argument("--output-root", required=True, type=Path)
    run.add_argument("--runtime-output-root", required=True, type=Path)
    run.add_argument("--runtime-diagnostics-root", required=True, type=Path)
    seal = subparsers.add_parser("seal-dev3")
    seal.add_argument("--scene-root", required=True, type=Path)
    seal.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run-scene":
        report = _run_scene(args)
        output = {
            "schema": SCENE_SCHEMA,
            "scene_id": report["scene_id"],
            "confirmed_track_count": report["confirmed_track_count"],
            "gt_access": False,
        }
    else:
        report = seal_dev3(scene_root=args.scene_root, output_manifest=args.out)
        output = {
            "schema": SEAL_SCHEMA,
            "scene_count": report["scene_count"],
            "confirmed_track_count": report["confirmed_track_count"],
            "gt_access": False,
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
