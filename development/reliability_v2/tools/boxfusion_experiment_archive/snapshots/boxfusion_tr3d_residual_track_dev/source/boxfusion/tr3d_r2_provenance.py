"""Fail-closed provenance helpers for the causal TR3D R2a observer.

R2a consumes real ScanNet RGB-D frames, whereas the parent TR3D cache only
binds the exported prefix point file.  This module closes that gap by hashing
the canonical prefix-manifest row and every selected RGB/depth/pose and
calibration artifact.  The artifact tree is content based and deterministic;
absolute paths are validated separately but are not used as hash ordering
keys.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from tools.tr3d_data import FrameBundle, PREFIX_SCHEMA


R2A_CLOCK_POLICY = "g0_post_frame_tail_guard_v1"
R2A_POSE_POLICY = "previous_valid_inf_only_v1"
R2A_TIMESTAMP_SEMANTICS = "zero_based_scannet_dataset_index"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    result = str(value)
    if result != result.lower() or _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return result


def _integer_list(row: Mapping[str, Any], name: str) -> list[int]:
    raw = row.get(name)
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw
    ):
        raise ValueError(f"prefix manifest {name} must be an integer list")
    values = [int(value) for value in raw]
    if any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError(
            f"prefix manifest {name} must be unique and nonnegative"
        )
    return values


def validate_prefix_manifest_row(
    row: Mapping[str, Any],
    *,
    expected_scene_id: str | None = None,
    expected_prefix_id: str | None = None,
    require_exported: bool = True,
) -> dict[str, Any]:
    """Validate and return a detached canonical R2a prefix row."""

    if not isinstance(row, Mapping):
        raise ValueError("prefix manifest row must be a mapping")
    detached = json.loads(canonical_json_bytes(dict(row)).decode("utf-8"))
    if detached.get("schema") != PREFIX_SCHEMA:
        raise ValueError("unsupported trajectory-prefix manifest schema")
    scene_id = detached.get("scene_id")
    prefix_id = detached.get("tag")
    if not isinstance(scene_id, str) or _SCENE_RE.fullmatch(scene_id) is None:
        raise ValueError("invalid prefix manifest scene_id")
    if not isinstance(prefix_id, str) or not prefix_id:
        raise ValueError("invalid prefix manifest tag")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError("prefix manifest scene mismatch")
    if expected_prefix_id is not None and prefix_id != expected_prefix_id:
        raise ValueError("prefix manifest tag mismatch")
    if require_exported and detached.get("status") != "exported":
        raise ValueError("R2a requires an exported trajectory prefix")
    if detached.get("clock_policy") != R2A_CLOCK_POLICY:
        raise ValueError("prefix manifest does not use the frozen-G0 clock")
    if detached.get("pose_policy") != R2A_POSE_POLICY:
        raise ValueError("prefix manifest does not use G0 pose carry-forward")
    if detached.get("source_timestamp_semantics") != R2A_TIMESTAMP_SEMANTICS:
        raise ValueError("prefix manifest timestamp semantics mismatch")

    frame_ids = _integer_list(detached, "frame_ids")
    used_frame_ids = _integer_list(detached, "used_frame_ids")
    timestamps = _integer_list(detached, "source_timestamps")
    used_timestamps = _integer_list(detached, "used_source_timestamps")
    if not frame_ids or not (
        frame_ids == used_frame_ids
        and timestamps == used_timestamps
        and len(frame_ids) == len(timestamps)
    ):
        raise ValueError("prefix manifest frame/timestamp lists disagree")
    if timestamps != sorted(timestamps):
        raise ValueError("prefix timestamps must be strictly increasing")
    stride = detached.get("frame_stride")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("prefix manifest frame_stride must be positive")
    if any(value % stride != 0 for value in timestamps):
        raise ValueError("prefix timestamps violate the keyframe stride")

    source_frame_count = detached.get("source_frame_count")
    if (
        isinstance(source_frame_count, bool)
        or not isinstance(source_frame_count, int)
        or source_frame_count < 1
    ):
        raise ValueError("prefix manifest source_frame_count must be positive")
    tail_guard = detached.get("tail_guard_frames")
    if (
        isinstance(tail_guard, bool)
        or not isinstance(tail_guard, int)
        or tail_guard < 1
    ):
        raise ValueError("prefix manifest tail_guard_frames must be positive")
    if tail_guard != stride:
        raise ValueError(
            "prefix manifest tail guard disagrees with frozen-G0 stride"
        )
    processed_frame_count = max(1, source_frame_count - tail_guard)
    recorded_processed_count = detached.get("processed_frame_count")
    if (
        isinstance(recorded_processed_count, bool)
        or not isinstance(recorded_processed_count, int)
        or recorded_processed_count != processed_frame_count
    ):
        raise ValueError(
            "processed_frame_count disagrees with frozen-G0 tail guard"
        )
    fraction = detached.get("fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError("prefix manifest fraction must be finite in (0,1]")
    full_timestamps = list(range(0, processed_frame_count, stride))
    prefix_count = max(
        1,
        min(
            len(full_timestamps),
            math.ceil(len(full_timestamps) * float(fraction)),
        ),
    )
    expected_timestamps = full_timestamps[:prefix_count]
    if timestamps != expected_timestamps:
        raise ValueError(
            "prefix timestamps disagree with the frozen-G0 schedule"
        )
    # ScanNet .sens exports used by frozen G0 name frames with their
    # zero-based dataset index.  The row contains no independent complete
    # source-id map, so accepting a different id sequence would make exact
    # schedule reconstruction impossible and could admit future frames.
    if frame_ids != expected_timestamps:
        raise ValueError(
            "prefix frame ids disagree with the frozen-G0 timestamp map"
        )
    if detached.get("sampled_frame_count") != len(frame_ids):
        raise ValueError("sampled_frame_count disagrees with frame list")
    if detached.get("last_frame_id") != frame_ids[-1]:
        raise ValueError("last_frame_id disagrees with frame list")
    if detached.get("last_source_timestamp") != timestamps[-1]:
        raise ValueError("last_source_timestamp disagrees with timestamp list")

    provenance = detached.get("pose_provenance")
    if not isinstance(provenance, list) or len(provenance) != len(frame_ids):
        raise ValueError("pose_provenance must align with selected frames")
    for index, (frame_id, timestamp, item) in enumerate(
        zip(frame_ids, timestamps, provenance)
    ):
        if not isinstance(item, Mapping):
            raise ValueError(f"pose_provenance[{index}] must be a mapping")
        if item.get("frame_id") != frame_id or item.get(
            "source_timestamp"
        ) != timestamp:
            raise ValueError("pose provenance frame/timestamp mismatch")
        if item.get("input_pose_frame_id") != frame_id:
            raise ValueError("input pose lineage disagrees with frame id")
        resolution = item.get("pose_resolution")
        if resolution not in {"direct", "carry_forward"}:
            raise ValueError("unsupported pose-resolution mode")
        source_timestamp = item.get("resolved_pose_source_timestamp")
        source_frame = item.get("resolved_pose_frame_id")
        if (
            isinstance(source_timestamp, bool)
            or not isinstance(source_timestamp, int)
            or source_timestamp < 0
            or source_timestamp > timestamp
            or isinstance(source_frame, bool)
            or not isinstance(source_frame, int)
            or source_frame < 0
        ):
            raise ValueError("invalid resolved pose lineage")
        _require_sha256(item.get("input_pose_sha256"), "input pose hash")
        _require_sha256(item.get("resolved_pose_sha256"), "resolved pose hash")
        if resolution == "direct" and (
            source_timestamp != timestamp or source_frame != frame_id
        ):
            raise ValueError("direct pose lineage must be identity")
        if resolution == "carry_forward" and source_timestamp >= timestamp:
            raise ValueError("carried pose must come from an earlier frame")
    return detached


def load_prefix_manifest(
    path: str | Path,
    *,
    prefix_id: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    manifest_path = Path(path)
    for line_number, raw in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            decoded = json.loads(raw)
            row = validate_prefix_manifest_row(
                decoded, expected_prefix_id=prefix_id
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"{manifest_path}:{line_number}: {error}"
            ) from error
        scene_id = row["scene_id"]
        if scene_id in result:
            raise ValueError(f"duplicate prefix manifest scene {scene_id}")
        result[scene_id] = row
    if not result:
        raise ValueError("prefix manifest is empty")
    return result


def _resolved_bundle_path(
    bundle: FrameBundle, mapping_name: str, frame_id: int
) -> Path:
    mapping = getattr(bundle, mapping_name)
    if frame_id not in mapping:
        raise ValueError(
            f"{bundle.scene_id}: {mapping_name} is missing frame {frame_id}"
        )
    path = Path(mapping[frame_id])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def frame_artifact_tree(
    row: Mapping[str, Any],
    bundle: FrameBundle,
) -> tuple[str, list[dict[str, Any]]]:
    """Hash the exact selected RGB-D, input/resolved poses and calibration."""

    checked = validate_prefix_manifest_row(
        row,
        expected_scene_id=bundle.scene_id,
        expected_prefix_id=str(row.get("tag")),
    )
    records: list[dict[str, Any]] = []
    calibration = {
        "intrinsic_depth": Path(bundle.intrinsic_depth),
        "intrinsic_color": Path(bundle.intrinsic_color),
        "extrinsic_depth": Path(bundle.extrinsic_depth),
        "extrinsic_color": Path(bundle.extrinsic_color),
    }
    for name, path in sorted(calibration.items()):
        resolved = path.resolve()
        records.append({
            "artifact": f"calibration/{name}",
            "sha256": sha256_file(resolved),
        })

    for frame_id, timestamp, pose_item in zip(
        checked["used_frame_ids"],
        checked["used_source_timestamps"],
        checked["pose_provenance"],
    ):
        for modality in ("color", "depth"):
            path = _resolved_bundle_path(bundle, modality, frame_id)
            records.append({
                "artifact": f"frame/{timestamp:08d}/{modality}",
                "frame_id": frame_id,
                "sha256": sha256_file(path),
            })
        input_pose = _resolved_bundle_path(bundle, "pose", frame_id)
        input_sha = sha256_file(input_pose)
        if input_sha != pose_item["input_pose_sha256"]:
            raise ValueError("input pose content changed after prefix export")
        resolved_frame = int(pose_item["resolved_pose_frame_id"])
        resolved_pose = _resolved_bundle_path(bundle, "pose", resolved_frame)
        resolved_sha = sha256_file(resolved_pose)
        if resolved_sha != pose_item["resolved_pose_sha256"]:
            raise ValueError("resolved pose content changed after prefix export")
        records.extend((
            {
                "artifact": f"frame/{timestamp:08d}/input_pose",
                "frame_id": frame_id,
                "sha256": input_sha,
            },
            {
                "artifact": f"frame/{timestamp:08d}/resolved_pose",
                "frame_id": resolved_frame,
                "sha256": resolved_sha,
            },
        ))
    records.sort(key=lambda item: str(item["artifact"]))
    names = [str(item["artifact"]) for item in records]
    if len(names) != len(set(names)):
        raise ValueError("frame artifact tree contains duplicate keys")
    return canonical_json_sha256(records), records


def load_resolved_poses(
    row: Mapping[str, Any],
    bundle: FrameBundle,
) -> dict[int, np.ndarray]:
    """Load manifest-authoritative resolved poses keyed by frame id."""

    checked = validate_prefix_manifest_row(
        row,
        expected_scene_id=bundle.scene_id,
        expected_prefix_id=str(row.get("tag")),
    )
    output: dict[int, np.ndarray] = {}
    for frame_id, item in zip(
        checked["used_frame_ids"], checked["pose_provenance"]
    ):
        source_frame = int(item["resolved_pose_frame_id"])
        path = _resolved_bundle_path(bundle, "pose", source_frame)
        if sha256_file(path) != item["resolved_pose_sha256"]:
            raise ValueError("resolved pose content changed after export")
        matrix = np.loadtxt(path, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{path}: resolved pose must be finite [4,4]")
        output[int(frame_id)] = matrix
    return output


def code_artifact_tree_sha256(paths: Sequence[str | Path]) -> str:
    records = []
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"name": path.name, "sha256": sha256_file(path)})
    names = [item["name"] for item in records]
    if len(names) != len(set(names)):
        raise ValueError("R2 code artifact basenames must be unique")
    return canonical_json_sha256(records)


__all__ = [
    "R2A_CLOCK_POLICY",
    "R2A_POSE_POLICY",
    "R2A_TIMESTAMP_SEMANTICS",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "code_artifact_tree_sha256",
    "frame_artifact_tree",
    "load_prefix_manifest",
    "load_resolved_poses",
    "sha256_file",
    "validate_prefix_manifest_row",
]
