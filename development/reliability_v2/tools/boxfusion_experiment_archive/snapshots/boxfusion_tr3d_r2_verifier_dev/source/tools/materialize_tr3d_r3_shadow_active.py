#!/usr/bin/env python3
"""Create an isolated R3 primary-rule shadow-active prediction tree.

This command is deliberately a *materializer*, not an inference launcher.  It
reads a manifest-pinned G0 prediction tree and immutable R3 observer sidecars,
then writes a new prediction namespace.  It never reads ground truth,
counterfactual reports, CLIP features, or an observer output directory.

The output is shadow-only: creating it proves that the pre-registered replay
can be represented by ordinary BoxFusion prediction pickles; it does not turn
the validation audit into authorization for a formal active method.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import struct
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import verify_frozen_anchor_manifest  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    code_artifact_tree_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    sha256_file,
)
from boxfusion.tr3d_r2_cache import tr3d_r2_cache_path  # noqa: E402
from boxfusion.tr3d_r2b_cache import tr3d_r2b_cache_path  # noqa: E402
from boxfusion.tr3d_r3_cache import (  # noqa: E402
    load_tr3d_r3_cache,
    tr3d_r3_cache_path,
)
from boxfusion.tr3d_residual_cache import tr3d_residual_cache_path  # noqa: E402
from tools.run_tr3d_r3_near_observer import (  # noqa: E402
    REPORT_SCHEMA as R3_EXPORT_SCHEMA,
    _code_hash as current_r3_code_hash,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


MANIFEST_SCHEMA = "boxfusion.tr3d_r3_shadow_active_manifest.v1"
CONFIG_SCHEMA = "boxfusion.tr3d_r3_shadow_active_config.v1"
PICKLE_PROTOCOL = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--r3-export-report", type=Path, required=True)
    parser.add_argument("--r3-cache-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--resume", action="store_true")
    return parser


def _snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_name": payload["anchor_name"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "scene_list_sha256": payload["scene_list_sha256"],
    }


def _sha_tree(rows: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(rows.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _config() -> dict[str, Any]:
    from boxfusion.tr3d_r3_active import active_config

    config = active_config()
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("ground_truth_access")
        or config.get("clip_access")
        or config.get("candidate_geometry_frame") != "scannet_unaligned_world"
        or config.get("axis_alignment_applied_by_materializer")
    ):
        raise ValueError("R3 active API exposes an unsafe materialization config")
    return config


def _code_hash() -> str:
    # Bind the materializer and the pure replacement implementation.  The
    # delayed import keeps this file importable while that isolated API lands.
    from boxfusion import tr3d_r3_active

    return code_artifact_tree_sha256((Path(__file__), Path(tr3d_r3_active.__file__)))


def _load_prediction(path: Path) -> list[list[tuple[Any, np.ndarray, Any]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - manifest-pinned local result
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"{path}: expected a one-scene BoxFusion list")
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"{path}: row {index} must be a 3-tuple")
        label, corners, score = row
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"{path}: row {index} label must be integer")
        geometry = np.asarray(corners)
        if (
            geometry.shape != (8, 3)
            or geometry.dtype.hasobject
            or not np.isfinite(geometry).all()
        ):
            raise ValueError(f"{path}: row {index} geometry must be finite numeric [8,3]")
        if isinstance(score, bool) or not isinstance(score, (float, np.floating)):
            raise ValueError(f"{path}: row {index} score must be floating point")
        if not math.isfinite(float(score)):
            raise ValueError(f"{path}: row {index} score is non-finite")
    return payload


def _label_bytes(value: Any) -> bytes:
    return pickle.dumps(value, protocol=PICKLE_PROTOCOL)


def _score_bytes(value: Any) -> bytes:
    if isinstance(value, np.floating):
        return np.asarray(value).tobytes()
    return struct.pack("!d", float(value))


def _geometry_equal(left: object, right: object) -> bool:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    return bool(
        lhs.dtype == rhs.dtype
        and lhs.shape == rhs.shape
        and lhs.strides == rhs.strides
        and lhs.flags.c_contiguous == rhs.flags.c_contiguous
        and lhs.flags.f_contiguous == rhs.flags.f_contiguous
        and lhs.tobytes(order="A") == rhs.tobytes(order="A")
    )


def _validate_invariants(
    source: list[list[tuple[Any, np.ndarray, Any]]],
    active: list[list[tuple[Any, np.ndarray, Any]]],
) -> dict[str, int | bool]:
    # Validate output syntax first, without trusting the active API.
    if not isinstance(active, list) or len(active) != 1 or not isinstance(active[0], list):
        raise ValueError("active API returned a malformed BoxFusion payload")
    if len(active[0]) != len(source[0]):
        raise ValueError("active API changed prediction count")
    changed = 0
    for index, (before, after) in enumerate(zip(source[0], active[0])):
        if not isinstance(after, tuple) or len(after) != 3:
            raise ValueError(f"active row {index} is not a 3-tuple")
        if type(after[0]) is not type(before[0]) or _label_bytes(after[0]) != _label_bytes(before[0]):
            raise ValueError(f"active row {index} changed label bytes")
        if type(after[2]) is not type(before[2]) or _score_bytes(after[2]) != _score_bytes(before[2]):
            raise ValueError(f"active row {index} changed score bytes")
        geometry = np.asarray(after[1])
        if (
            geometry.shape != (8, 3)
            or geometry.dtype.hasobject
            or not np.isfinite(geometry).all()
        ):
            raise ValueError(f"active row {index} has invalid geometry")
        changed += int(not _geometry_equal(before[1], after[1]))
    return {
        "rows": len(source[0]),
        "changed_rows": changed,
        "count_unchanged": True,
        "order_unchanged": True,
        "labels_unchanged": True,
        "scores_unchanged": True,
    }


def _materialize_payload(
    source: list[list[tuple[Any, np.ndarray, Any]]], cache: Any
) -> tuple[list[list[tuple[Any, np.ndarray, Any]]], dict[str, Any]]:
    """Delayed adapter to the dependency-light active replacement API."""

    from boxfusion.tr3d_r3_active import (
        materialize_shadow_active_prediction,
        validate_shadow_active_prediction,
    )

    result = materialize_shadow_active_prediction(source, cache)
    if isinstance(result, tuple) and len(result) == 2:
        payload, summary = result
    elif hasattr(result, "payload") and hasattr(result, "summary"):
        payload, summary = result.payload, result.summary
    else:
        raise TypeError("materialize_shadow_active_prediction returned an unsupported result")
    validated = validate_shadow_active_prediction(source, payload, cache)

    def plain(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            result: Any = dict(value)
        elif hasattr(value, "as_dict"):
            result = value.as_dict()
        elif is_dataclass(value):
            result = asdict(value)
        else:
            raise TypeError("active materialization summary must be a mapping/dataclass")
        if not isinstance(result, Mapping):
            raise TypeError("active materialization summary must resolve to a mapping")
        # Round-trip through strict JSON to recursively normalize tuples and
        # reject hidden ndarray/non-finite values before writing a manifest.
        return json.loads(
            json.dumps(dict(result), sort_keys=True, allow_nan=False)
        )

    materialized_summary = plain(summary)
    validated_summary = plain(validated)
    if materialized_summary != validated_summary:
        raise ValueError("active API materialization/validation summaries disagree")
    return payload, validated_summary


def _write_bytes_create_only(path: Path, encoded: bytes) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable shadow-active prediction exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    _write_bytes_create_only(path, encoded)


def _load_export(path: Path, scenes: Sequence[str]) -> dict[str, Any]:
    export = json.loads(path.read_text(encoding="utf-8"))
    if export.get("schema") != R3_EXPORT_SCHEMA:
        raise ValueError("unsupported R3 export schema")
    if (
        not export.get("observer_only")
        or export.get("mutation_enabled")
        or int(export.get("applied_count", -1)) != 0
        or export.get("ground_truth_access")
        or export.get("clip_access")
        or not export.get("clip_semantics_unchanged")
    ):
        raise ValueError("R3 export violates its observer-only contract")
    ordered = [str(row.get("scene_id")) for row in export.get("scenes", [])]
    if ordered != list(scenes) or int(export.get("scene_count", -1)) != len(scenes):
        raise ValueError("R3 export ordered scene set mismatch")
    if len(set(ordered)) != len(ordered):
        raise ValueError("R3 export contains duplicate scenes")
    config = export.get("r3_config")
    if not isinstance(config, dict) or canonical_json_sha256(config) != export.get(
        "r3_config_sha256"
    ):
        raise ValueError("R3 export config hash mismatch")
    if current_r3_code_hash() != export.get("r3_code_sha256"):
        raise ValueError("current R3 observer code differs from immutable export")
    rows = export.get("scenes", [])
    if any(
        not isinstance(row.get("r3_sidecar_sha256"), str)
        or len(row["r3_sidecar_sha256"]) != 64
        for row in rows
    ):
        raise ValueError("R3 export has an invalid sidecar SHA")
    return export


def _claim_namespace(root: Path, manifest: Path, *, resume: bool) -> None:
    resolved_root = root.resolve()
    resolved_manifest = manifest.resolve()
    if resolved_manifest == resolved_root or resolved_root in resolved_manifest.parents:
        raise ValueError("manifest must be outside the prediction-only output root")
    if manifest.is_symlink():
        raise ValueError(f"manifest must not be a symlink: {manifest}")
    if manifest.exists() and not manifest.is_file():
        raise ValueError(f"manifest must be a regular file: {manifest}")
    if manifest.exists() and manifest.stat().st_mode & 0o222:
        raise ValueError(f"existing manifest is not immutable: {manifest}")
    if manifest.exists() and not resume:
        raise FileExistsError(f"shadow-active manifest already exists: {manifest}")
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"output root must be a real directory: {root}")
        if not resume:
            raise FileExistsError(f"shadow-active output root already exists: {root}")
    else:
        if resume:
            raise FileNotFoundError(f"resume output root is absent: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()


def _existing_names(root: Path, scenes: Sequence[str]) -> set[str]:
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    found = {path.name for path in root.iterdir()}
    extra = sorted(found - expected)
    if extra:
        raise ValueError(f"shadow-active output root has orphan/extra entries: {extra}")
    invalid = sorted(
        path.name
        for path in root.iterdir()
        if path.name in expected
        and (
            path.is_symlink()
            or not path.is_file()
            or bool(path.stat().st_mode & 0o222)
        )
    )
    if invalid:
        raise ValueError(
            "shadow-active predictions must be regular immutable files: "
            f"{invalid}"
        )
    return found


def _load_bound_cache(
    *,
    scene_id: str,
    prefix_id: str,
    export: Mapping[str, Any],
    frozen_manifest: Path,
    frozen_root: Path,
    r3_root: Path,
    scans_root: Path,
    prefix_rows: Mapping[str, Mapping[str, Any]],
) -> Any:
    parent_root = Path(str(export["parent_cache_root"])).resolve()
    prefix_manifest = Path(str(export["prefix_manifest"])).resolve()
    config = export["r3_config"]
    r2a_enabled = bool(config.get("r2a_enabled"))
    r2b_enabled = bool(config.get("r2b_enabled"))
    r2a_root = Path(str(export["r2a_cache_root"])).resolve() if r2a_enabled else None
    r2b_root = Path(str(export["r2b_cache_root"])).resolve() if r2b_enabled else None
    frames_root = Path(str(export["frames_root"])).resolve() if r2a_enabled else None
    manifest_row_sha = frame_tree_sha = ""
    if r2a_enabled:
        row = prefix_rows[scene_id]
        manifest_row_sha = canonical_json_sha256(row)
        bundle = discover_frame_bundle(frames_root, scene_id)
        frame_tree_sha, _ = frame_artifact_tree(row, bundle)
    source_path = frozen_root / f"{scene_id}_boxes.pkl"
    source = _load_prediction(source_path)
    corners = (
        np.stack([row[1] for row in source[0]])
        if source[0]
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    scores = np.asarray([float(row[2]) for row in source[0]], dtype=np.float64)
    evidence = export.get("parent_evidence_hashes", {})
    return load_tr3d_r3_cache(
        tr3d_r3_cache_path(r3_root, scene_id, prefix_id),
        parent_tr3d_cache_path=tr3d_residual_cache_path(parent_root, scene_id, prefix_id),
        frozen_anchor_manifest_path=frozen_manifest,
        anchor_prediction_path=source_path,
        anchor_corners_world=corners,
        anchor_scores=scores,
        axis_alignment_metadata_path=scans_root / scene_id / f"{scene_id}.txt",
        expected_checkpoint_sha256=str(export["expected_parent_checkpoint_sha256"]),
        expected_config_sha256=str(export["expected_parent_config_sha256"]),
        expected_r3_config_sha256=str(export["r3_config_sha256"]),
        expected_r3_code_sha256=str(export["r3_code_sha256"]),
        parent_r2a_cache_path=(
            tr3d_r2_cache_path(r2a_root, scene_id, prefix_id) if r2a_root else None
        ),
        parent_r2b_cache_path=(
            tr3d_r2b_cache_path(r2b_root, scene_id, prefix_id) if r2b_root else None
        ),
        expected_prefix_manifest_row_sha256=manifest_row_sha,
        expected_frame_artifact_tree_sha256=frame_tree_sha,
        expected_r2_config_sha256=str(evidence.get("r2_config_sha256", "")),
        expected_r2_code_sha256=str(evidence.get("r2_code_sha256", "")),
        expected_feature_checkpoint_sha256=str(
            evidence.get("feature_checkpoint_sha256", "")
        ),
        expected_feature_config_sha256=str(evidence.get("feature_config_sha256", "")),
        expected_feature_code_sha256=str(evidence.get("feature_code_sha256", "")),
        expected_scene_id=scene_id,
        expected_prefix_id=prefix_id,
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    frozen_manifest = args.frozen_manifest.resolve()
    export_path = args.r3_export_report.resolve()
    r3_root = args.r3_cache_root.resolve()
    scene_list = args.scene_list.resolve()
    scans_root = args.scans_root.resolve()
    # Keep the unresolved final component so symlink outputs can be refused.
    output_root = args.output_root.expanduser().absolute()
    manifest_path = args.manifest.expanduser().absolute()
    scenes = read_scene_list(scene_list)
    if len(scenes) not in (10, 100):
        raise ValueError("shadow-active materialization requires fixed10 or full100")

    before = verify_frozen_anchor_manifest(frozen_manifest)
    before_snapshot = _snapshot(before)
    frozen_positions = {
        scene_id: index for index, scene_id in enumerate(before["scene_ids"])
    }
    if any(scene_id not in frozen_positions for scene_id in scenes) or [
        frozen_positions[scene_id] for scene_id in scenes
    ] != sorted(frozen_positions[scene_id] for scene_id in scenes):
        raise ValueError(
            "materialization scenes are not an ordered frozen-manifest subset"
        )
    export = _load_export(export_path, scenes)
    fixed_paths = {
        "frozen_manifest": frozen_manifest,
        "r3_cache_root": r3_root,
        "scene_list": scene_list,
        "scans_root": scans_root,
    }
    for name, expected in fixed_paths.items():
        if Path(str(export.get(name, ""))).resolve() != expected:
            raise ValueError(f"R3 export {name} path mismatch")
    if export.get("prefix_id") != args.prefix_id:
        raise ValueError("R3 export prefix mismatch")
    if export.get("frozen_manifest_sha256") != sha256_file(frozen_manifest):
        raise ValueError("R3 export frozen manifest bytes mismatch")
    if export.get("frozen_prediction_tree_sha256") != before["prediction_tree_sha256"]:
        raise ValueError("R3 export frozen prediction tree mismatch")
    export_rows = {str(row["scene_id"]): row for row in export["scenes"]}
    prefix_rows: Mapping[str, Mapping[str, Any]] = {}
    if bool(export["r3_config"].get("r2a_enabled")):
        prefix_rows = load_prefix_manifest(
            Path(str(export["prefix_manifest"])).resolve(), prefix_id=args.prefix_id
        )

    protected_roots = (frozen_root := Path(str(before["reference_result_root"])).resolve(), r3_root)
    if any(
        output_root == protected
        or output_root in protected.parents
        or protected in output_root.parents
        for protected in protected_roots
    ):
        raise ValueError("output root must be isolated from frozen/R3 input trees")
    if manifest_path in {
        frozen_manifest,
        export_path,
        scene_list,
    }:
        raise ValueError("manifest path aliases an immutable input")

    _claim_namespace(output_root, manifest_path, resume=bool(args.resume))
    existing = _existing_names(output_root, scenes)
    expected_names = {f"{scene}_boxes.pkl" for scene in scenes}
    if manifest_path.exists() and existing != expected_names:
        raise ValueError("complete manifest exists but prediction tree is incomplete")
    config = _config()
    config_sha = canonical_json_sha256(config)
    code_sha = _code_hash()
    per_scene: list[dict[str, Any]] = []
    output_hashes: dict[str, str] = {}
    total_rows = total_changed = total_applied = resumed = 0

    for scene_id in scenes:
        source_path = frozen_root / f"{scene_id}_boxes.pkl"
        source = _load_prediction(source_path)
        sidecar_path = tr3d_r3_cache_path(r3_root, scene_id, args.prefix_id)
        if sha256_file(sidecar_path) != export_rows[scene_id]["r3_sidecar_sha256"]:
            raise ValueError(f"{scene_id}: immutable R3 sidecar SHA mismatch")
        cache = _load_bound_cache(
            scene_id=scene_id,
            prefix_id=args.prefix_id,
            export=export,
            frozen_manifest=frozen_manifest,
            frozen_root=frozen_root,
            r3_root=r3_root,
            scans_root=scans_root,
            prefix_rows=prefix_rows,
        )
        active, api_summary = _materialize_payload(source, cache)
        invariant = _validate_invariants(source, active)
        encoded = pickle.dumps(active, protocol=PICKLE_PROTOCOL)
        # Refuse API objects whose pickle round trip changes any promised row.
        round_trip = pickle.loads(encoded)  # noqa: S301 - bytes created above
        _validate_invariants(active, round_trip)
        target = output_root / f"{scene_id}_boxes.pkl"
        if target.name in existing:
            observed = _load_prediction(target)
            _validate_invariants(active, observed)
            if any(
                not _geometry_equal(expected[1], actual[1])
                for expected, actual in zip(active[0], observed[0])
            ):
                raise ValueError(f"{scene_id}: resumed active geometry differs")
            if target.read_bytes() != encoded:
                raise ValueError(f"{scene_id}: resumed active pickle bytes differ")
            resumed += 1
        else:
            _write_bytes_create_only(target, encoded)
        output_sha = sha256_file(target)
        output_hashes[target.name] = output_sha
        applied = int(
            api_summary.get(
                "selected_count",
                api_summary.get("applied_count", invariant["changed_rows"]),
            )
        )
        if applied < invariant["changed_rows"] or applied > invariant["rows"]:
            raise ValueError(f"{scene_id}: active API summary has an invalid applied_count")
        per_scene.append(
            {
                "scene_id": scene_id,
                "source_prediction_sha256": sha256_file(source_path),
                "r3_sidecar_sha256": sha256_file(sidecar_path),
                "active_prediction_sha256": output_sha,
                "rows": invariant["rows"],
                "changed_rows": invariant["changed_rows"],
                "applied_count": applied,
                "resumed": target.name in existing,
                "api_summary": api_summary,
            }
        )
        total_rows += int(invariant["rows"])
        total_changed += int(invariant["changed_rows"])
        total_applied += applied

    if _existing_names(output_root, scenes) != expected_names:
        raise RuntimeError("shadow-active output set is incomplete")
    after = verify_frozen_anchor_manifest(frozen_manifest)
    after_snapshot = _snapshot(after)
    if before_snapshot != after_snapshot:
        raise RuntimeError("frozen G0 changed during shadow-active materialization")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "shadow_only": True,
        "formal_active_authorized": False,
        "ground_truth_access": False,
        "counterfactual_report_access": False,
        "clip_access": False,
        "labels_scores_order_count_unchanged": True,
        "replacement_geometry_frame": "scannet_unaligned_world",
        "config": config,
        "config_sha256": config_sha,
        "code_sha256": code_sha,
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": sha256_file(frozen_manifest),
        "frozen_prediction_tree_sha256": before["prediction_tree_sha256"],
        "r3_export_report": str(export_path),
        "r3_export_report_sha256": sha256_file(export_path),
        "r3_cache_root": str(r3_root),
        "r3_config_sha256": export["r3_config_sha256"],
        "r3_code_sha256": export["r3_code_sha256"],
        "scene_list": str(scene_list),
        "scene_list_sha256": sha256_file(scene_list),
        "prefix_id": args.prefix_id,
        "output_root": str(output_root),
        "output_prediction_tree_sha256": _sha_tree(output_hashes),
        "counts": {
            "scenes": len(scenes),
            "rows": total_rows,
            "applied_replacements": total_applied,
            "byte_changed_rows": total_changed,
            "resumed_scenes": resumed,
        },
        "frozen_anchor": {"before": before_snapshot, "after": after_snapshot},
        "scenes": per_scene,
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Resume metadata itself is intentionally not part of semantic identity.
        normalized = dict(manifest)
        normalized["counts"] = dict(normalized["counts"], resumed_scenes=0)
        normalized["scenes"] = [dict(row, resumed=False) for row in normalized["scenes"]]
        existing_normalized = dict(existing_manifest)
        existing_normalized["counts"] = dict(
            existing_normalized.get("counts", {}), resumed_scenes=0
        )
        existing_normalized["scenes"] = [
            dict(row, resumed=False) for row in existing_normalized.get("scenes", [])
        ]
        if existing_normalized != normalized:
            raise ValueError("existing shadow-active manifest disagrees with replay")
        return existing_manifest
    _write_json_create_only(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = materialize(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
