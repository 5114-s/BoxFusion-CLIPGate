#!/usr/bin/env python3
"""Export observer-only M1/M2 candidates to the strict oracle contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_fused_oracle import read_scene_ids  # noqa: E402
from tools.report_trifusion_oracles import (  # noqa: E402
    CORNER_FRAME,
    SUPPLEMENTAL_CANDIDATE_SCHEMA,
    load_supplemental_candidates,
)


DIAGNOSTIC_SCHEMA = (
    "boxfusion.trifusion.missing_graph_observer.v1"
)
OUTPUT_SUFFIX = "_supplemental_candidates.npz"


def _scalar_text(
    value: np.ndarray, *, name: str, path: Path
) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    result = array.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"{path}: {name} must be a scalar string")
    return result


def export_scene(
    *,
    scene_id: str,
    diagnostic_path: Path,
) -> dict[str, np.ndarray]:
    required = {
        "scene_id",
        "trifusion_missing_diagnostics_schema",
        "trifusion_missing_enabled",
        "trifusion_missing_mutation_enabled",
        "trifusion_missing_candidate_ids",
        "trifusion_missing_sources",
        "trifusion_missing_corners",
        "trifusion_missing_scores",
        "trifusion_missing_labels",
        "trifusion_missing_feature_names",
        "trifusion_missing_features",
        "trifusion_missing_valid",
        "trifusion_missing_verified",
        "trifusion_missing_confirmed",
        "trifusion_missing_applied",
        "trifusion_missing_reasons",
        "trifusion_missing_unique_views",
        "trifusion_missing_node_counts",
        "trifusion_missing_edge_counts",
        "trifusion_missing_point_counts",
        "trifusion_missing_frame_ids_json",
    }
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{diagnostic_path}: missing M1/M2 fields "
                f"{sorted(missing)}"
            )
        raw = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    stored_scene = _scalar_text(
        raw["scene_id"], name="scene_id", path=diagnostic_path
    )
    if stored_scene != scene_id:
        raise ValueError(
            f"{diagnostic_path}: scene {stored_scene!r} != {scene_id!r}"
        )
    schema = _scalar_text(
        raw["trifusion_missing_diagnostics_schema"],
        name="trifusion_missing_diagnostics_schema",
        path=diagnostic_path,
    )
    if schema != DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"{diagnostic_path}: unsupported missing graph schema {schema!r}"
        )
    enabled = np.asarray(raw["trifusion_missing_enabled"])
    mutation = np.asarray(
        raw["trifusion_missing_mutation_enabled"]
    )
    if enabled.shape != () or enabled.dtype != np.bool_ or not enabled.item():
        raise ValueError(f"{diagnostic_path}: missing graph was not enabled")
    if mutation.shape != () or mutation.dtype != np.bool_ or mutation.item():
        raise ValueError(
            f"{diagnostic_path}: missing graph is not observer-only"
        )

    corners = np.asarray(
        raw["trifusion_missing_corners"], dtype=np.float32
    )
    if (
        corners.ndim != 3
        or corners.shape[1:] != (8, 3)
        or not np.isfinite(corners).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: invalid missing candidate corners"
        )
    rows = len(corners)

    def vector(name: str, *, dtype=None) -> np.ndarray:
        array = np.asarray(raw[name], dtype=dtype)
        if array.shape != (rows,) or array.dtype.hasobject:
            raise ValueError(
                f"{diagnostic_path}: {name} must have shape [{rows}]"
            )
        return array

    track_ids = vector(
        "trifusion_missing_candidate_ids", dtype=np.int64
    )
    if len(np.unique(track_ids)) != rows:
        raise ValueError(
            f"{diagnostic_path}: missing candidate IDs are not unique"
        )
    sources = vector("trifusion_missing_sources").astype(str)
    scores = vector(
        "trifusion_missing_scores", dtype=np.float32
    )
    labels = vector("trifusion_missing_labels").astype(str)
    valid = vector("trifusion_missing_valid").astype(bool)
    verified = vector("trifusion_missing_verified").astype(bool)
    confirmed = vector("trifusion_missing_confirmed").astype(bool)
    applied = vector("trifusion_missing_applied").astype(bool)
    reasons = vector("trifusion_missing_reasons").astype(str)
    unique_views = vector(
        "trifusion_missing_unique_views", dtype=np.int64
    )
    node_counts = vector(
        "trifusion_missing_node_counts", dtype=np.int64
    )
    edge_counts = vector(
        "trifusion_missing_edge_counts", dtype=np.int64
    )
    point_counts = vector(
        "trifusion_missing_point_counts", dtype=np.int64
    )
    frame_ids_json = vector(
        "trifusion_missing_frame_ids_json"
    ).astype(str)
    if (
        not np.isfinite(scores).all()
        or np.any((scores < 0.0) | (scores > 1.0))
        or np.any(verified & ~valid)
        or np.any(~confirmed)
        or np.any(applied)
        or np.any(unique_views < 2)
        or np.any(node_counts < 2)
        or np.any(edge_counts < 1)
        or np.any(point_counts < 1)
    ):
        raise ValueError(
            f"{diagnostic_path}: invalid M1/M2 observer contract"
        )
    feature_names = np.asarray(
        raw["trifusion_missing_feature_names"]
    )
    features = np.asarray(
        raw["trifusion_missing_features"], dtype=np.float32
    )
    if (
        feature_names.ndim != 1
        or feature_names.dtype.hasobject
        or features.shape != (rows, len(feature_names))
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: invalid M1/M2 feature contract"
        )
    ids = np.asarray(
        [
            f"{scene_id}:missing_graph:track:{int(track_id)}"
            for track_id in track_ids
        ],
        dtype="<U128",
    )
    return {
        "schema": np.asarray(SUPPLEMENTAL_CANDIDATE_SCHEMA),
        "format_version": np.asarray(1, dtype=np.int64),
        "scene_id": np.asarray(scene_id),
        "corner_frame": np.asarray(CORNER_FRAME),
        "candidate_corners": corners,
        "candidate_ids": ids,
        "candidate_sources": sources.astype("<U32"),
        "candidate_valid": valid.astype(np.bool_),
        "candidate_verified": verified.astype(np.bool_),
        "candidate_scores": scores,
        "candidate_labels": labels.astype("<U96"),
        "candidate_feature_names": feature_names.astype("<U96"),
        "candidate_features": features,
        "candidate_confirmed": confirmed.astype(np.bool_),
        "candidate_reasons": reasons.astype("<U64"),
        "candidate_unique_views": unique_views,
        "candidate_node_counts": node_counts,
        "candidate_edge_counts": edge_counts,
        "candidate_point_counts": point_counts,
        "candidate_frame_ids_json": frame_ids_json.astype("<U512"),
        "observer_only": np.asarray(True, dtype=np.bool_),
        "uses_ground_truth": np.asarray(False, dtype=np.bool_),
    }


def export_directory(
    *,
    diagnostics_root: Path,
    scene_list: Path,
    output_root: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    scenes = read_scene_ids(scene_list)
    if not scenes:
        raise ValueError(f"scene list is empty: {scene_list}")
    if output_root.resolve() == diagnostics_root.resolve():
        raise ValueError("output root must differ from diagnostics root")
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = valid = verified = 0
    for scene_id in scenes:
        destination = output_root / f"{scene_id}{OUTPUT_SUFFIX}"
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite supplemental artifact: {destination}"
            )
        payload = export_scene(
            scene_id=scene_id,
            diagnostic_path=(
                diagnostics_root / f"{scene_id}_tracks.npz"
            ),
        )
        np.savez_compressed(destination, **payload)
        # Reload with the independent strict oracle parser before counting.
        checked = load_supplemental_candidates(
            destination, expected_scene_id=scene_id
        )
        candidates += len(checked.candidate_corners)
        valid += int(np.sum(checked.candidate_valid))
        verified += int(np.sum(checked.candidate_verified))
    return {
        "schema": "boxfusion.trifusion.supplemental_export_summary",
        "format_version": 1,
        "scenes": len(scenes),
        "candidates": candidates,
        "valid": valid,
        "verified": verified,
        "output_root": str(output_root),
        "uses_ground_truth": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)
    summary = export_directory(
        diagnostics_root=args.diagnostics_root,
        scene_list=args.scene_list,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    rendered = json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False
    )
    print(rendered)
    if args.summary_json is not None:
        if args.summary_json.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite summary: {args.summary_json}"
            )
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
