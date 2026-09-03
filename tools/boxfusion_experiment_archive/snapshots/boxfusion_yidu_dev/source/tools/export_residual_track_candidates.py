#!/usr/bin/env python3
"""Export observer-only residual tracks to the shared oracle contract."""

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


DIAGNOSTIC_SCHEMA = "boxfusion.residual_mask_track_observer.v1"
OUTPUT_SUFFIX = "_supplemental_candidates.npz"


def _scalar(value: np.ndarray, *, name: str, path: Path):
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar")
    result = array.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return result


def export_scene(
    *,
    scene_id: str,
    diagnostic_path: Path,
) -> dict[str, np.ndarray]:
    required = {
        "scene_id",
        "residual_track_diagnostics_schema",
        "residual_track_enabled",
        "residual_track_observer_only",
        "residual_track_mutation_enabled",
        "residual_track_failed",
        "residual_track_candidate_ids",
        "residual_track_sources",
        "residual_track_corners",
        "residual_track_scores",
        "residual_track_labels",
        "residual_track_feature_names",
        "residual_track_features",
        "residual_track_graph_contract_valid",
        "residual_track_graph_confirmed",
        "residual_track_applied",
        "residual_track_reasons",
        "residual_track_unique_views",
        "residual_track_node_counts",
        "residual_track_edge_counts",
        "residual_track_point_counts",
        "residual_track_frame_ids_json",
        "residual_track_provider_counts_json",
        "residual_track_mixed_provider",
        "yidu_zero_write_verified",
    }
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{diagnostic_path}: missing residual fields "
                f"{sorted(missing)}"
            )
        raw = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    if _scalar(raw["scene_id"], name="scene_id", path=diagnostic_path) != scene_id:
        raise ValueError(f"{diagnostic_path}: scene ID mismatch")
    schema = _scalar(
        raw["residual_track_diagnostics_schema"],
        name="residual_track_diagnostics_schema",
        path=diagnostic_path,
    )
    if schema != DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"{diagnostic_path}: unsupported residual schema {schema!r}"
        )
    contracts = {
        "enabled": bool(
            _scalar(
                raw["residual_track_enabled"],
                name="residual_track_enabled",
                path=diagnostic_path,
            )
        ),
        "observer_only": bool(
            _scalar(
                raw["residual_track_observer_only"],
                name="residual_track_observer_only",
                path=diagnostic_path,
            )
        ),
        "mutation": bool(
            _scalar(
                raw["residual_track_mutation_enabled"],
                name="residual_track_mutation_enabled",
                path=diagnostic_path,
            )
        ),
        "failed": bool(
            _scalar(
                raw["residual_track_failed"],
                name="residual_track_failed",
                path=diagnostic_path,
            )
        ),
        "zero_write": bool(
            _scalar(
                raw["yidu_zero_write_verified"],
                name="yidu_zero_write_verified",
                path=diagnostic_path,
            )
        ),
    }
    if contracts != {
        "enabled": True,
        "observer_only": True,
        "mutation": False,
        "failed": False,
        "zero_write": True,
    }:
        raise ValueError(
            f"{diagnostic_path}: residual zero-write contract failed: "
            f"{contracts}"
        )

    corners = np.asarray(raw["residual_track_corners"], dtype=np.float32)
    if (
        corners.ndim != 3
        or corners.shape[1:] != (8, 3)
        or not np.isfinite(corners).all()
    ):
        raise ValueError(f"{diagnostic_path}: invalid residual corners")
    rows = len(corners)

    def vector(name: str, *, dtype=None) -> np.ndarray:
        array = np.asarray(raw[name], dtype=dtype)
        if array.shape != (rows,) or array.dtype.hasobject:
            raise ValueError(
                f"{diagnostic_path}: {name} must have shape [{rows}]"
            )
        return array

    track_ids = vector("residual_track_candidate_ids", dtype=np.int64)
    if len(np.unique(track_ids)) != rows:
        raise ValueError(f"{diagnostic_path}: duplicate residual IDs")
    scores = vector("residual_track_scores", dtype=np.float32)
    labels = vector("residual_track_labels").astype(str)
    sources = vector("residual_track_sources").astype(str)
    valid = vector("residual_track_graph_contract_valid").astype(bool)
    confirmed = vector("residual_track_graph_confirmed").astype(bool)
    applied = vector("residual_track_applied").astype(bool)
    reasons = vector("residual_track_reasons").astype(str)
    unique_views = vector(
        "residual_track_unique_views", dtype=np.int64
    )
    node_counts = vector(
        "residual_track_node_counts", dtype=np.int64
    )
    edge_counts = vector(
        "residual_track_edge_counts", dtype=np.int64
    )
    point_counts = vector(
        "residual_track_point_counts", dtype=np.int64
    )
    frame_ids_json = vector(
        "residual_track_frame_ids_json"
    ).astype(str)
    provider_counts_json = vector(
        "residual_track_provider_counts_json"
    ).astype(str)
    mixed_provider = vector(
        "residual_track_mixed_provider"
    ).astype(bool)
    if (
        not np.isfinite(scores).all()
        or np.any((scores < 0.0) | (scores > 1.0))
        or np.any(~valid)
        or np.any(~confirmed)
        or np.any(applied)
        or np.any(unique_views < 2)
        or np.any(node_counts < 2)
        or np.any(edge_counts < 1)
        or np.any(point_counts < 1)
    ):
        raise ValueError(
            f"{diagnostic_path}: invalid residual observer contract"
        )
    feature_names = np.asarray(raw["residual_track_feature_names"])
    features = np.asarray(
        raw["residual_track_features"], dtype=np.float32
    )
    if (
        feature_names.ndim != 1
        or feature_names.dtype.hasobject
        or features.shape != (rows, len(feature_names))
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"{diagnostic_path}: invalid residual features")
    for encoded in provider_counts_json:
        decoded = json.loads(str(encoded))
        if (
            not isinstance(decoded, dict)
            or sum(int(value) for value in decoded.values()) <= 0
        ):
            raise ValueError(
                f"{diagnostic_path}: invalid provider provenance"
            )

    candidate_ids = np.asarray(
        [
            f"{scene_id}:residual_track:track:{int(track_id)}"
            for track_id in track_ids
        ],
        dtype="<U160",
    )
    return {
        "schema": np.asarray(SUPPLEMENTAL_CANDIDATE_SCHEMA),
        "format_version": np.asarray(1, dtype=np.int64),
        "scene_id": np.asarray(scene_id),
        "corner_frame": np.asarray(CORNER_FRAME),
        "candidate_corners": corners,
        "candidate_ids": candidate_ids,
        "candidate_sources": sources.astype("<U48"),
        "candidate_valid": valid.astype(np.bool_),
        "candidate_verified": confirmed.astype(np.bool_),
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
        "candidate_provider_counts_json": provider_counts_json.astype(
            "<U512"
        ),
        "candidate_mixed_provider": mixed_provider.astype(np.bool_),
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
    candidate_count = 0
    mixed_count = 0
    for scene_id in scenes:
        destination = output_root / f"{scene_id}{OUTPUT_SUFFIX}"
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        payload = export_scene(
            scene_id=scene_id,
            diagnostic_path=diagnostics_root / f"{scene_id}_tracks.npz",
        )
        np.savez_compressed(destination, **payload)
        checked = load_supplemental_candidates(
            destination, expected_scene_id=scene_id
        )
        candidate_count += len(checked.candidate_corners)
        mixed_count += int(np.sum(payload["candidate_mixed_provider"]))
    return {
        "schema": "boxfusion.residual_track.export_summary.v1",
        "scenes": len(scenes),
        "candidates": candidate_count,
        "mixed_provider_candidates": mixed_count,
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
            raise FileExistsError(args.summary_json)
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
