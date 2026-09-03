#!/usr/bin/env python3
"""Build an immutable train-only R3 veto-calibration dataset.

Ground truth is used only by this offline builder.  The resulting feature
matrix contains the frozen six inference-time features plus train labels; the
online calibrator never receives GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import verify_frozen_anchor_manifest  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    code_artifact_tree_sha256,
    load_prefix_manifest,
    sha256_file,
)
from boxfusion.tr3d_r3_active import (  # noqa: E402
    active_config_sha256,
    primary_candidate_rows,
)
from boxfusion.tr3d_r3_calibrator import candidate_features  # noqa: E402
from boxfusion.tr3d_r3_calibration_dataset import (  # noqa: E402
    R3CalibrationDataset,
    SAFE_NEUTRAL_CLASS,
    label_dataset_global_leave_one_out,
    write_dataset,
)
from tools.audit_tr3d_r3_near_correction import _scene_from_r3_cache  # noqa: E402
from tools.materialize_tr3d_r3_shadow_active import (  # noqa: E402
    _load_bound_cache,
    _load_export,
    _load_prediction,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r3_veto_dataset_build.v1"
EXPECTED_TRAIN100_SHA256 = "c83575e05df28ccc2fe21dd113692b4162d941431cdbd9dad782f0ad8889ff12"
EXPECTED_VAL100_SHA256 = "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
EXPECTED_OFFICIAL_TRAIN_SHA256 = "96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--r3-export-report", type=Path, required=True)
    parser.add_argument("--r3-cache-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--forbidden-scene-list", type=Path, required=True)
    parser.add_argument("--official-train-scene-list", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _tree_hash(rows: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(rows.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R3 dataset report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def build(args: argparse.Namespace) -> tuple[R3CalibrationDataset, dict[str, Any]]:
    frozen_path = args.frozen_manifest.resolve()
    export_path = args.r3_export_report.resolve()
    r3_root = args.r3_cache_root.resolve()
    scene_list = args.scene_list.resolve()
    forbidden_list = args.forbidden_scene_list.resolve()
    official_train_list = args.official_train_scene_list.resolve()
    scans_root = args.scans_root.resolve()
    gt_root = args.gt_root.resolve()
    scenes = read_scene_list(scene_list)
    forbidden = set(read_scene_list(forbidden_list))
    official_train = set(read_scene_list(official_train_list))
    if sha256_file(scene_list) != EXPECTED_TRAIN100_SHA256:
        raise ValueError("train100 scene-list SHA differs from the frozen protocol")
    if sha256_file(forbidden_list) != EXPECTED_VAL100_SHA256:
        raise ValueError("forbidden val100 SHA differs from the frozen protocol")
    if sha256_file(official_train_list) != EXPECTED_OFFICIAL_TRAIN_SHA256:
        raise ValueError("official ScanNet train-list SHA differs from the frozen protocol")
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise ValueError("R3 calibration requires exactly 100 unique train scenes")
    if len(forbidden) != 100:
        raise ValueError("R3 calibration requires the exact frozen val100 list")
    outside_official = sorted(set(scenes) - official_train)
    if outside_official:
        raise ValueError(f"calibration scenes are outside official train: {outside_official}")
    overlap = sorted(set(scenes) & forbidden)
    if overlap:
        raise ValueError(f"train-only calibration overlaps forbidden validation: {overlap}")
    manifest = verify_frozen_anchor_manifest(frozen_path, required_scene_count=100)
    if manifest["scene_ids"] != scenes:
        raise ValueError("frozen train anchor order differs from train scene list")
    export = _load_export(export_path, scenes)
    collection_artifact = manifest.get("artifacts", {}).get("collection_manifest")
    if not isinstance(collection_artifact, Mapping):
        raise ValueError("train anchor lacks its audited collection manifest")
    collection_path = Path(str(collection_artifact.get("path", ""))).resolve()
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if collection.get("schema") != "boxfusion.g0_sgcdet_train_collection_manifest.v1":
        raise ValueError("unsupported train anchor collection schema")
    anchor_contract = {
        "score_threshold": float(collection["score_thresh"]),
        "minimum_extent_m": float(collection["minimum_extent"]),
        "quality_detector_blend": float(collection["b6_detector_blend"]),
        "selective_gate": dict(collection["g0_gate"]),
        "quality_checkpoint_sha256": str(collection["b6_checkpoint_sha256"]),
        "yoloe_checkpoint_sha256": str(collection["yoloe_checkpoint_sha256"]),
    }
    fixed_paths = {
        "frozen_manifest": frozen_path,
        "r3_cache_root": r3_root,
        "scene_list": scene_list,
        "scans_root": scans_root,
    }
    for name, expected in fixed_paths.items():
        if Path(str(export.get(name, ""))).resolve() != expected:
            raise ValueError(f"R3 train export {name} path mismatch")
    if export.get("prefix_id") != args.prefix_id:
        raise ValueError("R3 train export prefix mismatch")
    if export.get("frozen_manifest_sha256") != sha256_file(frozen_path):
        raise ValueError("R3 train export frozen manifest changed")
    prefix_rows: Mapping[str, Mapping[str, Any]] = {}
    if bool(export["r3_config"].get("r2a_enabled")):
        prefix_rows = load_prefix_manifest(
            Path(str(export["prefix_manifest"])).resolve(), prefix_id=args.prefix_id
        )
    frozen_root = Path(manifest["reference_result_root"]).resolve()
    export_rows = {str(row["scene_id"]): row for row in export["scenes"]}

    scene_ids: list[str] = []
    anchor_offsets = [0]
    gt_offsets = [0]
    anchor_boxes: list[np.ndarray] = []
    anchor_scores: list[np.ndarray] = []
    gt_boxes: list[np.ndarray] = []
    sample_scene: list[int] = []
    sample_anchor: list[int] = []
    proposal_ids: list[int] = []
    candidate_boxes: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    tp_deltas: list[np.ndarray] = []
    gt_hashes: dict[str, str] = {}
    r3_hashes: dict[str, str] = {}
    per_scene: list[dict[str, Any]] = []

    for scene_index, scene_id in enumerate(scenes):
        source_path = frozen_root / f"{scene_id}_boxes.pkl"
        source = _load_prediction(source_path)
        sidecar_path = r3_root / scene_id / f"{args.prefix_id}.npz"
        expected_sha = str(export_rows[scene_id]["r3_sidecar_sha256"])
        if sha256_file(sidecar_path) != expected_sha:
            raise ValueError(f"{scene_id}: R3 train sidecar SHA mismatch")
        cache = _load_bound_cache(
            scene_id=scene_id,
            prefix_id=args.prefix_id,
            export=export,
            frozen_manifest=frozen_path,
            frozen_root=frozen_root,
            r3_root=r3_root,
            scans_root=scans_root,
            prefix_rows=prefix_rows,
        )
        scene = _scene_from_r3_cache(
            scene_id=scene_id,
            frozen_root=frozen_root,
            gt_root=gt_root,
            scans_root=scans_root,
            cache=cache,
        )
        selected = np.asarray(primary_candidate_rows(source, cache), dtype=np.int64)
        selected_features = candidate_features(source, cache, selected)
        for local, proposal_row in enumerate(selected.tolist()):
            anchor_index = int(cache.anchor_index[proposal_row])
            sample_scene.append(scene_index)
            sample_anchor.append(anchor_index)
            proposal_ids.append(int(cache.proposal_ids[proposal_row]))
            candidate_boxes.append(np.array(scene.candidate_boxes[proposal_row], copy=True))
            features.append(np.array(selected_features[local], copy=True))
            labels.append(SAFE_NEUTRAL_CLASS)
            tp_deltas.append(np.zeros(3, dtype=np.int8))
        scene_ids.append(scene_id)
        anchor_boxes.append(np.array(scene.anchor_boxes, copy=True))
        anchor_scores.append(np.array(scene.anchor_scores, copy=True))
        gt_boxes.append(np.array(scene.gt_boxes, copy=True))
        anchor_offsets.append(anchor_offsets[-1] + len(scene.anchor_boxes))
        gt_offsets.append(gt_offsets[-1] + len(scene.gt_boxes))
        gt_path = gt_root / f"{scene_id}_bbox.npy"
        gt_hashes[gt_path.name] = sha256_file(gt_path)
        r3_hashes[f"{scene_id}/{args.prefix_id}.npz"] = expected_sha
        per_scene.append(
            {
                "scene_id": scene_id,
                "anchors": len(scene.anchor_boxes),
                "ground_truth": len(scene.gt_boxes),
                "near_candidates": len(scene.candidate_boxes),
                "primary_samples": len(selected),
                "gain": 0,
                "safe_neutral": 0,
                "harm": 0,
            }
        )

    dataset = R3CalibrationDataset(
        scene_ids=np.asarray(scene_ids),
        anchor_offsets=np.asarray(anchor_offsets, dtype=np.int64),
        anchor_boxes=(
            np.concatenate(anchor_boxes, axis=0)
            if anchor_boxes else np.empty((0, 6), dtype=np.float64)
        ),
        anchor_scores=(
            np.concatenate(anchor_scores, axis=0)
            if anchor_scores else np.empty((0,), dtype=np.float64)
        ),
        gt_offsets=np.asarray(gt_offsets, dtype=np.int64),
        gt_boxes=(
            np.concatenate(gt_boxes, axis=0)
            if gt_boxes else np.empty((0, 6), dtype=np.float64)
        ),
        sample_scene_index=np.asarray(sample_scene, dtype=np.int64),
        sample_anchor_index=np.asarray(sample_anchor, dtype=np.int64),
        proposal_ids=np.asarray(proposal_ids, dtype=np.int64),
        candidate_boxes=(
            np.stack(candidate_boxes)
            if candidate_boxes else np.empty((0, 6), dtype=np.float64)
        ),
        features=(
            np.stack(features)
            if features else np.empty((0, 6), dtype=np.float64)
        ),
        labels=np.asarray(labels, dtype=np.int8),
        tp_deltas=(
            np.stack(tp_deltas)
            if tp_deltas else np.empty((0, 3), dtype=np.int8)
        ),
        ap_deltas=np.zeros((len(labels), 3), dtype=np.float64),
        provenance={
            "ground_truth_access": "offline_train_dataset_builder_only",
            "label_policy": "global_rank_aware_joint_raw_primary_leave_one_out_ap_and_tp",
            "validation_scene_overlap": 0,
            "global_anchor_scores_unique": True,
            "scene_list": str(scene_list),
            "scene_list_sha256": sha256_file(scene_list),
            "forbidden_scene_list": str(forbidden_list),
            "forbidden_scene_list_sha256": sha256_file(forbidden_list),
            "official_train_scene_list": str(official_train_list),
            "official_train_scene_list_sha256": sha256_file(official_train_list),
            "frozen_manifest": str(frozen_path),
            "frozen_manifest_sha256": sha256_file(frozen_path),
            "r3_export_report": str(export_path),
            "r3_export_report_sha256": sha256_file(export_path),
            "r3_sidecar_tree_sha256": _tree_hash(r3_hashes),
            "gt_tree_sha256": _tree_hash(gt_hashes),
            "builder_code_sha256": code_artifact_tree_sha256(
                (
                    Path(__file__),
                    _ROOT / "boxfusion" / "tr3d_r3_calibrator.py",
                    _ROOT / "boxfusion" / "tr3d_r3_calibration_dataset.py",
                    _ROOT / "boxfusion" / "tr3d_r3_active.py",
                    _ROOT / "tools" / "audit_tr3d_r3_near_correction.py",
                    _ROOT / "tools" / "materialize_tr3d_r3_shadow_active.py",
                )
            ),
            "tr3d_checkpoint_training_overlap": True,
            "independent_calibration_proof": False,
            "formal_independent_activation_authorized": False,
            "inference_lineage_contract": {
                "prefix_id": args.prefix_id,
                "parent_checkpoint_sha256": str(
                    export["expected_parent_checkpoint_sha256"]
                ),
                "parent_config_sha256": str(
                    export["expected_parent_config_sha256"]
                ),
                "r3_config_sha256": str(export["r3_config_sha256"]),
                "r3_code_sha256": str(export["r3_code_sha256"]),
                "primary_active_config_sha256": active_config_sha256(),
                "training_prefix_manifest_sha256": sha256_file(
                    Path(str(export["prefix_manifest"])).resolve()
                ),
                "anchor_distribution_contract": anchor_contract,
            },
            "warning": (
                "epoch12 TR3D was trained on official ScanNet train; these "
                "train100 proposal scores are in-sample for the proposal model"
            ),
        },
    ).validate()
    global_labels, global_tp_deltas, global_ap_deltas = (
        label_dataset_global_leave_one_out(dataset)
    )
    dataset = R3CalibrationDataset(
        **{
            **dataset.__dict__,
            "labels": global_labels,
            "tp_deltas": global_tp_deltas,
            "ap_deltas": global_ap_deltas,
        }
    ).validate()
    for scene_index, row in enumerate(per_scene):
        scene_labels = dataset.labels[dataset.sample_scene_index == scene_index]
        row["gain"] = int(np.count_nonzero(scene_labels == 0))
        row["safe_neutral"] = int(np.count_nonzero(scene_labels == 1))
        row["harm"] = int(np.count_nonzero(scene_labels == 2))
    counts = np.bincount(dataset.labels.astype(np.int64), minlength=3)
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "train_only": True,
        "ground_truth_access": True,
        "online_inference_ground_truth_access": False,
        "validation_scene_overlap": 0,
        "scene_count": dataset.scene_count,
        "sample_count": dataset.sample_count,
        "class_counts": {
            "gain": int(counts[0]),
            "safe_neutral": int(counts[1]),
            "harm": int(counts[2]),
        },
        "dataset": str(args.output.expanduser().absolute()),
        "provenance": dict(dataset.provenance),
        "scenes": per_scene,
    }
    return dataset, report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().absolute()
    report_path = args.report.expanduser().absolute()
    if output.exists() or report_path.exists():
        raise FileExistsError("R3 calibration dataset/report namespace already exists")
    dataset, report = build(args)
    write_dataset(output, dataset)
    report["dataset_sha256"] = sha256_file(output)
    _write_json_create_only(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
