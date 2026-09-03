#!/usr/bin/env python3
"""Build a leakage-checked train-only C3 source-gate dataset.

Ground truth is read only here, never by the runtime observer/appender.  Every
row is joined to an immutable terminal TR3D proposal by proposal identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_c3_online_active import FEATURE_NAMES, candidate_features
from boxfusion.tr3d_c3_online_identity import PARENT_SCORE_ROUTE, ROUTE, SCHEMA
from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file
from boxfusion.tr3d_residual_cache import load_tr3d_residual_cache


DATASET_SCHEMA = "boxfusion.tr3d_c3_source_gate_dataset.v1"


def read_scenes(path: Path) -> tuple[str, ...]:
    rows = tuple(
        row.strip() for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    )
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"scene list is empty or contains duplicates: {path}")
    return rows


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    extent = np.maximum(
        np.minimum(left[:, None, 3:], right[None, :, 3:])
        - np.maximum(left[:, None, :3], right[None, :, :3]),
        0.0,
    )
    intersection = np.prod(extent, axis=2)
    left_volume = np.prod(left[:, 3:] - left[:, :3], axis=1)
    right_volume = np.prod(right[:, 3:] - right[:, :3], axis=1)
    union = left_volume[:, None] + right_volume[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def gt_minmax(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 6 or not np.isfinite(value[:, :6]).all():
        raise ValueError(f"invalid ScanNet GT: {path}")
    center, size = value[:, :3], value[:, 3:6]
    if np.any(size <= 0):
        raise ValueError(f"non-positive ScanNet GT extent: {path}")
    return np.concatenate((center - size / 2, center + size / 2), axis=1)


def transformed_minmax(corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    aligned = corners @ transform[:3, :3].T + transform[None, None, :3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def unaligned_to_aligned(aligned_to_unaligned: np.ndarray) -> np.ndarray:
    inverse = np.asarray(aligned_to_unaligned, dtype=np.float64)
    if inverse.shape != (4, 4) or not np.isfinite(inverse).all():
        raise ValueError("invalid cached aligned_to_unaligned transform")
    forward = np.linalg.inv(inverse)
    identity = np.eye(4, dtype=np.float64)
    if not (
        np.allclose(forward @ inverse, identity, rtol=0.0, atol=1e-10)
        and np.allclose(inverse @ forward, identity, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("cached ScanNet alignment transform is not invertible")
    return forward


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing C3 dataset: {path}") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing C3 dataset report: {path}") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build(args: argparse.Namespace) -> dict:
    train_list = args.train_scene_list.resolve()
    forbidden_list = args.forbidden_validation_scene_list.resolve()
    train_scenes = read_scenes(train_list)
    forbidden = read_scenes(forbidden_list)
    overlap = sorted(set(train_scenes) & set(forbidden))
    if overlap or len(forbidden) < 100:
        raise ValueError(f"invalid train/validation partition; overlap={overlap[:5]}")

    features: list[np.ndarray] = []
    target_iou: list[float] = []
    scene_ids: list[str] = []
    proposal_ids: list[int] = []
    scene_counts: dict[str, int] = {}
    observed_route: str | None = None
    for scene_id in train_scenes:
        diagnostic_path = args.diagnostics_root / f"{scene_id}_c3_online_identity.json"
        payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != SCHEMA
            or payload.get("scene_id") != scene_id
            or payload.get("route") not in {ROUTE, PARENT_SCORE_ROUTE}
            or not payload.get("complete")
            or not payload.get("observer_only")
            or payload.get("mutation_enabled")
            or payload.get("ground_truth_access")
            or payload.get("candidate_generation_is_live") is not False
        ):
            raise ValueError(f"{diagnostic_path}: invalid C3 observer contract")
        if observed_route is None:
            observed_route = str(payload["route"])
        elif payload.get("route") != observed_route:
            raise ValueError("C3 training diagnostics mix candidate routes")
        parent_path = Path(str(payload.get("parent_cache", ""))).resolve()
        if sha256_file(parent_path) != payload.get("parent_cache_sha256"):
            raise ValueError(f"{scene_id}: parent cache hash mismatch")
        with np.load(parent_path, allow_pickle=False) as archive:
            checkpoint_sha = str(np.asarray(archive["checkpoint_sha256"]).item())
            config_sha = str(np.asarray(archive["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path,
            expected_scene_id=scene_id,
            expected_prefix_id="p100",
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
        )
        gt = gt_minmax(args.ground_truth_root / f"{scene_id}_bbox.npy")
        accepted = 0
        seen: set[str] = set()
        for candidate in payload.get("candidates", ()):
            identity = str(candidate.get("identity_key", ""))
            if not identity or identity in seen:
                raise ValueError(f"{scene_id}: duplicate/empty candidate identity")
            seen.add(identity)
            if not bool(candidate.get("online_yoloe_mask2_depth")):
                continue
            row = int(candidate.get("parent_row", -1))
            proposal_id = int(candidate.get("proposal_id", -1))
            if row < 0 or row >= len(parent.proposal_ids) or int(parent.proposal_ids[row]) != proposal_id:
                raise ValueError(f"{scene_id}: candidate/parent identity mismatch")
            prediction = transformed_minmax(
                np.asarray(parent.corners_world[row:row + 1], dtype=np.float64),
                unaligned_to_aligned(parent.aligned_to_unaligned),
            )
            maximum = float(np.max(box_iou(prediction, gt), initial=0.0))
            features.append(candidate_features(candidate))
            target_iou.append(maximum)
            scene_ids.append(scene_id)
            proposal_ids.append(proposal_id)
            accepted += 1
        scene_counts[scene_id] = accepted

    if len(features) < 20 or len(set(scene_ids)) < 5:
        raise ValueError("C3 training dataset is too small (need >=20 rows and >=5 scenes)")
    arrays = {
        "features": np.asarray(features, dtype=np.float32),
        "target_iou": np.asarray(target_iou, dtype=np.float32),
        "scene_ids": np.asarray(scene_ids, dtype=f"<U{max(map(len, scene_ids))}"),
        "proposal_ids": np.asarray(proposal_ids, dtype=np.int64),
        "feature_names": np.asarray(FEATURE_NAMES),
        "schema": np.asarray(DATASET_SCHEMA),
        "route": np.asarray(observed_route),
        "train_scene_list_sha256": np.asarray(sha256_file(train_list)),
        "forbidden_validation_scene_list_sha256": np.asarray(sha256_file(forbidden_list)),
    }
    atomic_npz(args.output, arrays)
    report = {
        "schema": DATASET_SCHEMA,
        "complete": True,
        "train_only": True,
        "ground_truth_used_only_for_training": True,
        "validation_predictions_used_for_training": False,
        "route": observed_route,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "train_scene_list": str(train_list),
        "train_scene_list_sha256": sha256_file(train_list),
        "forbidden_validation_scene_list": str(forbidden_list),
        "forbidden_validation_scene_list_sha256": sha256_file(forbidden_list),
        "validation_overlap_count": 0,
        "scenes": len(train_scenes),
        "samples": len(features),
        "positive_iou25": int(np.count_nonzero(np.asarray(target_iou) >= 0.25)),
        "positive_iou50": int(np.count_nonzero(np.asarray(target_iou) >= 0.50)),
        "per_scene_samples": scene_counts,
    }
    atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scene-list", type=Path, required=True)
    parser.add_argument("--forbidden-validation-scene-list", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
