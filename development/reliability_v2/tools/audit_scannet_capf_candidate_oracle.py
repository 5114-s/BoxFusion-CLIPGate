#!/usr/bin/env python3
"""Post-hoc Candidate/Face Oracle for the GT-free CAPF shadow bank."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from boxfusion.capf import _box_corners, local_faces_to_box  # noqa: E402
from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    _voc_ap,
    aligned_iou_matrix,
    load_axis_alignment,
    load_gt_minmax,
    load_scene_list,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_capf_candidate_oracle.v1"
SHADOW_SCHEMA = "boxfusion.capf_candidate_shadow.v1"
THRESHOLDS = (0.15, 0.25, 0.50)
ARMS = {"C1": 1, "C3": 3, "C6": 6}


class AuditError(ValueError):
    pass


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"invalid prediction file: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise AuditError(f"invalid prediction schema: {path}")
    rows = payload[0]
    labels = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    corners = np.asarray([row[1] for row in rows], dtype=np.float64)
    scores = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    if len(rows) == 0:
        corners = np.empty((0, 8, 3), dtype=np.float64)
    if corners.shape != (len(rows), 8, 3):
        raise AuditError(f"invalid corners in {path}")
    if not np.isfinite(corners).all() or not np.isfinite(scores).all():
        raise AuditError(f"non-finite prediction in {path}")
    return labels, corners, scores


def _aligned_minmax(corners: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    if len(corners) == 0:
        return np.empty((0, 6), dtype=np.float64)
    aligned = corners @ alignment[:3, :3].T + alignment[:3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def _official_real_score_evaluate(
    iou_by_scene: Sequence[np.ndarray],
    scores_by_scene: Sequence[np.ndarray],
    gt_counts: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    lengths = np.asarray([len(value) for value in iou_by_scene], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    scores = np.concatenate(scores_by_scene) if int(offsets[-1]) else np.empty(0)
    order = np.argsort(-scores)
    scene_indices = np.searchsorted(offsets[1:], order, side="right")
    local_indices = order - offsets[scene_indices]
    matched = [np.zeros(int(count), dtype=bool) for count in gt_counts]
    tp = np.zeros(len(order), dtype=np.float64)
    for rank, (scene_index, local_index) in enumerate(
        zip(scene_indices.tolist(), local_indices.tolist())
    ):
        overlaps = iou_by_scene[scene_index][local_index]
        if overlaps.size == 0:
            continue
        target = int(np.argmax(overlaps))
        if overlaps[target] > threshold and not matched[scene_index][target]:
            matched[scene_index][target] = True
            tp[rank] = 1.0
    fp = 1.0 - tp
    ap, recall, precision = _voc_ap(tp, fp, int(sum(gt_counts)))
    return {
        "ap": float(ap),
        "ap_points": float(100.0 * ap),
        "greedy_tp": int(tp.sum()),
        "recall": float(recall),
        "precision": float(precision),
    }


def _bounded_faces(
    anchor: np.ndarray,
    faces: np.ndarray,
    face_index: int,
    proposed_value: float,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    axis, side = divmod(int(face_index), 2)
    old_value = float(faces[axis, side])
    maximum_shift = min(
        float(cfg["max_face_shift_m"]),
        float(cfg["max_face_shift_ratio"]) * float(anchor[3 + axis]),
    )
    value = float(
        np.clip(proposed_value, old_value - maximum_shift, old_value + maximum_shift)
    )
    if abs(value - old_value) < float(cfg["min_candidate_shift_m"]):
        return None
    opposite = float(faces[axis, 1 - side])
    if side == 0:
        value = min(value, opposite - float(cfg["min_extent_m"]))
    else:
        value = max(value, opposite + float(cfg["min_extent_m"]))
    if abs(value - old_value) < float(cfg["min_candidate_shift_m"]):
        return None
    result = faces.copy()
    result[axis, side] = value
    if np.any(result[:, 1] - result[:, 0] < float(cfg["min_extent_m"])):
        return None
    return result


def _enumerate_candidate_corners(
    snapshot: dict[str, Any] | None,
    budget: int,
    cfg: dict[str, Any],
    native_corners: np.ndarray,
) -> np.ndarray:
    if snapshot is None or not snapshot.get("face_options"):
        return native_corners[None]
    anchor = np.asarray(snapshot["anchor_box_xyzlhw"], dtype=np.float64)
    rotation = np.asarray(snapshot["anchor_rotation"], dtype=np.float64)
    initial_faces = np.asarray(snapshot["anchor_faces"], dtype=np.float64)
    options = list(snapshot["face_options"])
    boxes = [anchor]
    seen_boxes = {tuple(np.rint(anchor * 1.0e7).astype(np.int64).tolist())}
    frontier = [(initial_faces, 0)]
    seen_states = {(0, tuple(np.rint(initial_faces.reshape(-1) * 1.0e7).astype(np.int64)))}
    for _ in range(budget):
        next_frontier = []
        for faces, used_mask in frontier:
            for option in options:
                face_index = int(option["face_index"])
                bit = 1 << face_index
                if used_mask & bit:
                    continue
                new_faces = _bounded_faces(
                    anchor,
                    faces,
                    face_index,
                    float(option.get("proposed_value", option["face_value"])),
                    cfg,
                )
                if new_faces is None:
                    continue
                new_mask = used_mask | bit
                state_key = (
                    new_mask,
                    tuple(np.rint(new_faces.reshape(-1) * 1.0e7).astype(np.int64)),
                )
                if state_key in seen_states:
                    continue
                seen_states.add(state_key)
                next_frontier.append((new_faces, new_mask))
                box = local_faces_to_box(anchor, rotation, new_faces)
                box_key = tuple(np.rint(box * 1.0e7).astype(np.int64).tolist())
                if box_key not in seen_boxes:
                    seen_boxes.add(box_key)
                    boxes.append(box)
        frontier = next_frontier
        if not frontier:
            break
    corners = np.stack([_box_corners(box, rotation) for box in boxes])
    # The serialized terminal corners are authoritative for the native choice.
    corners[0] = native_corners
    return corners


def _same_target_selection(
    native: np.ndarray, pools: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = native.copy()
    native_target_iou = np.zeros(len(native), dtype=np.float64)
    selected_target_iou = np.zeros(len(native), dtype=np.float64)
    for row_index, pool in enumerate(pools):
        if native.shape[1] == 0:
            continue
        target = int(np.argmax(native[row_index]))
        native_target_iou[row_index] = native[row_index, target]
        candidate_index = int(np.argmax(pool[:, target]))
        selected[row_index] = pool[candidate_index]
        selected_target_iou[row_index] = pool[candidate_index, target]
    return selected, native_target_iou, selected_target_iou


def _crossings(
    native_values: Sequence[np.ndarray],
    selected_values: Sequence[np.ndarray],
    threshold: float,
) -> dict[str, int]:
    native = np.concatenate(native_values) if native_values else np.empty(0)
    selected = np.concatenate(selected_values) if selected_values else np.empty(0)
    native_pass = native > threshold
    selected_pass = selected > threshold
    return {
        "up": int(np.count_nonzero(~native_pass & selected_pass)),
        "down": int(np.count_nonzero(native_pass & ~selected_pass)),
        "unchanged_pass": int(np.count_nonzero(native_pass & selected_pass)),
    }


def _evaluate_all_thresholds(
    matrices: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    gt_counts: Sequence[int],
) -> dict[str, Any]:
    return {
        f"{threshold:.2f}": _official_real_score_evaluate(
            matrices, scores, gt_counts, threshold
        )
        for threshold in THRESHOLDS
    }


def _matching_capacity(
    pools_by_scene: Sequence[Sequence[np.ndarray]],
    native_by_scene: Sequence[np.ndarray],
) -> dict[str, Any]:
    result = {}
    for threshold in THRESHOLDS:
        native_total = 0
        pool_total = 0
        for pools, native in zip(pools_by_scene, native_by_scene):
            if pools:
                envelope = np.stack([pool.max(axis=0) for pool in pools])
            else:
                envelope = np.empty((0, native.shape[1]), dtype=np.float64)
            native_total += len(strict_maximum_matching(native, threshold))
            pool_total += len(strict_maximum_matching(envelope, threshold))
        result[f"{threshold:.2f}"] = {
            "native_maximum_matching": int(native_total),
            "candidate_maximum_matching": int(pool_total),
            "additional_matches": int(pool_total - native_total),
        }
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scenes = load_scene_list(args.scene_list)
    sealed: dict[str, dict[str, Any]] = {}
    shadow_predictions = {}
    reference_predictions = {}
    candidate_cfg = None
    parity = {
        "scenes_with_row_score_difference": [],
        "scenes_with_row_corner_difference": [],
        "row_score_difference_count": 0,
        "row_corner_difference_count": 0,
        "max_abs_score_difference": 0.0,
        "max_abs_corner_difference": 0.0,
    }

    # Complete all GT-free validation before opening the first GT file.
    for scene in scenes:
        labels, corners, scores = _load_prediction(
            args.shadow_root / f"{scene}_boxes.pkl"
        )
        ref_labels, ref_corners, ref_scores = _load_prediction(
            args.reference_native_root / f"{scene}_boxes.pkl"
        )
        sidecar_path = args.candidate_root / f"{scene}_capf_candidates.json"
        if not sidecar_path.is_file() or sidecar_path.is_symlink():
            raise AuditError(f"missing candidate sidecar: {sidecar_path}")
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != SHADOW_SCHEMA
            or payload.get("scene_id") != scene
            or payload.get("gt_access") is not False
            or payload.get("online_writeback") is not False
            or payload.get("final_row_count") != len(corners)
        ):
            raise AuditError(f"invalid candidate sidecar contract: {sidecar_path}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(corners):
            raise AuditError(f"invalid candidate rows: {sidecar_path}")
        stored_corners = np.asarray(
            [row["native_corners_world"] for row in rows], dtype=np.float64
        )
        stored_scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        if not np.array_equal(stored_corners, corners) or not np.array_equal(
            stored_scores, scores
        ):
            raise AuditError(f"sidecar/native row mismatch: {scene}")
        if len(corners) != len(ref_corners) or not np.array_equal(labels, ref_labels):
            raise AuditError(f"shadow/reference identity mismatch: {scene}")
        # The shadow run is authoritative because the sidecars are sealed
        # against its exact rows.  A separately generated historical baseline
        # can differ through nondeterministic optimizer ordering, so report
        # those differences rather than mixing reference rows into the oracle.
        score_different = scores != ref_scores
        corner_different = np.any(corners != ref_corners, axis=(1, 2))
        if np.any(score_different):
            parity["scenes_with_row_score_difference"].append(scene)
            parity["row_score_difference_count"] += int(np.count_nonzero(score_different))
            parity["max_abs_score_difference"] = max(
                parity["max_abs_score_difference"],
                float(np.max(np.abs(scores - ref_scores))),
            )
        if np.any(corner_different):
            parity["scenes_with_row_corner_difference"].append(scene)
            parity["row_corner_difference_count"] += int(np.count_nonzero(corner_different))
            parity["max_abs_corner_difference"] = max(
                parity["max_abs_corner_difference"],
                float(np.max(np.abs(corners - ref_corners))),
            )
        if candidate_cfg is None:
            candidate_cfg = dict(payload["candidate_generation"])
        elif candidate_cfg != payload["candidate_generation"]:
            raise AuditError("candidate-generation config changed across scenes")
        sealed[scene] = payload
        shadow_predictions[scene] = (labels, corners, scores)
        reference_predictions[scene] = (ref_labels, ref_corners, ref_scores)

    gt_counts = []
    scores_by_scene = []
    reference_scores_by_scene = []
    native_iou_by_scene = []
    reference_iou_by_scene = []
    proxy_iou_by_scene = []
    pool_iou: dict[str, list[list[np.ndarray]]] = {name: [] for name in ARMS}
    candidate_counts = {name: 0 for name in ARMS}
    snapshot_count = 0
    option_count = 0
    proxy_losses = []
    proxy_deltas = []

    for scene in scenes:
        alignment = load_axis_alignment(args.scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(args.gt_root / f"{scene}_bbox.npy")
        _, native_corners, scores = shadow_predictions[scene]
        _, reference_corners, reference_scores = reference_predictions[scene]
        native_aligned = _aligned_minmax(native_corners, alignment)
        reference_aligned = _aligned_minmax(reference_corners, alignment)
        native_iou = aligned_iou_matrix(native_aligned, gt)
        reference_iou = aligned_iou_matrix(reference_aligned, gt)
        native_iou_by_scene.append(native_iou)
        reference_iou_by_scene.append(reference_iou)
        scores_by_scene.append(scores)
        reference_scores_by_scene.append(reference_scores)
        gt_counts.append(len(gt))

        proxy_rows = native_iou.copy()
        scene_pools = {name: [] for name in ARMS}
        for row_index, row in enumerate(sealed[scene]["rows"]):
            snapshot = row.get("candidate_snapshot")
            if snapshot is not None:
                snapshot_count += 1
                option_count += len(snapshot.get("face_options", []))
                proxy_box = np.asarray(
                    snapshot["proxy_selected_box_xyzlhw"], dtype=np.float64
                )
                proxy_corners = _box_corners(
                    proxy_box,
                    np.asarray(snapshot["anchor_rotation"], dtype=np.float64),
                )[None]
                proxy_rows[row_index] = aligned_iou_matrix(
                    _aligned_minmax(proxy_corners, alignment), gt
                )[0]
                updates = snapshot.get("proxy_selected_updates", [])
                if updates and len(gt):
                    target = int(np.argmax(native_iou[row_index]))
                    proxy_losses.append(
                        float(sum(item["median_loss_improvement"] for item in updates))
                    )
                    proxy_deltas.append(
                        float(proxy_rows[row_index, target] - native_iou[row_index, target])
                    )
            for arm, budget in ARMS.items():
                candidate_corners = _enumerate_candidate_corners(
                    snapshot,
                    budget,
                    candidate_cfg,
                    native_corners[row_index],
                )
                candidate_counts[arm] += len(candidate_corners)
                candidate_boxes = _aligned_minmax(candidate_corners, alignment)
                scene_pools[arm].append(aligned_iou_matrix(candidate_boxes, gt))
        proxy_iou_by_scene.append(proxy_rows)
        for arm in ARMS:
            pool_iou[arm].append(scene_pools[arm])

    native_eval = _evaluate_all_thresholds(
        native_iou_by_scene, scores_by_scene, gt_counts
    )
    reference_eval = _evaluate_all_thresholds(
        reference_iou_by_scene, reference_scores_by_scene, gt_counts
    )
    proxy_eval = _evaluate_all_thresholds(
        proxy_iou_by_scene, scores_by_scene, gt_counts
    )

    arms_report = {}
    for arm in ARMS:
        selected_by_scene = []
        native_target_by_scene = []
        selected_target_by_scene = []
        for native, pools in zip(native_iou_by_scene, pool_iou[arm]):
            selected, native_target, selected_target = _same_target_selection(
                native, pools
            )
            selected_by_scene.append(selected)
            native_target_by_scene.append(native_target)
            selected_target_by_scene.append(selected_target)
        evaluation = _evaluate_all_thresholds(
            selected_by_scene, scores_by_scene, gt_counts
        )
        for key in evaluation:
            evaluation[key]["delta_ap_points"] = (
                evaluation[key]["ap_points"] - native_eval[key]["ap_points"]
            )
        deltas = np.concatenate(selected_target_by_scene) - np.concatenate(
            native_target_by_scene
        )
        arms_report[arm] = {
            "face_budget": ARMS[arm],
            "candidate_count_including_native": int(candidate_counts[arm]),
            "same_target_official_evaluation": evaluation,
            "same_target_geometry": {
                "mean_delta_iou": float(np.mean(deltas)),
                "improved_rows": int(np.count_nonzero(deltas > 1.0e-9)),
                "degraded_rows": int(np.count_nonzero(deltas < -1.0e-9)),
                "crossings": {
                    f"{threshold:.2f}": _crossings(
                        native_target_by_scene,
                        selected_target_by_scene,
                        threshold,
                    )
                    for threshold in THRESHOLDS
                },
            },
            "maximum_matching_capacity": _matching_capacity(
                pool_iou[arm], native_iou_by_scene
            ),
        }

    proxy_delta = np.asarray(proxy_deltas, dtype=np.float64)
    correlation = None
    correlation_p = None
    if len(proxy_losses) >= 3 and np.std(proxy_losses) > 0 and np.std(proxy_delta) > 0:
        value = spearmanr(proxy_losses, proxy_delta)
        correlation = float(value.statistic)
        correlation_p = float(value.pvalue)
    proxy_report = {
        "official_evaluation": proxy_eval,
        "updated_rows": len(proxy_delta),
        "same_target_improved_rows": int(np.count_nonzero(proxy_delta > 1.0e-9)),
        "same_target_degraded_rows": int(np.count_nonzero(proxy_delta < -1.0e-9)),
        "same_target_unchanged_rows": int(np.count_nonzero(np.abs(proxy_delta) <= 1.0e-9)),
        "mean_same_target_delta_iou": float(np.mean(proxy_delta)) if len(proxy_delta) else 0.0,
        "proxy_loss_delta_iou_spearman": correlation,
        "proxy_loss_delta_iou_spearman_p": correlation_p,
    }
    return {
        "schema": SCHEMA,
        "oracle_only": True,
        "deployable": False,
        "gt_used_only_after_candidate_seal": True,
        "scene_count": len(scenes),
        "prediction_count": int(sum(len(value) for value in scores_by_scene)),
        "gt_count": int(sum(gt_counts)),
        "candidate_bank": {
            "terminal_snapshot_count": snapshot_count,
            "unique_face_option_count": option_count,
            "candidate_generation": candidate_cfg,
        },
        "reference_native_evaluation": reference_eval,
        "shadow_reference_row_parity": parity,
        "shadow_native_evaluation": native_eval,
        "proxy_shadow": proxy_report,
        "raw_candidate_oracle": arms_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--shadow-root",
        type=Path,
        default=ROOT / "results/scannet_t05_boxer_capf_oracle_shadow_v2_topk3_real_score05",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=ROOT / "diagnostics/t05_boxer_capf_oracle_shadow_v2/candidate_bank",
    )
    parser.add_argument(
        "--reference-native-root",
        type=Path,
        default=ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=ROOT / "evaluation/data_util/scannet_train_detection_data",
    )
    parser.add_argument(
        "--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports/capf_candidate_oracle/CAPF_CANDIDATE_ORACLE_OFFICIAL100.json",
    )
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise AuditError(f"refusing to overwrite report: {args.out}")
    report = audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.out)
    print(json.dumps({
        "out": os.fspath(args.out),
        "native": {k: v["ap_points"] for k, v in report["shadow_native_evaluation"].items()},
        "proxy": {k: v["ap_points"] for k, v in report["proxy_shadow"]["official_evaluation"].items()},
        "oracle": {
            arm: {
                k: v["ap_points"]
                for k, v in data["same_target_official_evaluation"].items()
            }
            for arm, data in report["raw_candidate_oracle"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
