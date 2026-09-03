#!/usr/bin/env python3
"""Paper100 identity-constrained oracle for sealed L0 F3/F4 view banks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    load_scene_list,
    official_constant_evaluate,
    strict_maximum_matching,
)
from tools.audit_scannet_fastsam_f1_paper100_oracle import (  # noqa: E402
    EXPECTED_BASELINE_AP_POINTS,
    REQUIRED_ADDITIONAL_MATCHES,
    THRESHOLDS,
)


SCHEMA = "boxfusion.scannet_l0_f3_f4_perview_paper100_oracle.v1"
SEAL_SCHEMA = "boxfusion.scannet_l0_f3_f4_perview_paper100.seal.v1"
PROTOCOL_ID = "L0-F3-CONFIRMED-F4-PERVIEW-HYPOTHESIS-BANK-PAPER100-V1"
HYPOTHESES = ("H0", "HL", "HLG", "HB")
TARGET_DELTA_AP_POINTS = 10.0
DEFAULT_SEAL = ROOT / "logs/scannet_l0_f3_f4_perview_paper100_score05/final/L0_F3_F4_PERVIEW_PAPER100.json"
DEFAULT_OUT = ROOT / "reports/l0_f3_f4_perview_paper100_oracle/L0_F3_F4_PERVIEW_PAPER100_ORACLE.json"
_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [+1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)


class L0OracleError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise L0OracleError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise L0OracleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise L0OracleError(f"{label} must contain one JSON object")
    return value


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise L0OracleError(f"{label} is not numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise L0OracleError(f"{label} must be finite shape {shape}")
    return result


def _aligned_geometry(
    row: Mapping[str, Any],
    name: str,
    alignment: np.ndarray,
    label: str,
) -> np.ndarray:
    if row.get("valid") is not True:
        raise L0OracleError(f"sealed available geometry is invalid: {label}")
    if name == "HB":
        corners = _array(row.get("world_corners"), (8, 3), f"{label}.world_corners")
    else:
        lower = _array(row.get("q02"), (3,), f"{label}.q02")
        upper = _array(row.get("q98"), (3,), f"{label}.q98")
        if np.any(upper <= lower):
            raise L0OracleError(f"sealed geometry is degenerate: {label}")
        center = (lower + upper) * 0.5
        extent = upper - lower
        corners = center[None, :] + _SIGNS * extent[None, :] * 0.5
    transformed = corners @ alignment[:3, :3].T + alignment[:3, 3]
    result = np.concatenate((transformed.min(axis=0), transformed.max(axis=0)))
    if np.any(result[3:] <= result[:3]):
        raise L0OracleError(f"aligned geometry is degenerate: {label}")
    return result


def _evaluation_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"matched_gt_masks", "evaluation_order"}
    }


def _source_map(f4: Mapping[str, Any], scene: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for frame in f4.get("frames", []):
        if not isinstance(frame, Mapping):
            raise L0OracleError(f"invalid F4 frame: {scene}")
        for source in frame.get("sources", []):
            if not isinstance(source, Mapping) or not isinstance(source.get("source_id"), str):
                raise L0OracleError(f"invalid F4 source: {scene}")
            source_id = str(source["source_id"])
            if source_id in result:
                raise L0OracleError(f"duplicate F4 source: {source_id}")
            result[source_id] = source
    return result


def audit(
    *,
    scene_list: Path,
    seal_path: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    expected_seal_schema: str = SEAL_SCHEMA,
    expected_protocol_id: str = PROTOCOL_ID,
    report_schema: str = SCHEMA,
    seal_track_count_key: str = "confirmed_track_count",
    identity_description: str = (
        "one confirmed F3 track contributes at most one F4 per-view geometry"
    ),
) -> dict[str, Any]:
    scenes = load_scene_list(scene_list)
    seal_sha = _sha(seal_path)
    seal = _json(seal_path, "sealed L0 receipt")
    seal_rows = seal.get("scenes")
    if (
        len(scenes) != 100
        or seal.get("schema") != expected_seal_schema
        or seal.get("protocol_id") != expected_protocol_id
        or seal.get("complete") is not True
        or seal.get("overall_pass") is not True
        or seal.get("scene_order") != scenes
        or seal.get("contracts", {}).get("ground_truth_access") is not False
        or not isinstance(seal_rows, list)
        or len(seal_rows) != 100
    ):
        raise L0OracleError("sealed L0 paper100 receipt differs")

    baseline_matrices: list[np.ndarray] = []
    candidate_max_matrices: list[np.ndarray] = []
    candidate_geometry_rows: list[list[list[tuple[str, str, np.ndarray]]]] = []
    track_ids_by_scene: list[list[int]] = []
    gt_counts: list[int] = []
    scene_reports: dict[str, Any] = {}
    geometry_total = 0
    track_total = 0

    for scene_index, (scene, seal_row) in enumerate(zip(scenes, seal_rows)):
        if (
            not isinstance(seal_row, Mapping)
            or seal_row.get("scene_id") != scene
            or seal_row.get("scene_index") != scene_index
        ):
            raise L0OracleError(f"sealed L0 scene order differs: {scene}")
        f4_receipt = seal_row.get("f4")
        tracks = seal_row.get("tracks")
        if not isinstance(f4_receipt, Mapping) or not isinstance(tracks, list):
            raise L0OracleError(f"sealed L0 scene ledger differs: {scene}")
        f4_path = Path(str(f4_receipt.get("path", "")))
        if _sha(f4_path) != f4_receipt.get("sha256"):
            raise L0OracleError(f"sealed F4 hash differs after L0 seal: {scene}")
        f4 = _json(f4_path, f"F4 scene {scene}")
        source_map = _source_map(f4, scene)
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(baseline_root / f"{scene}_boxes.pkl", alignment)

        scene_geometry_rows: list[list[tuple[str, str, np.ndarray]]] = []
        scene_max_rows: list[np.ndarray] = []
        scene_track_ids: list[int] = []
        for track in tracks:
            if not isinstance(track, Mapping) or type(track.get("track_id")) is not int:
                raise L0OracleError(f"invalid sealed track: {scene}")
            geometry_rows: list[tuple[str, str, np.ndarray]] = []
            for source_receipt in track.get("sources", []):
                if not isinstance(source_receipt, Mapping):
                    raise L0OracleError(f"invalid sealed source receipt: {scene}")
                source_id = str(source_receipt.get("source_id", ""))
                available = source_receipt.get("available_hypotheses")
                source = source_map.get(source_id)
                hypotheses = source.get("hypotheses") if isinstance(source, Mapping) else None
                if (
                    not isinstance(available, list)
                    or not available
                    or any(name not in HYPOTHESES for name in available)
                    or not isinstance(hypotheses, Mapping)
                ):
                    raise L0OracleError(f"sealed source geometry bank differs: {source_id}")
                for name in available:
                    box = _aligned_geometry(
                        hypotheses[name], name, alignment, f"{source_id}.{name}"
                    )
                    geometry_rows.append((source_id, name, aligned_iou_matrix(box[None, :], gt)[0]))
            if not geometry_rows:
                raise L0OracleError(f"sealed track has no geometry: {scene}:{track['track_id']}")
            scene_geometry_rows.append(geometry_rows)
            scene_max_rows.append(
                np.maximum.reduce([item[2] for item in geometry_rows])
            )
            scene_track_ids.append(int(track["track_id"]))
            geometry_total += len(geometry_rows)
        candidate_max = (
            np.stack(scene_max_rows, axis=0)
            if scene_max_rows
            else np.empty((0, len(gt)), dtype=np.float64)
        )
        baseline_matrices.append(aligned_iou_matrix(native, gt))
        candidate_max_matrices.append(candidate_max)
        candidate_geometry_rows.append(scene_geometry_rows)
        track_ids_by_scene.append(scene_track_ids)
        gt_counts.append(len(gt))
        track_total += len(scene_track_ids)
        scene_reports[scene] = {
            "native_prediction_count": len(native),
            "gt_count": len(gt),
            "confirmed_track_identity_count": len(scene_track_ids),
            "per_view_geometry_count": sum(len(row) for row in scene_geometry_rows),
        }

    expected_counts = seal.get("counts", {})
    if (
        track_total != expected_counts.get(seal_track_count_key)
        or geometry_total != expected_counts.get("valid_geometry_count")
    ):
        raise L0OracleError("L0 oracle census differs from no-GT seal")

    baseline_evaluations = {
        threshold: official_constant_evaluate(baseline_matrices, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    for threshold, evaluation in baseline_evaluations.items():
        expected = EXPECTED_BASELINE_AP_POINTS[f"{threshold:.2f}"]
        if abs(float(evaluation["ap_points"]) - expected) > 1.0e-9:
            raise L0OracleError(
                f"official baseline AP reproduction differs at {threshold}: "
                f"{evaluation['ap_points']} != {expected}"
            )

    per_threshold: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        baseline = baseline_evaluations[threshold]
        suffix_matrices = []
        selections: dict[str, Any] = {}
        winner_counts: Counter[str] = Counter()
        native_matching = candidate_matching = union_matching = selected_total = 0
        for scene, native, candidate_max, geometry_banks, track_ids, matched in zip(
            scenes,
            baseline_matrices,
            candidate_max_matrices,
            candidate_geometry_rows,
            track_ids_by_scene,
            baseline["matched_gt_masks"],
        ):
            native_pairs = strict_maximum_matching(native, threshold)
            candidate_pairs = strict_maximum_matching(candidate_max, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, candidate_max), axis=0), threshold
            )
            suffix_pairs = strict_maximum_matching(
                candidate_max, threshold, ~np.asarray(matched, dtype=np.bool_)
            )
            selected_rows = []
            selected_receipts = []
            for candidate_index, gt_index in suffix_pairs:
                bank = geometry_banks[candidate_index]
                winner = sorted(
                    enumerate(bank),
                    key=lambda pair: (
                        -float(pair[1][2][gt_index]),
                        pair[0],
                    ),
                )[0][1]
                source_id, name, iou_row = winner
                selected_rows.append(iou_row)
                winner_counts[name] += 1
                selected_receipts.append(
                    {
                        "candidate_index": candidate_index,
                        "track_id": track_ids[candidate_index],
                        "source_id": source_id,
                        "hypothesis": name,
                        "target_gt_index": gt_index,
                        "target_iou": float(iou_row[gt_index]),
                    }
                )
            suffix = (
                np.stack(selected_rows, axis=0)
                if selected_rows
                else np.empty((0, candidate_max.shape[1]), dtype=np.float64)
            )
            suffix_matrices.append(suffix)
            selections[scene] = selected_receipts
            native_matching += len(native_pairs)
            candidate_matching += len(candidate_pairs)
            union_matching += len(union_pairs)
            selected_total += len(suffix_pairs)
        combined = [
            np.concatenate((native, suffix), axis=0)
            for native, suffix in zip(baseline_matrices, suffix_matrices)
        ]
        evaluation = official_constant_evaluate(combined, gt_counts, threshold)
        delta = float(evaluation["ap_points"]) - float(baseline["ap_points"])
        additional = union_matching - native_matching
        per_threshold[f"{threshold:.2f}"] = {
            "iou_threshold": threshold,
            "identity_constraint": identity_description,
            "geometry_choice": "GT-assisted source-view and H0/HL/HLG/HB; oracle only",
            "native_maximum_matching_count": native_matching,
            "candidate_maximum_matching_count": candidate_matching,
            "union_maximum_matching_count": union_matching,
            "additional_union_matching_over_native": additional,
            "required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
            "passes_144_match_capacity_gate": additional >= REQUIRED_ADDITIONAL_MATCHES,
            "baseline_official_constant_score": _evaluation_json(baseline),
            "gt_selected_candidate_suffix": {
                "oracle_only": True,
                "deployable": False,
                "native_prefix_unchanged": True,
                "formal_score": 1.0,
                "selected_candidate_count": selected_total,
                "selected_hypothesis_counts": dict(sorted(winner_counts.items())),
                "official_evaluation": _evaluation_json(evaluation),
                "delta_ap_points": delta,
                "passes_plus10_ap": delta > TARGET_DELTA_AP_POINTS,
                "per_scene_selection": selections,
            },
        }

    ap50 = per_threshold["0.50"]
    ap50_delta = float(ap50["gt_selected_candidate_suffix"]["delta_ap_points"])
    gate = bool(
        ap50_delta > TARGET_DELTA_AP_POINTS
        and int(ap50["additional_union_matching_over_native"])
        >= REQUIRED_ADDITIONAL_MATCHES
    )
    return {
        "schema": report_schema,
        "protocol_id": expected_protocol_id,
        "complete": True,
        "oracle_only": True,
        "deployable": False,
        "seal": {"path": os.fspath(seal_path.resolve()), "sha256": seal_sha},
        "counts": {
            "scene_count": 100,
            "gt_count": sum(gt_counts),
            "confirmed_track_identity_count": track_total,
            "per_view_geometry_count": geometry_total,
        },
        "per_threshold": per_threshold,
        "decision": {
            "passes_l0_ap50_plus10_and_144_match_gate": gate,
            "authorize_gt_free_best_view_selector_experiment": gate,
            "authorize_accuracy_claim": False,
            "next_step": (
                "freeze_gt_free_past_only_best_view_selector"
                if gate
                else "f3_confirmation_or_identity_compression_already_below_target"
            ),
        },
        "scenes": scene_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--gt-root", type=Path, default=ROOT / "evaluation/data_util/scannet_train_detection_data")
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise L0OracleError(f"refusing to overwrite L0 oracle report: {args.out}")
    report = audit(
        scene_list=args.scene_list,
        seal_path=args.seal,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": os.fspath(args.out), "counts": report["counts"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
