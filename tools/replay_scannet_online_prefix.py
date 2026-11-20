#!/usr/bin/env python3
"""Replay final ReCaR-3D on cached native prefixes and measure online quality."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "development" / "mvsr_multiview_audit_20261012_source"
sys.path.insert(0, str(ROOT))
# The isolated source tree must precede the repository package.  It contains
# the final three-factor MVSR implementation used by the audited prediction
# run, whereas the repository-level package is retained only for local tools.
sys.path.insert(0, str(SOURCE))

from boxfusion.online_candidate_map import EvidenceFrame, NativeFrame  # noqa: E402
from boxfusion.reliability_candidate_map import build_reliability_candidate_map  # noqa: E402
from tools.audit_final_ledger import scannet_inputs  # noqa: E402
from tools.true_fusion_audit_core import aabb_iou  # noqa: E402


THRESHOLDS = (0.15, 0.25, 0.50)
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def slices(lengths: np.ndarray):
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    return [(int(offsets[i]), int(offsets[i + 1])) for i in range(len(lengths))]


def load_native(path: Path):
    with np.load(path, allow_pickle=False) as z:
        values = {key: np.asarray(z[key]) for key in z.files}
    parts = slices(values["native_lengths"])
    rows = []
    for i, (low, high) in enumerate(parts):
        rows.append((
            int(values["frame_ids"][i]),
            NativeFrame(
                ids=values["native_ids"][low:high],
                corners=values["native_corners"][low:high],
                scores=values["native_scores"][low:high],
                camera_to_world=values["camera_to_world"][i],
                intrinsic=values["intrinsic"][i],
                width=int(values["width"][i]),
                height=int(values["height"][i]),
            ),
        ))
    return rows


def load_evidence(path: Path):
    with np.load(path, allow_pickle=False) as z:
        values = {key: np.asarray(z[key]) for key in z.files}
    proposal_parts = slices(values["proposal_lengths"])
    anchor_parts = slices(values["anchor_lengths"])
    rows = []
    for i, ((pl, ph), (al, ah)) in enumerate(zip(proposal_parts, anchor_parts)):
        rows.append((
            int(values["frame_ids"][i]),
            EvidenceFrame(
                proposal_ids=values["proposal_ids"][pl:ph],
                proposal_boxes_2d=values["proposal_boxes_2d"][pl:ph],
                proposal_corners=values["proposal_corners"][pl:ph],
                proposal_scores=values["proposal_scores"][pl:ph],
                anchor_ids=values["anchor_ids"][al:ah],
                anchor_corners=values["anchor_corners"][al:ah],
                anchor_scores=values["anchor_scores"][al:ah],
            ),
        ))
    return rows


def write_prediction(path: Path, boxes: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = [[(0, np.array(box, copy=True), float(score)) for box, score in zip(boxes, scores)]]
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def matched_gt(boxes: np.ndarray, gt: np.ndarray, threshold: float) -> np.ndarray:
    if not len(gt) or not len(boxes):
        return np.zeros(len(gt), dtype=bool)
    return aabb_iou(boxes, gt).max(axis=0) > threshold


def percentile(values: list[int], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-prefix-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terminal-readout",
        action="store_true",
        help=(
            "At every prefix checkpoint, apply the host ScanNet terminal "
            "native-box extent filter and read the causal state on the "
            "surviving native identities."
        ),
    )
    parser.add_argument(
        "--native-min-extent-m",
        type=float,
        default=0.30,
        help="Minimum extent on every native AABB axis for terminal readout.",
    )
    args = parser.parse_args()
    scenes = [row.strip() for row in args.scene_list.read_text().splitlines() if row.strip() and not row.startswith("#")]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected ScanNet official100")
    config = yaml.safe_load(args.config.read_text())
    state_config = dict(config["online_candidate_map"]["state"])
    state_config.update({
        "mvsr_only": False,
        "selected_calr_v1": True,
        "audit_controls": False,
        "audit_direct_plr": False,
        "audit_raw_iou_max": False,
        "audit_mvsr_factors": False,
        "audit_plr_controls": False,
    })
    state_config["m2"] = {**state_config["m2"], "strength_variant": "no_visibility", "support_mode": "max"}
    gts, alignments, transform = scannet_inputs()
    per_scene = {}
    pooled = {str(t): {"delays_keyframes": [], "delays_frames": [], "first_output_keyframes": [], "eligible": 0, "detected": 0} for t in THRESHOLDS}
    checkpoint_totals = {
        str(f): {
            "keyframes": 0,
            "scenes": 0,
            "native_records": 0,
            "recar_records": 0,
        }
        for f in FRACTIONS
    }
    for scene_index, scene in enumerate(scenes, 1):
        native_rows = load_native(args.native_prefix_root / f"{scene}.npz")
        evidence_rows = load_evidence(args.evidence_root / f"{scene}.npz")
        if [row[0] for row in native_rows] != [row[0] for row in evidence_rows]:
            raise RuntimeError(f"{scene}: native/evidence frame ids differ")
        total = len(native_rows)
        checkpoint_indices = {min(total - 1, max(0, int(np.ceil(total * f)) - 1)): f for f in FRACTIONS}
        state = build_reliability_candidate_map(scene, state_config)
        first_evidence = {t: np.full(len(gts[scene]), -1, dtype=np.int64) for t in THRESHOLDS}
        first_output = {t: np.full(len(gts[scene]), -1, dtype=np.int64) for t in THRESHOLDS}
        output_frame = {t: np.full(len(gts[scene]), -1, dtype=np.int64) for t in THRESHOLDS}
        for ordinal, ((frame_id, native), (_, evidence)) in enumerate(zip(native_rows, evidence_rows)):
            snapshot = state.update(frame_id, native, evidence)
            evidence_boxes = np.concatenate((evidence.proposal_corners, evidence.anchor_corners), axis=0)
            aligned_evidence = transform(evidence_boxes, alignments[scene])
            aligned_output = transform(snapshot.boxes, alignments[scene])
            for threshold in THRESHOLDS:
                seen = matched_gt(aligned_evidence, gts[scene], threshold)
                first_evidence[threshold][(first_evidence[threshold] < 0) & seen] = ordinal
                detected = matched_gt(aligned_output, gts[scene], threshold)
                fresh = (first_output[threshold] < 0) & detected
                first_output[threshold][fresh] = ordinal
                output_frame[threshold][fresh] = frame_id
            if ordinal in checkpoint_indices:
                fraction = checkpoint_indices[ordinal]
                tag = f"p{int(round(100 * fraction))}"
                if args.terminal_readout:
                    extents = np.ptp(native.corners, axis=1)
                    valid = np.all(
                        extents >= float(args.native_min_extent_m), axis=1
                    )
                    readout_ids = native.ids[valid]
                    readout_corners = native.corners[valid]
                    readout_scores = native.scores[valid]
                    readout = state.materialize_terminal(
                        frame_id,
                        readout_ids,
                        readout_corners,
                        readout_scores,
                    )
                else:
                    readout_corners = native.corners
                    readout_scores = native.scores
                    readout = snapshot
                write_prediction(
                    args.output / "predictions" / f"base_{tag}" / f"{scene}_boxes.pkl",
                    readout_corners,
                    readout_scores,
                )
                write_prediction(
                    args.output / "predictions" / f"recar_{tag}" / f"{scene}_boxes.pkl",
                    readout.boxes,
                    readout.scores,
                )
                checkpoint_totals[str(fraction)]["keyframes"] += ordinal + 1
                checkpoint_totals[str(fraction)]["scenes"] += 1
                checkpoint_totals[str(fraction)]["native_records"] += len(
                    readout_corners
                )
                checkpoint_totals[str(fraction)]["recar_records"] += len(
                    readout.boxes
                )
        scene_delay = {}
        for threshold in THRESHOLDS:
            eligible = first_evidence[threshold] >= 0
            detected = first_output[threshold] >= 0
            both = eligible & detected
            delays_k = np.maximum(first_output[threshold][both] - first_evidence[threshold][both], 0).tolist()
            # Gap is normally 25, but use actual cached frame ids for exactness.
            ids = np.asarray([row[0] for row in native_rows], dtype=np.int64)
            delays_f = np.maximum(ids[first_output[threshold][both]] - ids[first_evidence[threshold][both]], 0).tolist()
            first_k = first_output[threshold][detected].tolist()
            bucket = pooled[str(threshold)]
            bucket["delays_keyframes"].extend(delays_k)
            bucket["delays_frames"].extend(delays_f)
            bucket["first_output_keyframes"].extend(first_k)
            bucket["eligible"] += int(eligible.sum())
            bucket["detected"] += int((eligible & detected).sum())
            scene_delay[str(threshold)] = {
                "eligible_gt": int(eligible.sum()), "detected_eligible_gt": int((eligible & detected).sum()),
                "median_delay_keyframes": percentile(delays_k, 50), "p90_delay_keyframes": percentile(delays_k, 90),
                "median_delay_frames": percentile(delays_f, 50), "p90_delay_frames": percentile(delays_f, 90),
            }
        per_scene[scene] = {"keyframes": total, "delay": scene_delay}
        if scene_index % 10 == 0:
            print(f"replayed {scene_index}/100", flush=True)
    summary = {}
    for threshold in THRESHOLDS:
        bucket = pooled[str(threshold)]
        summary[str(threshold)] = {
            "eligible_gt": bucket["eligible"], "detected_eligible_gt": bucket["detected"],
            "eligible_detection_rate": bucket["detected"] / max(bucket["eligible"], 1),
            "median_delay_keyframes": percentile(bucket["delays_keyframes"], 50),
            "p90_delay_keyframes": percentile(bucket["delays_keyframes"], 90),
            "p95_delay_keyframes": percentile(bucket["delays_keyframes"], 95),
            "median_delay_frames": percentile(bucket["delays_frames"], 50),
            "p90_delay_frames": percentile(bucket["delays_frames"], 90),
            "median_first_output_keyframe_from_start": percentile(bucket["first_output_keyframes"], 50),
        }
    report = {
        "schema": "boxfusion.scannet100.online_prefix_quality.v2",
        "causal": True, "scene_end_inference": False,
        "readout_protocol": (
            "prefix_terminal_native_extent_filter_and_state_readout"
            if args.terminal_readout
            else "provisional_keyframe_snapshot"
        ),
        "native_min_extent_m": (
            float(args.native_min_extent_m)
            if args.terminal_readout
            else None
        ),
        "future_observations_used": False,
        "history_replayed_at_readout": False,
        "delay_origin": "first keyframe whose frozen auxiliary proposal-or-anchor evidence overlaps the GT at the same IoU threshold",
        "checkpoints": checkpoint_totals,
        "delay_summary": summary, "per_scene": per_scene,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replay_results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
