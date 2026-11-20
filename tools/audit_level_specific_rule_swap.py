#!/usr/bin/env python3
"""Diagnostic rule-swap audit for post-NMS proposals and pre-NMS anchors.

The proposal-side swap is count- and score-multiset matched to current PLR.
The anchor-side track rule is reported at its natural confirmed count because
the exact PLR association rule often produces fewer confirmed hypotheses than
CALR's voxel rule; padding with unconfirmed anchors would invalidate the test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "development/final_controls"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FINAL))
from boxfusion.m1_anchor_online import OnlineAnchorRecovery, deterministic_tail_score
from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap


def read_rows(path: Path):
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    return [(int(x[0]), np.asarray(x[1], dtype=np.float64).reshape(8, 3), float(x[2])) for x in rows]


def pair_iou(one, two):
    lo = np.maximum(one.min(0), two.min(0))
    hi = np.minimum(one.max(0), two.max(0))
    inter = np.maximum(0.0, hi - lo).prod()
    union = np.ptp(one, axis=0).prod() + np.ptp(two, axis=0).prod() - inter
    return float(inter / union) if union > 0 else 0.0


def matrix_iou(boxes):
    if not boxes:
        return np.empty((0, 0))
    value = np.stack(boxes)
    return aabb_iou(value, value)


def medoid(observations):
    ordered = sorted(observations.values(), key=lambda x: (x[0], x[1]))
    boxes = [row[3] for row in ordered]
    sums = matrix_iou(boxes).sum(1)
    index = max(range(len(ordered)), key=lambda i: (float(sums[i]), ordered[i][2], -ordered[i][0], -ordered[i][1]))
    return ordered[index]


@dataclass
class Track:
    track_id: int
    observations: dict = field(default_factory=dict)
    last_ordinal: int = -1

    @property
    def latest(self):
        return max(self.observations.values(), key=lambda row: (row[0], row[1]))

    def add(self, ordinal, frame, source_id, score, box):
        current = self.observations.get(frame)
        if current is None or (-score, source_id) < (-current[2], current[1]):
            self.observations[frame] = (ordinal, source_id, score, box.copy())
        self.last_ordinal = max(self.last_ordinal, ordinal)
        if len(self.observations) > 12:
            oldest = min(self.observations, key=lambda key: self.observations[key][0])
            del self.observations[oldest]


def anchor_track_candidates(frame_ids, lengths, ids, corners, scores):
    offsets = np.r_[0, np.cumsum(lengths)]
    tracks, confirmed = {}, {}
    next_id = 0
    for ordinal, frame in enumerate(frame_ids):
        expired = [key for key, track in tracks.items() if ordinal - track.last_ordinal > 10]
        for key in expired:
            del tracks[key]
        grid = {}
        for key, track in tracks.items():
            cell = tuple(np.floor(track.latest[3].mean(0) / 0.5).astype(np.int64))
            grid.setdefault(cell, []).append(key)
        lo, hi = int(offsets[ordinal]), int(offsets[ordinal + 1])
        for index in np.argsort(-scores[lo:hi], kind="stable") + lo:
            box, center = corners[index], corners[index].mean(0)
            cell = np.floor(center / 0.5).astype(np.int64)
            nearby = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nearby.extend(grid.get(tuple(cell + (dx, dy, dz)), ()))
            best = None
            for key in nearby:
                latest = tracks[key].latest[3]
                distance = float(np.linalg.norm(latest.mean(0) - center))
                if distance > 0.50:
                    continue
                overlap = pair_iou(latest, box)
                rank = (-overlap, distance, key)
                if overlap >= 0.10 and (best is None or rank < best[0]):
                    best = (rank, key)
            if best is None:
                if len(tracks) >= 4096:
                    continue
                key = next_id
                next_id += 1
                track = Track(key)
                track.add(ordinal, int(frame), int(ids[index]), float(scores[index]), box)
                tracks[key] = track
                grid.setdefault(tuple(cell), []).append(key)
            else:
                key = best[1]
                track = tracks[key]
                old_cell = tuple(np.floor(track.latest[3].mean(0) / 0.5).astype(np.int64))
                track.add(ordinal, int(frame), int(ids[index]), float(scores[index]), box)
                new_cell = tuple(np.floor(track.latest[3].mean(0) / 0.5).astype(np.int64))
                if old_cell != new_cell:
                    grid[old_cell].remove(key)
                    grid.setdefault(new_cell, []).append(key)
            if len(track.observations) >= 3:
                confirmed[key] = track
    candidates = []
    for key, track in confirmed.items():
        ordinal, source_id, score, box = medoid(track.observations)
        raw_mean = float(np.mean([row[2] for row in track.observations.values()]))
        candidates.append((raw_mean, int(ordinal), int(source_id), box, float(score), key))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[5]))
    return candidates[:640], {"tracks_created": next_id, "confirmed": len(confirmed)}


def proposal_voxel_candidates(scene, frame_ids, lengths, ids, corners, scores, native, target_count, target_scores):
    state = OnlineAnchorRecovery(scene, voxel_m=0.3, min_views=3, max_active=4096, max_births=4096)
    offsets = np.r_[0, np.cumsum(lengths)]
    for ordinal, frame in enumerate(frame_ids):
        lo, hi = int(offsets[ordinal]), int(offsets[ordinal + 1])
        state.update(ordinal, int(frame), ids[lo:hi], corners[lo:hi], scores[lo:hi])
    candidates = state.rows()
    if candidates and native:
        matrix = aabb_iou(np.stack([row["box"] for row in candidates]), np.stack([row[1] for row in native]))
        candidates = [row for index, row in enumerate(candidates) if not np.any(matrix[index] >= 0.25)]
    candidates.sort(key=lambda row: (-row["raw_score"], row["frame_id"], row["anchor_id"]))
    selected = []
    for row in candidates:
        if selected and np.any(aabb_iou(np.asarray(row["box"])[None], np.stack([x["box"] for x in selected])) >= 0.50):
            continue
        selected.append(row)
        if len(selected) >= target_count:
            break
    prices = sorted((float(value) for value in target_scores), reverse=True)[:len(selected)]
    rows = [(0, np.asarray(row["box"]), prices[index]) for index, row in enumerate(selected)]
    return rows, {"confirmed_voxels": len(state.rows()), "selected": len(rows)}


def process_scene(payload):
    scene, evidence_path, component_root, factorial_root = payload
    with np.load(evidence_path, allow_pickle=False) as z:
        values = {key: np.asarray(z[key]) for key in z.files}
    native = read_rows(component_root / "native_factor_no_visibility" / f"{scene}_boxes.pkl")
    plr = read_rows(component_root / "births_plr_v1" / f"{scene}_boxes.pkl")
    prefix_plr = read_rows(factorial_root / "plr_mvsr" / f"{scene}_boxes.pkl")
    full = read_rows(factorial_root / "full" / f"{scene}_boxes.pkl")
    current_calr = full[len(prefix_plr):]
    voxel_rows, voxel_diag = proposal_voxel_candidates(
        scene, values["frame_ids"], values["proposal_lengths"], values["proposal_ids"],
        values["proposal_corners"], values["proposal_scores"], native,
        len(plr), [row[2] for row in plr],
    )
    tracks, track_diag = anchor_track_candidates(
        values["frame_ids"], values["anchor_lengths"], values["anchor_ids"],
        values["anchor_corners"], values["anchor_scores"],
    )
    track_rows = [
        (0, row[3], deterministic_tail_score(row[4], scene, int(values["frame_ids"][row[1]]), row[2]))
        for row in tracks
    ]
    return {
        "scene": scene, "native": native, "plr": plr,
        "plr_matched": sorted(plr, key=lambda row: -row[2])[:len(voxel_rows)],
        "current_calr": current_calr,
        "prefix_plr": prefix_plr, "voxel_proposal": voxel_rows,
        "track_anchor": track_rows, "voxel_diag": voxel_diag, "track_diag": track_diag,
    }


def as_prediction(rows, transform, alignment):
    boxes = np.stack([row[1] for row in rows]) if rows else np.empty((0, 8, 3))
    return transform(boxes, alignment), np.asarray([row[2] for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = FINAL / "experiments/mvsr_design_20261005"
    components = experiment / "components"
    factorial = experiment / "factorial_no_visibility"
    evidence = FINAL / "evidence/scannet_recar3d_key_controls"
    gts, alignments, transform = scannet_inputs()
    scenes = list(gts)
    tasks = [(scene, evidence / f"{scene}.npz", components, factorial) for scene in scenes]
    if args.workers == 1:
        rows = [process_scene(task) for task in tasks]
    else:
        with Pool(args.workers) as pool:
            rows = list(pool.imap(process_scene, tasks, chunksize=1))
    arms = {name: {} for name in ("proposal_track", "proposal_voxel", "anchor_voxel", "anchor_track")}
    counts = {name: 0 for name in arms}
    per_scene = {}
    for value in rows:
        scene = value["scene"]
        native = value["native"]
        plr = value["plr"]
        calr = value["current_calr"]
        prefix_plr = value["prefix_plr"]
        # Proposal comparison shares native+CALR and the exact PLR score multiset.
        proposal_prefix = native + calr
        assembled = {
            "proposal_track": proposal_prefix + value["plr_matched"],
            "proposal_voxel": proposal_prefix + value["voxel_proposal"],
            "anchor_voxel": prefix_plr + calr,
            "anchor_track": prefix_plr + value["track_anchor"],
        }
        per_scene[scene] = {
            "proposal_original_budget": len(plr),
            "proposal_matched_budget": len(value["plr_matched"]),
            "anchor_voxel_count": len(calr),
            "anchor_track_count": len(value["track_anchor"]),
            "proposal_voxel": value["voxel_diag"], "anchor_track": value["track_diag"],
        }
        for name, arm_rows in assembled.items():
            arms[name][scene] = as_prediction(arm_rows, transform, alignments[scene])
            counts[name] += len(arm_rows)
    metrics = {
        name: {str(t): class_agnostic_ap(pred, gts, t)["ap"] for t in (0.15, 0.25, 0.5)}
        for name, pred in arms.items()
    }
    result = {
        "schema": "boxfusion.scannet100.level_specific_rule_swap.v1",
        "scene_count": len(scenes), "metrics": metrics, "prediction_counts": counts,
        "per_scene": per_scene,
        "proposal_control": "same per-scene attainable output count=min(current PLR count, confirmed proposal-voxel count), same top PLR score multiset; terminal selection; GT-free",
        "anchor_control": "natural count under PLR gates; no padding with unconfirmed anchors; terminal readout; GT-free",
        "causal_state": True, "strict_online_output_for_controls": False,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# ScanNet-100 level-specific rule-swap audit", "",
        "| Candidate source | Evidence organizer | Predictions | AP15 | AP25 | AP50 |",
        "|---|---|---:|---:|---:|---:|",
        f"| post-NMS proposals | trajectory (PLR rule, matched subset) | {counts['proposal_track']:,} | {metrics['proposal_track']['0.15']:.4f} | {metrics['proposal_track']['0.25']:.4f} | {metrics['proposal_track']['0.5']:.4f} |",
        f"| post-NMS proposals | 0.3 m voxel consensus | {counts['proposal_voxel']:,} | {metrics['proposal_voxel']['0.15']:.4f} | {metrics['proposal_voxel']['0.25']:.4f} | {metrics['proposal_voxel']['0.5']:.4f} |",
        f"| pre-NMS anchors | voxel consensus (current CALR) | {counts['anchor_voxel']:,} | {metrics['anchor_voxel']['0.15']:.4f} | {metrics['anchor_voxel']['0.25']:.4f} | {metrics['anchor_voxel']['0.5']:.4f} |",
        f"| pre-NMS anchors | trajectory association | {counts['anchor_track']:,} | {metrics['anchor_track']['0.15']:.4f} | {metrics['anchor_track']['0.25']:.4f} | {metrics['anchor_track']['0.5']:.4f} |",
        "", "The proposal comparison is matched per scene at the largest budget attainable by both rules and reuses the same top PLR score multiset; current PLR is truncated only where the voxel rule confirms fewer hypotheses. The anchor trajectory rule naturally confirms fewer hypotheses than CALR in some scenes and is therefore reported at its natural count. Padding either control with unconfirmed candidates would no longer test the stated rule. Both swapped controls update state causally but use a scene-end readout and must not be described as strict-online output arms.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
