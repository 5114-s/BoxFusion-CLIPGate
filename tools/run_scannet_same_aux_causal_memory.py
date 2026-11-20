#!/usr/bin/env python3
"""Evaluate a bounded strict-causal score-memory baseline on ScanNet-100.

The baseline consumes the exact frozen per-keyframe WeDetect-Uni/Boxer
proposal and anchor streams used by ReCaR-3D.  It deliberately omits PLR
multi-keyframe confirmation/medoid selection and CALR multi-keyframe
consensus.  Each branch keeps the strongest raw-score exemplar per 0.30 m
world-space voxel in bounded causal state.

Two readouts are materialized:

* ``simple_matched_*`` uses only the *number* of final PLR/CALR outputs in
  each scene as the branch budget.  Candidate identity and selection never
  consult ReCaR outputs, future candidates, or ground truth.
* ``simple_fixed_caps_latest`` uses the deployment caps (12/640) directly.

The matched readout is the primary source/compute/output-count control.  Its
state is independently deployable and strictly causal; count matching is a
terminal evaluation control and is stated as such in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "development/final_controls"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FINAL))

from boxfusion.m1_anchor_online import deterministic_tail_score  # noqa: E402
from boxfusion.online_candidate_map import (  # noqa: E402
    _aabb_iou_matrix,
    _size_price,
)
from tools.audit_final_ledger import scannet_inputs  # noqa: E402
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap  # noqa: E402


THRESHOLDS = (0.15, 0.25, 0.50)


@dataclass
class Record:
    box: np.ndarray
    raw_score: float
    frame_id: int
    source_id: int
    ordinal: int

    @property
    def rank(self) -> tuple:
        center_key = tuple(np.floor(self.box.mean(axis=0) / 0.30).astype(np.int64))
        return (-self.raw_score, self.frame_id, self.source_id, center_key)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in payload[0]
    ]


def write_rows(path: Path, rows: list[tuple[int, np.ndarray, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            [[(int(label), np.array(box, copy=True), float(score))
              for label, box, score in rows]],
            handle,
        )


def same_geometry(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1])
        for a, b in zip(left, right)
    )


def voxel_key(box: np.ndarray) -> tuple[int, int, int]:
    return tuple(np.floor(np.asarray(box).mean(axis=0) / 0.30).astype(np.int64))


def keep_strongest(state: dict, key, record: Record) -> None:
    current = state.get(key)
    if current is None or record.rank < current.rank:
        state[key] = record


def enforce_capacity(state: dict, capacity: int) -> int:
    if len(state) <= capacity:
        return 0
    ordered = sorted(state.items(), key=lambda item: item[1].rank)
    dropped = len(state) - capacity
    state.clear()
    state.update(ordered[:capacity])
    return dropped


class ProposalScoreMemory:
    """Single-observation bounded score memory for post-NMS proposals."""

    def __init__(self) -> None:
        self.active: dict[tuple[int, int, int], Record] = {}
        self.archive: dict[tuple[int, int, int], Record] = {}
        self.peak_active = 0
        self.peak_archive = 0
        self.capacity_drops = 0
        self.frames_seen = 0
        self.observations = 0

    def update(self, ordinal, frame_id, ids, boxes, scores) -> None:
        expired = [
            key for key, row in self.active.items()
            if int(ordinal) - row.ordinal > 10
        ]
        for key in expired:
            del self.active[key]
        order = np.argsort(-np.asarray(scores), kind="stable")[:150]
        self.observations += len(order)
        for index in order:
            row = Record(
                np.asarray(boxes[index], dtype=np.float64).reshape(8, 3).copy(),
                float(scores[index]), int(frame_id), int(ids[index]), int(ordinal),
            )
            key = voxel_key(row.box)
            keep_strongest(self.active, key, row)
            keep_strongest(self.archive, key, row)
        self.capacity_drops += enforce_capacity(self.active, 1024)
        self.capacity_drops += enforce_capacity(self.archive, 1024)
        self.peak_active = max(self.peak_active, len(self.active))
        self.peak_archive = max(self.peak_archive, len(self.archive))
        self.frames_seen += 1

    def rows(self, native_rows, quota: int) -> list[tuple[int, np.ndarray, float]]:
        candidates = sorted(self.archive.values(), key=lambda row: row.rank)
        if candidates and native_rows:
            boxes = np.stack([row.box for row in candidates])
            native = np.stack([row[1] for row in native_rows])
            covered = np.any(_aabb_iou_matrix(boxes, native) >= 0.25, axis=1)
            candidates = [row for row, drop in zip(candidates, covered) if not drop]
        if candidates:
            boxes = np.stack([row.box for row in candidates])
            overlaps = _aabb_iou_matrix(boxes, boxes)
        else:
            overlaps = np.empty((0, 0), dtype=np.float64)
        selected: list[int] = []
        for index in range(len(candidates)):
            if selected and np.any(overlaps[index, selected] >= 0.50):
                continue
            selected.append(index)
            if len(selected) >= int(quota):
                break
        return [
            (0, candidates[index].box.copy(), float(_size_price(candidates[index].box)))
            for index in selected
        ]

    def diagnostics(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "observations": self.observations,
            "active_states": len(self.active),
            "archive_states": len(self.archive),
            "peak_active": self.peak_active,
            "peak_archive": self.peak_archive,
            "active_capacity": 1024,
            "archive_capacity": 1024,
            "ttl_keyframes": 10,
            "capacity_drops": self.capacity_drops,
        }


class AnchorScoreMemory:
    """Strongest-exemplar voxel memory for pre-NMS anchors."""

    def __init__(self) -> None:
        self.state: dict[tuple[int, int, int], Record] = {}
        self.peak_states = 0
        self.capacity_drops = 0
        self.frames_seen = 0
        self.observations = 0

    def update(self, ordinal, frame_id, ids, boxes, scores) -> None:
        order = np.argsort(-np.asarray(scores), kind="stable")[:300]
        self.observations += len(order)
        for index in order:
            row = Record(
                np.asarray(boxes[index], dtype=np.float64).reshape(8, 3).copy(),
                float(scores[index]), int(frame_id), int(ids[index]), int(ordinal),
            )
            keep_strongest(self.state, voxel_key(row.box), row)
        self.capacity_drops += enforce_capacity(self.state, 4096)
        self.peak_states = max(self.peak_states, len(self.state))
        self.frames_seen += 1

    def rows(self, scene: str, quota: int) -> list[tuple[int, np.ndarray, float]]:
        selected = sorted(self.state.values(), key=lambda row: row.rank)[: int(quota)]
        return [
            (
                0,
                row.box.copy(),
                float(deterministic_tail_score(
                    row.raw_score, scene, row.frame_id, row.source_id
                )),
            )
            for row in selected
        ]

    def diagnostics(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "observations": self.observations,
            "states": len(self.state),
            "peak_states": self.peak_states,
            "capacity": 4096,
            "output_capacity": 640,
            "ttl_keyframes": None,
            "capacity_drops": self.capacity_drops,
        }


def load_evidence(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def replay_scene(path: Path):
    values = load_evidence(path)
    proposal = ProposalScoreMemory()
    anchor = AnchorScoreMemory()
    p_offsets = np.r_[0, np.cumsum(values["proposal_lengths"], dtype=np.int64)]
    a_offsets = np.r_[0, np.cumsum(values["anchor_lengths"], dtype=np.int64)]
    start = time.perf_counter()
    for ordinal, frame_id in enumerate(values["frame_ids"]):
        p0, p1 = int(p_offsets[ordinal]), int(p_offsets[ordinal + 1])
        a0, a1 = int(a_offsets[ordinal]), int(a_offsets[ordinal + 1])
        proposal.update(
            ordinal, int(frame_id), values["proposal_ids"][p0:p1],
            values["proposal_corners"][p0:p1], values["proposal_scores"][p0:p1],
        )
        anchor.update(
            ordinal, int(frame_id), values["anchor_ids"][a0:a1],
            values["anchor_corners"][a0:a1], values["anchor_scores"][a0:a1],
        )
    elapsed = time.perf_counter() - start
    return proposal, anchor, elapsed


def prediction(rows, transform, alignment):
    boxes = (
        np.stack([row[1] for row in rows])
        if rows else np.empty((0, 8, 3), dtype=np.float64)
    )
    scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    return transform(boxes, alignment), scores


def added_quality(rows, native_count, gt, transform, alignment, threshold):
    boxes, scores = prediction(rows, transform, alignment)
    matrix = (
        aabb_iou(boxes, gt)
        if len(boxes) and len(gt) else np.zeros((len(boxes), len(gt)))
    )
    matched = np.zeros(len(gt), dtype=bool)
    result = {"tp": 0, "fp": 0, "duplicate": 0}
    for index in np.argsort(-scores, kind="stable"):
        values = matrix[index]
        target = int(values.argmax()) if len(values) else -1
        passes = target >= 0 and float(values[target]) > threshold
        duplicate = bool(passes and matched[target])
        true_positive = bool(passes and not matched[target])
        if true_positive:
            matched[target] = True
        if index >= native_count:
            result["tp" if true_positive else "fp"] += 1
            result["duplicate"] += int(duplicate)
    native_overlap = matrix[:native_count]
    added_overlap = matrix[native_count:]
    native_covered = (
        np.any(native_overlap > threshold, axis=0)
        if len(native_overlap) else np.zeros(len(gt), dtype=bool)
    )
    added_covered = (
        np.any(added_overlap > threshold, axis=0)
        if len(added_overlap) else np.zeros(len(gt), dtype=bool)
    )
    result["new_unique_gt"] = int(np.sum(added_covered & ~native_covered))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--evidence-root", type=Path,
        default=FINAL / "experiments/mvsr_unified_samepass_20261010/evidence",
    )
    parser.add_argument(
        "--canonical-components", type=Path,
        default=FINAL / "experiments/scannet_factorial_1787_canonical_20261012/components",
    )
    parser.add_argument(
        "--mvsr-components", type=Path,
        default=FINAL / "experiments/mvsr_unified_samepass_20261010/components",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "results.json").exists():
        raise RuntimeError(f"refusing to overwrite completed output: {args.output}")

    scenes = [
        line.strip() for line in args.scene_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the unique official ScanNet-100 scene list")
    roots = {
        "base": args.output / "predictions/base",
        "simple_matched_native": args.output / "predictions/simple_matched_native",
        "simple_matched_latest": args.output / "predictions/simple_matched_latest",
        "simple_matched_maximum": args.output / "predictions/simple_matched_maximum",
        "simple_fixed_caps_latest": args.output / "predictions/simple_fixed_caps_latest",
        "simple_fixed_caps_maximum": args.output / "predictions/simple_fixed_caps_maximum",
        "full_recar": args.output / "predictions/full_recar",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    gts, alignments, transform = scannet_inputs()
    predictions = {name: {} for name in roots}
    counts = {name: 0 for name in roots}
    branch_counts = {
        name: {"native": 0, "proposal": 0, "anchor": 0}
        for name in roots
    }
    quality = {
        name: {
            str(t): {"tp": 0, "fp": 0, "duplicate": 0, "new_unique_gt": 0}
            for t in THRESHOLDS
        }
        for name in roots if name != "base"
    }
    diagnostics = {}
    hashes = {str(args.scene_list.resolve()): sha256(args.scene_list)}
    total_replay_seconds = 0.0

    for scene in scenes:
        filename = f"{scene}_boxes.pkl"
        paths = {
            "native": args.canonical_components / "native" / filename,
            "plr": args.canonical_components / "plr" / filename,
            "calr": args.canonical_components / "calr" / filename,
            "latest": args.mvsr_components / "native_factor_no_visibility_latest" / filename,
            "maximum": args.mvsr_components / "native_factor_no_visibility_max" / filename,
            "evidence": args.evidence_root / f"{scene}.npz",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[str(path.resolve())] = sha256(path)
        native = read_rows(paths["native"])
        plr_target = read_rows(paths["plr"])
        calr_target = read_rows(paths["calr"])
        latest = read_rows(paths["latest"])
        maximum = read_rows(paths["maximum"])
        if not same_geometry(native, latest) or not same_geometry(native, maximum):
            raise RuntimeError(f"native geometry changed in MVSR component: {scene}")

        proposal_memory, anchor_memory, elapsed = replay_scene(paths["evidence"])
        total_replay_seconds += elapsed
        matched_proposals = proposal_memory.rows(native, len(plr_target))
        matched_anchors = anchor_memory.rows(scene, len(calr_target))
        capped_proposals = proposal_memory.rows(native, 12)
        capped_anchors = anchor_memory.rows(scene, 640)
        if len(matched_proposals) != len(plr_target):
            raise RuntimeError(
                f"proposal memory cannot meet matched budget in {scene}: "
                f"{len(matched_proposals)} != {len(plr_target)}"
            )
        if len(matched_anchors) != len(calr_target):
            raise RuntimeError(
                f"anchor memory cannot meet matched budget in {scene}: "
                f"{len(matched_anchors)} != {len(calr_target)}"
            )

        outputs = {
            "base": native,
            "simple_matched_native": native + matched_proposals + matched_anchors,
            "simple_matched_latest": latest + matched_proposals + matched_anchors,
            "simple_matched_maximum": maximum + matched_proposals + matched_anchors,
            "simple_fixed_caps_latest": latest + capped_proposals + capped_anchors,
            "simple_fixed_caps_maximum": maximum + capped_proposals + capped_anchors,
            "full_recar": maximum + plr_target + calr_target,
        }
        for name, rows in outputs.items():
            write_rows(roots[name] / filename, rows)
            predictions[name][scene] = prediction(rows, transform, alignments[scene])
            counts[name] += len(rows)
            branch_counts[name]["native"] += len(native)
            if name.startswith("simple_matched"):
                branch_counts[name]["proposal"] += len(matched_proposals)
                branch_counts[name]["anchor"] += len(matched_anchors)
            elif name in {
                "simple_fixed_caps_latest", "simple_fixed_caps_maximum"
            }:
                branch_counts[name]["proposal"] += len(capped_proposals)
                branch_counts[name]["anchor"] += len(capped_anchors)
            elif name == "full_recar":
                branch_counts[name]["proposal"] += len(plr_target)
                branch_counts[name]["anchor"] += len(calr_target)
            if name != "base":
                for threshold in THRESHOLDS:
                    row = added_quality(
                        rows, len(native), gts[scene], transform,
                        alignments[scene], threshold,
                    )
                    for key, value in row.items():
                        quality[name][str(threshold)][key] += int(value)
        diagnostics[scene] = {
            "target_counts": {
                "plr": len(plr_target), "calr": len(calr_target),
            },
            "matched_counts": {
                "proposal": len(matched_proposals), "anchor": len(matched_anchors),
            },
            "fixed_cap_counts": {
                "proposal": len(capped_proposals), "anchor": len(capped_anchors),
            },
            "proposal_memory": proposal_memory.diagnostics(),
            "anchor_memory": anchor_memory.diagnostics(),
            "replay_seconds": elapsed,
        }

    exact_ap = {
        name: {
            str(t): class_agnostic_ap(values, gts, t)["ap"]
            for t in THRESHOLDS
        }
        for name, values in predictions.items()
    }
    result = {
        "schema": "boxfusion.scannet100.same_aux_strict_causal_score_memory.v1",
        "scene_count": len(scenes),
        "native_box_count": branch_counts["base"]["native"],
        "strict_causal_state": True,
        "future_frames_used_for_state_update": False,
        "ground_truth_used": False,
        "same_frozen_auxiliary_stream_as_recar": True,
        "same_per_frame_input_caps": {"post_nms_proposals": 150, "pre_nms_anchors": 300},
        "matched_readout": (
            "per-scene terminal PLR/CALR output counts only; candidate identities "
            "and ranking are independently selected by raw-score causal memory"
        ),
        "fixed_cap_readout": {"proposal": 12, "anchor": 640},
        "counts": counts,
        "branch_counts": branch_counts,
        "exact_ap": exact_ap,
        "added_quality": quality,
        "total_cpu_replay_seconds": total_replay_seconds,
        "per_scene_diagnostics": diagnostics,
        "input_sha256": hashes,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(
        json.dumps({
            key: result[key] for key in (
                "schema", "scene_count", "native_box_count", "strict_causal_state",
                "future_frames_used_for_state_update", "ground_truth_used",
                "same_frozen_auxiliary_stream_as_recar", "same_per_frame_input_caps",
                "matched_readout", "fixed_cap_readout", "counts", "branch_counts",
                "total_cpu_replay_seconds", "input_sha256",
            )
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ScanNet-100 same-auxiliary-stream causal score-memory baseline", "",
        "All arms use the canonical 1,787 native boxes and the exact frozen "
        "WeDetect-Uni/Boxer candidate stream used by ReCaR-3D.", "",
        "| Arm | Native/Proposal/Anchor | Total | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in roots:
        branch = branch_counts[name]
        metric = exact_ap[name]
        lines.append(
            f"| {name} | {branch['native']:,}/{branch['proposal']:,}/{branch['anchor']:,} "
            f"| {counts[name]:,} | {metric['0.15']:.4f} | {metric['0.25']:.4f} "
            f"| {metric['0.5']:.4f} |"
        )
    lines += [
        "",
        "The score-memory state is strictly causal and bounded. The matched arm "
        "uses only ReCaR's per-scene branch output counts as a terminal evaluation "
        "budget; it does not reuse ReCaR-selected boxes or ground truth. The fixed-cap "
        "arm is fully standalone and uses the deployment caps 12/640.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
