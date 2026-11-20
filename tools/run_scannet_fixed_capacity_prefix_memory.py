#!/usr/bin/env python3
"""Strict per-keyframe fixed-capacity memory control on ScanNet-100.

This control consumes the exact cached per-keyframe native map and frozen
WeDetect-Uni/Boxer proposal/anchor streams used by the final ReCaR-3D run.
It keeps a bounded strongest-score voxel memory, but deliberately omits PLR
multi-keyframe track confirmation/medoid selection and CALR multi-keyframe
voxel confirmation.  Capacities and output caps are fixed before evaluation.

The memory output is materialized after every keyframe.  AP is reported at
25/50/75/100 percent sequence prefixes.  HCSR is shared with ReCaR-3D so the
comparison isolates the recovery-state organization rather than reranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "development/mvsr_multiview_audit_20261012_source"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

from boxfusion.m1_anchor_online import deterministic_tail_score  # noqa: E402
from boxfusion.online_candidate_map import (  # noqa: E402
    EvidenceFrame,
    NativeFrame,
    _aabb_iou_matrix,
    _size_price,
)
from boxfusion.reliability_candidate_map import (  # noqa: E402
    build_reliability_candidate_map,
)
from tools.audit_final_ledger import scannet_inputs  # noqa: E402
from tools.true_fusion_audit_core import class_agnostic_ap  # noqa: E402


THRESHOLDS = (0.15, 0.25, 0.50)
FRACTIONS = (0.25, 0.50, 0.75, 1.00)


@dataclass
class Record:
    box: np.ndarray
    raw_score: float
    frame_id: int
    source_id: int
    ordinal: int

    @property
    def rank(self) -> tuple:
        center_key = tuple(
            np.floor(self.box.mean(axis=0) / 0.30).astype(np.int64)
        )
        return (-self.raw_score, self.frame_id, self.source_id, center_key)


@dataclass
class ActiveCell:
    exemplar: Record
    last_seen_ordinal: int

    @property
    def rank(self) -> tuple:
        return self.exemplar.rank


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slices(lengths: np.ndarray) -> list[tuple[int, int]]:
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    return [
        (int(offsets[index]), int(offsets[index + 1]))
        for index in range(len(lengths))
    ]


def load_native(path: Path) -> list[tuple[int, NativeFrame]]:
    with np.load(path, allow_pickle=False) as values:
        arrays = {key: np.asarray(values[key]) for key in values.files}
    rows = []
    for index, (low, high) in enumerate(slices(arrays["native_lengths"])):
        rows.append((
            int(arrays["frame_ids"][index]),
            NativeFrame(
                ids=arrays["native_ids"][low:high],
                corners=arrays["native_corners"][low:high],
                scores=arrays["native_scores"][low:high],
                camera_to_world=arrays["camera_to_world"][index],
                intrinsic=arrays["intrinsic"][index],
                width=int(arrays["width"][index]),
                height=int(arrays["height"][index]),
            ),
        ))
    return rows


def load_evidence(path: Path) -> list[tuple[int, EvidenceFrame]]:
    with np.load(path, allow_pickle=False) as values:
        arrays = {key: np.asarray(values[key]) for key in values.files}
    proposal_parts = slices(arrays["proposal_lengths"])
    anchor_parts = slices(arrays["anchor_lengths"])
    rows = []
    for index, ((p0, p1), (a0, a1)) in enumerate(
        zip(proposal_parts, anchor_parts)
    ):
        rows.append((
            int(arrays["frame_ids"][index]),
            EvidenceFrame(
                proposal_ids=arrays["proposal_ids"][p0:p1],
                proposal_boxes_2d=arrays["proposal_boxes_2d"][p0:p1],
                proposal_corners=arrays["proposal_corners"][p0:p1],
                proposal_scores=arrays["proposal_scores"][p0:p1],
                anchor_ids=arrays["anchor_ids"][a0:a1],
                anchor_corners=arrays["anchor_corners"][a0:a1],
                anchor_scores=arrays["anchor_scores"][a0:a1],
            ),
        ))
    return rows


def voxel_key(box: np.ndarray) -> tuple[int, int, int]:
    return tuple(
        np.floor(np.asarray(box).mean(axis=0) / 0.30).astype(np.int64)
    )


def enforce_capacity(state: dict, capacity: int) -> int:
    if len(state) <= capacity:
        return 0
    ordered = sorted(state.items(), key=lambda item: item[1].rank)
    dropped = len(state) - capacity
    state.clear()
    state.update(ordered[:capacity])
    return dropped


class ProposalScoreMemory:
    """Bounded single-observation memory for post-NMS proposals."""

    def __init__(
        self,
        *,
        active_capacity: int,
        archive_capacity: int,
        ttl_keyframes: int,
        output_capacity: int,
    ) -> None:
        self.active_capacity = int(active_capacity)
        self.archive_capacity = int(archive_capacity)
        self.ttl_keyframes = int(ttl_keyframes)
        self.output_capacity = int(output_capacity)
        self.active: dict[tuple[int, int, int], ActiveCell] = {}
        self.archive: dict[tuple[int, int, int], Record] = {}
        self.peak_active = 0
        self.peak_archive = 0
        self.peak_combined = 0
        self.capacity_drops = 0
        self.observations = 0

    def update(
        self,
        ordinal: int,
        frame_id: int,
        ids: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        expired = [
            key
            for key, cell in self.active.items()
            if int(ordinal) - int(cell.last_seen_ordinal) > self.ttl_keyframes
        ]
        for key in expired:
            del self.active[key]
        order = np.argsort(-np.asarray(scores), kind="stable")[:150]
        self.observations += len(order)
        for index in order:
            row = Record(
                np.asarray(boxes[index], dtype=np.float64).reshape(8, 3).copy(),
                float(scores[index]),
                int(frame_id),
                int(ids[index]),
                int(ordinal),
            )
            key = voxel_key(row.box)
            cell = self.active.get(key)
            if cell is None:
                self.active[key] = ActiveCell(row, int(ordinal))
            else:
                if row.rank < cell.exemplar.rank:
                    cell.exemplar = row
                cell.last_seen_ordinal = int(ordinal)
            archived = self.archive.get(key)
            if archived is None or row.rank < archived.rank:
                self.archive[key] = row
        self.capacity_drops += enforce_capacity(
            self.active, self.active_capacity
        )
        self.capacity_drops += enforce_capacity(
            self.archive, self.archive_capacity
        )
        self.peak_active = max(self.peak_active, len(self.active))
        self.peak_archive = max(self.peak_archive, len(self.archive))
        self.peak_combined = max(
            self.peak_combined, len(self.active) + len(self.archive)
        )

    def rows(
        self, native_rows: list[tuple[int, np.ndarray, float]]
    ) -> list[tuple[int, np.ndarray, float]]:
        candidates = sorted(self.archive.values(), key=lambda row: row.rank)
        if candidates and native_rows:
            boxes = np.stack([row.box for row in candidates])
            native = np.stack([row[1] for row in native_rows])
            covered = np.any(
                _aabb_iou_matrix(boxes, native) >= 0.25, axis=1
            )
            candidates = [
                row for row, drop in zip(candidates, covered) if not drop
            ]
        selected: list[Record] = []
        for row in candidates:
            if selected:
                overlap = _aabb_iou_matrix(
                    row.box.reshape(1, 8, 3),
                    np.stack([value.box for value in selected]),
                )
                if np.any(overlap >= 0.50):
                    continue
            selected.append(row)
            if len(selected) >= self.output_capacity:
                break
        return [
            (0, row.box.copy(), float(_size_price(row.box)))
            for row in selected
        ]

    def counts(self) -> dict[str, int]:
        return {
            "active": len(self.active),
            "archive": len(self.archive),
            "combined": len(self.active) + len(self.archive),
        }

    def diagnostics(self) -> dict[str, int]:
        return {
            **self.counts(),
            "peak_active": self.peak_active,
            "peak_archive": self.peak_archive,
            "peak_combined": self.peak_combined,
            "active_capacity": self.active_capacity,
            "archive_capacity": self.archive_capacity,
            "ttl_keyframes": self.ttl_keyframes,
            "output_capacity": self.output_capacity,
            "capacity_drops": self.capacity_drops,
            "observations": self.observations,
        }


class AnchorScoreMemory:
    """Bounded strongest-exemplar voxel memory for pre-NMS anchors."""

    def __init__(self, *, capacity: int, output_capacity: int) -> None:
        self.capacity = int(capacity)
        self.output_capacity = int(output_capacity)
        self.state: dict[tuple[int, int, int], Record] = {}
        self.peak_states = 0
        self.capacity_drops = 0
        self.observations = 0

    def update(
        self,
        ordinal: int,
        frame_id: int,
        ids: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        order = np.argsort(-np.asarray(scores), kind="stable")[:300]
        self.observations += len(order)
        for index in order:
            row = Record(
                np.asarray(boxes[index], dtype=np.float64).reshape(8, 3).copy(),
                float(scores[index]),
                int(frame_id),
                int(ids[index]),
                int(ordinal),
            )
            key = voxel_key(row.box)
            current = self.state.get(key)
            if current is None or row.rank < current.rank:
                self.state[key] = row
        self.capacity_drops += enforce_capacity(self.state, self.capacity)
        self.peak_states = max(self.peak_states, len(self.state))

    def rows(self, scene: str) -> list[tuple[int, np.ndarray, float]]:
        selected = sorted(
            self.state.values(), key=lambda row: row.rank
        )[: self.output_capacity]
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

    def diagnostics(self) -> dict[str, int]:
        return {
            "states": len(self.state),
            "peak_states": self.peak_states,
            "capacity": self.capacity,
            "output_capacity": self.output_capacity,
            "capacity_drops": self.capacity_drops,
            "observations": self.observations,
        }


def write_prediction(
    path: Path, rows: list[tuple[int, np.ndarray, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = [[
        (int(label), np.array(box, copy=True), float(score))
        for label, box, score in rows
    ]]
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prediction(rows, transform, alignment):
    boxes = (
        np.stack([row[1] for row in rows])
        if rows else np.empty((0, 8, 3), dtype=np.float64)
    )
    scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    return transform(boxes, alignment), scores


def percentile(values: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / (
            "development/final_controls/experiments/"
            "mvsr_multiview_audit_20261012/config.yaml"
        ),
    )
    parser.add_argument(
        "--native-prefix-root",
        type=Path,
        default=ROOT / (
            "development/final_controls/experiments/"
            "mvsr_multiview_audit_20261012/native_prefix"
        ),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / (
            "development/final_controls/experiments/"
            "mvsr_unified_samepass_20261010/evidence"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-min-extent-m", type=float, default=0.30)
    parser.add_argument("--proposal-active-capacity", type=int, default=1024)
    parser.add_argument("--proposal-archive-capacity", type=int, default=1024)
    parser.add_argument("--proposal-ttl-keyframes", type=int, default=10)
    parser.add_argument("--proposal-output-capacity", type=int, default=12)
    parser.add_argument("--anchor-capacity", type=int, default=4096)
    parser.add_argument("--anchor-output-capacity", type=int, default=640)
    args = parser.parse_args()
    result_path = args.output / "results.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite completed output: {args.output}")

    scenes = [
        line.strip()
        for line in args.scene_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the unique official ScanNet-100 scene list")

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
    state_config["m2"] = {
        **state_config["m2"],
        "strength_variant": "no_visibility",
        "support_mode": "max",
    }

    gts, alignments, transform = scannet_inputs()
    systems = ("base", "fixed_memory", "full_recar")
    predictions = {
        fraction: {name: {} for name in systems}
        for fraction in FRACTIONS
    }
    checkpoints = {
        fraction: {
            name: {"native": 0, "proposal": 0, "anchor": 0, "total": 0}
            for name in systems
        }
        for fraction in FRACTIONS
    }
    per_scene = {}
    all_keyframe_outputs = {"proposal": [], "anchor": [], "total": []}
    replay_seconds = 0.0
    hashes = {
        str(args.scene_list.resolve()): sha256(args.scene_list),
        str(args.config.resolve()): sha256(args.config),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for scene_index, scene in enumerate(scenes, 1):
        native_path = args.native_prefix_root / f"{scene}.npz"
        evidence_path = args.evidence_root / f"{scene}.npz"
        if not native_path.is_file() or not evidence_path.is_file():
            raise FileNotFoundError(
                native_path if not native_path.is_file() else evidence_path
            )
        native_rows = load_native(native_path)
        evidence_rows = load_evidence(evidence_path)
        if [row[0] for row in native_rows] != [row[0] for row in evidence_rows]:
            raise RuntimeError(f"{scene}: native/evidence frame ids differ")

        full_state = build_reliability_candidate_map(scene, state_config)
        proposal = ProposalScoreMemory(
            active_capacity=args.proposal_active_capacity,
            archive_capacity=args.proposal_archive_capacity,
            ttl_keyframes=args.proposal_ttl_keyframes,
            output_capacity=args.proposal_output_capacity,
        )
        anchor = AnchorScoreMemory(
            capacity=args.anchor_capacity,
            output_capacity=args.anchor_output_capacity,
        )
        total_keyframes = len(native_rows)
        checkpoint_indices = {
            min(
                total_keyframes - 1,
                max(0, int(np.ceil(total_keyframes * fraction)) - 1),
            ): fraction
            for fraction in FRACTIONS
        }
        scene_trace = []
        started = time.perf_counter()
        for ordinal, ((frame_id, native), (_, evidence)) in enumerate(
            zip(native_rows, evidence_rows)
        ):
            full_state.update(frame_id, native, evidence)
            proposal.update(
                ordinal,
                frame_id,
                evidence.proposal_ids,
                evidence.proposal_corners,
                evidence.proposal_scores,
            )
            anchor.update(
                ordinal,
                frame_id,
                evidence.anchor_ids,
                evidence.anchor_corners,
                evidence.anchor_scores,
            )

            extents = np.ptp(native.corners, axis=1)
            valid = np.all(
                extents >= float(args.native_min_extent_m), axis=1
            )
            ids = native.ids[valid]
            corners = native.corners[valid]
            raw_scores = native.scores[valid]
            hcsr_scores, _ = full_state.m2.materialize(ids, raw_scores)
            native_base = [
                (0, np.array(box, copy=True), float(score))
                for box, score in zip(corners, raw_scores)
            ]
            native_hcsr = [
                (0, np.array(box, copy=True), float(score))
                for box, score in zip(corners, hcsr_scores)
            ]
            proposal_rows = proposal.rows(native_base)
            anchor_rows = anchor.rows(scene)
            memory_rows = native_hcsr + proposal_rows + anchor_rows

            output_counts = {
                "proposal": len(proposal_rows),
                "anchor": len(anchor_rows),
                "total": len(memory_rows),
            }
            for name, value in output_counts.items():
                all_keyframe_outputs[name].append(value)
            state_counts = proposal.counts()
            scene_trace.append({
                "ordinal": ordinal,
                "frame_id": int(frame_id),
                "native_output": len(native_hcsr),
                "proposal_output": len(proposal_rows),
                "anchor_output": len(anchor_rows),
                "total_output": len(memory_rows),
                "proposal_active_state": state_counts["active"],
                "proposal_archive_state": state_counts["archive"],
                "anchor_state": len(anchor.state),
            })

            if ordinal not in checkpoint_indices:
                continue
            fraction = checkpoint_indices[ordinal]
            tag = f"p{int(round(100 * fraction))}"
            full_snapshot = full_state.materialize_terminal(
                frame_id, ids, corners, raw_scores
            )
            full_rows = [
                (0, np.array(box, copy=True), float(score))
                for box, score in zip(full_snapshot.boxes, full_snapshot.scores)
            ]
            rows_by_system = {
                "base": native_base,
                "fixed_memory": memory_rows,
                "full_recar": full_rows,
            }
            full_source_counts = {
                source: full_snapshot.sources.count(source)
                for source in ("native", "m1p", "m1a")
            }
            branch_by_system = {
                "base": {
                    "native": len(native_base), "proposal": 0, "anchor": 0,
                },
                "fixed_memory": {
                    "native": len(native_hcsr),
                    "proposal": len(proposal_rows),
                    "anchor": len(anchor_rows),
                },
                "full_recar": {
                    "native": full_source_counts["native"],
                    "proposal": full_source_counts["m1p"],
                    "anchor": full_source_counts["m1a"],
                },
            }
            for name, rows in rows_by_system.items():
                write_prediction(
                    args.output / "predictions" / f"{name}_{tag}"
                    / f"{scene}_boxes.pkl",
                    rows,
                )
                predictions[fraction][name][scene] = prediction(
                    rows, transform, alignments[scene]
                )
                branch = branch_by_system[name]
                for source in ("native", "proposal", "anchor"):
                    checkpoints[fraction][name][source] += branch[source]
                checkpoints[fraction][name]["total"] += len(rows)

        replay_seconds += time.perf_counter() - started
        per_scene[scene] = {
            "keyframes": total_keyframes,
            "proposal_memory": proposal.diagnostics(),
            "anchor_memory": anchor.diagnostics(),
            "keyframe_trace": scene_trace,
        }
        if scene_index % 10 == 0:
            print(f"replayed {scene_index}/100", flush=True)

    exact_ap = {
        str(fraction): {
            name: {
                str(threshold): class_agnostic_ap(
                    predictions[fraction][name], gts, threshold
                )["ap"]
                for threshold in THRESHOLDS
            }
            for name in systems
        }
        for fraction in FRACTIONS
    }
    global_state_peaks = {
        "proposal_active": max(
            row["proposal_memory"]["peak_active"]
            for row in per_scene.values()
        ),
        "proposal_archive": max(
            row["proposal_memory"]["peak_archive"]
            for row in per_scene.values()
        ),
        "proposal_combined": max(
            row["proposal_memory"]["peak_combined"]
            for row in per_scene.values()
        ),
        "anchor": max(
            row["anchor_memory"]["peak_states"]
            for row in per_scene.values()
        ),
        "total_simple_state": max(
            row["proposal_memory"]["peak_combined"]
            + row["anchor_memory"]["peak_states"]
            for row in per_scene.values()
        ),
    }
    output_summary = {
        name: {
            "observations": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": percentile(values, 95),
            "max": int(max(values)),
        }
        for name, values in all_keyframe_outputs.items()
    }
    protocol = {
        "strict_causal": True,
        "future_observations_used": False,
        "ground_truth_used": False,
        "readout_after_every_keyframe": True,
        "checkpoint_ap_fractions": list(FRACTIONS),
        "same_frozen_auxiliary_stream_as_recar": True,
        "same_hcsr_state_and_scores_as_recar": True,
        "hierarchical_confirmation_used_by_simple_memory": False,
        "native_min_extent_m": float(args.native_min_extent_m),
        "capacities_fixed_before_evaluation": {
            "proposal_active": args.proposal_active_capacity,
            "proposal_archive": args.proposal_archive_capacity,
            "anchor": args.anchor_capacity,
        },
        "output_caps_fixed_before_evaluation": {
            "proposal_per_scene_readout": args.proposal_output_capacity,
            "anchor_per_scene_readout": args.anchor_output_capacity,
        },
        "proposal_memory": (
            "strongest raw-score exemplar per 0.30 m world voxel; active "
            "last-seen time is refreshed by every observation; bounded archive"
        ),
        "anchor_memory": (
            "strongest raw-score exemplar per 0.30 m world voxel; bounded state"
        ),
        "candidate_selection_uses_future_or_target_recar_counts": False,
        "actual_output_counts_equal_caps_required": False,
    }
    result = {
        "schema": "boxfusion.scannet100.fixed_capacity_per_keyframe_memory.v1",
        "scene_count": len(scenes),
        "keyframe_count": len(all_keyframe_outputs["total"]),
        "protocol": protocol,
        "checkpoints": {str(key): value for key, value in checkpoints.items()},
        "exact_ap": exact_ap,
        "global_state_peaks": global_state_peaks,
        "all_keyframe_output_summary": output_summary,
        "cpu_replay_seconds": replay_seconds,
        "per_scene": per_scene,
        "input_sha256": hashes,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(
        json.dumps({
            "schema": result["schema"],
            "scene_count": result["scene_count"],
            "keyframe_count": result["keyframe_count"],
            "protocol": protocol,
            "checkpoints": result["checkpoints"],
            "global_state_peaks": global_state_peaks,
            "all_keyframe_output_summary": output_summary,
            "cpu_replay_seconds": replay_seconds,
            "input_sha256": hashes,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ScanNet-100 fixed-capacity per-keyframe causal memory", "",
        "The simple memory and ReCaR-3D consume the same frozen auxiliary "
        "stream and share the same final three-factor HCSR. The simple memory "
        "uses no multi-keyframe confirmation. It is materialized after every "
        "keyframe; capacities and 12/640 output caps are fixed before evaluation.",
        "",
        "| Prefix | System | Native | Proposal | Anchor | Total | AP15 | AP25 | AP50 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fraction in FRACTIONS:
        for name in systems:
            count = checkpoints[fraction][name]
            metric = exact_ap[str(fraction)][name]
            lines.append(
                f"| {int(100 * fraction)}% | {name} | {count['native']:,} | "
                f"{count['proposal']:,} | {count['anchor']:,} | "
                f"{count['total']:,} | {metric['0.15']:.4f} | "
                f"{metric['0.25']:.4f} | {metric['0.5']:.4f} |"
            )
    lines += [
        "",
        "## Actual peak state occupancy", "",
        "| State | Configured capacity | Observed peak |",
        "|---|---:|---:|",
        f"| Proposal active | {args.proposal_active_capacity:,} | "
        f"{global_state_peaks['proposal_active']:,} |",
        f"| Proposal archive | {args.proposal_archive_capacity:,} | "
        f"{global_state_peaks['proposal_archive']:,} |",
        f"| Anchor voxel memory | {args.anchor_capacity:,} | "
        f"{global_state_peaks['anchor']:,} |",
        f"| Combined simple-memory state | "
        f"{args.proposal_active_capacity + args.proposal_archive_capacity + args.anchor_capacity:,} | "
        f"{global_state_peaks['total_simple_state']:,} |",
        "",
        "## Per-keyframe materialized output counts", "",
        "| Branch | Mean | Median | P95 | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("proposal", "anchor", "total"):
        values = output_summary[name]
        lines.append(
            f"| {name} | {values['mean']:.2f} | {values['median']:.2f} | "
            f"{values['p95']:.2f} | {values['max']:,} |"
        )
    lines += [
        "",
        f"CPU replay time: {replay_seconds:.2f} s. No detector inference, "
        "future observation, ground truth, or ReCaR output-count matching was "
        "used by the simple-memory state or selection.",
    ]
    report = "\n".join(lines) + "\n"
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
