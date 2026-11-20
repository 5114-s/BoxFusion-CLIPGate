#!/usr/bin/env python3
"""Evaluate strict-causal CALR controls from frozen lifted-anchor caches.

The script never runs an image model.  It replays each scene's ordered Top-M
lifted anchors through bounded online states, combines each resulting anchor
tail with the frozen final PLR+MVSR prefix, and evaluates all arms together.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "development/final_controls"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FINAL))

from boxfusion.m1_anchor_online import (  # noqa: E402
    OnlineAnchorRecovery,
    deterministic_tail_score,
)
from tools.audit_final_ledger import scannet_inputs  # noqa: E402
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap  # noqa: E402


THRESHOLDS = (0.15, 0.25, 0.50)
TOPK = (50, 100, 200)
FP_PER_100_GT = (1, 5, 10, 20)
ARMS = ("streaming_topk", "nA1", "nA2", "nA3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (tuple, list)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction payload: {path}")
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in payload[0]
    ]


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([[(int(a), np.array(b, copy=True), float(c)) for a, b, c in rows]], handle)


def same_rows(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
        for a, b in zip(left, right)
    )


def load_evidence(path: Path):
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def replay_voxel(scene: str, values, min_views: int):
    state = OnlineAnchorRecovery(
        scene, voxel_m=0.30, min_views=min_views,
        max_active=4096, max_births=640,
    )
    offsets = np.r_[0, np.cumsum(values["anchor_lengths"], dtype=np.int64)]
    for ordinal, frame_id in enumerate(values["frame_ids"]):
        low, high = int(offsets[ordinal]), int(offsets[ordinal + 1])
        state.update(
            ordinal, int(frame_id), values["anchor_ids"][low:high],
            values["anchor_corners"][low:high], values["anchor_scores"][low:high],
        )
    rows = [
        (0, np.asarray(row["box"], dtype=np.float64), float(row["score"]))
        for row in state.rows()
    ]
    return rows, state.diagnostics()


def replay_streaming_topk(scene: str, values):
    """Bounded causal strongest-exemplar memory without view confirmation.

    The 0.3 m voxel key is retained only as the same spatial deduplication unit
    used by CALR.  A voxel is eligible after one observation; its strongest
    exemplar is updated online.  At most 4096 voxel states and 640 outputs are
    retained, both by deterministic raw-score rank.
    """
    state = {}
    offsets = np.r_[0, np.cumsum(values["anchor_lengths"], dtype=np.int64)]
    dropped = 0
    peak = 0
    for ordinal, frame_id in enumerate(values["frame_ids"]):
        low, high = int(offsets[ordinal]), int(offsets[ordinal + 1])
        for index in range(low, high):
            box = np.asarray(values["anchor_corners"][index], dtype=np.float64)
            raw = float(values["anchor_scores"][index])
            anchor_id = int(values["anchor_ids"][index])
            key = tuple(np.floor(box.mean(0) / 0.30).astype(np.int64).tolist())
            rank = (-raw, int(frame_id), anchor_id, key)
            current = state.get(key)
            if current is None or rank < current[0]:
                state[key] = (rank, box.copy(), raw, int(frame_id), anchor_id)
        if len(state) > 4096:
            ordered = sorted(state.items(), key=lambda item: item[1][0])
            keep = dict(ordered[:4096])
            dropped += len(state) - len(keep)
            state = keep
        peak = max(peak, len(state))
    selected = sorted(state.values(), key=lambda value: value[0])[:640]
    rows = [
        (
            0, value[1],
            deterministic_tail_score(value[2], scene, value[3], value[4]),
        )
        for value in selected
    ]
    return rows, {
        "frames_seen": int(len(values["frame_ids"])),
        "states": len(state), "peak_states": peak,
        "max_states": 4096, "outputs": len(rows), "max_outputs": 640,
        "capacity_drops": dropped, "minimum_distinct_frames": 1,
    }


def prediction(rows, transform, alignment):
    boxes = (
        np.stack([row[1] for row in rows])
        if rows else np.empty((0, 8, 3), dtype=np.float64)
    )
    return transform(boxes, alignment), np.asarray([row[2] for row in rows])


def quality(boxes, scores, prefix_count, gt, threshold):
    overlap = (
        aabb_iou(boxes, gt)
        if len(boxes) and len(gt) else np.zeros((len(boxes), len(gt)))
    )
    matched = np.zeros(len(gt), dtype=bool)
    total = {"tp": 0, "fp": 0, "duplicate": 0}
    added = {"tp": 0, "fp": 0, "duplicate": 0}
    for row in np.argsort(-scores, kind="stable"):
        values = overlap[row]
        target = int(values.argmax()) if len(values) else -1
        passes = target >= 0 and float(values[target]) > threshold
        is_tp = bool(passes and not matched[target])
        is_duplicate = bool(passes and matched[target])
        if is_tp:
            matched[target] = True
        bucket = total
        bucket["tp" if is_tp else "fp"] += 1
        bucket["duplicate"] += int(is_duplicate)
        if row >= prefix_count:
            bucket = added
            bucket["tp" if is_tp else "fp"] += 1
            bucket["duplicate"] += int(is_duplicate)
    return total, added


def truncate_predictions(predictions, k: int):
    output = {}
    for scene, (boxes, scores) in predictions.items():
        keep = np.argsort(-scores, kind="stable")[:k]
        output[scene] = boxes[keep], scores[keep]
    return output


def ranked_flags(predictions, gts, threshold):
    scores, locations, matrices = [], [], {}
    matched = {scene: np.zeros(len(gt), dtype=bool) for scene, gt in gts.items()}
    for scene, (boxes, confidence) in predictions.items():
        matrices[scene] = (
            aabb_iou(boxes, gts[scene])
            if len(boxes) and len(gts[scene])
            else np.zeros((len(boxes), len(gts[scene])))
        )
        scores.extend(confidence.tolist())
        locations.extend((scene, row) for row in range(len(boxes)))
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    flags = np.zeros(len(order), dtype=np.int64)
    for rank, source in enumerate(order):
        scene, row = locations[int(source)]
        values = matrices[scene][row]
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not matched[scene][target]:
                matched[scene][target] = True
                flags[rank] = 1
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp = np.cumsum(flags)
    fp = np.cumsum(1 - flags)
    return sorted_scores, ends, tp, fp


def pr_curve(predictions, gts, threshold):
    scores, _, tp, fp = ranked_flags(predictions, gts, threshold)
    del scores
    total_gt = sum(len(value) for value in gts.values())
    return tp / total_gt, tp / np.maximum(tp + fp, 1)


def render_pr(curves, output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    colors = {
        "streaming_topk": "#757575", "nA1": "#59A14F",
        "nA2": "#F28E2B", "nA3": "#4E79A7",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    for ax, threshold in zip(axes, THRESHOLDS):
        for arm in ARMS:
            recall, precision = curves[arm][str(threshold)]
            ax.plot(recall, precision, lw=1.25, color=colors[arm], label=arm)
        ax.set_title(f"3D AABB IoU = {threshold:.2f}", fontsize=8)
        ax.set_xlabel("Recall")
        ax.grid(color="#E5E7EB", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Precision")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.2)
    stem = output / "calr_causal_controls_pr"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    experiment = FINAL / "experiments/mvsr_design_20261005"
    factorial = experiment / "factorial_no_visibility"
    evidence_root = FINAL / "evidence/scannet_recar3d_key_controls"
    gts, alignments, transform = scannet_inputs()
    scenes = list(gts)
    total_gt = sum(len(value) for value in gts.values())
    predictions = {arm: {} for arm in ARMS}
    counts = {arm: {"predictions": 0, "added": 0} for arm in ARMS}
    per_scene = {}
    hashes = {}
    quality_totals = {
        arm: {
            str(t): {
                "all": {"tp": 0, "fp": 0, "duplicate": 0},
                "added": {"tp": 0, "fp": 0, "duplicate": 0},
            } for t in THRESHOLDS
        } for arm in ARMS
    }

    for index, scene in enumerate(scenes, 1):
        prefix_path = factorial / "plr_mvsr" / f"{scene}_boxes.pkl"
        full_path = factorial / "full" / f"{scene}_boxes.pkl"
        evidence_path = evidence_root / f"{scene}.npz"
        prefix, full = read_rows(prefix_path), read_rows(full_path)
        if not same_rows(prefix, full[:len(prefix)]):
            raise RuntimeError(f"frozen prefix mismatch: {scene}")
        frozen_nA3 = full[len(prefix):]
        values = load_evidence(evidence_path)
        outputs, diagnostics = {}, {}
        outputs["streaming_topk"], diagnostics["streaming_topk"] = (
            replay_streaming_topk(scene, values)
        )
        for views in (1, 2, 3):
            outputs[f"nA{views}"], diagnostics[f"nA{views}"] = replay_voxel(
                scene, values, views
            )
        if not same_rows(outputs["nA3"], frozen_nA3):
            raise RuntimeError(f"nA=3 replay does not reproduce frozen CALR-v1: {scene}")

        per_scene[scene] = {
            arm: {"added": len(rows), "diagnostics": diagnostics[arm]}
            for arm, rows in outputs.items()
        }
        for arm, added in outputs.items():
            rows = prefix + added
            write_rows(args.output / "predictions" / arm / f"{scene}_boxes.pkl", rows)
            boxes, scores = prediction(rows, transform, alignments[scene])
            predictions[arm][scene] = boxes, scores
            counts[arm]["predictions"] += len(rows)
            counts[arm]["added"] += len(added)
            for threshold in THRESHOLDS:
                all_q, added_q = quality(boxes, scores, len(prefix), gts[scene], threshold)
                for group, source in (("all", all_q), ("added", added_q)):
                    for key in source:
                        quality_totals[arm][str(threshold)][group][key] += source[key]
        for path in (prefix_path, full_path, evidence_path):
            hashes[str(path.resolve())] = sha256(path)
        if index % 10 == 0:
            print(f"replayed {index}/100", flush=True)

    metrics, topk, fixed_fp, curves = {}, {}, {}, {}
    for arm in ARMS:
        metrics[arm], topk[arm], fixed_fp[arm], curves[arm] = {}, {}, {}, {}
        for threshold in THRESHOLDS:
            key = str(threshold)
            metrics[arm][key] = class_agnostic_ap(
                predictions[arm], gts, threshold
            )["ap"]
            curves[arm][key] = tuple(
                value.tolist() for value in pr_curve(predictions[arm], gts, threshold)
            )
            topk[arm][key] = {}
            for k in TOPK:
                result = class_agnostic_ap(
                    truncate_predictions(predictions[arm], k), gts, threshold
                )
                topk[arm][key][str(k)] = {
                    "ap": result["ap"],
                    "recall": 100.0 * result["tp"] / total_gt,
                    "tp": int(result["tp"]), "fp": int(result["fp"]),
                }
            scores, ends, tp, fp = ranked_flags(predictions[arm], gts, threshold)
            fixed_fp[arm][key] = {}
            for budget in FP_PER_100_GT:
                allowed = int(np.floor(budget * total_gt / 100))
                valid = ends[fp[ends] <= allowed]
                last = int(valid[-1]) if len(valid) else None
                fixed_fp[arm][key][str(budget)] = {
                    "recall": 100.0 * int(tp[last]) / total_gt if last is not None else 0.0,
                    "tp": int(tp[last]) if last is not None else 0,
                    "fp": int(fp[last]) if last is not None else 0,
                    "fp_budget": allowed,
                    "confidence_cutoff": float(scores[last]) if last is not None else None,
                }

    result = {
        "schema": "boxfusion.scannet100.calr_causal_controls.v1",
        "scene_count": len(scenes), "ground_truth_boxes": total_gt,
        "strictly_causal_replay": True, "models_run": False,
        "shared_prefix": "frozen final PLR + three-factor MVSR Maximum",
        "parameters": {
            "voxel_m": 0.30, "anchors_per_keyframe": 300,
            "max_states": 4096, "max_outputs": 640,
            "ttl": None,
            "ttl_note": "Final CALR-v1 has no expiration rule; all controls match it.",
            "score_range": [0.040001, 0.049999],
            "streaming_topk": (
                "one strongest exemplar per 0.3 m voxel; eligible after one "
                "observation; online top-4096 state and top-640 output"
            ),
        },
        "counts": counts, "metrics": metrics, "quality": quality_totals,
        "topk_per_scene": topk, "fixed_fp_per_100_gt": fixed_fp,
        "per_scene": per_scene, "input_sha256": hashes,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "arm", "added_boxes", "total_predictions", "AP15", "AP25", "AP50",
            "added_TP15", "added_FP15", "added_duplicates15",
            "added_precision15", "added_recall15",
        ])
        for arm in ARMS:
            q = quality_totals[arm]["0.15"]["added"]
            writer.writerow([
                arm, counts[arm]["added"], counts[arm]["predictions"],
                metrics[arm]["0.15"], metrics[arm]["0.25"], metrics[arm]["0.5"],
                q["tp"], q["fp"], q["duplicate"],
                q["tp"] / max(q["tp"] + q["fp"], 1), q["tp"] / total_gt,
            ])
    with (args.output / "topk_recall_ap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "iou", "top_k_per_scene", "ap", "recall", "tp", "fp"])
        for arm in ARMS:
            for threshold in THRESHOLDS:
                for k in TOPK:
                    row = topk[arm][str(threshold)][str(k)]
                    writer.writerow([arm, threshold, k, row["ap"], row["recall"], row["tp"], row["fp"]])
    with (args.output / "fixed_fp_recall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "iou", "fp_per_100_gt", "fp_budget", "tp", "fp", "recall", "confidence_cutoff"])
        for arm in ARMS:
            for threshold in THRESHOLDS:
                for budget in FP_PER_100_GT:
                    row = fixed_fp[arm][str(threshold)][str(budget)]
                    writer.writerow([arm, threshold, budget, row["fp_budget"], row["tp"], row["fp"], row["recall"], row["confidence_cutoff"]])
    for arm in ARMS:
        for threshold in THRESHOLDS:
            recall, precision = curves[arm][str(threshold)]
            with (args.output / f"pr_{arm}_iou{int(threshold*100):02d}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["recall", "precision"])
                writer.writerows(zip(recall, precision))
    render_pr(curves, args.output)

    lines = [
        "# ScanNet-100 CALR strict-causal controls", "",
        "All arms replay the same ordered lifted-anchor caches and share the frozen final PLR+MVSR prefix. No visual model was run.", "",
        "| Arm | Added boxes | AP15 | AP25 | AP50 | Added TP/FP/dup. @0.15 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        q = quality_totals[arm]["0.15"]["added"]
        lines.append(
            f"| {arm} | {counts[arm]['added']:,} | {metrics[arm]['0.15']:.4f} | "
            f"{metrics[arm]['0.25']:.4f} | {metrics[arm]['0.5']:.4f} | "
            f"{q['tp']:,}/{q['fp']:,}/{q['duplicate']:,} |"
        )
    lines += [
        "", "`nA3` is asserted byte-for-byte equal to the frozen final CALR-v1 suffix in every scene.",
        "The final CALR-v1 implementation has no TTL and no native-map or branch-internal NMS; the controls preserve that actual behavior rather than adding an unreported rule.",
        "Fixed Top-50/100/200 and fixed-FP results are in the CSV files and `results.json`.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

