#!/usr/bin/env python3
"""Build the final ScanNet-100 recovery-budget and online-delay evidence.

This is an offline audit of the frozen final no-visibility run.  It never
modifies prediction roots and does not rerun CuTR, WeDetect-Uni, or Boxer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from boxfusion.m1_anchor_online import OnlineAnchorRecovery
from boxfusion.online_candidate_map import OnlineProposalRecovery
from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.50)
FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_rows(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = payload[0]
    return [
        (int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score))
        for label, box, score in rows
    ]


def prediction(rows: list[tuple[int, np.ndarray, float]], transform, alignment):
    boxes = (
        np.stack([row[1] for row in rows])
        if rows else np.empty((0, 8, 3), dtype=np.float64)
    )
    scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    return transform(boxes, alignment), scores


def select_fraction(rows, fraction: float):
    count = int(math.ceil(float(fraction) * len(rows) - 1e-12))
    order = sorted(range(len(rows)), key=lambda i: (-rows[i][2], i))
    return [rows[i] for i in order[:count]]


def quality_counts(boxes, scores, sources, gt, threshold):
    overlaps = (
        aabb_iou(boxes, gt)
        if len(boxes) and len(gt)
        else np.zeros((len(boxes), len(gt)), dtype=np.float64)
    )
    matched = np.zeros(len(gt), dtype=bool)
    result = {
        "all": {"tp": 0, "fp": 0, "duplicate": 0},
        "calr": {"tp": 0, "fp": 0, "duplicate": 0},
    }
    for row in np.argsort(-scores, kind="stable"):
        group = "calr" if sources[row] == "calr" else None
        values = overlaps[row]
        target = int(values.argmax()) if len(values) else -1
        passes = target >= 0 and float(values[target]) > threshold
        if passes and not matched[target]:
            matched[target] = True
            outcome = "tp"
        else:
            outcome = "fp"
        duplicate = bool(passes and matched[target] and outcome == "fp")
        result["all"][outcome] += 1
        if duplicate:
            result["all"]["duplicate"] += 1
        if group:
            result[group][outcome] += 1
            if duplicate:
                result[group]["duplicate"] += 1
    return result


def pr_curve(predictions, gts, threshold):
    pooled = []
    total_gt = sum(len(value) for value in gts.values())
    for scene, (boxes, scores) in predictions.items():
        gt = gts[scene]
        overlap = (
            aabb_iou(boxes, gt)
            if len(boxes) and len(gt)
            else np.zeros((len(boxes), len(gt)), dtype=np.float64)
        )
        used = np.zeros(len(gt), dtype=bool)
        for row in np.argsort(-scores, kind="stable"):
            target = int(overlap[row].argmax()) if len(gt) else -1
            tp = target >= 0 and overlap[row, target] > threshold and not used[target]
            if tp:
                used[target] = True
            pooled.append((float(scores[row]), int(tp)))
    pooled.sort(key=lambda row: -row[0])
    labels = np.asarray([row[1] for row in pooled], dtype=np.float64)
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    return tp / (total_gt + 1e-6), tp / np.maximum(tp + fp, 1.0)


def offsets(lengths):
    return np.r_[0, np.cumsum(np.asarray(lengths, dtype=np.int64))]


def update_first(values, frame_ordinal, boxes, gt, threshold):
    if not len(boxes) or not len(gt):
        return
    overlap = aabb_iou(boxes, gt)
    hit = overlap.max(axis=0) > threshold
    unset = (values < 0) & hit
    values[unset] = frame_ordinal


def replay_delay(scene, cache_path, native_rows, gt, transform, alignment, threshold):
    with np.load(cache_path, allow_pickle=False) as z:
        frame_ids = np.asarray(z["frame_ids"], dtype=np.int64)
        proposal_lengths = np.asarray(z["proposal_lengths"], dtype=np.int64)
        proposal_ids = np.asarray(z["proposal_ids"], dtype=np.int64)
        proposal_corners = np.asarray(z["proposal_corners"], dtype=np.float64)
        proposal_scores = np.asarray(z["proposal_scores"], dtype=np.float64)
        anchor_lengths = np.asarray(z["anchor_lengths"], dtype=np.int64)
        anchor_ids = np.asarray(z["anchor_ids"], dtype=np.int64)
        anchor_corners = np.asarray(z["anchor_corners"], dtype=np.float64)
        anchor_scores = np.asarray(z["anchor_scores"], dtype=np.float64)
    poff, aoff = offsets(proposal_lengths), offsets(anchor_lengths)
    plr = OnlineProposalRecovery(
        use_children=False, proposal_min_views=3, child_min_views=2,
        ttl_keyframes=10, match_iou=0.10, match_center_m=0.50,
        native_dedup_iou=0.25, self_nms_iou=0.50, max_tracks=1024,
        max_births=12, max_observations_per_track=12,
        max_observations_per_frame=192,
    )
    calr = OnlineAnchorRecovery(
        scene, voxel_m=0.3, min_views=3, max_active=4096, max_births=640
    )
    native_boxes = prediction(native_rows, transform, alignment)[0]
    native_covered = (
        aabb_iou(native_boxes, gt).max(axis=0) > threshold
        if len(native_boxes) and len(gt) else np.zeros(len(gt), dtype=bool)
    )
    first_evidence = {
        "plr": np.full(len(gt), -1, dtype=np.int64),
        "calr": np.full(len(gt), -1, dtype=np.int64),
    }
    first_recovery = {
        "plr": np.full(len(gt), -1, dtype=np.int64),
        "calr": np.full(len(gt), -1, dtype=np.int64),
    }
    empty_native = np.empty((0, 8, 3), dtype=np.float64)
    for ordinal, frame_id in enumerate(frame_ids):
        pl, ph = int(poff[ordinal]), int(poff[ordinal + 1])
        al, ah = int(aoff[ordinal]), int(aoff[ordinal + 1])
        pbox = proposal_corners[pl:ph]
        abox = anchor_corners[al:ah]
        update_first(first_evidence["plr"], ordinal, transform(pbox, alignment), gt, threshold)
        update_first(first_evidence["calr"], ordinal, transform(abox, alignment), gt, threshold)
        plr.update(
            ordinal, int(frame_id), proposal_ids=proposal_ids[pl:ph],
            proposal_corners=pbox, proposal_scores=proposal_scores[pl:ph],
            native_corners=empty_native,
        )
        calr.update(
            ordinal, int(frame_id), anchor_ids[al:ah], abox, anchor_scores[al:ah]
        )
        p_rows = plr.rows()
        a_rows = calr.rows()
        p_current = (
            np.stack([row["box"] for row in p_rows])
            if p_rows else np.empty((0, 8, 3), dtype=np.float64)
        )
        a_current = (
            np.stack([row["box"] for row in a_rows])
            if a_rows else np.empty((0, 8, 3), dtype=np.float64)
        )
        update_first(first_recovery["plr"], ordinal, transform(p_current, alignment), gt, threshold)
        update_first(first_recovery["calr"], ordinal, transform(a_current, alignment), gt, threshold)
    rows = []
    for branch in ("plr", "calr"):
        eligible = (~native_covered) & (first_evidence[branch] >= 0)
        recovered = eligible & (first_recovery[branch] >= 0)
        for target in np.flatnonzero(recovered):
            rows.append({
                "scene": scene,
                "gt_index": int(target),
                "branch": branch,
                "iou_threshold": threshold,
                "first_evidence_ordinal": int(first_evidence[branch][target]),
                "first_recovery_ordinal": int(first_recovery[branch][target]),
                "delay_keyframes": int(first_recovery[branch][target] - first_evidence[branch][target]),
                "first_evidence_frame": int(frame_ids[first_evidence[branch][target]]),
                "first_recovery_frame": int(frame_ids[first_recovery[branch][target]]),
            })
        rows.append({
            "scene": scene, "branch": branch, "iou_threshold": threshold,
            "summary": True, "eligible_gt": int(eligible.sum()),
            "recovered_gt": int(recovered.sum()),
        })
    return rows


def summarize_delay(rows):
    result = {}
    for branch in ("plr", "calr"):
        observations = [r for r in rows if r.get("branch") == branch and not r.get("summary")]
        summaries = [r for r in rows if r.get("branch") == branch and r.get("summary")]
        values = np.asarray([r["delay_keyframes"] for r in observations], dtype=np.float64)
        eligible = sum(r["eligible_gt"] for r in summaries)
        recovered = sum(r["recovered_gt"] for r in summaries)
        result[branch] = {
            "eligible_gt": eligible,
            "recovered_gt": recovered,
            "recovery_fraction": recovered / eligible if eligible else 0.0,
            "median_keyframes": float(np.median(values)) if len(values) else None,
            "p90_keyframes": float(np.percentile(values, 90)) if len(values) else None,
            "p95_keyframes": float(np.percentile(values, 95)) if len(values) else None,
            "mean_keyframes": float(values.mean()) if len(values) else None,
        }
    return result


def render(budget_rows, delay_rows, output):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    ax = axes[0]
    x = np.asarray([r["budget_percent"] for r in budget_rows])
    colors = ("#2774AE", "#E67E22", "#6F4E9C")
    for threshold, color in zip(THRESHOLDS, colors):
        ax.plot(x, [r[f"ap{int(threshold*100)}"] for r in budget_rows],
                marker="o", lw=1.5, ms=4, color=color, label=f"AP{int(threshold*100)}")
    ax.set_xlabel("Retained CALR terminal budget (%)")
    ax.set_ylabel("Average precision")
    ax.set_title(
        "a  CALR gains saturate\nwith terminal output budget",
        loc="left", fontweight="bold", fontsize=8.5,
    )
    ax.grid(axis="y", color="#E4E7EA", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    quality_axis = ax.twinx()
    yields = []
    for row in budget_rows:
        q = row["quality"]["0.15"]["calr"]
        yields.append(100.0 * q["tp"] / (q["tp"] + q["fp"]) if q["tp"] + q["fp"] else np.nan)
    quality_axis.plot(
        x, yields, color="#5F6368", marker="s", ms=3.4,
        lw=1.0, ls="--", label="TPs / 100 CALR records @0.15",
    )
    quality_axis.set_ylabel(
        "TPs per 100 added\nCALR records", color="#5F6368"
    )
    quality_axis.tick_params(axis="y", colors="#5F6368", labelsize=7)
    quality_axis.spines["top"].set_visible(False)
    quality_axis.set_ylim(0, max(2.2, np.nanmax(yields) * 1.2))
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = quality_axis.get_legend_handles_labels()
    ax.legend(
        handles_left + handles_right, labels_left + labels_right,
        frameon=False, ncol=2, fontsize=6.5, loc="center right",
    )
    ax.text(
        0.02, 0.035, "0% = PLR + MVSR; 100% = full ReCaR-3D",
        transform=ax.transAxes, fontsize=6.4, color="#454B52",
    )

    ax = axes[1]
    for branch, color in (("plr", "#E87522"), ("calr", "#C58A00")):
        values = np.sort(np.asarray([
            r["delay_keyframes"] for r in delay_rows
            if r.get("branch") == branch and not r.get("summary")
        ], dtype=np.float64))
        summaries = [
            r for r in delay_rows
            if r.get("branch") == branch and r.get("summary")
        ]
        eligible = sum(int(r["eligible_gt"]) for r in summaries)
        recovered = sum(int(r["recovered_gt"]) for r in summaries)
        if len(values):
            if recovered != len(values) or eligible < recovered:
                raise RuntimeError(
                    f"invalid {branch} delay population: "
                    f"eligible={eligible}, recovered={recovered}, rows={len(values)}"
                )
            y = np.arange(1, len(values) + 1) / eligible
            ax.step(values, y, where="post", lw=1.7, color=color,
                    label=f"{branch.upper()}: {recovered}/{eligible} confirmed")
    ax.set_xlabel("Keyframes from first source evidence to confirmation")
    ax.set_ylabel("Fraction of evidence-eligible native\nmisses confirmed")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "b  Causal branch-confirmation\nbehavior",
        loc="left", fontweight="bold", fontsize=8.5,
    )
    ax.grid(axis="both", color="#E4E7EA", lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.0, w_pad=2.0)
    stem = output / "online_recovery_behavior"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-iou", type=float, default=0.15)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    experiment = FINAL / "experiments/mvsr_design_20261005"
    components = experiment / "components"
    evidence = FINAL / "evidence/scannet_recar3d_key_controls"
    suffix = "_boxes.pkl"
    available_scenes = {path.name[:-len(suffix)] for path in
                        (components / "native_factor_no_visibility").glob("*_boxes.pkl")}
    if len(available_scenes) != 100:
        raise RuntimeError(f"expected 100 scenes, found {len(available_scenes)}")
    gts, alignments, transform = scannet_inputs()
    # Preserve the frozen protocol order. Several tail scores tie across
    # scenes, so lexicographically reordering scenes can change AP by a few
    # hundredths even though predictions are identical.
    scenes = list(gts)
    if available_scenes != set(gts):
        raise RuntimeError("final components and frozen ScanNet GT scene sets differ")

    per_fraction = {fraction: {} for fraction in FRACTIONS}
    quality = {fraction: {threshold: {"all": {"tp": 0, "fp": 0, "duplicate": 0},
                                      "calr": {"tp": 0, "fp": 0, "duplicate": 0}}
                         for threshold in THRESHOLDS} for fraction in FRACTIONS}
    counts = {fraction: {"calr": 0, "predictions": 0} for fraction in FRACTIONS}
    hashes = {}
    native_by_scene = {}
    for scene in scenes:
        paths = {
            "native": components / "native_factor_no_visibility" / f"{scene}_boxes.pkl",
            "prefix": experiment / "factorial_no_visibility/plr_mvsr" / f"{scene}_boxes.pkl",
            "full": experiment / "factorial_no_visibility/full" / f"{scene}_boxes.pkl",
            "evidence": evidence / f"{scene}.npz",
        }
        native = read_rows(paths["native"])
        prefix = read_rows(paths["prefix"])
        full = read_rows(paths["full"])
        if len(full) < len(prefix) or any(
            left[0] != right[0]
            or not np.array_equal(left[1], right[1])
            or left[2] != right[2]
            for left, right in zip(prefix, full[:len(prefix)])
        ):
            raise RuntimeError(f"final PLR+MVSR prefix is not a strict prefix of full: {scene}")
        # The final selected CALR-v1 is the exact suffix of the frozen full
        # prediction.  Do not use components/births_calr_v2, which is the
        # archived negative CALR-v2 control despite the legacy component name.
        calr = full[len(prefix):]
        native_by_scene[scene] = native
        for path in paths.values():
            hashes[str(path.resolve())] = sha256(path)
        for fraction in FRACTIONS:
            chosen = select_fraction(calr, fraction)
            rows = prefix + chosen
            boxes, scores = prediction(rows, transform, alignments[scene])
            sources = np.asarray(["prefix"] * len(prefix) + ["calr"] * len(chosen))
            per_fraction[fraction][scene] = (boxes, scores)
            counts[fraction]["calr"] += len(chosen)
            counts[fraction]["predictions"] += len(rows)
            for threshold in THRESHOLDS:
                scene_quality = quality_counts(boxes, scores, sources, gts[scene], threshold)
                for group in ("all", "calr"):
                    for key in ("tp", "fp", "duplicate"):
                        quality[fraction][threshold][group][key] += scene_quality[group][key]

    budget_rows = []
    for fraction in FRACTIONS:
        aps = {
            threshold: class_agnostic_ap(per_fraction[fraction], gts, threshold)["ap"]
            for threshold in THRESHOLDS
        }
        row = {
            "budget_fraction": fraction, "budget_percent": int(100 * fraction),
            "calr_boxes": counts[fraction]["calr"],
            "mean_calr_boxes_per_scene": counts[fraction]["calr"] / len(scenes),
            "predictions": counts[fraction]["predictions"],
            **{f"ap{int(t*100)}": aps[t] for t in THRESHOLDS},
            "quality": {str(t): quality[fraction][t] for t in THRESHOLDS},
        }
        budget_rows.append(row)

    delay_rows = []
    for index, scene in enumerate(scenes, 1):
        delay_rows.extend(replay_delay(
            scene, evidence / f"{scene}.npz", native_by_scene[scene],
            gts[scene], transform, alignments[scene], args.delay_iou,
        ))
        if index % 10 == 0:
            print(f"delay replay {index}/100", flush=True)
    delay_summary = summarize_delay(delay_rows)

    with (args.output / "budget_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "budget_percent", "calr_boxes", "mean_calr_boxes_per_scene",
            "total_predictions", "AP15", "AP25", "AP50",
            "CALR_TP15", "CALR_FP15", "CALR_duplicate15",
            "CALR_precision15", "CALR_recall15",
            "CALR_TP25", "CALR_FP25", "CALR_duplicate25",
            "CALR_precision25", "CALR_recall25",
            "CALR_TP50", "CALR_FP50", "CALR_duplicate50",
            "CALR_precision50", "CALR_recall50",
        ])
        for row in budget_rows:
            values = [
                row["budget_percent"], row["calr_boxes"],
                row["mean_calr_boxes_per_scene"], row["predictions"],
                row["ap15"], row["ap25"], row["ap50"],
            ]
            for threshold in THRESHOLDS:
                q = row["quality"][str(threshold)]["calr"]
                precision = q["tp"] / max(q["tp"] + q["fp"], 1)
                recall = q["tp"] / sum(len(value) for value in gts.values())
                values.extend([
                    q["tp"], q["fp"], q["duplicate"], precision, recall,
                ])
            writer.writerow(values)
    with (args.output / "recovery_delay.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["scene", "gt_index", "branch", "iou_threshold",
                  "first_evidence_ordinal", "first_recovery_ordinal",
                  "delay_keyframes", "first_evidence_frame", "first_recovery_frame"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in delay_rows:
            if not row.get("summary"):
                writer.writerow({key: row[key] for key in fields})
    curves = {}
    for fraction in (0.0, 1.0):
        recall, precision = pr_curve(per_fraction[fraction], gts, 0.15)
        name = f"budget_{int(100*fraction)}"
        curves[name] = {"recall": recall.tolist(), "precision": precision.tolist()}
        with (args.output / f"pr_iou15_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(["recall", "precision"])
            writer.writerows(zip(recall, precision))

    result = {
        "schema": "boxfusion.scannet100.targeted_recovery_evidence.v1",
        "scene_count": len(scenes), "thresholds": THRESHOLDS,
        "budget_definition": "per-scene top ceil(fraction*N_final_CALR-v1) by serialized CALR tail score; fixed prefix is final PLR+MVSR",
        "budget": budget_rows,
        "delay": {
            "iou_threshold": args.delay_iou,
            "population": "terminal-native-missed GTs with source evidence",
            "endpoint": "first correct branch-state confirmation; PLR native-map hide/unhide is not reconstructed",
            "summary": delay_summary,
        },
        "input_sha256": hashes,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    render(budget_rows, delay_rows, args.output)

    lines = [
        "# ScanNet-100 targeted recovery evidence", "",
        "All results reuse the frozen final three-factor, no-visibility run. No detector or lifting model was rerun.", "",
        "## CALR terminal budget", "",
        "| Retained budget | CALR boxes | Mean/scene | AP15 | AP25 | AP50 | CALR TP/FP/duplicate @0.15 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_rows:
        q = row["quality"]["0.15"]["calr"]
        lines.append(
            f"| {row['budget_percent']}% | {row['calr_boxes']:,} | {row['mean_calr_boxes_per_scene']:.2f} | "
            f"{row['ap15']:.4f} | {row['ap25']:.4f} | {row['ap50']:.4f} | "
            f"{q['tp']:,}/{q['fp']:,}/{q['duplicate']:,} |"
        )
    lines += ["", "## Causal branch-confirmation delay", "",
              "| Branch | Evidence-eligible native misses | Eventually confirmed | Fraction | Median | P90 | P95 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for branch in ("plr", "calr"):
        row = delay_summary[branch]
        lines.append(
            f"| {branch.upper()} | {row['eligible_gt']:,} | {row['recovered_gt']:,} | "
            f"{100*row['recovery_fraction']:.2f}% | {row['median_keyframes']:.1f} | "
            f"{row['p90_keyframes']:.1f} | {row['p95_keyframes']:.1f} |"
        )
    lines += [
        "", "Delay is measured in processed keyframes from the first source candidate overlapping a terminal-native-missed GT to the first overlapping record in that branch's causal state. Because the frozen run did not serialize every native-map prefix, this is a branch-confirmation delay rather than a claim about first visibility after final output assembly.",
        "", "Duplicate means an added prediction overlaps a GT above the stated IoU threshold after that GT has already been claimed by a higher-scored prediction under the evaluator's greedy order.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
