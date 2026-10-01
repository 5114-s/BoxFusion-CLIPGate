#!/usr/bin/env python3
"""Audit scene replay and strictly bind selected cases to causal rerun traces."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment


def read(path: Path):
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    return ([int(r[0]) for r in rows],
            np.asarray([r[1] for r in rows], dtype=np.float64).reshape(-1, 8, 3),
            np.asarray([r[2] for r in rows], dtype=np.float64))


def aabb_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(-1, 8, 3)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 8, 3)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    l0, l1 = left.min(1), left.max(1)
    r0, r1 = right.min(1), right.max(1)
    lo = np.maximum(l0[:, None], r0[None])
    hi = np.minimum(l1[:, None], r1[None])
    inter = np.prod(np.maximum(hi - lo, 0), axis=2)
    lv = np.prod(np.maximum(l1 - l0, 0), axis=1)[:, None]
    rv = np.prod(np.maximum(r1 - r0, 0), axis=1)[None]
    return inter / np.maximum(lv + rv - inter, 1e-12)


def audit_scene(replay_path: Path, reference_path: Path) -> dict:
    """Match by geometry instead of row position and retain replay differences."""
    replay_labels, replay_boxes, replay_scores = read(replay_path)
    ref_labels, ref_boxes, ref_scores = read(reference_path)
    same_count = len(replay_labels) == len(ref_labels)
    same_label_multiset = Counter(replay_labels) == Counter(ref_labels)
    result = {
        "replay_count": len(replay_labels),
        "reference_count": len(ref_labels),
        "count_equal": same_count,
        "label_multiset_equal": same_label_multiset,
        "rowwise_exact": False,
    }
    if not (same_count and same_label_multiset):
        result["audit_passed"] = False
        return result
    if not replay_labels:
        result.update({"audit_passed": True, "rowwise_exact": True,
                       "matched_iou_min": 1.0, "matched_iou_median": 1.0,
                       "matched_iou_p05": 1.0, "matches_below_0_99": 0,
                       "matched_score_abs_max": 0.0})
        return result
    overlaps = aabb_iou(replay_boxes, ref_boxes)
    compatible = np.equal.outer(replay_labels, ref_labels)
    costs = 1.0 - overlaps
    costs[~compatible] = 1e6
    rr, cc = linear_sum_assignment(costs)
    matched_iou = overlaps[rr, cc]
    matched_score_error = np.abs(replay_scores[rr] - ref_scores[cc])
    rowwise_corner_error = float(np.max(np.abs(replay_boxes - ref_boxes)))
    rowwise_score_error = float(np.max(np.abs(replay_scores - ref_scores)))
    rowwise_exact = (replay_labels == ref_labels and rowwise_corner_error <= 1e-5
                     and rowwise_score_error <= 1e-6)
    result.update({
        "audit_passed": bool(np.all(compatible[rr, cc])),
        "rowwise_exact": bool(rowwise_exact),
        "rowwise_max_corner_abs_error": rowwise_corner_error,
        "rowwise_max_score_abs_error": rowwise_score_error,
        "matched_iou_min": float(np.min(matched_iou)),
        "matched_iou_p05": float(np.quantile(matched_iou, .05)),
        "matched_iou_median": float(np.median(matched_iou)),
        "matches_below_0_99": int(np.sum(matched_iou < .99)),
        "matched_score_abs_max": float(np.max(matched_score_error)),
    })
    return result


def selected_case_audit(case: dict, diagnostics_path: Path) -> dict:
    diagnostic = json.loads(diagnostics_path.read_text())
    state = diagnostic["state"]
    source = case["source"]
    trace_name = "plr" if source == "plr" else "calr" if source == "calr" else "native"
    rows = state["qualitative_trace"][trace_name]
    target = np.asarray(case["box"], dtype=np.float64)
    overlaps = aabb_iou(np.asarray([row["box"] for row in rows]), target[None])[:, 0]
    index = int(np.argmax(overlaps))
    row = rows[index]
    iou = float(overlaps[index])
    score_checks = {}
    if source == "plr":
        score_checks["birth_score_abs_error"] = abs(float(row["score"]) - float(case["score"]))
    elif source == "native":
        score_checks["original_score_abs_error"] = abs(
            float(row["original_score"]) - float(case["original_score"]))
        score_checks["reranked_score_abs_error"] = abs(
            float(row["reranked_score"]) - float(case["reranked_score"]))
    causal = bool(diagnostic["strictly_causal"] and diagnostic["online_incremental"])
    correct_config = bool(state.get("selected_calr_v1") is True
                          and state["m1p"]["use_children"] is False)
    enough_support = True
    if source == "plr":
        enough_support = len(set(map(int, row["evidence_frame_ids"]))) >= 3
    elif source == "calr":
        enough_support = len(set(map(int, row["support_frame_ids"]))) >= 3
    score_ok = all(error <= 1e-4 for error in score_checks.values())
    passed = causal and correct_config and enough_support and iou >= .999 and score_ok
    return {
        "passed": bool(passed),
        "source": source,
        "trace_index": index,
        "selected_to_trace_iou": iou,
        "strictly_causal_online": causal,
        "correct_final_configuration": correct_config,
        "support_requirement_passed": enough_support,
        **score_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [r.strip() for r in args.scenes.read_text().splitlines() if r.strip()]
    scene_report = {}
    for scene in scenes:
        scene_report[scene] = {
            "native": audit_scene(args.native/f"{scene}_boxes.pkl",
                                  args.components/"native_native"/f"{scene}_boxes.pkl"),
            "full": audit_scene(args.full/f"{scene}_boxes.pkl",
                                args.factorial/"p_a_m2"/f"{scene}_boxes.pkl"),
        }
    selection = json.loads(args.selection.read_text())
    case_report = {
        panel: selected_case_audit(case, args.diagnostics/f"{case['scene']}.json")
        for panel, case in selection["cases"].items()
    }
    scene_audit_passed = all(
        block[arm]["audit_passed"] for block in scene_report.values()
        for arm in ("native", "full"))
    exact = all(
        block[arm]["rowwise_exact"] for block in scene_report.values()
        for arm in ("native", "full"))
    selected_passed = all(row["passed"] for row in case_report.values())
    payload = {
        "schema": "boxfusion.recar3d.qualitative_parity.v2",
        "all_selected_cases_passed": selected_passed,
        "scene_source_audit_passed": scene_audit_passed,
        "scene_replay_rowwise_exact": exact,
        "interpretation": (
            "Rendering is gated by strict selected-case/source/trace parity. Whole-scene "
            "replay is retained as a count-, label-, and geometry-matched nondeterminism audit."
        ),
        "selected_cases": case_report,
        "scenes": scene_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    if not selected_passed:
        raise RuntimeError(f"selected-case trace parity failed; see {args.output}")


if __name__ == "__main__":
    main()
