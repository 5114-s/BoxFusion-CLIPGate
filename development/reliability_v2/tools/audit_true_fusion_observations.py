#!/usr/bin/env python3
"""Offline GT audit of captured original observations and actual PFO inputs.

Never consumes kf_map as observations. Results are a scoped identity diagnostic,
not a proof that every association is correct or that AP headroom is zero.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.true_fusion_audit_core import (
    aabb_iou, assign_observations, assess_membership, summarize_memberships,
    class_agnostic_ap,
)
from tools.audit_ca1m_nms_child_headroom import verify_anchor

THRESHOLDS = (0.15, 0.25, 0.50)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def new_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def check_ap_anchor(root):
    root = Path(root)
    sys.path.insert(0, str(root / "utils"))
    spec = importlib.util.spec_from_file_location("true_fusion_anchor_eval", root / "utils/eval_det.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    boxes = np.stack((signs, signs + 3.0)).astype(float)
    gt = {"a": boxes, "b": boxes[:1]}
    predictions = {"a": (boxes[[0, 0, 1]], [0.9, 0.8, 0.2]),
                   "b": (np.stack((boxes[0] + 20, boxes[0])), [0.95, 0.1])}
    anchor_pred = {scene: list(zip(value[0], value[1])) for scene, value in predictions.items()}
    errors = []
    for threshold in THRESHOLDS:
        _, _, expected = module.eval_det_cls(anchor_pred, gt, ovthresh=threshold,
                                             get_iou_func=module.get_iou_obb_v2)
        actual = class_agnostic_ap(predictions, gt, threshold)["ap"] / 100
        errors.append(abs(actual - expected))
    if max(errors) > 1e-10:
        raise ValueError(f"AP evaluator parity failed: {errors}")
    return {"max_error": max(errors), "source_sha256": sha(root / "utils/eval_det.py")}


def summed(items):
    statuses = Counter()
    keys = ("groups", "fully_identified_groups", "mixed_known_groups",
            "mixed_with_two_2frame_groups", "mixed_with_two_3frame_groups")
    output = {key: 0 for key in keys}
    for item in items:
        statuses.update(item["status_counts"])
        for key in keys:
            output[key] += item[key]
    output["status_counts"] = dict(sorted(statuses.items()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/ca1m_clean_root"))
    parser.add_argument("--anchor-root", type=Path,
                        default=Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.scenes) != len(set(args.scenes)):
        raise ValueError("Duplicate scenes")
    hashes, results, predictions, ground_truth = {}, {}, {}, {}
    def remember(path):
        hashes[str(path.resolve())] = sha(path)
        return path
    for scene in args.scenes:
        directory = args.run_dir / scene
        manifest = json.loads(remember(directory / "scene.json").read_text())
        if manifest.get("completed") is not True or manifest["scene_id"] != scene:
            raise ValueError(f"{scene}: incomplete or mismatched capture")
        raw_path = directory / manifest.get("raw_file", "observations.npz")
        final_path = directory / manifest.get("final_file", "final.npz")
        event_path = directory / manifest.get("events_file", "fusion_events.json")
        with np.load(remember(raw_path), allow_pickle=False) as source:
            raw = {key: source[key] for key in source.files}
        with np.load(remember(final_path), allow_pickle=False) as source:
            final = {key: source[key] for key in source.files}
        events = json.loads(remember(event_path).read_text())
        if isinstance(events, dict):
            events = events["events"]
        gt_path = args.data_root / scene / "after_filter_boxes.npy"
        gt = np.load(remember(gt_path), allow_pickle=False)
        ground_truth[scene] = gt
        predictions[scene] = (final["corners"], final["scores"])
        n = len(raw["corners"])
        if not np.array_equal(raw["init_ids"], np.arange(n)):
            raise ValueError(f"{scene}: IDs are not raw observation indices")
        matrix = aabb_iou(raw["corners"], gt)
        actual_sets, retained_sets = [], []
        seen = set()
        for event in events:
            selected = list(event["selected_ids"])
            source_ids = list(event["source_ids"])
            if not selected or not set(selected).issubset(source_ids):
                raise ValueError(f"{scene}: actual selection outside retained source")
            if min(source_ids) < 0 or max(source_ids) >= n:
                raise ValueError(f"{scene}: unknown source index")
            if np.any(raw["frame_ids"][source_ids] > event["frame_id"]):
                raise ValueError(f"{scene}: future observation input")
            actual_sets.append(selected)
            retained_sets.append(source_ids)
            key = tuple(sorted(set(selected)))
            seen.add(key)
        final_lists = manifest["final_fusion_lists"]
        if len(final_lists) != len(final["corners"]):
            raise ValueError(f"{scene}: final row/membership mismatch")
        result = {
            "raw_observations": n, "raw_distinct_frames": len(np.unique(raw["frame_ids"])),
            "gt": len(gt), "final_rows": len(final_lists), "pfo_events": len(events),
            "pfo_events_updated": sum(bool(event["updated"]) for event in events),
            "unique_pfo_selected_sets": len(seen), "diagnostics": {},
        }
        for threshold in THRESHOLDS:
            for policy in ("unique", "best"):
                key = f"{threshold:.2f}/{policy}"
                assignments = assign_observations(matrix, threshold, policy=policy)
                groups = {
                    "actual_pfo_events": actual_sets,
                    "unique_actual_pfo_sets": sorted(seen),
                    "pre_topk_retained_sources": retained_sets,
                    "final_retained_memberships": final_lists,
                }
                entry = {
                    "observation_assignment": {
                        "assigned": int(np.count_nonzero(assignments >= 0)),
                        "unknown": int(np.count_nonzero(assignments == -1)),
                        "ambiguous": int(np.count_nonzero(assignments == -2)),
                    },
                    **{label: summarize_memberships(ids, assignments, raw["frame_ids"])
                       for label, ids in groups.items()},
                }
                # This measures a *retained-set* refusion opportunity, not a TP.
                eligible = []
                for event, actual in zip(events, entry["actual_pfo_events"]["details"]):
                    if actual["status"] != "mixed_known":
                        continue
                    source = assess_membership(event["source_ids"], assignments, raw["frame_ids"])
                    eligible.append({
                        "event_id": event["event_id"], "row": event["row"],
                        "frame_id": event["frame_id"], "updated": event["updated"],
                        "selected": actual, "retained_source": source,
                        "refittable_gt_groups": [int(gt_id) for gt_id, value in source["known_groups"].items()
                                                  if value["distinct_frames"] >= 3],
                    })
                entry["actual_mixed_events"] = eligible
                entry["mixed_events_with_refittable_group"] = sum(bool(item["refittable_gt_groups"]) for item in eligible)
                result["diagnostics"][key] = entry
        results[scene] = result
    metric = verify_anchor(args.anchor_root)
    ap_parity = check_ap_anchor(args.anchor_root)
    aggregate = {}
    for threshold in THRESHOLDS:
        for policy in ("unique", "best"):
            key = f"{threshold:.2f}/{policy}"
            entries = [result["diagnostics"][key] for result in results.values()]
            aggregate[key] = {
                label: summed(entry[label] for entry in entries)
                for label in ("actual_pfo_events", "unique_actual_pfo_sets",
                              "pre_topk_retained_sources", "final_retained_memberships")
            }
            aggregate[key]["mixed_events_with_refittable_group"] = sum(
                entry["mixed_events_with_refittable_group"] for entry in entries)
    payload = {
        "schema": "boxfusion.true_fusion_gt_audit.v1", "completed": True,
        "scenes": args.scenes, "metric": metric, "ap_parity": ap_parity,
        "scope": "CA1M CuTR+Boxer+TopK3 threshold=.15 native branch, offline diagnostic",
        "limitations": [
            "GT assignment by 3D overlap is a proxy, not original pixel-level identity.",
            "Unique means exactly one GT exceeds threshold, not certainty of identity.",
            "Final retained membership is not full lifetime history or necessarily last PFO input.",
            "Event counts and unique selected-set counts are not independent tracks.",
            "A limited sample and fixed retained inputs cannot prove all demixing mechanisms impossible.",
            "Baseline AP below is sample-only; no active demixing AP or online speed claim.",
        ],
        "aggregate": aggregate, "per_scene": results,
        "sample_native_ap": {f"{threshold:.2f}": class_agnostic_ap(predictions, ground_truth, threshold)
                             for threshold in THRESHOLDS},
        "input_sha256": hashes,
    }
    new_json(args.output, payload)
    print(json.dumps({"scenes": len(results), "aggregate": aggregate,
                      "sample_native_ap": payload["sample_native_ap"]}, indent=2))


if __name__ == "__main__":
    main()
