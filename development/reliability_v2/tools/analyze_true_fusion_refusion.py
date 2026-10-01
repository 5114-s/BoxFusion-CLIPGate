#!/usr/bin/env python3
"""Compare GT-cleaned native PFO against identity replay, including scoped AP.

Terminal counterfactuals only replace a row whose init_id AND exact xyzlhw/R
identify one recorded, geometry-updating PFO event. Earlier events that would
require rerunning future association are deliberately not patched into EOF.
No boxes are added, no scores are changed, and all repetitions are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.audit_true_fusion_observations import sha, new_json, check_ap_anchor

THRESHOLDS = (0.15, 0.25, 0.50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/ca1m_clean_root"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {}
    def remember(path):
        path = Path(path).resolve()
        hashes[str(path)] = sha(path)
        return path
    replay = json.loads(remember(args.replay).read_text())
    if replay.get("completed") is not True:
        raise ValueError("Incomplete replay")
    audit_path = remember(replay["audit_path"])
    if replay["input_sha256"].get(str(audit_path)) != hashes[str(audit_path)]:
        raise ValueError("Audit changed since replay")
    audit = json.loads(audit_path.read_text())
    audit_hashes = audit["input_sha256"]
    repeats = replay["repeats"]
    if repeats < 3:
        raise ValueError("Insufficient identity repetitions")
    cached, base_predictions, gt_all = {}, {}, {}
    scene_ids = list(replay["scenes"])
    for scene in scene_ids:
        directory = args.run_dir / scene
        manifest = json.loads(remember(directory / "scene.json").read_text())
        with np.load(remember(directory / manifest["raw_file"]), allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
        with np.load(remember(directory / manifest["final_file"]), allow_pickle=False) as data:
            final = {key: data[key] for key in data.files}
        events = json.loads(remember(directory / manifest["events_file"]).read_text())
        gt = np.load(remember(args.data_root / scene / "after_filter_boxes.npy"), allow_pickle=False)
        for path in (directory / "scene.json", directory / manifest["raw_file"],
                     directory / manifest["final_file"], directory / manifest["events_file"],
                     args.data_root / scene / "after_filter_boxes.npy"):
            resolved = str(path.resolve())
            if audit_hashes.get(resolved) != hashes[resolved]:
                raise ValueError(f"Analysis input is not the one used by audit: {path}")
            expected_replay_hash = replay["input_sha256"].get(resolved)
            if expected_replay_hash is not None and expected_replay_hash != hashes[resolved]:
                raise ValueError(f"Analysis input changed since replay: {path}")
        gt_all[scene] = gt
        base_predictions[scene] = (final["corners"], final["scores"])
        event_to_final, unresolved = {}, []
        for row in range(len(final["scores"])):
            history = [event for event in events if event["updated"]
                       and int(event["init_id"]) == int(final["init_ids"][row])]
            matches = []
            if history:
                latest_id = max(int(event["event_id"]) for event in history)
                latest = [event for event in history if int(event["event_id"]) == latest_id]
                matches = [event for event in latest
                           if np.array_equal(np.asarray(event["post_box_xyzlhw"], dtype=final["boxes_xyzlhw"].dtype), final["boxes_xyzlhw"][row])
                           and np.array_equal(np.asarray(event["post_rotation"], dtype=final["rotations"].dtype), final["rotations"][row])]
            if len(matches) == 1:
                event_to_final.setdefault(int(matches[0]["event_id"]), []).append(row)
            elif len(matches) > 1:
                unresolved.append(row)
        cached[scene] = {"raw": raw, "final": final, "events": events,
                         "event_to_final": event_to_final, "ambiguous_final_links": unresolved}
    anchor_parity = check_ap_anchor(Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation"))
    native_ap = {f"{threshold:.2f}": class_agnostic_ap(base_predictions, gt_all, threshold)
                 for threshold in THRESHOLDS}
    policies = {}
    for policy, policy_data in replay["by_policy"].items():
        identities = [{scene: (value[0].copy(), value[1].copy()) for scene, value in base_predictions.items()}
                      for _ in range(repeats)]
        cleaned = [{scene: (value[0].copy(), value[1].copy()) for scene, value in base_predictions.items()}
                   for _ in range(repeats)]
        event_results, terminal_rows = [], []
        seen_events, seen_rows = set(), set()
        for scene, scene_data in policy_data["scenes"].items():
            raw = cached[scene]["raw"]
            gt = gt_all[scene]
            for event in scene_data["mixed_events"]:
                event_key = (scene, int(event["event_id"]))
                if event_key in seen_events:
                    raise ValueError("Duplicate replay event in policy")
                seen_events.add(event_key)
                if event["B_identity"] is None:
                    continue
                groups = [group for group in event["C_groups"] if group["runs"]]
                if not groups:
                    raise ValueError("Identity replay has no cleaned group counterpart")
                # Frozen deterministic owner selection using GT identity labels
                # for partition only, never candidate/GT IoU to select a winner.
                def rank(group):
                    ids = sorted(set(group["source_ids"]))
                    return (-group["distinct_frames"], -len(ids),
                            -float(np.sum(raw["scores"][ids])), int(group["gt_id"]))
                groups = sorted(groups, key=rank)
                chosen = groups[0]
                target = int(chosen["gt_id"])
                if not 0 <= target < len(gt):
                    raise ValueError("Invalid GT identity")
                A = aabb_iou([event["A_captured"]["corners"]], gt)[0, target]
                B = [float(aabb_iou([run["corners"]], gt)[0, target])
                     for run in event["B_identity"]["runs"]]
                C = [float(aabb_iou([run["corners"]], gt)[0, target]) for run in chosen["runs"]]
                if len(B) != repeats or len(C) != repeats:
                    raise ValueError("Missing repeats")
                rows = cached[scene]["event_to_final"].get(int(event["event_id"]), [])
                result = {
                    "scene_id": scene, "event_id": event["event_id"], "frame_id": event["frame_id"],
                    "selected_gt_group": target, "source_ids": chosen["source_ids"],
                    "source_distinct_frames": chosen["distinct_frames"],
                    "other_native_executable_groups": len(groups) - 1,
                    "A_target_iou": float(A), "B_target_iou": B, "C_target_iou": C,
                    "C_minus_B_iou": (np.asarray(C) - B).tolist(),
                    "identity_noise": event["B_identity"]["noise"],
                    "clean_noise": chosen["noise"],
                    "exact_terminal_rows": rows,
                    "crossings_C_minus_B": {
                        f"{threshold:.2f}": ((np.asarray(C) > threshold).astype(int)
                                             - (np.asarray(B) > threshold).astype(int)).tolist()
                        for threshold in THRESHOLDS},
                }
                event_results.append(result)
                for row in rows:
                    if (scene, row) in seen_rows:
                        raise ValueError("More than one counterfactual patches the same final row")
                    seen_rows.add((scene, row))
                    terminal_rows.append({"scene_id": scene, "row": row,
                                          "event_id": event["event_id"], "gt_group": target})
                    for repeat in range(repeats):
                        identities[repeat][scene][0][row] = event["B_identity"]["runs"][repeat]["corners"]
                        cleaned[repeat][scene][0][row] = chosen["runs"][repeat]["corners"]
        metrics = []
        for repeat in range(repeats):
            for scene in scene_ids:
                for arm in (identities[repeat], cleaned[repeat]):
                    if not np.array_equal(arm[scene][1], base_predictions[scene][1]):
                        raise ValueError("Terminal counterfactual changed scores")
                    if len(arm[scene][0]) != len(base_predictions[scene][0]):
                        raise ValueError("Terminal counterfactual changed row count")
            B = {f"{threshold:.2f}": class_agnostic_ap(identities[repeat], gt_all, threshold)
                 for threshold in THRESHOLDS}
            C = {f"{threshold:.2f}": class_agnostic_ap(cleaned[repeat], gt_all, threshold)
                 for threshold in THRESHOLDS}
            metrics.append({"repeat": repeat, "identity_B": B, "cleaned_C": C,
                            "C_minus_B_AP": {key: C[key]["ap"] - B[key]["ap"] for key in B},
                            "B_minus_native_AP": {key: B[key]["ap"] - native_ap[key]["ap"] for key in B}})
        policies[policy] = {
            "eligible_events": len(event_results), "patched_terminal_rows": len(terminal_rows),
            "new_rows": 0, "terminal_rows": terminal_rows, "event_quality": event_results,
            "sample_terminal_counterfactual_ap": metrics,
            "mean_C_minus_B_AP": {f"{threshold:.2f}": float(np.mean([
                item["C_minus_B_AP"][f"{threshold:.2f}"] for item in metrics])) for threshold in THRESHOLDS},
        }
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f"Input changed during analysis: {path}")
    output = {
        "schema": "boxfusion.true_fusion_refusion_analysis.v1", "completed": True,
        "scenes": scene_ids, "sample_native_AP": native_ap, "anchor_AP_parity": anchor_parity,
        "by_policy": policies,
        "limitations": [
            "This is GT-assisted sample-only geometry refusion, not a deployable online policy.",
            "Terminal patches require exact identity/parameter links to one real last geometry-changing event.",
            "Earlier counterfactual events would require causal downstream re-execution and are not patched at EOF.",
            "Association after even the last geometry-changing event is not re-executed; this is not full causal-route AP.",
            "Fixed count, score and ordering; no split births are implemented or evaluated.",
            "Groups with fewer than three retained original observations are not refitted by native PFO.",
            "GT overlap identities are imperfect proxies; results are not a global mathematical upper bound.",
            "Zero patched rows means no executable terminal edit in this guarded set, not zero headroom for all demixing.",
            "All three repeats retained; no best policy, threshold or repeat selected for the headline.",
        ], "input_sha256": hashes,
    }
    new_json(args.output, output)
    print(json.dumps({"sample_native_AP": native_ap, "by_policy": {
        key: {name: value[name] for name in ("eligible_events", "patched_terminal_rows", "mean_C_minus_B_AP")}
        for key, value in policies.items()}}, indent=2))


if __name__ == "__main__":
    main()
