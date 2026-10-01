#!/usr/bin/env python3
"""Event-level GT-cleaned refusion using the *unmodified* native PFO.

A is the recorded post-fusion geometry. B replays the recorded source list;
C independently replays each uniquely GT-assigned retained-source group with
at least three unique observations (the native PFO gate). Distinct frames are
a separate confirmation diagnostic, not an added execution requirement.
The 0.15/unique audit is primary, with 0.25/unique and 0.50/unique sensitivities.
The audit chooses the
groups, but no GT box geometry enters a replay or output. This is an offline,
GT-assisted geometry-capacity diagnostic, not an active split/birth module,
prediction export, AP evaluation, or online-speed measurement.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import io
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY = "0.15/unique"
POLICIES = (POLICY, "0.25/unique", "0.50/unique")
MIN_OBSERVATIONS = 3
INDEPENDENT_FRAMES = 3


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def require_close(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=0.0, atol=1e-7)):
        error = float(np.max(np.abs(actual - expected))) if actual.shape == expected.shape else None
        raise RuntimeError(f"Identity replay {name} mismatch: shape={actual.shape}/{expected.shape}, error={error}")
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def repeat_noise(runs, captured_corners=None):
    if not runs:
        return None
    corners = np.asarray([run["corners"] for run in runs], dtype=np.float64)
    boxes = np.asarray([run["box_xyzlhw"] for run in runs], dtype=np.float64)
    pair_corner = 0.0
    pair_params = 0.0
    for i in range(len(runs)):
        for j in range(i):
            pair_corner = max(pair_corner, float(np.linalg.norm(corners[i] - corners[j], axis=-1).max()))
            pair_params = max(pair_params, float(np.abs(boxes[i] - boxes[j]).max()))
    result = {"repeats": len(runs), "pairwise_max_corner_l2_m": pair_corner,
              "pairwise_max_xyzlhw_abs": pair_params,
              "all_selected_ids_identical": all(run["selected_ids"] == runs[0]["selected_ids"] for run in runs)}
    if captured_corners is not None:
        errors = np.linalg.norm(corners - np.asarray(captured_corners)[None], axis=-1).max(axis=-1)
        result["each_max_corner_l2_to_captured_m"] = errors.tolist()
        result["max_corner_l2_to_captured_m"] = float(errors.max())
    return result


class Instances(SimpleNamespace):
    """Only the field access interface consumed by native BoxFusion.boxfusion."""

    def __len__(self):
        return len(self.pred_boxes_3d.tensor)

    def has(self, name):
        return hasattr(self, name)

    def get(self, name):
        return getattr(self, name)


class Manager:
    def __init__(self, source_ids):
        self.fusion_list = [list(source_ids)]
        self.dynamic_object_branch = None
        self.updated = False
        self.added_fusion_ids = []

    def check_if_fusion(self, ids):
        return False

    def update_fusion_flag(self, row):
        if row != 0:
            raise RuntimeError("Expected single-row local replay")
        self.updated = True

    def add_fusion_ind(self, ids):
        self.added_fusion_ids.append(list(ids))


def make_observations(raw, prefix):
    import torch
    from boxfusion.boxes import GeneralInstance3DBoxes
    def tensor(key):
        return torch.from_numpy(np.array(raw[key][:prefix], copy=True))
    return Instances(pred_boxes_3d=GeneralInstance3DBoxes(tensor("boxes_xyzlhw"), tensor("rotations")),
                     cam_pose=tensor("cam_poses"), scores=tensor("scores"),
                     pred_boxes=tensor("boxes2d"), projected_boxes=tensor("projected_boxes"),
                     frame_id=tensor("frame_ids"), init_id=tensor("init_ids"))


def make_target(event):
    import torch
    from boxfusion.boxes import GeneralInstance3DBoxes
    xyz = torch.as_tensor(np.asarray(event["pre_box_xyzlhw"])[None], dtype=torch.float32, device="cuda")
    rotation = torch.as_tensor(np.asarray(event["pre_rotation"])[None], dtype=torch.float32, device="cuda")
    # Native PFO does not read the global row score; use a constant inert field.
    return Instances(pred_boxes_3d=GeneralInstance3DBoxes(xyz, rotation),
                     scores=torch.ones(1, device="cuda"),
                     init_id=torch.as_tensor([int(event.get("init_id", -1))], device="cuda"))


def build_fuser(manifest, protocol, scene_dir, scene, remember):
    from boxfusion.box_fusion import BoxFusion
    config = copy.deepcopy(manifest.get("effective_capture_config", manifest.get("effective_config")))
    if not isinstance(config, dict):
        raise ValueError(f"{scene}: no captured effective configuration")
    config["data"]["datadir"] = str(Path(protocol.get("data_root", f"/tmp/ca1m_clean_root/{scene}")))
    pst_path = Path(config["box_fusion"]["pst_path"])
    if not pst_path.is_absolute():
        pst_path = ROOT / pst_path
    config["box_fusion"]["pst_path"] = str(remember(pst_path))
    remember(Path(config["data"]["datadir"]) / "K_depth.txt")
    fuser = BoxFusion(config)
    if (fuser.vapf_lite.enabled or fuser.capf.enabled
            or fuser.maskdepth_pfo_cfg.get("enabled", False)):
        raise RuntimeError("Native-PFO replay rejects enabled non-native geometry branches")
    if not fuser.reliable_view_cfg["enabled"]:
        raise RuntimeError("Expected captured Reliable-View Top-K baseline")
    if fuser.reliable_view_cfg["min_views"] != 3 or fuser.reliable_view_cfg["top_k"] != 3:
        raise RuntimeError("This diagnostic keeps the original Top-K3/min-3 gate")
    expected = np.load(remember(scene_dir / manifest["pst_file"]), allow_pickle=False)
    if (fuser.PST.dtype != expected.dtype or fuser.PST.shape != expected.shape
            or fuser.PST.tobytes() != expected.tobytes()):
        raise RuntimeError("Native PST differs from the actually captured PST")
    return fuser, expected


def replay(fuser, expected_pst, raw, event, source_ids, *, identity):
    import torch
    source_ids = [int(value) for value in source_ids]
    frames = np.asarray(raw["frame_ids"])
    if np.any(np.diff(frames) < 0):
        raise ValueError("Raw observations are not chronologically ordered")
    prefix = int(np.count_nonzero(frames <= event["frame_id"]))
    if not source_ids or min(source_ids) < 0 or max(source_ids) >= prefix:
        raise ValueError("Counterfactual source is not a nonempty past/current-only subset")
    if len(set(source_ids)) < MIN_OBSERVATIONS:
        raise ValueError("Cannot run native refusion with fewer than three unique source observations")
    target = make_target(event)
    observations = make_observations(raw, prefix)
    manager = Manager(source_ids)
    fuser.K = np.asarray(event["K"]).copy()
    fuser.H, fuser.W = int(event["H"]), int(event["W"])
    if (int(fuser.fusion_iters) != int(event["fusion_iters"])
            or int(fuser.pst_size) != int(event["pst_size"])):
        raise RuntimeError("PFO iteration/PST budget differs from captured execution")
    if fuser.PST.tobytes() != expected_pst.tobytes():
        raise RuntimeError("PST mutated between repetitions")
    original_evaluate = fuser.evaluate_iou
    original_native_code = type(fuser).boxfusion.__code__
    signature = inspect.signature(original_evaluate)
    trace = []
    iterations = []

    def observed_evaluate(self, *args, **kwargs):
        caller = inspect.currentframe().f_back
        try:
            if caller.f_code is not original_native_code:
                raise RuntimeError("PFO is not being called by native boxfusion")
            local = caller.f_locals
            iterations.append(int(local["n"]))
            if int(local["n"]) == 0:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                selected = [int(i) for i in local["fusion_idx"]]
                first = {
                    "selected_ids": selected, "view_weights": array(local["view_weights"]),
                    "pfo_init_box_xyzlhw": array(values["box_3d"]),
                    "pfo_rotation": array(values["box_rot"]),
                    "scores_box": array(values["scores_box"]),
                    "search_size": array(values["search_size"]),
                    "selected_frame_ids": [int(frames[i]) for i in selected],
                }
                if identity:
                    if selected != [int(i) for i in event["selected_ids"]]:
                        raise RuntimeError(f"Identity Top-K mismatch: {selected} != {event['selected_ids']}")
                    if [int(i) for i in local["source_fusion_idx"]] != event["source_ids"]:
                        raise RuntimeError("Identity source list changed")
                    first["identity_input_errors"] = {
                        key: require_close(first[key], event[key], key)
                        for key in ("pfo_init_box_xyzlhw", "pfo_rotation", "scores_box", "view_weights", "search_size")
                    }
                    if bool(values["use_view_weights"]) != bool(event["use_view_weights"]):
                        raise RuntimeError("Identity weighting mode changed")
                trace.append(first)
        finally:
            del caller
        return original_evaluate(*args, **kwargs)

    fuser.evaluate_iou = MethodType(observed_evaluate, fuser)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            fuser.boxfusion(target, observations, manager, beta=float(event.get("beta", 0.9)))
        torch.cuda.synchronize()
    finally:
        fuser.evaluate_iou = original_evaluate
    if len(trace) > 1 or (identity and len(trace) != 1):
        raise RuntimeError(f"Unexpected number of first-iteration native calls: {len(trace)}")
    if (not np.array_equal(observations.pred_boxes_3d.tensor.numpy(), raw["boxes_xyzlhw"][:prefix])
            or not np.array_equal(observations.pred_boxes_3d.R.numpy(), raw["rotations"][:prefix])):
        raise RuntimeError("Native replay mutated the raw source geometry")
    xyz = array(target.pred_boxes_3d.tensor)[0]
    rotation = array(target.pred_boxes_3d.R)[0]
    corners = array(target.pred_boxes_3d.corners)[0]
    if not np.isfinite(corners).all() or not np.isfinite(xyz).all() or not np.isfinite(rotation).all():
        raise RuntimeError("Native refusion produced nonfinite output")
    initial = trace[0] if trace else {"selected_ids": [], "view_weights": [], "selected_frame_ids": []}
    independent = len(set(initial["selected_frame_ids"])) >= INDEPENDENT_FRAMES
    return {
        "box_xyzlhw": xyz, "rotation": rotation, "corners": corners,
        "source_ids": source_ids, **initial,
        "source_distinct_frames": len(set(int(frames[i]) for i in source_ids)),
        "source_unique_observations": len(set(source_ids)),
        "native_executable": len(set(source_ids)) >= MIN_OBSERVATIONS,
        "source_independent_3frame": len(set(int(frames[i]) for i in source_ids)) >= INDEPENDENT_FRAMES,
        "selected_distinct_frames": len(set(initial["selected_frame_ids"])),
        "independent_three_view": independent,
        "confirmation_status": "three_distinct_selected_frames" if independent else "not_independent",
        "pfo_called": bool(trace), "pfo_iterations": len(iterations),
        "updated": bool(manager.updated), "pre_box_fallback": not manager.updated,
        "geometry_changed": (not np.array_equal(xyz, np.asarray(event["pre_box_xyzlhw"], dtype=xyz.dtype))
                             or not np.array_equal(rotation, np.asarray(event["pre_rotation"], dtype=rotation.dtype))),
        "raw_input_prefix_length": prefix, "future_source_count": 0,
        "native_log_tail_if_skipped": log.getvalue()[-2000:] if not trace else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3 to measure native CUDA variation")
    args.run_dir, args.audit, args.output = args.run_dir.resolve(), args.audit.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    input_hashes = {}
    def remember(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha(path)
        if str(path) in input_hashes and input_hashes[str(path)] != observed:
            raise RuntimeError(f"Input changed during replay preparation: {path}")
        input_hashes[str(path)] = observed
        return path
    audit = json.loads(remember(args.audit).read_text())
    if audit.get("completed") is not True or audit.get("schema") != "boxfusion.true_fusion_gt_audit.v1":
        raise ValueError("Expected a completed true-fusion GT audit")
    source_paths = [Path(__file__).resolve(), ROOT / "boxfusion/box_fusion.py", ROOT / "boxfusion/boxes.py",
                    ROOT / "boxfusion/reliable_views.py", ROOT / "tools/true_fusion_audit_core.py"]
    sources = {str(p): sha(p) for p in source_paths}
    summary = {key: 0 for key in ("mixed_events", "eligible_events", "under_supported_events", "known_groups",
                                  "eligible_groups", "under_supported_groups", "identity_replays", "clean_group_replays",
                                  "clean_group_pfo_calls", "clean_group_independent_three_view_replays")}
    output_scenes = {}
    audited_hashes = audit.get("input_sha256", {})
    for scene in audit["scenes"]:
        if not isinstance(scene, str) or not scene.isdecimal():
            raise ValueError("Invalid CA-1M scene ID in audit")
        scene_dir = args.run_dir / scene
        manifest = json.loads(remember(scene_dir / "scene.json").read_text())
        if (manifest.get("completed") is not True or manifest["scene_id"] != scene
                or manifest.get("raw_prefix_immutable") is not True):
            raise ValueError(f"{scene}: incomplete / nonimmutable capture")
        captured_sources = manifest.get("source_sha256", {})
        for path in source_paths[1:4]:
            expected = captured_sources.get(str(path))
            if expected is None or expected != sources[str(path)]:
                raise RuntimeError(f"{scene}: native refusion source differs from original capture: {path}")
        raw_path = remember(scene_dir / manifest["raw_file"])
        events_path = remember(scene_dir / manifest["events_file"])
        for path in (scene_dir / "scene.json", raw_path, events_path):
            path = Path(path).resolve()
            if audited_hashes.get(str(path)) != input_hashes[str(path)]:
                raise RuntimeError(f"{scene}: capture is not the one used by supplied audit: {path}")
        with np.load(raw_path, allow_pickle=False) as payload:
            raw = {key: payload[key] for key in payload.files}
        n = len(raw["frame_ids"])
        if not np.array_equal(raw["init_ids"], np.arange(n)):
            raise ValueError(f"{scene}: native init IDs differ from raw indices")
        events = json.loads(events_path.read_text())
        if isinstance(events, dict):
            events = events["events"]
        by_id = {int(event["event_id"]): event for event in events}
        if len(by_id) != len(events):
            raise ValueError(f"{scene}: duplicate captured event IDs")
        mixed = [(policy, item) for policy in POLICIES
                 for item in audit["per_scene"][scene]["diagnostics"][policy]["actual_mixed_events"]]
        scene_result = {"mixed_events": [], "mixed_event_count": len(mixed)}
        fuser = expected_pst = None
        for policy, item in mixed:
            event = by_id[int(item["event_id"])]
            if int(item["frame_id"]) != int(event["frame_id"]) or int(item["row"]) != int(event["row"]):
                raise ValueError("Audit event mapping differs from captured event")
            source_ids = [int(v) for v in event["source_ids"]]
            if any(v < 0 or v >= n or raw["frame_ids"][v] > event["frame_id"] for v in source_ids):
                raise ValueError("Captured source has future / invalid observation IDs")
            groups = []
            known = item["retained_source"]["known_groups"]
            for gt_id, group in sorted(known.items(), key=lambda pair: int(pair[0])):
                member_set = set(int(v) for v in group["observation_ids"])
                if not member_set.issubset(source_ids):
                    raise ValueError("GT-assigned source group extends beyond actual retained sources")
                ids = [v for v in source_ids if v in member_set]
                distinct = len(set(int(raw["frame_ids"][v]) for v in ids))
                if distinct != int(group["distinct_frames"]):
                    raise ValueError("Audited group distinct-frame count differs from raw observations")
                executable = len(set(ids)) >= MIN_OBSERVATIONS
                groups.append({"gt_id": int(gt_id), "source_ids": ids, "distinct_frames": distinct,
                               "unique_observations": len(set(ids)), "native_executable": executable,
                               "independent_3frame": distinct >= INDEPENDENT_FRAMES,
                               "status": "eligible" if executable else "under_support", "runs": []})
            eligible = [group for group in groups if group["status"] == "eligible"]
            summary["mixed_events"] += 1
            summary["known_groups"] += len(groups)
            summary["eligible_groups"] += len(eligible)
            summary["under_supported_groups"] += len(groups) - len(eligible)
            result = {
                "policy": policy,
                "event_id": int(event["event_id"]), "row": int(event["row"]),
                "frame_id": int(event["frame_id"]), "source_ids": source_ids,
                "selected_ids": list(event["selected_ids"]),
                "A_captured": {"box_xyzlhw": event["post_box_xyzlhw"], "rotation": event["post_rotation"],
                               "corners": event["post_corners"], "updated": bool(event["updated"])},
                "pre": {"box_xyzlhw": event["pre_box_xyzlhw"], "rotation": event["pre_rotation"],
                        "corners": event["pre_corners"]},
                "B_identity": None, "C_groups": groups,
                "status": "eligible" if eligible else "under_support",
                "unknown_or_ambiguous_source_observations": int(item["retained_source"]["unknown_observations"])
                                                            + int(item["retained_source"]["ambiguous_observations"]),
            }
            if not eligible:
                summary["under_supported_events"] += 1
                scene_result["mixed_events"].append(result)
                continue
            summary["eligible_events"] += 1
            if fuser is None:
                protocol = json.loads(remember(scene_dir / "protocol.json").read_text())
                fuser, expected_pst = build_fuser(manifest, protocol, scene_dir, scene, remember)
            baseline_runs = [replay(fuser, expected_pst, raw, event, source_ids, identity=True)
                             for _ in range(args.repeats)]
            summary["identity_replays"] += len(baseline_runs)
            result["B_identity"] = {"runs": baseline_runs,
                                    "noise": repeat_noise(baseline_runs, event["post_corners"]),
                                    "native_inputs_verified_atol": 1e-7}
            for group in eligible:
                group["runs"] = [replay(fuser, expected_pst, raw, event, group["source_ids"], identity=False)
                                 for _ in range(args.repeats)]
                group["status"] = "replayed"
                group["noise"] = repeat_noise(group["runs"])
                summary["clean_group_replays"] += len(group["runs"])
                summary["clean_group_pfo_calls"] += sum(run["pfo_called"] for run in group["runs"])
                summary["clean_group_independent_three_view_replays"] += sum(run["independent_three_view"] for run in group["runs"])
            scene_result["mixed_events"].append(result)
            print(f"REFUSION policy={policy} scene={scene} event={event['event_id']} groups={len(eligible)}/{len(groups)} "
                  f"identity_corner_error={result['B_identity']['noise']['max_corner_l2_to_captured_m']:.6g}", flush=True)
        output_scenes[scene] = scene_result
    changed = [path for path, expected in {**sources, **input_hashes}.items() if sha(path) != expected]
    if changed:
        raise RuntimeError(f"Source/input changed during offline refusion: {changed}")
    by_policy = {}
    for policy in POLICIES:
        policy_scenes = {}
        for scene, value in output_scenes.items():
            events_for_policy = [item for item in value["mixed_events"] if item["policy"] == policy]
            policy_scenes[scene] = {"mixed_events": events_for_policy, "mixed_event_count": len(events_for_policy)}
        entries = [item for value in policy_scenes.values() for item in value["mixed_events"]]
        groups = [group for item in entries for group in item["C_groups"]]
        identity_runs = [run for item in entries if item["B_identity"] is not None for run in item["B_identity"]["runs"]]
        clean_runs = [run for group in groups for run in group["runs"]]
        policy_summary = {
            "mixed_events": len(entries), "eligible_events": sum(item["B_identity"] is not None for item in entries),
            "under_supported_events": sum(item["B_identity"] is None for item in entries),
            "known_groups": len(groups), "native_executable_groups": sum(g["native_executable"] for g in groups),
            "under_supported_groups": sum(not g["native_executable"] for g in groups),
            "independent_3frame_source_groups": sum(g["independent_3frame"] for g in groups),
            "identity_replays": len(identity_runs), "clean_group_replays": len(clean_runs),
            "clean_group_pfo_calls": sum(run["pfo_called"] for run in clean_runs),
            "clean_group_independent_three_view_replays": sum(run["independent_three_view"] for run in clean_runs),
            "clean_group_not_independent_replays": sum(not run["independent_three_view"] for run in clean_runs),
        }
        by_policy[policy] = {"summary": policy_summary, "scenes": policy_scenes}
    payload = {
        "schema": "boxfusion.true_fusion_refusion_oracle.v1", "completed": True,
        "gt_assisted": True, "policy": POLICY, "repeats": args.repeats,
        "primary_policy": POLICY, "policies": list(POLICIES),
        "native_minimum_unique_source_observations": MIN_OBSERVATIONS,
        "independent_frame_confirmation_diagnostic": INDEPENDENT_FRAMES,
        "native_topk": 3, "native_minimum_views": 3,
        "summary": by_policy[POLICY]["summary"], "scenes": by_policy[POLICY]["scenes"],
        "by_policy": by_policy, "all_policy_execution_totals": summary,
        "limitations": [
            "This is event-level geometry-capacity analysis, not an active track split, birth, AP, or speed test.",
            "Groups use overlap-based GT identity proxies; 0.15/unique is primary and 0.25/unique/0.50/unique are sensitivities.",
            "No GT box geometry is loaded, fitted, or exported by this tool.",
            "Native execution requires three unique observations, NOT three distinct frames; under_support only means fewer than three unique observations.",
            "Top-K is never altered to force temporal independence; fewer than three selected frames is marked not_independent.",
            "No future raw observations are supplied to native refusion; every replay only gets its event-time prefix.",
            "C uses the same native pre-box fallback when native PFO does not update; no GT or group-mean fallback is invented.",
            "All identity and cleaned-group repetitions are retained; no best repetition is selected.",
            "B-vs-capture includes CUDA atomic variability and possible tiny GPU corner batch-shape rounding differences.",
            "Unknown/ambiguous sources are removed only by the offline GT grouping; this is not an online selection rule.",
        ],
        "run_dir": str(args.run_dir), "audit_path": str(args.audit),
        "input_sha256": input_hashes, "source_sha256": sources,
        "production_predictions_written": False, "input_and_source_hashes_unchanged": True,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"completed": True, "by_policy": {k: v["summary"] for k, v in by_policy.items()},
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
