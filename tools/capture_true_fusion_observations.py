#!/usr/bin/env python3
"""Capture actual native observation/Top-K/PFO provenance without changing it.

Only this standalone process is instrumented.  Production files, model outputs,
matching, scores, and the selected views are not rewritten.  Every original
observation is frozen when BoxManager first registers its frame, before native
association, then checked against the complete per-frame history at every
fusion boundary and terminal save.  GT annotation access is forbidden.

Output schema: scene.json indexes observations.npz, fusion_events.json,
fusion_calls.json, final.npz, and PST.npy.  Source/selected IDs are the actual
raw-history indices used by production; the capture requires raw init_ids to
equal these indices.  Events record only real PFO calls, at iteration n == 0.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW_KEYS = ("corners", "boxes_xyzlhw", "rotations", "cam_poses", "scores",
            "boxes2d", "projected_boxes", "frame_ids", "init_ids")


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


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


def write_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(jsonable(payload), handle, ensure_ascii=False, allow_nan=False)


def write_npz(path, payload):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **payload)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def forbid_gt(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        if "after_filter_boxes.npy" in name or "full_annotations.json" in name:
            raise RuntimeError("GT annotation read forbidden during true-fusion capture")


def canonical_corners(boxes, rotations):
    """Derive world corners from immutable parameters, without batch GPU math.

    GeneralInstance3DBoxes.corners uses batched GPU matmul. Its rounding can
    change as the growing history changes the batch shape, although xyzlhw/R
    remain bitwise identical. That derived numerical difference is not a raw
    observation mutation. The order below is exactly that native convention.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    signs = np.asarray([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    local = boxes[:, None, 3:] * signs[None] * 0.5
    rotated = np.sum(rotations[:, None, :, :] * local[:, :, None, :], axis=-1)
    return rotated + boxes[:, None, :3]


def instance_arrays(instances):
    if instances is None:
        raise RuntimeError("Cannot capture absent Instances3D")
    required = ("pred_boxes_3d", "cam_pose", "scores", "pred_boxes",
                "projected_boxes", "frame_id", "init_id")
    missing = [key for key in required if not instances.has(key)]
    if missing:
        raise RuntimeError(f"Required raw observation fields missing: {missing}")
    box = instances.pred_boxes_3d
    payload = {
        "boxes_xyzlhw": array(box.tensor),
        "rotations": array(box.R), "cam_poses": array(instances.cam_pose),
        "scores": array(instances.scores), "boxes2d": array(instances.pred_boxes),
        "projected_boxes": array(instances.projected_boxes),
        "frame_ids": array(instances.frame_id), "init_ids": array(instances.init_id),
    }
    payload["corners"] = canonical_corners(payload["boxes_xyzlhw"], payload["rotations"])
    count = len(instances)
    for key, value in payload.items():
        if value.ndim < 1 or len(value) != count:
            raise RuntimeError(f"Misaligned {key}: {value.shape} vs {count} instances")
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite {key} in original production observations")
    if payload["corners"].shape != (count, 8, 3):
        raise RuntimeError("Unexpected native corner shape")
    return payload


def geometry_arrays(instances):
    box = instances.pred_boxes_3d
    return {
        "corners": array(box.corners), "scores": array(instances.scores),
        "init_ids": array(instances.init_id), "boxes_xyzlhw": array(box.tensor),
        "rotations": array(box.R),
    }


class Capture:
    def __init__(self, scene, run_dir):
        self.scene, self.run_dir = scene, Path(run_dir)
        self.raw = None
        self.raw_checks = []
        self.events = []
        self.calls = []
        self.active_call = None
        self.final = None
        self.pst = None
        self.patch_targets = []
        self.birth_native_corners = []
        self.corner_numeric_diagnostics = []

    def capture_new(self, instances, all_num, box_num, frame_id):
        observed = instance_arrays(instances)
        native_corners = array(instances.pred_boxes_3d.corners)
        self.birth_native_corners.append(native_corners)
        self.corner_numeric_diagnostics.append({
            "boundary": "birth", "frame_id": int(frame_id),
            "max_native_vs_float64_corner_error": float(np.max(np.abs(native_corners - observed["corners"])))
            if native_corners.size else 0.0,
        })
        expected_start = 0 if self.raw is None else len(self.raw["init_ids"])
        if int(all_num) != expected_start or int(box_num) != len(observed["init_ids"]):
            raise RuntimeError("New observation registration is not a contiguous history append")
        expected_ids = np.arange(expected_start, expected_start + int(box_num))
        if not np.array_equal(observed["init_ids"], expected_ids):
            raise RuntimeError("Native init_ids do not equal actual raw-history indices")
        if not np.all(observed["frame_ids"] == int(frame_id)):
            raise RuntimeError("New observation frame IDs differ from actual demo frame")
        if self.raw is None:
            self.raw = observed
        else:
            self.raw = {key: np.concatenate((self.raw[key], observed[key]), axis=0)
                        for key in RAW_KEYS}
        self.raw_checks.append({"boundary": "new_observation_registration",
                                "frame_id": int(frame_id), "raw_count": len(self.raw["init_ids"]),
                                "new_count": int(box_num), "immutable": True})

    def check_raw(self, instances, boundary, frame_id):
        observed = instance_arrays(instances)
        native_corners = array(instances.pred_boxes_3d.corners)
        birth_corners = np.concatenate(self.birth_native_corners, axis=0)
        self.corner_numeric_diagnostics.append({
            "boundary": boundary, "frame_id": int(frame_id),
            "max_native_vs_birth_gpu_corner_error": float(np.max(np.abs(native_corners - birth_corners)))
            if native_corners.size and native_corners.shape == birth_corners.shape else None,
            "max_native_vs_float64_corner_error": float(np.max(np.abs(native_corners - observed["corners"])))
            if native_corners.size else 0.0,
        })
        if self.raw is None:
            raise RuntimeError("History encountered before original observation registration")
        changed = [key for key in RAW_KEYS if not exact(self.raw[key], observed[key])]
        if changed:
            detail = []
            for key in changed:
                a, b = self.raw[key], observed[key]
                changed_rows = []
                if a.shape == b.shape:
                    changed_rows = [i for i in range(len(a)) if not exact(a[i], b[i])][:10]
                detail.append({"field": key, "frozen_shape": list(a.shape),
                               "current_shape": list(b.shape), "first_changed_rows": changed_rows})
            write_json(self.run_dir / "raw_immutability_failure.json", {
                "boundary": boundary, "frame_id": int(frame_id), "changes": detail,
                "original_observation_claim_valid": False,
            })
            raise RuntimeError(f"Original observation history mutated at {boundary}: {changed}")
        self.raw_checks.append({"boundary": boundary, "frame_id": int(frame_id),
                                "raw_count": len(observed["init_ids"]), "immutable": True})

    def capture_pfo(self, fuser, locals_, bound):
        if self.active_call is None:
            raise RuntimeError("PFO evaluation outside instrumented native boxfusion")
        row = int(locals_["i"])
        source = [int(v) for v in locals_["source_fusion_idx"]]
        selected = np.asarray(locals_["fusion_idx"], dtype=np.int64)
        raw_n = len(self.raw["init_ids"])
        if any(v < 0 or v >= raw_n for v in source) or np.any((selected < 0) | (selected >= raw_n)):
            raise RuntimeError("PFO references outside captured original observation history")
        if any(int(v) not in source for v in selected):
            raise RuntimeError("Actual selected PFO index was not in source fusion list")
        if len(selected) != int(bound["num_of_boxes"]):
            raise RuntimeError("PFO count does not match actual selection")
        for parameter, raw_key in (("corners_2d", "projected_boxes"),
                                   ("camera_poses", "cam_poses")):
            if not exact(array(bound[parameter]), self.raw[raw_key][selected]):
                raise RuntimeError(f"PFO {parameter} differs from frozen selected source observations")
        pst = array(fuser.PST)
        if self.pst is None:
            self.pst = pst
            with (self.run_dir / "PST.npy").open("xb") as handle:
                np.save(handle, pst, allow_pickle=False)
        elif not exact(self.pst, pst):
            raise RuntimeError("PST changed during native run; single-PST replay would be invalid")
        pre = self.active_call["pre"]
        event = {
            "event_id": len(self.events), "call_id": self.active_call["call_id"],
            "frame_id": self.active_call["frame_id"], "row": row,
            "init_id": int(pre["init_ids"][row]), "source_ids": source,
            "selected_ids": selected.tolist(),
            "source_frame_ids": self.raw["frame_ids"][source].tolist(),
            "selected_frame_ids": self.raw["frame_ids"][selected].tolist(),
            "pre_corners": pre["corners"][row].copy(),
            "pre_box_xyzlhw": pre["boxes_xyzlhw"][row].copy(),
            "pre_rotation": pre["rotations"][row].copy(),
            "view_weights": array(locals_["view_weights"]),
            "scores_box": array(bound["scores_box"]),
            "pfo_init_box_xyzlhw": array(bound["box_3d"]),
            "pfo_rotation": array(bound["box_rot"]),
            "search_size": array(bound["search_size"]),
            "K": array(fuser.K), "H": int(fuser.H), "W": int(fuser.W),
            "pst_size": int(fuser.pst_size), "pst_file": "PST.npy",
            "use_view_weights": bool(bound.get("use_view_weights", False)),
            "num_of_boxes": int(bound["num_of_boxes"]),
            "fusion_iters": int(fuser.fusion_iters),
            "center_scaling_coefficient": float(fuser.center_scaling_coefficient),
            "shape_scaling_coefficient": float(fuser.shape_scaling_coefficient),
            "beta": self.active_call["beta"],
        }
        if any(self.events[e]["row"] == row for e in self.active_call["event_ids"]):
            raise RuntimeError("Multiple first-iteration PFO captures for one native row")
        self.active_call["event_ids"].append(event["event_id"])
        self.events.append(event)

    def capture_save(self, data, filename, local):
        if self.final is not None:
            raise RuntimeError("More than one terminal prediction save; expected static CA-1M")
        if Path(filename).resolve() != self.run_dir / "predictions" / f"{self.scene}_boxes.pkl":
            raise RuntimeError(f"Unexpected prediction output: {filename}")
        manager = local["box_manager"]
        native = local["all_pred_box"]
        frame = int(local["count"]) - 1
        self.check_raw(local["per_frame_ins"], "terminal_save", frame)
        full = geometry_arrays(native)
        if len(manager.fusion_list) != len(full["init_ids"]):
            raise RuntimeError("Terminal native rows do not align with fusion_list")
        valid_mask = np.asarray(local["terminal_valid_mask"], dtype=bool)
        if valid_mask.shape != (len(native),):
            raise RuntimeError("Terminal source mask does not align with native rows")
        indices = np.flatnonzero(valid_mask)
        if not isinstance(data, list) or len(data) != 1 or len(data[0]) != len(indices):
            raise RuntimeError("Native saved prediction count differs from terminal row mapping")
        rows = data[0]
        saved_corners = np.stack([np.asarray(row[1]) for row in rows]) if rows else np.empty((0, 8, 3))
        saved_scores = np.asarray([row[2] for row in rows], dtype=full["scores"].dtype)
        if (not np.array_equal(saved_corners, full["corners"][indices])
                or not np.array_equal(saved_scores, full["scores"][indices])):
            raise RuntimeError("Terminal corners/scores no longer correspond to native row geometry")
        lists = [[int(v) for v in manager.fusion_list[int(i)]] for i in indices]
        final = {key: value[indices].copy() for key, value in full.items()}
        final["native_row_indices"] = indices.astype(np.int64)
        self.final = {"arrays": final, "fusion_lists": lists,
                      "full_native": full, "full_fusion_lists": copy.deepcopy(manager.fusion_list),
                      "frame_id": frame, "prediction_path": str(filename),
                      "saved_row_mapping_exact": True}

    def install(self):
        # Imports live here so --help / syntax checks do not initialize CUDA.
        from boxfusion.box_fusion import BoxFusion
        from boxfusion.box_manager import BoxManager
        import tools.utils as utils

        original_new = BoxManager.init_new_predictions
        original_fusion = BoxFusion.boxfusion
        original_evaluate = BoxFusion.evaluate_iou
        original_save = utils.save_box
        fusion_signature = inspect.signature(original_fusion)
        evaluate_signature = inspect.signature(original_evaluate)

        def new_predictions(manager, box_num, all_num):
            caller = inspect.currentframe().f_back
            try:
                local = caller.f_locals
                if Path(caller.f_code.co_filename).resolve() != ROOT / "demo.py":
                    raise RuntimeError("Observation registration not called directly by demo.run")
                if self.raw is not None:
                    self.check_raw(local["per_frame_ins"], "pre_new_registration", local["count"])
                self.capture_new(local["pred_instances"], all_num, box_num, local["count"])
            finally:
                del caller
            return original_new(manager, box_num, all_num)

        def fusion(fuser, all_pred_box, per_frame_box, box_manager, *args, **kwargs):
            if self.active_call is not None:
                raise RuntimeError("Nested native boxfusion invocation")
            caller = inspect.currentframe().f_back
            try:
                frame_id = int(caller.f_locals["count"])
            finally:
                del caller
            bound = fusion_signature.bind(fuser, all_pred_box, per_frame_box, box_manager, *args, **kwargs)
            bound.apply_defaults()
            self.check_raw(per_frame_box, "pre_boxfusion", frame_id)
            pre = geometry_arrays(all_pred_box)
            call = {"call_id": len(self.calls), "frame_id": frame_id, "pre": pre,
                    "pre_fusion_lists": copy.deepcopy(box_manager.fusion_list),
                    "event_ids": [], "beta": float(bound.arguments["beta"])}
            if len(call["pre_fusion_lists"]) != len(pre["init_ids"]):
                raise RuntimeError("Pre-fusion rows and manager lists do not align")
            self.active_call = call
            try:
                result = original_fusion(fuser, all_pred_box, per_frame_box, box_manager, *args, **kwargs)
                self.check_raw(per_frame_box, "post_boxfusion", frame_id)
                post = geometry_arrays(all_pred_box)
                if not exact(pre["init_ids"], post["init_ids"]):
                    raise RuntimeError("Native PFO unexpectedly reordered identities")
                call["post"] = post
                call["post_fusion_lists"] = copy.deepcopy(box_manager.fusion_list)
                for event_id in call["event_ids"]:
                    event = self.events[event_id]
                    row = event["row"]
                    event["post_corners"] = post["corners"][row].copy()
                    event["post_box_xyzlhw"] = post["boxes_xyzlhw"][row].copy()
                    event["post_rotation"] = post["rotations"][row].copy()
                    event["updated"] = (not exact(pre["boxes_xyzlhw"][row], post["boxes_xyzlhw"][row])
                                        or not exact(pre["rotations"][row], post["rotations"][row]))
                self.calls.append(call)
                return result
            finally:
                self.active_call = None

        def evaluate(fuser, *args, **kwargs):
            caller = inspect.currentframe().f_back
            try:
                if caller.f_code is not original_fusion.__code__:
                    raise RuntimeError("evaluate_iou caller is not unmodified native boxfusion")
                local = caller.f_locals
                if int(local["n"]) == 0:
                    bound = evaluate_signature.bind(fuser, *args, **kwargs)
                    bound.apply_defaults()
                    self.capture_pfo(fuser, local, bound.arguments)
            finally:
                del caller
            return original_evaluate(fuser, *args, **kwargs)

        def save(data, filename):
            caller = inspect.currentframe().f_back
            try:
                if Path(caller.f_code.co_filename).resolve() != ROOT / "demo.py":
                    raise RuntimeError("save_box caller is not demo.run")
                self.capture_save(data, filename, caller.f_locals)
            finally:
                del caller
            if Path(filename).exists():
                raise FileExistsError(filename)
            # The original serializer receives the exact unmodified payload.
            return original_save(data, filename)

        for owner, name, replacement in ((BoxManager, "init_new_predictions", new_predictions),
                                          (BoxFusion, "boxfusion", fusion),
                                          (BoxFusion, "evaluate_iou", evaluate),
                                          (utils, "save_box", save)):
            self.patch_targets.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)

    def restore(self):
        for owner, name, original in reversed(self.patch_targets):
            setattr(owner, name, original)
        self.patch_targets.clear()

    def finish(self, config, source_hashes, input_hashes):
        if self.final is None or self.raw is None:
            raise RuntimeError("Demo exited without a verified terminal save / original observations")
        prediction = Path(self.final["prediction_path"])
        if not prediction.is_file() or prediction.stat().st_size == 0:
            raise RuntimeError("Original save_box did not produce a nonempty prediction")
        write_npz(self.run_dir / "observations.npz", self.raw)
        write_npz(self.run_dir / "birth_native_corners.npz", {
            "corners": np.concatenate(self.birth_native_corners, axis=0),
        })
        write_json(self.run_dir / "corner_numeric_diagnostics.json", self.corner_numeric_diagnostics)
        write_npz(self.run_dir / "final.npz", self.final["arrays"])
        write_json(self.run_dir / "fusion_events.json", self.events)
        write_json(self.run_dir / "fusion_calls.json", self.calls)
        write_json(self.run_dir / "raw_immutability.json", self.raw_checks)
        write_json(self.run_dir / "scene.json", {
            "schema": "boxfusion.true_fusion_capture.v1", "scene_id": self.scene, "completed": True,
            "raw_file": "observations.npz", "events_file": "fusion_events.json",
            "calls_file": "fusion_calls.json", "final_file": "final.npz",
            "pst_file": "PST.npy" if self.pst is not None else None,
            "final_fusion_lists": self.final["fusion_lists"],
            "final_frame_id": self.final["frame_id"],
            "raw_count": len(self.raw["init_ids"]), "final_count": len(self.final["arrays"]["init_ids"]),
            "fusion_calls": len(self.calls), "pfo_events": len(self.events),
            "raw_prefix_immutable": True, "saved_row_mapping_exact": True,
            "raw_corners_semantics": "fixed float64 CPU derivation from immutable xyzlhw/R, native corner order",
            "birth_gpu_corners_file": "birth_native_corners.npz",
            "derived_gpu_corner_rounding_is_not_state_mutation": True,
            "source_hashes_unchanged": True, "input_hashes_unchanged": True,
            "ground_truth_read": False, "production_files_modified": False,
            "capture_timing_is_production_fps": False,
            "ids_semantics": "actual per_frame_ins row indices; init_ids equality checked",
            "updated_semantics": "exact xyzlhw/R value change across the native boxfusion call",
            "source_ids_semantics": "original source_fusion_idx from real PFO caller locals",
            "selected_ids_semantics": "actual fusion_idx on first evaluate_iou call (n=0)",
            "prediction_file": str(prediction.relative_to(self.run_dir)),
            "prediction_sha256": digest(prediction), "effective_capture_config": config,
            "source_sha256": source_hashes, "input_sha256": input_hashes,
        })


def isolated_config(config, run_dir):
    result = copy.deepcopy(config)
    if result.get("dataset") != "CA1M" or not result.get("eval", False):
        raise ValueError("Capture requires an evaluated CA1M configuration")
    if not result.get("box_fusion", {}).get("use", False):
        raise ValueError("Capture requires native box_fusion.use=true")
    if (result.get("causal_dynamic_branch", {}) or {}).get("mode", "disabled") != "disabled":
        raise ValueError("Capture is for the static native baseline, not a dynamic branch")
    changes = [{"key": "data.output_dir", "old": result["data"]["output_dir"],
                "new": str(run_dir / "predictions")}]
    result["data"]["output_dir"] = str(run_dir / "predictions")

    def redirect(node, trail=()):
        for key, value in list(node.items()):
            if isinstance(value, dict):
                redirect(value, (*trail, key))
            elif isinstance(value, str) and ("diagnostic" in key or key == "events_root"):
                target = str(run_dir / "diagnostics" / "_".join((*trail, key)))
                changes.append({"key": ".".join((*trail, key)), "old": value, "new": target})
                node[key] = target

    redirect(result)
    return result, changes


def gather_hashes(args, config):
    datadir = Path(config["data"]["datadir"])
    if "example" in str(datadir):
        raise ValueError("Example datadir is not a CA1M sequence-selector root")
    scene_root = datadir.parent.parent / args.scene
    required = [scene_root / "K_depth.txt", scene_root / "K_rgb.txt", scene_root / "all_poses.npy"]
    for subdir in ("rgb", "depth"):
        frames = sorted((scene_root / subdir).glob("*.png"))
        if not frames:
            raise ValueError(f"No {subdir} input frames under {scene_root}")
        required.extend(frames)
    sources = [Path(__file__).resolve(), ROOT / "demo.py", ROOT / "tools/utils.py", args.config]
    sources.extend(sorted((ROOT / "boxfusion").glob("*.py")))
    model_files = [args.model_path, args.clip_path, args.class_txt, ROOT / "data/class_features.pt",
                   ROOT / config["box_fusion"]["pst_path"]]
    boxer = (config.get("lifting", {}).get("boxer", {}) or {})
    if config.get("lifting", {}).get("backend") == "boxer":
        model_files.append(Path(boxer["checkpoint"]))
        model_files.append(Path(boxer["official_root"]) / "ckpts" /
                           "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
    for path in (*sources, *required, *model_files):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return ({str(Path(p).resolve()): digest(p) for p in sources},
            {str(Path(p).resolve()): digest(p) for p in (*required, *model_files)}, scene_root.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/ca1m_thr15.yaml")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/cutr_rgbd.pth")
    parser.add_argument("--clip-path", type=Path, default=ROOT / "models/open_clip_pytorch_model.bin")
    parser.add_argument("--class-txt", type=Path, default=ROOT / "data/panoptic_categories_nomerge.txt")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args()
    if not args.scene.isdecimal():
        parser.error("--scene must be a numeric CA-1M scene ID")
    args.run_dir = args.run_dir.resolve()
    for key in ("config", "model_path", "clip_path", "class_txt"):
        setattr(args, key, getattr(args, key).resolve())
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite capture directory: {args.run_dir}")
    config = yaml.safe_load(args.config.read_text())
    isolated, changes = isolated_config(config, args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "predictions").mkdir()
    capture = Capture(args.scene, args.run_dir)
    old_argv, old_cwd = sys.argv[:], Path.cwd()
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        sys.addaudithook(forbid_gt)
        source_hashes, input_hashes, scene_root = gather_hashes(args, config)
        write_json(args.run_dir / "input_sha256.json", input_hashes)
        write_json(args.run_dir / "source_sha256.json", source_hashes)
        config_path = args.run_dir / "capture_config.yaml"
        with config_path.open("x") as handle:
            yaml.safe_dump(isolated, handle, sort_keys=False)
        config_hash = digest(config_path)
        write_json(args.run_dir / "protocol.json", {
            "schema": "boxfusion.true_fusion_capture.v1", "scene": args.scene,
            "data_root": str(scene_root), "configuration_changes": changes,
            "capture_config_sha256": config_hash, "ground_truth_allowed": False,
            "only_runtime_wrappers": ["BoxManager.init_new_predictions", "BoxFusion.boxfusion",
                                      "BoxFusion.evaluate_iou", "tools.utils.save_box"],
        })
        os.environ.setdefault("MPLCONFIGDIR", str(args.run_dir / "mplconfig"))
        capture.install()
        sys.argv = [str(ROOT / "demo.py"), "CA1M", "--config", str(config_path),
                    "--seq", args.scene, "--model-path", str(args.model_path),
                    "--clip_path", str(args.clip_path), "--class_txt", str(args.class_txt),
                    "--device", args.device]
        try:
            runpy.run_path(str(ROOT / "demo.py"), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        finally:
            capture.restore()
        changed = [path for path, expected in {**source_hashes, **input_hashes}.items()
                   if digest(path) != expected]
        if changed or digest(config_path) != config_hash:
            raise RuntimeError(f"Input/source mutated during capture: {changed}")
        capture.finish(isolated, source_hashes, input_hashes)
        print(f"TRUE_FUSION_CAPTURE_COMPLETE scene={args.scene} raw={len(capture.raw['init_ids'])} "
              f"events={len(capture.events)} final={len(capture.final['arrays']['init_ids'])}", flush=True)
    except BaseException as error:
        if not (args.run_dir / "failure.json").exists():
            write_json(args.run_dir / "failure.json", {
                "completed": False, "error": repr(error), "traceback": traceback.format_exc(),
                "raw_observations_registered": 0 if capture.raw is None else len(capture.raw["init_ids"]),
                "pfo_events_captured": len(capture.events),
            })
        raise
    finally:
        capture.restore()
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
