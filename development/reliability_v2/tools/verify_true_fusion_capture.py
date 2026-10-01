#!/usr/bin/env python3
"""Read-only consistency verification for completed true-fusion captures.

This checks saved evidence against itself and current source files. It does not
replay inference, prove continuous file immutability, or rehash every original
RGB-D frame/model weight. Pickles must be locally generated trusted captures.
Only the explicitly new receipt JSON is written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import yaml


RAW_KEYS = ("corners", "boxes_xyzlhw", "rotations", "cam_poses", "scores",
            "boxes2d", "projected_boxes", "frame_ids", "init_ids")
GEOMETRY_KEYS = ("corners", "boxes_xyzlhw", "rotations", "scores", "init_ids")
HEX = set("0123456789abcdef")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def integer(value, name, *, minimum=0):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
            f"{name}: expected an integer")
    require(int(value) >= minimum, f"{name}: below {minimum}")
    return int(value)


def indices(values, count, name):
    require(isinstance(values, (list, tuple, np.ndarray)), f"{name}: not an index list")
    result = [integer(value, name) for value in values]
    require(all(value < count for value in result), f"{name}: outside 0..{count - 1}")
    return result


def finite(value, name, *, shape=None):
    array = np.asarray(value)
    require(array.dtype.kind in "iuf", f"{name}: nonnumeric dtype")
    require(np.isfinite(array).all(), f"{name}: nonfinite values")
    if shape is not None:
        require(array.shape == shape, f"{name}: {array.shape}, expected {shape}")
    return array


def same(left, right, name):
    require(np.array_equal(np.asarray(left), np.asarray(right)), f"{name}: values/order mismatch")


def canonical_corners(boxes, rotations):
    """Independently derive corners in the released native sign convention."""
    boxes = np.asarray(boxes, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    signs = np.asarray([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
    local = boxes[:, None, 3:] * signs[None] * 0.5
    return np.sum(rotations[:, None, :, :] * local[:, :, None, :], axis=-1) + boxes[:, None, :3]


class Evidence:
    def __init__(self):
        self.read_hashes = {}

    def remember(self, path):
        path = Path(path).resolve(strict=True)
        require(path.is_file(), f"Not a regular evidence file: {path}")
        current = digest(path)
        previous = self.read_hashes.setdefault(str(path), current)
        require(previous == current, f"Evidence changed between reads: {path}")
        return path

    def json(self, path):
        return json.loads(self.remember(path).read_text())

    def npz(self, path):
        with np.load(self.remember(path), allow_pickle=False) as source:
            return {key: source[key] for key in source.files}

    def unchanged(self):
        for path, expected in self.read_hashes.items():
            require(digest(path) == expected, f"Evidence/source changed during verification: {path}")


def artifact(directory, name):
    require(isinstance(name, str) and name, "Missing artifact filename")
    target = (directory / name).resolve(strict=True)
    require(target.is_relative_to(directory), f"Artifact escapes scene directory: {name}")
    return target


def hash_manifest(value, name):
    require(isinstance(value, dict) and bool(value), f"{name}: empty/malformed hash manifest")
    result = {}
    for filename, expected in value.items():
        require(isinstance(filename, str) and Path(filename).is_absolute(), f"{name}: nonabsolute path")
        require(isinstance(expected, str) and len(expected) == 64 and set(expected) <= HEX,
                f"{name}: invalid SHA256 for {filename}")
        canonical = str(Path(filename).resolve())
        require(canonical not in result, f"{name}: duplicate resolved path {canonical}")
        result[canonical] = expected
    return result


def geometry(payload, count, name, *, json_input=False):
    require(isinstance(payload, dict), f"{name}: not a geometry dictionary")
    result = {}
    shapes = {"corners": (count, 8, 3), "boxes_xyzlhw": (count, 6),
              "rotations": (count, 3, 3), "scores": (count,), "init_ids": (count,)}
    for key in GEOMETRY_KEYS:
        require(key in payload, f"{name}: missing {key}")
        array = np.asarray(payload[key])
        if json_input and count == 0 and array.shape == (0,):
            array = array.reshape(shapes[key])
        result[key] = finite(array, f"{name}.{key}", shape=shapes[key])
    for row_id in result["init_ids"]:
        integer(row_id, f"{name}.init_ids")
    for key, value in payload.items():
        require(np.asarray(value).ndim >= 1 and len(value) == count,
                f"{name}.{key}: unaligned array length")
    return result
def verify_scene(run_dir, scene, evidence):
    directory = (run_dir / scene).resolve(strict=True)
    require(directory.is_relative_to(run_dir), f"{scene}: scene outside run directory")
    manifest = evidence.json(directory / "scene.json")
    require(manifest.get("schema") == "boxfusion.true_fusion_capture.v1", f"{scene}: unsupported schema")
    require(manifest.get("scene_id") == scene, f"{scene}: scene ID mismatch")
    for key in ("completed", "raw_prefix_immutable", "saved_row_mapping_exact",
                "source_hashes_unchanged", "input_hashes_unchanged",
                "derived_gpu_corner_rounding_is_not_state_mutation"):
        require(manifest.get(key) is True, f"{scene}: {key} is not true")
    for key in ("ground_truth_read", "production_files_modified", "capture_timing_is_production_fps"):
        require(manifest.get(key) is False, f"{scene}: {key} is not false")
    require(manifest.get("raw_corners_semantics") ==
            "fixed float64 CPU derivation from immutable xyzlhw/R, native corner order",
            f"{scene}: unsupported raw corner semantics")

    source_hashes = hash_manifest(manifest["source_sha256"], f"{scene}.source_sha256")
    input_hashes = hash_manifest(manifest["input_sha256"], f"{scene}.input_sha256")
    require(source_hashes == hash_manifest(evidence.json(directory / "source_sha256.json"), "source ledger"),
            f"{scene}: source ledgers disagree")
    require(input_hashes == hash_manifest(evidence.json(directory / "input_sha256.json"), "input ledger"),
            f"{scene}: input ledgers disagree")
    require(not any("after_filter_boxes.npy" in path or "full_annotations.json" in path
                    for path in input_hashes), f"{scene}: GT annotation present in inference input ledger")

    protocol = evidence.json(directory / "protocol.json")
    require(protocol.get("scene") == scene and protocol.get("ground_truth_allowed") is False,
            f"{scene}: capture protocol mismatch")
    config_path = evidence.remember(directory / "capture_config.yaml")
    require(digest(config_path) == protocol["capture_config_sha256"], f"{scene}: capture config hash mismatch")
    config = yaml.safe_load(config_path.read_text())
    require(config == manifest["effective_capture_config"], f"{scene}: effective configuration mismatch")
    normalized_config = copy.deepcopy(config)
    for change in protocol["configuration_changes"]:
        keys = change["key"].split(".")
        require(change["key"] == "data.output_dir" or "diagnostic" in keys[-1] or keys[-1] == "events_root",
                f"{scene}: undocumented non-output configuration change")
        node = normalized_config
        for key in keys[:-1]:
            node = node[key]
        require(node[keys[-1]] == change["new"], f"{scene}: redirect manifest mismatch")
        node[keys[-1]] = change["old"]
    original_configs = [path for path in source_hashes if Path(path).suffix in (".yaml", ".yml")]
    require(len(original_configs) == 1, f"{scene}: cannot identify unique original config")
    original_config = yaml.safe_load(evidence.remember(original_configs[0]).read_text())
    require(original_config == normalized_config, f"{scene}: non-output config changed during capture")

    raw = evidence.npz(artifact(directory, manifest["raw_file"]))
    count = integer(manifest["raw_count"], "raw_count")
    require(set(RAW_KEYS) <= set(raw), f"{scene}: missing raw arrays")
    for key, array in raw.items():
        finite(array, f"raw.{key}")
        require(array.ndim >= 1 and len(array) == count, f"{scene}: raw.{key} row count mismatch")
    for key, shape in {"corners": (count, 8, 3), "boxes_xyzlhw": (count, 6),
                       "rotations": (count, 3, 3), "cam_poses": (count, 4, 4),
                       "scores": (count,), "frame_ids": (count,), "init_ids": (count,)}.items():
        finite(raw[key], f"raw.{key}", shape=shape)
    same(raw["init_ids"], np.arange(count), f"{scene}: raw observation IDs")
    for value in (*raw["frame_ids"], *raw["init_ids"]):
        integer(value, "raw frame/observation ID")
    require(np.all(np.diff(raw["frame_ids"]) >= 0), f"{scene}: raw frames out of order")
    expected_corners = canonical_corners(raw["boxes_xyzlhw"], raw["rotations"])
    require(raw["corners"].dtype == np.float64, f"{scene}: canonical corners not float64")
    same(raw["corners"], expected_corners, f"{scene}: CPU canonical corner formula")
    final_frame = integer(manifest["final_frame_id"], "final_frame_id")
    require(np.all(raw["frame_ids"] <= final_frame), f"{scene}: raw observations after terminal frame")

    birth = evidence.npz(artifact(directory, manifest["birth_gpu_corners_file"]))
    birth_corners = finite(birth["corners"], "birth_gpu_corners", shape=(count, 8, 3))
    numeric = evidence.json(directory / "corner_numeric_diagnostics.json")
    require(isinstance(numeric, list), f"{scene}: malformed numeric diagnostics")
    # These derived GPU corners may differ due to growing-batch rounding.
    max_birth_error = float(np.max(np.abs(birth_corners - expected_corners))) if count else 0.0

    immutable = evidence.json(directory / "raw_immutability.json")
    require(isinstance(immutable, list) and bool(immutable), f"{scene}: missing immutability checks")
    previous_count = 0
    registered = 0
    call_boundaries = {"pre_boxfusion": [], "post_boxfusion": []}
    for item in immutable:
        require(item.get("immutable") is True, f"{scene}: raw immutability check failed")
        current_count = integer(item["raw_count"], "immutability.raw_count")
        frame = integer(item["frame_id"], "immutability.frame_id")
        require(previous_count <= current_count <= count, f"{scene}: non-prefix raw count")
        previous_count = current_count
        if item["boundary"] == "new_observation_registration":
            new_count = integer(item["new_count"], "registration.new_count")
            require(registered + new_count == current_count, f"{scene}: registration count mismatch")
            require(np.all(raw["frame_ids"][registered:current_count] == frame),
                    f"{scene}: registration frame mismatch")
            registered = current_count
        elif item["boundary"] in call_boundaries:
            call_boundaries[item["boundary"]].append((frame, current_count))
        elif item["boundary"] == "pre_new_registration":
            require(current_count == registered == np.count_nonzero(raw["frame_ids"] < frame),
                    f"{scene}: incorrect past-only prefix before new registration")
        else:
            require(item["boundary"] == "terminal_save", f"{scene}: unknown immutability boundary")
    require(registered == count and immutable[-1]["boundary"] == "terminal_save"
            and immutable[-1]["raw_count"] == count and immutable[-1]["frame_id"] == final_frame,
            f"{scene}: incomplete terminal raw checks")

    final_data = evidence.npz(artifact(directory, manifest["final_file"]))
    final_count = integer(manifest["final_count"], "final_count")
    final = geometry(final_data, final_count, "final")
    native_rows = finite(final_data["native_row_indices"], "native_row_indices", shape=(final_count,))
    for value in native_rows:
        integer(value, "native row index")
    require(np.all(np.diff(native_rows) > 0), f"{scene}: terminal row filter order mismatch")
    final_lists = manifest["final_fusion_lists"]
    require(len(final_lists) == final_count, f"{scene}: terminal membership length mismatch")
    for item in final_lists:
        indices(item, count, "final membership")
    prediction_path = evidence.remember(artifact(directory, manifest["prediction_file"]))
    require(digest(prediction_path) == manifest["prediction_sha256"], f"{scene}: prediction hash mismatch")
    with prediction_path.open("rb") as stream:
        predictions = pickle.load(stream)
        require(not stream.read(1), f"{scene}: trailing pickle data")
    require(isinstance(predictions, list) and len(predictions) == 1 and len(predictions[0]) == final_count,
            f"{scene}: prediction batch/count mismatch")
    for index, row in enumerate(predictions[0]):
        require(len(row) == 3 and row[0] == 0, f"{scene}: non-class-agnostic prediction row")
        same(row[1], final["corners"][index], f"{scene}: final pickle corners row {index}")
        same(row[2], final["scores"][index], f"{scene}: final pickle score row {index}")

    events = evidence.json(artifact(directory, manifest["events_file"]))
    calls = evidence.json(artifact(directory, manifest["calls_file"]))
    require(isinstance(events, list) and len(events) == integer(manifest["pfo_events"], "pfo_events"),
            f"{scene}: PFO event count mismatch")
    require(isinstance(calls, list) and len(calls) == integer(manifest["fusion_calls"], "fusion_calls"),
            f"{scene}: fusion call count mismatch")
    require([integer(event["event_id"], "event_id") for event in events] == list(range(len(events))),
            f"{scene}: missing/duplicate/out-of-order event IDs")
    require([integer(call["call_id"], "call_id") for call in calls] == list(range(len(calls))),
            f"{scene}: missing/duplicate/out-of-order call IDs")
    frame_calls = [integer(call["frame_id"], "call.frame_id") for call in calls]
    require(frame_calls == sorted(frame_calls) and all(value <= final_frame for value in frame_calls),
            f"{scene}: noncausal call chronology")
    for boundary, values in call_boundaries.items():
        require([frame for frame, _ in values] == frame_calls, f"{scene}: missing {boundary} proof records")
        for frame, raw_count in values:
            require(raw_count == np.count_nonzero(raw["frame_ids"] <= frame),
                    f"{scene}: incorrect raw prefix at {boundary}")
    referenced_events = []
    for call in calls:
        call_id = call["call_id"]
        pre_lists, post_lists = call["pre_fusion_lists"], call["post_fusion_lists"]
        pre = geometry(call["pre"], len(pre_lists), f"call{call_id}.pre", json_input=True)
        post = geometry(call["post"], len(post_lists), f"call{call_id}.post", json_input=True)
        same(pre["init_ids"], post["init_ids"], f"{scene}: native identity order across call {call_id}")
        same(pre["scores"], post["scores"], f"{scene}: score invariance across native PFO")
        for memberships in (pre_lists, post_lists):
            for item in memberships:
                group = indices(item, count, "call membership")
                require(np.all(raw["frame_ids"][group] <= call["frame_id"]),
                        f"{scene}: future call membership")
        call_events = indices(call["event_ids"], len(events), "call event IDs")
        require(len(set(call_events)) == len(call_events), f"{scene}: duplicate event within call")
        referenced_events.extend(call_events)
        seen_rows = set()
        for event_id in call_events:
            event = events[event_id]
            require(event["call_id"] == call_id and event["frame_id"] == call["frame_id"],
                    f"{scene}: event parent call/frame mismatch")
            require(event["beta"] == call["beta"], f"{scene}: event/call beta mismatch")
            row = integer(event["row"], "event.row")
            require(row < len(pre_lists) and row not in seen_rows, f"{scene}: invalid/duplicate PFO row")
            seen_rows.add(row)
            require(event["init_id"] == pre["init_ids"][row], f"{scene}: event native init ID mismatch")
            require(event["source_ids"] == pre_lists[row], f"{scene}: event/source parent membership mismatch")
            source = indices(event["source_ids"], count, "event source")
            selected = indices(event["selected_ids"], count, "event selected")
            require(bool(selected) and set(selected) <= set(source), f"{scene}: invalid selected input")
            require(len(selected) == integer(event["num_of_boxes"], "num_of_boxes"),
                    f"{scene}: event PFO input count mismatch")
            require(np.all(raw["frame_ids"][source] <= event["frame_id"]), f"{scene}: future PFO source")
            same(event["source_frame_ids"], raw["frame_ids"][source], "event source frame IDs")
            same(event["selected_frame_ids"], raw["frame_ids"][selected], "event selected frame IDs")
            for prefix, parent in (("pre", pre), ("post", post)):
                for event_key, parent_key in (("corners", "corners"), ("box_xyzlhw", "boxes_xyzlhw"),
                                              ("rotation", "rotations")):
                    same(event[f"{prefix}_{event_key}"], parent[parent_key][row],
                         f"{scene}: event {event_id} {prefix}.{parent_key}")
            updated = (not np.array_equal(pre["boxes_xyzlhw"][row], post["boxes_xyzlhw"][row])
                       or not np.array_equal(pre["rotations"][row], post["rotations"][row]))
            require(event["updated"] is updated, f"{scene}: event updated flag mismatch")
            for name, shape in (("pfo_init_box_xyzlhw", (6,)), ("pfo_rotation", (3, 3)),
                                ("search_size", (6,)),
                                ("scores_box", (len(selected),)), ("view_weights", (len(selected),))):
                finite(event[name], f"event.{name}", shape=shape)
            intrinsic = finite(event["K"], "event.K")
            require(intrinsic.shape in ((3, 3), (4, 4)), f"{scene}: invalid native camera intrinsics shape")
            expected_scores = event["view_weights"] if event["use_view_weights"] else raw["scores"][selected]
            same(event["scores_box"], expected_scores, f"{scene}: actual PFO score/weight input")
    require(sorted(referenced_events) == list(range(len(events))), f"{scene}: orphan/multiply-owned event")

    pst_sha = None
    if events:
        require(manifest.get("pst_file"), f"{scene}: PFO events without PST")
        pst_path = evidence.remember(artifact(directory, manifest["pst_file"]))
        pst_sha = digest(pst_path)
        pst = finite(np.load(pst_path, allow_pickle=False), "PST")
        require(pst.ndim == 2 and pst.shape[1] == 6, f"{scene}: PST shape mismatch")
        for event in events:
            require(event["pst_file"] == manifest["pst_file"], f"{scene}: mismatched event PST file")
            require(0 < integer(event["pst_size"], "pst_size") <= len(pst), f"{scene}: PST size mismatch")
            if "pst_sha256" in event:
                require(event["pst_sha256"] == pst_sha, f"{scene}: event PST hash mismatch")
        if "pst_sha256" in manifest:
            require(manifest["pst_sha256"] == pst_sha, f"{scene}: manifest PST hash mismatch")
    else:
        require(manifest.get("pst_file") is None, f"{scene}: PST declared without captured PFO event")
    return ({"scene": scene, "checkpass": True, "raw_observations": count,
             "final_rows": final_count, "pfo_events": len(events), "fusion_calls": len(calls),
             "immutable_records": len(immutable), "numeric_diagnostic_records": len(numeric),
             "birth_gpu_vs_cpu_max_abs_corner_error": max_birth_error,
             "canonical_raw_corner_formula_exact": True, "final_pickle_parity_exact": True,
             "pst_sha256": pst_sha}, source_hashes, input_hashes, normalized_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--scenes", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(len(args.scenes) == len(set(args.scenes)) and all(scene.isdecimal() for scene in args.scenes),
            "Expected unique numeric scene IDs")
    require(not args.output.exists(), f"Refusing to overwrite receipt: {args.output}")
    run_dir = args.run_dir.resolve(strict=True)
    evidence = Evidence()
    root_protocol = evidence.json(run_dir / "protocol.json")
    require(root_protocol.get("schema") == "boxfusion.true_fusion_pilot.v1", "Unsupported run protocol")
    require(all(scene in root_protocol["scenes"] for scene in args.scenes), "Scenes outside sealed run protocol")
    sources, input_hashes, normalized, results, pst_hashes = None, {}, None, [], set()
    for scene in args.scenes:
        result, scene_sources, scene_inputs, config = verify_scene(run_dir, scene, evidence)
        if sources is None:
            sources, normalized = scene_sources, config
        else:
            require(scene_sources == sources, f"{scene}: cross-scene source set/hash mismatch")
            require(config == normalized, f"{scene}: cross-scene non-output configuration mismatch")
        for path, expected in scene_inputs.items():
            require(input_hashes.setdefault(path, expected) == expected, f"Cross-scene shared input changed: {path}")
        if result["pst_sha256"] is not None:
            pst_hashes.add(result["pst_sha256"])
        results.append(result)
        print(f"VERIFIED scene={scene} raw={result['raw_observations']} events={result['pfo_events']} "
              f"final={result['final_rows']}", flush=True)
    require(len(pst_hashes) <= 1, "PST files differ across same-config scenes")
    current_sources = dict(sources or {})
    for path, expected in hash_manifest(root_protocol["input_sha256"], "run sealed source ledger").items():
        require(current_sources.setdefault(path, expected) == expected, f"Run/scene source hash disagreement: {path}")
    for path, expected in current_sources.items():
        evidence.remember(path)
        require(evidence.read_hashes[str(Path(path).resolve())] == expected, f"Current source no longer matches capture: {path}")
    evidence.remember(Path(__file__))
    evidence.unchanged()
    receipt = {
        "schema": "boxfusion.true_fusion_capture_verification.v1", "checkpass": True,
        "run_dir": str(run_dir), "scenes": args.scenes,
        "matches_complete_planned_scene_list": args.scenes == root_protocol["scenes"],
        "per_scene": results, "cross_scene_source_hashes_equal": True,
        "cross_scene_configuration_equal": True, "current_sources_match_capture_hashes": True,
        "read_artifacts_and_sources_unchanged_during_verification": True,
        "original_frame_and_model_files_rehashed": False,
        "original_input_hashes": input_hashes, "input_sha256": evidence.read_hashes,
        "limitations": [
            "Checks consistency of persisted capture evidence, not independent execution replay.",
            "Raw immutability/no-GT-read flags and boundary logs are capture assertions, not continuous monitoring proof.",
            "Every source file is checked now; original RGB-D/model files are not rehashed by this verifier.",
            "GPU-derived birth corners may differ numerically; frozen raw xyzlhw/R define canonical float64 corners.",
            "PFO initialization/weights are captured inputs; this tool does not independently replay Top-K or optimization.",
            "A partial scene receipt does not certify the complete planned pilot.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"TRUE_FUSION_CAPTURE_VERIFICATION_PASS scenes={len(results)} receipt={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"TRUE_FUSION_CAPTURE_VERIFICATION_FAILED: {error}", file=sys.stderr, flush=True)
        raise
