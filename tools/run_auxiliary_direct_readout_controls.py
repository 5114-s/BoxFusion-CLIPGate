#!/usr/bin/env python3
"""Current-only WeDetect-Uni/Boxer readout versus frozen recovery outputs.

No models, no GPU, no modifications to production inference. Existing causal
prefix exports supply identical native/HCSR and PLR/CALR reference components.
The new direct path has no temporal state and is read after every keyframe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev
from tools.bootstrap_final_recar_pairs import ap_from_pairs, scene_pairs
from tools.run_hcsr_current_zero_control import same_rows

core = ev.audit.core
DEFAULT = ROOT / "reports/scannet_auxiliary_direct_readout_20261102"
ARMS = (
    "base", "hcsr", "plr", "calr", "full", "memory256",
    "current_p_all", "current_p12", "current_a_all", "current_both_all",
    "current_p12_a_all", "replace_plr_current12", "replace_calr_current_all",
    "current_p_all_raw",
)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def current_rows(scene, frame, evidence):
    """Pure current-input function: no archive, confirmation or geometry edits."""
    porder = np.argsort(-evidence.proposal_scores, kind="stable")
    aorder = np.argsort(-evidence.anchor_scores, kind="stable")
    assert len(porder) <= 150 and len(aorder) <= 300
    proposals = [(0, evidence.proposal_corners[i].copy(),
                  float(core._size_price(evidence.proposal_corners[i]))) for i in porder]
    raw = [(0, evidence.proposal_corners[i].copy(),
            float(evidence.proposal_scores[i])) for i in porder]
    anchors = [(0, evidence.anchor_corners[i].copy(),
                core.deterministic_tail_score(float(evidence.anchor_scores[i]),
                    scene, frame, int(evidence.anchor_ids[i]))) for i in aorder]
    return proposals, proposals[:12], anchors, raw


def combine(native, proposals=(), anchors=()):
    return ev.case(list(native)+list(proposals)+list(anchors),
        ["native"]*len(native)+["proposal"]*len(proposals)+["anchor"]*len(anchors))


def self_test():
    from boxfusion.online_candidate_map import EvidenceFrame
    def evidence(xs):
        boxes = np.asarray([[[x+dx, y, z] for dx in (0., .4)
                             for y in (0., .4) for z in (0., .4)] for x in xs],
                           dtype=float).reshape(-1, 8, 3)
        return EvidenceFrame(proposal_ids=np.arange(len(xs)),
            proposal_boxes_2d=np.tile([0., 0., 10., 10.], (len(xs), 1)),
            proposal_corners=boxes, proposal_scores=np.full(len(xs), .7),
            anchor_ids=np.arange(len(xs)), anchor_corners=boxes,
            anchor_scores=np.full(len(xs), .7))
    original = evidence(list(range(15)))
    p, p12, a, raw = current_rows("test", 0, original)
    assert len(p) == len(a) == 15 and len(p12) == 12
    assert all(np.array_equal(p[i][1], original.proposal_corners[i]) for i in range(15))
    assert all(0.040001 <= row[2] <= .049999 for row in a)
    assert all(row[2] == .1 for row in p) and all(row[2] == .7 for row in raw)
    empty = current_rows("test", 25, evidence([]))
    assert all(len(rows) == 0 for rows in empty), "No old output may survive an empty current frame"
    # A second unrelated frame cannot be contaminated by a previous invocation.
    second = current_rows("test", 50, evidence([100]))
    assert second[0][0][1][:, 0].min() == 100
    assert same_rows(p, current_rows("test", 0, original)[0])
    ev.check_small_metrics()
    print("PASS: current-only, empty-frame clearing, preset top12, unchanged boxes, scores and metric tests", flush=True)


def materialize(out, protocol, scenes):
    parent = ROOT/protocol["parent_report"]
    previous = json.loads((parent/"results.json").read_text())
    factor = ROOT/protocol["factor_reference"]
    factor_manifest = json.loads((factor/"replay_manifest.json").read_text())
    inputs = {}
    prefixes = {str(f): {arm: {} for arm in ARMS} for f in protocol["prefix_fractions"]}
    trace_rows = []
    total_keyframes = 0
    started = time.perf_counter()
    for scene_index, scene in enumerate(scenes, 1):
        evidence_path = ROOT/protocol["evidence_root"]/f"{scene}.npz"
        native_path = ROOT/protocol["native_prefix_root"]/f"{scene}.npz"
        for path in (evidence_path, native_path):
            inputs[str(path)] = core.sha256(path)
            assert inputs[str(path)] == previous["input_sha256"][str(path)]
        evidence_frames = core.load_evidence(evidence_path)
        with np.load(native_path, allow_pickle=False) as z:
            frame_ids = z["frame_ids"].tolist()
        assert frame_ids == [f for f, _ in evidence_frames]
        assert len(frame_ids) == previous["per_scene"][scene]["keyframes"]
        checkpoints = {int(np.ceil(len(frame_ids)*f))-1: str(f)
                       for f in protocol["prefix_fractions"]}
        assert len(checkpoints) == 4
        reference_path = factor/"scenes"/f"{scene}.pkl"
        inputs[str(reference_path)] = core.sha256(reference_path)
        with reference_path.open("rb") as handle:
            reference = pickle.load(handle)
        assert reference["protocol_sha256"] == factor_manifest["protocol_sha256"]
        for ordinal, (frame, evidence) in enumerate(evidence_frames):
            p, p12, a, raw = current_rows(scene, frame, evidence)
            trace_rows.append({"scene": scene, "ordinal": ordinal, "frame": frame,
                "current_proposals": len(p), "current_top12_proposals": len(p12),
                "current_anchors": len(a), "persistent_direct_auxiliary_records": 0})
            if ordinal not in checkpoints:
                continue
            fraction = checkpoints[ordinal]
            tag = f"p{int(float(fraction)*100)}"
            loaded = {}
            for name in ("base", "full", "memory_a256"):
                path = parent/"predictions"/f"{name}_{tag}"/f"{scene}_boxes.pkl"
                inputs[str(path)] = core.sha256(path)
                loaded[name] = ev.audit.read_rows(path)
            counts = previous["per_scene"][scene]["keyframe_trace"][ordinal]["output_counts"]
            n, np_ = counts["full"]["native"], counts["full"]["proposal"]
            full = loaded["full"]
            native, plr, calr = full[:n], full[n:n+np_], full[n+np_:]
            assert len(loaded["base"]) == n
            assert all(np.array_equal(x[1], y[1]) for x, y in zip(native, loaded["base"]))
            assert same_rows(native+plr, reference["prefixes"][fraction]["p_track3_medoid"]["rows"])
            assert same_rows(native+calr, reference["prefixes"][fraction]["a3_k640"]["rows"])
            memory = loaded["memory_a256"]
            assert same_rows(native, memory[:n])
            mp = counts["memory_a256"]["proposal"]
            arms = {
                "base": combine(loaded["base"]), "hcsr": combine(native),
                "plr": combine(native, plr), "calr": combine(native, (), calr),
                "full": combine(native, plr, calr),
                "memory256": combine(native, memory[n:n+mp], memory[n+mp:]),
                "current_p_all": combine(native, p), "current_p12": combine(native, p12),
                "current_a_all": combine(native, (), a),
                "current_both_all": combine(native, p, a),
                "current_p12_a_all": combine(native, p12, a),
                "replace_plr_current12": combine(native, p12, calr),
                "replace_calr_current_all": combine(native, plr, a),
                "current_p_all_raw": combine(native, raw),
            }
            assert set(arms) == set(ARMS)
            for arm, value in arms.items():
                expected = loaded["base"] if arm == "base" else native
                assert same_rows(value["rows"][:n], expected)
                prefixes[fraction][arm][scene] = copy.deepcopy(value)
                core.write_prediction(out/"predictions"/f"{arm}_{tag}"/f"{scene}_boxes.pkl", value["rows"])
        total_keyframes += len(frame_ids)
        if scene_index % 10 == 0:
            print(f"materialized {scene_index}/100 scenes ({total_keyframes} keyframes)", flush=True)
    assert total_keyframes == 6817
    manifest = {"protocol_sha256": core.sha256(out/"protocol.json"),
        "input_sha256": inputs, "scene_count": len(scenes), "keyframes": total_keyframes,
        "materialization_seconds": time.perf_counter()-started,
        "shared_native_order_geometry_scores_checked_all_arms_prefixes": True,
        "full_matches_existing_single_factor_reference_all_prefixes": True,
        "model_inference": False, "gpu_used": False,
        "source_sha256": {str(path): core.sha256(path) for path in (Path(__file__),
            core.SOURCE/"boxfusion/m1_anchor_online.py", core.SOURCE/"boxfusion/online_candidate_map.py")}}
    write_csv(out/"current_keyframe_readout_trace.csv", trace_rows)
    ev.save_json(out/"materialization_manifest.json", manifest)
    return prefixes, manifest


def evaluate(out, protocol, scenes, prefixes, manifest):
    with contextlib.redirect_stdout(io.StringIO()):
        gts, alignments, transform = core.scannet_inputs()
    assert sum(len(gts[s]) for s in scenes) == 1433
    reference = json.loads((ROOT/protocol["parent_report"]/"results.json").read_text())
    factor = json.loads((ROOT/protocol["factor_reference"]/"results.json").read_text())
    points, terminal, workpoints = {}, {}, {}
    ap_rows, counts_rows, finite_rows, quality_rows = [], [], [], []
    for fraction, arms in prefixes.items():
        points[fraction] = {}
        for arm in ARMS:
            values = arms[arm]
            transformed = {s: ev.as_prediction(values[s], s, alignments, transform) for s in scenes}
            plain = {s: (v[0], v[1]) for s, v in transformed.items()}
            counts = Counter(source for v in values.values() for source in v["sources"])
            stable, legacy = {}, {}
            for threshold in protocol["iou_thresholds"]:
                key = str(threshold)
                stable[key] = ap_from_pairs(np.concatenate([
                    scene_pairs(*plain[s], gts[s], threshold) for s in scenes]), 1433)
                legacy[key] = core.class_agnostic_ap(plain, gts, threshold)["ap"]
            points[fraction][arm] = {"counts": dict(counts), "total": sum(counts.values()),
                                    "stable_ap": stable, "legacy_ap": legacy}
            ap_rows.append({"prefix": fraction, "arm": arm, "native": counts["native"],
                "proposal": counts["proposal"], "anchor": counts["anchor"],
                "total": sum(counts.values()),
                **{f"AP{int(t*100)}": stable[str(t)] for t in protocol["iou_thresholds"]},
                **{f"legacy_AP{int(t*100)}": legacy[str(t)] for t in protocol["iou_thresholds"]}})
            # Reproduce the existing reference AP; do not silently mix runs.
            if arm in ("base", "full", "memory256"):
                old_arm = "memory_a256" if arm == "memory256" else arm
                for t in protocol["iou_thresholds"]:
                    assert abs(legacy[str(t)]-reference["prefix_ap"][fraction][old_arm][str(t)]) < 1e-9
            if arm in ("plr", "calr"):
                old_arm = "p_track3_medoid" if arm == "plr" else "a3_k640"
                for t in protocol["iou_thresholds"]:
                    assert abs(stable[str(t)]-factor["prefix_ap"][fraction][old_arm]["stable_ap"][str(t)]) < 1e-9
            if fraction != "1.0":
                continue
            assert counts["native"] == 1787
            terminal[arm] = plain
            for scene, value in values.items():
                c = Counter(value["sources"])
                counts_rows.append({"scene": scene, "arm": arm, "native": c["native"],
                    "proposal": c["proposal"], "anchor": c["anchor"], "total": len(value["rows"])})
            metrics, curves, _ = ev.workpoints(transformed, gts,
                thresholds=tuple(protocol["iou_thresholds"]),
                topks=tuple(protocol["top_k_per_scene"]), fp_budgets=tuple(protocol["fp_per_scene"]))
            workpoints[arm] = metrics
            np.savez_compressed(out/f"pr_{arm}.npz", **{
                f"iou{t}_{k}": value for t, current in curves.items() for k, value in current.items()})
            for t, current in metrics.items():
                assert abs(current["stable_ap"]-stable[t]) < 1e-9
                for source, row in (("all", current["all_outputs"]), *current["per_source"].items()):
                    assert row["predictions"] == row["tp"]+row["fp"]
                    assert row["within_duplicate_fp"]+row["cross_duplicate_fp"] <= row["fp"]
                    quality_rows.append({"arm": arm, "iou": t, "source": source, **row})
                for kind, key in (("TopK", "topk"), ("oracle_FP_per_scene", "fixed_fp_per_scene")):
                    for budget, row in current[key].items():
                        finite_rows.append({"arm": arm, "iou": t, "type": kind, "budget": budget, **row})
        print(f"evaluated prefix {fraction} ({len(ARMS)} arms)", flush=True)
    for name, rows in (("prefix_ap.csv", ap_rows), ("terminal_output_counts.csv", counts_rows),
                       ("finite_workpoints.csv", finite_rows), ("tp_fp_duplicates.csv", quality_rows)):
        write_csv(out/name, rows)
    result = {"protocol": protocol, "protocol_sha256": manifest["protocol_sha256"],
        "scene_count": 100, "gt_count": 1433, "native_count": 1787, "keyframes": 6817,
        "prefix_results": points, "terminal_workpoints": workpoints,
        "validation": {"status": "PASS", "matched_native_geometry_scores_order": True,
            "reference_AP_reproduced_all_prefixes": True, "all_terminal_native_counts_1787": True,
            "TP_FP_and_duplicate_accounting_checked": True, "current_only_self_tests": True,
            "model_inference": False, "gpu_used": False}}
    ev.save_json(out/"point_results.json", result)
    print("Point estimates saved; starting 2000 paired scene bootstrap replicates", flush=True)
    comparisons = protocol["bootstrap_comparisons"]
    intervals = ev.audit.bootstrap(terminal, gts, scenes, comparisons,
        protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    ci_rows = []
    for name, (left, right) in comparisons.items():
        for t in protocol["iou_thresholds"]:
            key = str(t)
            row = intervals[name][key]
            row["point_ap"] = points["1.0"][left]["stable_ap"][key]-points["1.0"][right]["stable_ap"][key]
            row["legacy_point_ap"] = points["1.0"][left]["legacy_ap"][key]-points["1.0"][right]["legacy_ap"][key]
            ci_rows.append({"contrast": name, "iou": t, "point_AP": row["point_ap"],
                "CI95_low": row["ci95_ap"][0], "CI95_high": row["ci95_ap"][1],
                "p_positive": row["p_positive"]})
    result["paired_bootstrap"] = intervals
    write_csv(out/"paired_bootstrap.csv", ci_rows)
    ev.save_json(out/"results.json", result)
    ev.save_json(out/"validation.json", result["validation"])
    print(json.dumps({"status": "complete", "output": str(out),
        "terminal": {arm: points["1.0"][arm] for arm in ARMS}}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test_only:
        return
    out = args.output
    if (out/"results.json").exists():
        raise RuntimeError("Completed output exists; refusing to overwrite")
    protocol = json.loads((out/"protocol.json").read_text())
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines()
              if s.strip() and not s.startswith("#")]
    assert len(scenes) == len(set(scenes)) == 100
    prefixes, manifest = materialize(out, protocol, scenes)
    evaluate(out, protocol, scenes, prefixes, manifest)


if __name__ == "__main__":
    main()
