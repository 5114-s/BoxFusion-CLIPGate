#!/usr/bin/env python3
"""Eight module settings with an always-present auxiliary candidate readout.

When PLR is off, read current proposal Top-12 directly. When CALR is off,
read all current anchors directly. When HCSR is off, use native raw scores.
Recombine verified same-cache components; do not run models or use a GPU.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev
from tools.bootstrap_final_recar_pairs import ap_from_pairs, scene_pairs
from tools.run_hcsr_current_zero_control import same_rows

SOURCE = ROOT / "reports/scannet_auxiliary_direct_readout_20261102"
OUTPUT = ROOT / "reports/scannet_always_aux_factorial_20261102"
SETTINGS = (
    ("auxiliary_direct", 0, 0, 0),
    ("hcsr", 1, 0, 0),
    ("hcsr_plr", 1, 1, 0),
    ("hcsr_calr", 1, 0, 1),
    ("plr", 0, 1, 0),
    ("calr", 0, 0, 1),
    ("plr_calr", 0, 1, 1),
    ("full", 1, 1, 1),
)


def main():
    if (OUTPUT/"results.json").exists():
        raise RuntimeError("Refusing to overwrite completed results")
    reference = json.loads((SOURCE/"results.json").read_text())
    assert reference["validation"]["status"] == "PASS"
    scenes = [s.strip() for s in (ROOT/reference["protocol"]["scene_list"]).read_text().splitlines()
              if s.strip() and not s.startswith("#")]
    assert len(scenes) == len(set(scenes)) == 100
    with contextlib.redirect_stdout(io.StringIO()):
        gt, alignment, transform = ev.audit.core.scannet_inputs()
    gt_count = sum(len(gt[s]) for s in scenes)
    assert gt_count == 1433
    protocol = {
        "schema": "recar3d.always_auxiliary_factorial.v1",
        "client_date": "2026-11-02",
        "source_report": str(SOURCE),
        "status": "Requested summary after inspecting previous direct-readout and removal results; no new thresholds, scoring changes, or model runs",
        "hcsr_off": "Original native scores; native geometry and order unchanged",
        "plr_off": "Current committed keyframe proposal Top-12 directly read out, no temporal recovery state",
        "calr_off": "All current committed keyframe anchors directly read out (300 per scene here), no temporal recovery state",
        "auxiliary_frontend": "Same frozen WeDetect-Uni/Boxer evidence source always contributes auxiliary output, including the all-off setting",
        "output_scoring": "Same frozen proposal size-bucket and anchor deterministic tail mapping in every setting",
        "assembly": "native + proposal + anchor, without new global NMS or cross-branch deduplication",
        "matching": "Strict AABB IoU > threshold and score-sorted one-to-one best-GT matching",
        "metrics": "Principal AP uses deterministic stable sorting; legacy evaluator AP retained separately",
        "budget": "Preset Top-12 current proposals and all current anchors; actual counts not matched to historical recovery outputs",
        "scope": "100%-prefix/terminal component readout of the same frozen run, not separate front-end inference runs",
        "note": "Not the earlier off-means-no-auxiliary-output module matrix; does not establish superiority over persistent simple memory",
        "gpu_used": False, "new_model_inference": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ev.save_json(OUTPUT/"protocol.json", protocol)
    predictions = {name: {} for name, _, _, _ in SETTINGS}
    counts = {name: {"native": 0, "proposal": 0, "anchor": 0} for name in predictions}
    scene_counts = []
    hashes = {str(SOURCE/"results.json"): ev.audit.core.sha256(SOURCE/"results.json")}
    for scene in scenes:
        loaded = {}
        for name in ("base", "hcsr", "plr", "calr", "current_p12", "current_a_all", "full"):
            path = SOURCE/"predictions"/f"{name}_p100"/f"{scene}_boxes.pkl"
            hashes[str(path)] = ev.audit.core.sha256(path)
            loaded[name] = ev.audit.read_rows(path)
        n = len(loaded["base"])
        assert len(loaded["hcsr"]) == n
        assert all(np.array_equal(a[1], b[1]) for a, b in zip(loaded["base"], loaded["hcsr"]))
        for name in ("plr", "calr", "current_p12", "current_a_all", "full"):
            assert same_rows(loaded[name][:n], loaded["hcsr"])
        for name, h, p, a in SETTINGS:
            native = loaded["hcsr"] if h else loaded["base"]
            proposals = loaded["plr" if p else "current_p12"][n:]
            anchors = loaded["calr" if a else "current_a_all"][n:]
            rows = native+proposals+anchors
            if name == "full":
                assert same_rows(rows, loaded["full"])
            predictions[name][scene] = ev.audit.core.prediction(rows, transform, alignment[scene])
            c = {"native": n, "proposal": len(proposals), "anchor": len(anchors)}
            for source, value in c.items():
                counts[name][source] += value
            scene_counts.append({"scene": scene, "setting": name, **c, "total": len(rows)})
    results, table = {}, []
    reference_arms = {"hcsr": "current_p12_a_all", "hcsr_plr": "replace_calr_current_all",
                      "hcsr_calr": "replace_plr_current12", "full": "full"}
    for name, h, p, a in SETTINGS:
        stable, legacy = {}, {}
        for threshold in (.15, .25, .5):
            key = str(threshold)
            stable[key] = ap_from_pairs(np.concatenate([
                scene_pairs(*predictions[name][s], gt[s], threshold) for s in scenes]), gt_count)
            legacy[key] = ev.audit.core.class_agnostic_ap(predictions[name], gt, threshold)["ap"]
            if name in reference_arms:
                expected = reference["prefix_results"]["1.0"][reference_arms[name]]
                assert abs(stable[key]-expected["stable_ap"][key]) < 1e-9
                assert abs(legacy[key]-expected["legacy_ap"][key]) < 1e-9
        assert counts[name]["native"] == 1787
        results[name] = {"HCSR": bool(h), "PLR": bool(p), "CALR": bool(a),
                         "counts": counts[name], "stable_ap": stable, "legacy_ap": legacy}
        table.append({"setting": name, "HCSR": h, "PLR": p, "CALR": a,
            **counts[name], "total": sum(counts[name].values()),
            **{f"AP{int(t*100)}": stable[str(t)] for t in (.15, .25, .5)},
            **{f"legacy_AP{int(t*100)}": legacy[str(t)] for t in (.15, .25, .5)}})
    ev.save_json(OUTPUT/"results.json", {"protocol": protocol, "scene_count": 100,
        "native_count": 1787, "gt_count": gt_count, "settings": results,
        "input_sha256": hashes, "producer_sha256": ev.audit.core.sha256(Path(__file__)),
        "validation": {"status": "PASS", "identical_native_geometry_and_order": True,
            "reference_arms_reproduced_to_1e9": True, "auxiliary_output_present_all_settings": True}})
    for filename, rows in (("ablation_table.csv", table), ("per_scene_output_counts.csv", scene_counts)):
        with (OUTPUT/filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
