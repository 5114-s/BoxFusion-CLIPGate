#!/usr/bin/env python3
"""Validate the frozen-cache controls and make AP tie conventions explicit."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev
from tools.bootstrap_final_recar_pairs import scene_pairs, ap_from_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ev.DEFAULT)
    args = parser.parse_args()
    out = args.output
    protocol = json.loads((out/"protocol.json").read_text())
    sha = ev.audit.core.sha256(out/"protocol.json")
    cached = json.loads((out/"cached_results.json").read_text())
    sensitivity = json.loads((out/"sensitivity_results.json").read_text())
    manifest = json.loads((out/"replay_manifest.json").read_text())
    assert cached["protocol_sha256"] == sensitivity["protocol_sha256"] == manifest["protocol_sha256"] == sha
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines() if s.strip() and not s.startswith("#")]
    assert len(manifest["scenes"]) == len(scenes) == 100
    assert sum(v["keyframes"] for v in manifest["scenes"].values()) == 6817
    with contextlib.redirect_stdout(io.StringIO()):
        gts, alignments, transform = ev.audit.core.scannet_inputs()
    assert sum(len(g) for g in gts.values()) == 1433
    parent = json.loads((ROOT/protocol["parent_report"] / "results.json").read_text())
    for filename, expected_hash in parent["input_sha256"].items():
        assert ev.audit.core.sha256(Path(filename)) == expected_hash, filename
    cached_cases = ev.cached_cases(out, scenes)["1.0"]
    evaluated = {name: {s: ev.as_prediction(v, s, alignments, transform)[:2] for s, v in values.items()}
                 for name, values in cached_cases.items()}
    for scene in scenes:
        with (out/"replay"/f"{scene}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        assert payload["protocol_sha256"] == sha
        for f, values in payload["prefixes"].items():
            for original, other in (("full_current", "full_raw"), ("full_current", "full_temperature2"),
                                    ("memory_current", "memory_raw"), ("memory_current", "memory_temperature2")):
                left, right = values[original], values[other]
                assert left["sources"] == right["sources"] and len(left["rows"]) == len(right["rows"])
                assert all(np.array_equal(x[1], y[1]) for x, y in zip(left["rows"], right["rows"]))
                for x, y, source in zip(left["rows"], right["rows"], left["sources"]):
                    if source == "native":
                        assert x[2] == y[2]
        for arm in ("layered_common", "unified_common"):
            evaluated.setdefault(arm, {})[scene] = ev.as_prediction(payload["prefixes"]["1.0"][arm], scene, alignments, transform)[:2]
        trace = payload["summary"]["trace"]
        assert len(trace) == manifest["scenes"][scene]["keyframes"]
        assert [x["ordinal"] for x in trace] == list(range(len(trace)))
        for method in ("full", "memory", "layered_common", "unified_common"):
            for field in ("reachable_bytes", "serialized_bytes", "entries"):
                assert max(x[field][method] for x in trace) == payload["summary"]["peaks"][f"{method}_{field}"]
    stable = {}
    for arm, predictions in evaluated.items():
        stable[arm] = {}
        for t in ev.TS:
            pairs = np.concatenate([scene_pairs(*predictions[s], gts[s], t) for s in scenes])
            stable[arm][str(t)] = ap_from_pairs(pairs, 1433)
    contrasts = {"plr-minus-proposal-memory": ("plr", "proposal_memory"), **{
        f"calr-minus-anchor-memory{a}": ("calr", f"anchor_memory{a}") for a in (128, 256, 512, 640)}}
    tie_contrasts = {}
    for name, (left, right) in contrasts.items():
        tie_contrasts[name] = {}
        for t in ev.TS:
            row = cached["bootstrap"][name][str(t)]
            point = stable[left][str(t)]-stable[right][str(t)]
            row["point_ap_stable_tie_protocol"] = point
            tie_contrasts[name][str(t)] = {"legacy_difference": row["point_ap"], "stable_difference": point,
                                           "absolute_delta": abs(row["point_ap"]-point)}
    name = "layered-minus-unified-common"
    tie_contrasts[name] = {}
    for t in ev.TS:
        row = sensitivity["bootstrap"][name][str(t)]
        point = stable["layered_common"][str(t)]-stable["unified_common"][str(t)]
        row["point_ap_stable_tie_protocol"] = point
        tie_contrasts[name][str(t)] = {"legacy_difference": row["point_ap"], "stable_difference": point,
                                       "absolute_delta": abs(row["point_ap"]-point)}
    for method, thresholds in cached["terminal_workpoints"].items():
        for t, values in thresholds.items():
            assert abs(values["stable_ap"]-stable[method][t]) < 1e-9
            all_rows = values["all_outputs"]
            assert all_rows["tp"]+all_rows["fp"] == all_rows["predictions"]
            assert all_rows["within_duplicate_fp"]+all_rows["cross_duplicate_fp"] <= all_rows["fp"]
            assert all(values["topk"][str(a)]["recall_percent"] <= values["topk"][str(b)]["recall_percent"] for a, b in zip((20, 50, 100), (50, 100, 300)))
            assert all(values["fixed_fp_per_scene"][str(b)]["fp"] <= b*100 for b in (1, 5, 10))
    for method, policies in sensitivity["dedup"].items():
        for policy, data in policies.items():
            if policy.startswith("native_priority"):
                assert data["removed"]["removed_native"] == 0
            for row in data["metrics"].values():
                counts = row["all_outputs"]
                assert counts["tp"]+counts["fp"] == counts["predictions"] == data["removed"]["retained"]
                assert counts["within_duplicate_fp"]+counts["cross_duplicate_fp"] <= counts["fp"]
    ev.save_json(out/"cached_results.json", cached)
    ev.save_json(out/"sensitivity_results.json", sensitivity)
    summary = {"scenes": 100, "GT": 1433, "keyframes": 6817, "passed": True,
               "fixed_geometry_count_for_scoring_verified": True, "native_priority_never_deletes_native": True,
               "counts_and_memory_traces_verified": True, "native_hcsr_shared": True,
               "all_200_frozen_native_evidence_files_hash_matched": True,
               "new_model_inference": False, "stable_ap_by_arm": stable, "tie_contrasts": tie_contrasts,
               "maximum_contrast_delta_from_tie_protocol": max(x["absolute_delta"] for v in tie_contrasts.values() for x in v.values()),
               "source_for_reproduction_sha256": {str(p.relative_to(ROOT)): ev.audit.core.sha256(p) for p in (
                   ROOT/"tools/evaluate_scannet_mechanism_controls.py", ROOT/"tools/replay_scannet_unified_history_controls.py",
                   ROOT/"tools/validate_scannet_mechanism_controls.py")}}
    ev.save_json(out/"validation_and_tie_audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("stable_ap_by_arm", "source_for_reproduction_sha256")}, indent=2))


if __name__ == "__main__":
    main()
