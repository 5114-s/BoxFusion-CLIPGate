#!/usr/bin/env python3
"""Audit serialized predictions, dataset grouping and GT provenance."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_multiscan_branch_strong_memory as run
from tools.bootstrap_final_recar_pairs import scene_pairs, ap_from_pairs


def main():
    out = run.DEFAULT
    protocol = json.loads((out/"protocol.json").read_text())
    result = json.loads((out/"results.json").read_text())
    manifest = json.loads((out/"replay_manifest.json").read_text())
    prepared = ROOT/protocol["prepared_root"]
    data = json.loads((prepared/"manifest.json").read_text())
    scenes = [s["alias"] for s in data["scans"]]
    environments = Counter(s["scan_id"].rsplit("_", 1)[0] for s in data["scans"])
    assert len(scenes) == len(environments) == 50 and max(environments.values()) == 1
    fingerprints = dict(manifest["input_sha256"])
    extras = [prepared/"manifest.json", ROOT/protocol["experiment_root"]/"config.yaml",
              *sorted((ROOT/"development/final_controls/experiments/multiscan50_mvsr_temporal_patch_20261012/native_prefix").glob("*.npz"))]
    for path in extras:
        fingerprints[str(path)] = run.core.sha256(path)
    for path, sha in manifest["source_sha256"].items():
        assert run.core.sha256(Path(path)) == sha
    arms = list(result["protocols"]["nonstructural"]["arms"])
    predictions = {arm: {} for arm in arms}
    for scene in scenes:
        native = run.ev.audit.read_rows(out/"predictions/hcsr"/f"{scene}_boxes.pkl")
        for arm in arms:
            path = out/"predictions"/arm/f"{scene}_boxes.pkl"
            rows = run.ev.audit.read_rows(path)
            assert run.same_rows(native, rows[:len(native)])
            predictions[arm][scene] = (
                np.asarray([r[1] for r in rows]).reshape(-1, 8, 3),
                np.asarray([r[2] for r in rows]),
            )
    for label, filename in (("nonstructural", "gt_aabb_nonstructural.npy"),
                            ("scannet18", "gt_aabb_scannet18.npy")):
        gt = {}
        for scene in scenes:
            path = prepared/scene/filename
            fingerprints[str(path)] = run.core.sha256(path)
            gt[scene] = np.load(path).astype(np.float64)
            assert gt[scene].ndim == 3 and gt[scene].shape[1:] == (8, 3)
            assert np.isfinite(gt[scene]).all()
        total = sum(map(len, gt.values()))
        assert total == result["protocols"][label]["GT_count"]
        for arm in arms:
            for t in protocol["iou_thresholds"]:
                pairs = np.concatenate([scene_pairs(*predictions[arm][s], gt[s], t) for s in scenes])
                ap = ap_from_pairs(pairs, total)
                assert abs(ap-result["protocols"][label]["arms"][arm]["metrics"][str(t)]["stable_ap"]) < 1e-9
    with zipfile.ZipFile(out/"实验结果与结论.docx") as archive:
        xml = archive.read("word/document.xml").decode()
        assert "4736" in xml and "97.5%" in xml and "端到端FPS" in xml
    audit = {"status": "PASS", "scans": 50, "distinct_environments": 50,
             "exported_prediction_files_verified": 650,
             "recomputed_stable_AP_cells": len(arms)*2*len(protocol["iou_thresholds"]),
             "GT_counts": {p: x["GT_count"] for p, x in result["protocols"].items()},
             "source_files_unchanged": True, "native_prefix_scores_and_geometry_identical": True,
             "DOCX_verified": True, "all_data_sha256": fingerprints}
    run.ev.save_json(out/"source_and_GT_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "all_data_sha256"}))


if __name__ == "__main__":
    main()
