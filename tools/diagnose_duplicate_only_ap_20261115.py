#!/usr/bin/env python3
"""GT-oracle diagnostic: AP penalty of duplicate FPs, not a usable method."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_cross_branch_duplicates_20261115 import (
    OUT, SOURCE, TS, classify, sha, write_csv, save_json)
from tools import evaluate_scannet_mechanism_controls as ev
from tools.bootstrap_final_recar_pairs import ap_from_pairs, scene_pairs


def main():
    import csv
    result = json.loads((OUT / "results.json").read_text())
    counts = {r["scene"]: r for r in csv.DictReader(
        (SOURCE / "terminal_output_counts.csv").open(encoding="utf-8-sig")) if r["arm"] == "full"}
    scenes = (ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt").read_text().split()
    with contextlib.redirect_stdout(io.StringIO()):
        gts, aligns, transform = ev.audit.core.scannet_inputs()
    arms = ("concat", "oracle_remove_cross_duplicate_fp", "oracle_remove_all_duplicate_fp")
    pairs = {t: {a: [] for a in arms} for t in TS}
    removed = {t: {a: 0 for a in arms} for t in TS}
    for scene in scenes:
        rows = ev.audit.read_rows(SOURCE / "predictions/full_p100" / (scene + "_boxes.pkl"))
        sources = np.array([s for s in ("native", "proposal", "anchor") for _ in range(int(counts[scene][s]))])
        boxes, scores, _ = ev.as_prediction(ev.case(rows, sources), scene, aligns, transform)
        matrix = ev.audit.aabb_iou(boxes, gts[scene])
        for t in TS:
            status, _, _ = classify(matrix, scores, sources, t)
            for arm, mask in zip(arms, (np.ones(len(rows), bool), status != 3, status < 2)):
                p = scene_pairs(boxes[mask], scores[mask], gts[scene], t)
                assert np.array_equal(p[:, 1], (status[mask] == 1).astype(float))
                pairs[t][arm].append(p)
                removed[t][arm] += int((~mask).sum())
    output = []
    for t in TS:
        base = result["metrics"]["full/concat"][str(t)]
        for arm in arms:
            p = np.concatenate(pairs[t][arm])
            ap = ap_from_pairs(p, 1433)
            assert p[:, 1].sum() == base["tp"]
            if arm == "concat":
                assert abs(ap - base["AP"]) < 1e-9
            assert ap >= base["AP"] - 1e-9
            output.append({"arm": arm, "iou": t, "AP": ap, "delta_AP": ap - base["AP"],
                "removed_duplicate_fp": removed[t][arm], "tp": base["tp"],
                "recall_percent": base["recall_percent"], "GT_ORACLE_NOT_DEPLOYABLE": True})
    write_csv(OUT / "oracle_duplicate_only_AP.csv", output)
    save_json(OUT / "oracle_duplicate_only_protocol.json", {
        "warning": "Uses test GT and different retained sets at each evaluation IoU. Not a deployable method, fair baseline or one three-threshold prediction set.",
        "purpose": "Isolate AP cost of the evaluation-defined duplicate FP, deleting only these FP while preserving every TP, score and survivor order.",
        "source_results_sha256": sha(OUT / "results.json"), "script_sha256": sha(Path(__file__)),
        "result": output})
    for row in output:
        print(row)


if __name__ == "__main__":
    main()
