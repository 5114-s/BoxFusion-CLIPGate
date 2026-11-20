#!/usr/bin/env python3
"""CPU-only, frozen-output duplicate attribution and post-assembly NMS audit.

Uses the exact stable-sort Full predictions underlying the 2026-11-02
always-auxiliary ablation (47.22224 / 42.74162 / 22.15790).
No model inference, score changes, GT-assisted suppression or production edits.
"""
from __future__ import annotations

import contextlib
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev
from tools.bootstrap_final_recar_pairs import ap_from_pairs, scene_pairs

SOURCE = ROOT / "reports/scannet_auxiliary_direct_readout_20261102"
OUT = ROOT / "reports/scannet_cross_branch_duplicates_20261115"
TS = (.15, .25, .5)
SOURCES = ("native", "proposal", "anchor")
LABELS = dict(zip(SOURCES, ("Native", "PTR", "VCR")))
METHODS = ("base", "memory256", "full")
POLICIES = ("concat", *[f"{kind}_{t}" for kind in
            ("cross_only", "global_nms", "native_priority") for t in (.25, .5)])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def classify(matrix, scores, sources, threshold):
    """0: localization/background FP, 1: TP, 2: within-branch duplicate, 3: cross."""
    status = np.zeros(len(scores), dtype=int)
    target = np.full(len(scores), -1, dtype=int)
    owner = np.full(matrix.shape[1], -1, dtype=int)
    owner_for_prediction = np.full(len(scores), -1, dtype=int)
    for index in np.argsort(-scores, kind="stable"):
        if matrix.shape[1] and matrix[index].max() > threshold:
            gt = int(matrix[index].argmax())
            target[index] = gt
            if owner[gt] == -1:
                owner[gt] = index
                status[index] = 1
            else:
                first = owner[gt]
                owner_for_prediction[index] = first
                status[index] = 2 if sources[first] == sources[index] else 3
    return status, target, owner_for_prediction


def suppress(matrix, scores, sources, policy):
    if policy == "concat":
        return np.arange(len(scores), dtype=int)
    kind, raw = policy.rsplit("_", 1)
    threshold = float(raw)
    keep = list(np.flatnonzero(sources == "native")) if kind == "native_priority" else []
    for index in np.argsort(-scores, kind="stable"):
        if kind == "native_priority" and sources[index] == "native":
            continue
        competitors = [i for i in keep if kind != "cross_only" or sources[i] != sources[index]]
        if competitors and np.any(matrix[index, competitors] >= threshold):
            continue
        keep.append(int(index))
    return np.asarray(sorted(keep), dtype=int)


def self_test():
    # Identical same-source predictions survive cross-only NMS; a cross-source
    # prediction is suppressed. Stable order determines the winner of a tie.
    scores = np.array([.9, .8, .7])
    sources = np.array(["native", "native", "anchor"])
    matrix = np.ones((3, 3))
    assert suppress(matrix, scores, sources, "cross_only_0.25").tolist() == [0, 1]
    assert suppress(matrix, scores, sources, "global_nms_0.25").tolist() == [0]
    assert suppress(matrix, scores, sources, "native_priority_0.25").tolist() == [0, 1]
    status, target, owner = classify(np.array([[.6], [.6], [.6]]), scores, sources, .5)
    assert status.tolist() == [1, 2, 3]
    assert owner.tolist() == [-1, 0, 0]
    assert classify(np.array([[.5]]), scores[:1], sources[:1], .5)[0].tolist() == [0]
    assert suppress(matrix, scores, np.array(["anchor", "native", "anchor"]),
                    "native_priority_0.5").tolist() == [1]
    ev.check_small_metrics()


def main():
    self_test()
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = {
        "dataset": "ScanNet-100", "scope": "terminal frozen-output exploratory diagnostic",
        "source": str(SOURCE), "scene_list": "evaluation/data_util/meta_data/scannetv2_val.txt",
        "methods": list(METHODS), "policies": list(POLICIES),
        "evaluation_iou": list(TS), "nms_iou": [.25, .5],
        "nms_coordinates": "original unaligned world-coordinate AABBs, matching existing NMS diagnostic",
        "nms_comparison": ">= threshold", "evaluation_comparison": "> threshold",
        "cross_only": "Greedy score-order NMS suppresses only against kept DIFFERENT-source boxes; same-source boxes do not suppress each other",
        "global_nms": "Greedy score-order NMS across all records, including within-branch overlaps",
        "native_priority": "Keep all native records; auxiliary records are suppressed against all kept records",
        "fixed": "Each retained record keeps exact geometry, source, score and original assembly order; no model/state rerun",
        "matching": "Stable descending score; best-overlap GT, at most one TP per GT. Remaining matches to an already claimed GT are duplicate FP",
        "cross_pair_attribution": "Unique duplicate FP attributed to its source and source of the first TP on its best GT. A prediction is counted once; attribution depends on score order",
        "geometric_overlap": "Directed record incidence, not raw number of box pairs or number of duplicate objects; no GT involved",
        "multiplicity": "All predictions with IoU>threshold to each GT; GT coverage diagnostic, not one-to-one recall",
        "top_k": [50, 100], "bootstrap": {"replicates": 2000, "seed": 20261115,
        "unit": "scene", "interval": "paired percentile 95%; marginal, no multiplicity correction"},
        "status": "Thresholds inherited from the existing 20261029 NMS diagnostic; this is not a newly blinded confirmatory experiment",
        "model_inference": False, "gpu_used": False, "production_changes": False,
    }
    save_json(OUT / "protocol.json", protocol)
    scene_path = ROOT / protocol["scene_list"]
    scenes = scene_path.read_text().split()
    assert len(scenes) == len(set(scenes)) == 100
    with contextlib.redirect_stdout(io.StringIO()):
        gt_all, alignments, transform = ev.audit.core.scannet_inputs()
    gts = {s: gt_all[s] for s in scenes}
    assert sum(len(g) for g in gts.values()) == 1433
    reference = json.loads((SOURCE / "point_results.json").read_text())
    countrows = list(csv.DictReader((SOURCE / "terminal_output_counts.csv").open(encoding="utf-8-sig")))
    countmap = {(r["arm"], r["scene"]): r for r in countrows}
    paths = [scene_path, SOURCE / "terminal_output_counts.csv", SOURCE / "point_results.json",
             Path(__file__), ROOT / "tools/evaluate_scannet_mechanism_controls.py",
             ROOT / "tools/bootstrap_final_recar_pairs.py", ROOT / "tools/true_fusion_audit_core.py"]
    hashes = {str(p): sha(p) for p in paths}
    np.savez_compressed(OUT / "evaluation_gt_and_alignment.npz", **{
        **{s + "_gt": gts[s] for s in scenes}, **{s + "_alignment": alignments[s] for s in scenes}})
    aggregates, ap_pairs = {}, {}
    scene_stats, duplicate_details, overlap_details, source_stats, accounting = [], [], [], [], []
    for method in METHODS:
        for policy in POLICIES:
            aggregates[method + "/" + policy] = {t: Counter() for t in TS}
            ap_pairs[method + "/" + policy] = {t: {} for t in TS}
        for scene in scenes:
            path = SOURCE / "predictions" / (method + "_p100") / (scene + "_boxes.pkl")
            hashes[str(path)] = sha(path)
            rows = ev.audit.read_rows(path)
            counts = countmap[method, scene]
            sources = np.array([s for s in SOURCES for _ in range(int(counts[s]))])
            assert len(rows) == len(sources) == int(counts["total"])
            value = ev.case(rows, sources)
            boxes, scores, _ = ev.as_prediction(value, scene, alignments, transform)
            world_boxes = np.asarray([r[1] for r in rows]).reshape(-1, 8, 3)
            geometry = ev.audit.aabb_iou(world_boxes, world_boxes)
            target_matrix = ev.audit.aabb_iou(boxes, gts[scene])
            original = {t: classify(target_matrix, scores, sources, t) for t in TS}
            if method == "full":
                for threshold in (.25, .5):
                    for a in SOURCES:
                        for b in SOURCES:
                            if a == b:
                                continue
                            m = geometry[sources == b][:, sources == a]
                            overlap_details.append({"scene": scene, "nms_iou": threshold,
                                "reference": LABELS[a], "tested": LABELS[b],
                                "tested_count": int((sources == b).sum()),
                                "overlapping_records": int((m >= threshold).any(1).sum())})
                for t in TS:
                    status, targets, owner = original[t]
                    order = np.argsort(-scores, kind="stable")
                    rank = np.empty(len(scores), dtype=int)
                    rank[order] = np.arange(1, len(scores) + 1)
                    for i in np.flatnonzero(status >= 2):
                        first = owner[i]
                        duplicate_details.append({"scene": scene, "evaluation_iou": t,
                            "record_index": int(i), "branch": LABELS[sources[i]],
                            "first_tp_index": int(first), "first_tp_branch": LABELS[sources[first]],
                            "gt_index": int(targets[i]), "best_gt_iou": float(target_matrix[i, targets[i]]),
                            "score": float(scores[i]), "rank": int(rank[i]),
                            "duplicate_type": "cross" if status[i] == 3 else "within"})
            for policy in POLICIES:
                keep = suppress(geometry, scores, sources, policy)
                removed = np.ones(len(rows), dtype=bool)
                removed[keep] = False
                if policy.startswith(("global_nms_", "native_priority_")):
                    expected, _ = ev.nms_case(value, float(policy.rsplit("_", 1)[1]), policy.startswith("native_priority"))
                    assert len(expected["rows"]) == len(keep)
                    assert all(np.array_equal(rows[i][1], row[1]) and rows[i][2] == row[2]
                               for i, row in zip(keep, expected["rows"]))
                if policy.startswith("native_priority"):
                    assert not (removed & (sources == "native")).any()
                ss, cc, matrix = scores[keep], sources[keep], target_matrix[keep]
                for t in TS:
                    status, targets, owner = classify(matrix, ss, cc, t)
                    flags = (status == 1).astype(int)
                    pairs = np.column_stack((ss, flags))
                    assert np.array_equal(pairs, scene_pairs(boxes[keep], ss, gts[scene], t))
                    ap_pairs[method + "/" + policy][t][scene] = pairs
                    hits = matrix > t
                    multiplicity = hits.sum(0)
                    order = np.argsort(-ss, kind="stable")
                    stat = {"predictions": len(keep), "native": int((cc == "native").sum()),
                        "ptr": int((cc == "proposal").sum()), "vcr": int((cc == "anchor").sum()),
                        "removed": int(removed.sum()), "removed_native": int((removed & (sources == "native")).sum()),
                        "tp": int(flags.sum()), "fp": int(len(flags) - flags.sum()),
                        "within_duplicate_fp": int((status == 2).sum()),
                        "cross_duplicate_fp": int((status == 3).sum()),
                        "other_fp": int((status == 0).sum()),
                        "covering_predictions_sum": int(multiplicity.sum()),
                        "covered_gt": int((multiplicity > 0).sum()),
                        "multiply_covered_gt": int((multiplicity > 1).sum()),
                        "additional_unique_gt_coverage": int((hits[cc != "native"].any(0) & ~hits[cc == "native"].any(0)).sum()),
                        "top50_tp": int(flags[order[:50]].sum()), "top100_tp": int(flags[order[:100]].sum())}
                    assert stat["fp"] == stat["within_duplicate_fp"] + stat["cross_duplicate_fp"] + stat["other_fp"]
                    aggregates[method + "/" + policy][t].update(stat)
                    scene_stats.append({"method": method, "policy": policy, "scene": scene, "iou": t, **stat})
                    for source in SOURCES:
                        mask = cc == source
                        source_stats.append({"method": method, "policy": policy, "scene": scene, "iou": t,
                            "branch": LABELS[source], "predictions": int(mask.sum()),
                            "tp": int((mask & (status == 1)).sum()), "fp": int((mask & (status != 1)).sum()),
                            "within_duplicate_fp": int((mask & (status == 2)).sum()),
                            "cross_duplicate_fp": int((mask & (status == 3)).sum())})
                    if method == "full" and policy != "concat":
                        old = original[t][0]
                        surviving_old_tp = int(((old == 1) & ~removed).sum())
                        accounting.append({"scene": scene, "policy": policy, "iou": t,
                            "removed_old_tp": int(((old == 1) & removed).sum()),
                            "removed_old_within_dup": int(((old == 2) & removed).sum()),
                            "removed_old_cross_dup": int(((old == 3) & removed).sum()),
                            "removed_old_other_fp": int(((old == 0) & removed).sum()),
                            "new_tp_reassigned_after_removal": int(flags.sum()) - surviving_old_tp})
        print(f"Evaluated {method}: 100 scenes x {len(POLICIES)} readout policies", flush=True)
    results, metric_rows = {}, []
    for name, by_t in aggregates.items():
        results[name] = {}
        method, policy = name.split("/")
        for t, counter in by_t.items():
            row = dict(counter)
            row.update({"AP": ap_from_pairs(np.concatenate([ap_pairs[name][t][s] for s in scenes]), 1433),
                "recall_percent": 100 * row["tp"] / 1433,
                "precision_percent": 100 * row["tp"] / max(1, row["predictions"]),
                "cross_duplicate_rate_percent": 100 * row["cross_duplicate_fp"] / max(1, row["predictions"]),
                "within_duplicate_rate_percent": 100 * row["within_duplicate_fp"] / max(1, row["predictions"]),
                "mean_multiplicity_all_gt": row["covering_predictions_sum"] / 1433,
                "R50_percent": 100 * row["top50_tp"] / 1433,
                "R100_percent": 100 * row["top100_tp"] / 1433})
            if policy == "concat":
                assert abs(row["AP"] - reference["prefix_results"]["1.0"][method]["stable_ap"][str(t)]) < 1e-9
            results[name][str(t)] = row
            metric_rows.append({"method": method, "policy": policy, "iou": t, **row})
    assert results["full/concat"]["0.25"]["native"] == 1787
    assert results["full/concat"]["0.25"]["predictions"] == 26060
    write_csv(OUT / "metrics.csv", metric_rows)
    write_csv(OUT / "per_scene_metrics.csv", scene_stats)
    write_csv(OUT / "full_duplicate_records.csv", duplicate_details)
    write_csv(OUT / "geometric_overlap_per_scene.csv", overlap_details)
    write_csv(OUT / "per_branch_metrics.csv", source_stats)
    write_csv(OUT / "removed_prediction_accounting.csv", accounting)
    save_json(OUT / "point_results.json", results)
    save_json(OUT / "input_sha256.json", hashes)
    print("PASS: exact Full/base/memory AP reproduced; suppression checked against previous implementation", flush=True)
    draws = np.random.default_rng(20261115).integers(0, len(scenes), (2000, len(scenes)))
    gt_counts = np.array([len(gts[s]) for s in scenes])
    samples = {p: {t: [] for t in TS} for p in POLICIES if p != "concat"}
    for i, draw in enumerate(draws):
        selected = [scenes[int(j)] for j in draw]
        num_gt = int(gt_counts[draw].sum())
        for t in TS:
            values = {p: ap_from_pairs(np.concatenate([ap_pairs["full/" + p][t][s] for s in selected]), num_gt)
                      for p in POLICIES}
            for p in samples:
                samples[p][t].append(values[p] - values["concat"])
        if (i + 1) % 250 == 0:
            print(f"Paired scene bootstrap {i+1}/2000", flush=True)
    ci_rows = []
    for p, by_t in samples.items():
        for t, values in by_t.items():
            low, high = np.percentile(values, [2.5, 97.5])
            ci_rows.append({"policy": p, "iou": t,
                "delta_AP_vs_concat": results["full/" + p][str(t)]["AP"] - results["full/concat"][str(t)]["AP"],
                "ci95_low": float(low), "ci95_high": float(high),
                "fraction_positive": float(np.mean(np.asarray(values) > 0))})
    write_csv(OUT / "paired_bootstrap.csv", ci_rows)
    validation = {"status": "PASS", "unit_tests": True, "reference_AP_reproduced": True,
        "matching_checked_against_shared_evaluator": True, "native_count": 1787,
        "full_count": 26060, "scene_count": 100, "gt_count": 1433,
        "predictions_unchanged_on_disk": all(sha(p) == h for p, h in hashes.items()),
        "model_inference": False, "gpu_used": False}
    assert validation["predictions_unchanged_on_disk"]
    save_json(OUT / "results.json", {"protocol": protocol, "metrics": results,
        "paired_bootstrap": ci_rows, "validation": validation})
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
