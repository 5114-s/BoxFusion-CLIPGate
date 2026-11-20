#!/usr/bin/env python3
"""Read-only model evaluation for the prespecified mechanism/protocol controls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_recovery_origin_memory_budget_audit as audit
from tools.bootstrap_final_recar_pairs import ap_from_pairs

DEFAULT = ROOT / "reports/scannet_mechanism_protocol_controls_20261029"
TS = (.15, .25, .5)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def case(rows, sources):
    assert len(rows) == len(sources)
    return {"rows": rows, "sources": tuple(sources)}


def geometry_signature(box):
    points = np.asarray(box, dtype=np.float64)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return np.round(points[order], 6).tobytes()


def as_prediction(value, scene, alignments, transform):
    boxes, scores = audit.core.prediction(value["rows"], transform, alignments[scene])
    return boxes, scores, np.asarray(value["sources"])


def greedy_info(boxes, scores, sources, gt, threshold):
    matrix = audit.aabb_iou(boxes, gt)
    order = np.argsort(-scores, kind="stable")
    flags = np.zeros(len(order), dtype=np.int64)
    within = np.zeros(len(order), dtype=np.int64)
    cross = np.zeros(len(order), dtype=np.int64)
    matched = np.zeros(len(gt), dtype=bool)
    first_source = np.full(len(gt), "", dtype=object)
    for rank, index in enumerate(order):
        overlaps = matrix[index]
        if len(overlaps) and overlaps.max() > threshold:
            target = int(overlaps.argmax())
            if not matched[target]:
                matched[target] = True
                first_source[target] = sources[index]
                flags[rank] = 1
            elif first_source[target] == sources[index]:
                within[rank] = 1
            else:
                cross[rank] = 1
    assert int(within.sum() + cross.sum()) <= int(len(flags) - flags.sum())
    return matrix, order, flags, within, cross


def workpoints(predictions, gts, thresholds=(.25, .5), topks=(20, 50, 100, 300),
               fp_budgets=(1, 5, 10)):
    output, curves, threshold_rows = {}, {}, []
    total_gt = sum(len(v) for v in gts.values())
    for threshold in thresholds:
        top = {str(k): Counter() for k in topks}
        fixed = {str(b): Counter() for b in fp_budgets}
        macro_top = {str(k): [] for k in topks}
        macro_fp = {str(b): [] for b in fp_budgets}
        totals = Counter()
        by_source = {s: Counter() for s in ("native", "proposal", "anchor")}
        scored = []
        for scene, (boxes, scores, sources) in predictions.items():
            gt = gts[scene]
            matrix, order, flags, within, cross = greedy_info(boxes, scores, sources, gt, threshold)
            tp, fp = np.cumsum(flags), np.cumsum(1-flags)
            sorted_scores = scores[order]
            scored.append(np.column_stack((sorted_scores, flags)))
            totals.update({"tp": int(flags.sum()), "fp": int((1-flags).sum()),
                           "within_duplicate_fp": int(within.sum()),
                           "cross_duplicate_fp": int(cross.sum()), "predictions": len(flags)})
            hits = matrix > threshold
            multiplicity = hits.sum(0)
            totals["sum_predictions_covering_each_gt"] += int(multiplicity.sum())
            totals["gt_covered_by_at_least_two"] += int((multiplicity >= 2).sum())
            covered_native = hits[sources == "native"].any(0)
            covered_aux = hits[sources != "native"].any(0)
            totals["additional_unique_gt_max_iou_coverage"] += int((covered_aux & ~covered_native).sum())
            for source in by_source:
                mask = sources[order] == source
                by_source[source].update({"predictions": int(mask.sum()), "tp": int(flags[mask].sum()),
                                          "fp": int((1-flags[mask]).sum()),
                                          "within_duplicate_fp": int(within[mask].sum()),
                                          "cross_duplicate_fp": int(cross[mask].sum())})
            for k in topks:
                n = min(k, len(flags)); count = int(flags[:n].sum())
                top[str(k)].update({"tp": count, "fp": n-count, "predictions": n})
                macro_top[str(k)].append(count/max(len(gt), 1))
            ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True]) if len(flags) else np.empty(0, int)
            for budget in fp_budgets:
                valid = ends[fp[ends] <= budget]
                last = int(valid[-1]) if len(valid) else -1
                count = int(tp[last]) if last >= 0 else 0
                false = int(fp[last]) if last >= 0 else 0
                fixed[str(budget)].update({"tp": count, "fp": false, "predictions": last+1})
                macro_fp[str(budget)].append(count/max(len(gt), 1))
                threshold_rows.append({"scene": scene, "iou": threshold, "fp_budget": budget,
                                       "threshold": float(sorted_scores[last]) if last >= 0 else None,
                                       "tp": count, "fp": false, "predictions": last+1,
                                       "oracle_test_gt_used": True})
        pairs = np.concatenate(scored)
        pairs = pairs[np.argsort(-pairs[:, 0], kind="stable")]
        cumulative_tp = pairs[:, 1].cumsum()
        cumulative_fp = (1-pairs[:, 1]).cumsum()
        recall = cumulative_tp / total_gt
        precision = cumulative_tp / np.maximum(cumulative_tp+cumulative_fp, 1)
        for key, row in top.items():
            row["recall_percent"] = 100*row["tp"]/total_gt
            row["macro_recall_percent"] = 100*float(np.mean(macro_top[key]))
            row["precision_percent"] = 100*row["tp"]/max(row["predictions"], 1)
        for key, row in fixed.items():
            row["recall_percent"] = 100*row["tp"]/total_gt
            row["macro_recall_percent"] = 100*float(np.mean(macro_fp[key]))
        totals["recall_percent"] = 100*totals["tp"]/total_gt
        totals["precision_percent"] = 100*totals["tp"]/max(totals["predictions"], 1)
        totals["within_duplicate_rate_percent"] = 100*totals["within_duplicate_fp"]/max(totals["predictions"], 1)
        totals["cross_duplicate_rate_percent"] = 100*totals["cross_duplicate_fp"]/max(totals["predictions"], 1)
        totals["mean_predictions_covering_each_gt"] = totals["sum_predictions_covering_each_gt"]/total_gt
        plain_predictions = {s: (x[0], x[1]) for s, x in predictions.items()}
        output[str(threshold)] = {"legacy_ap": audit.core.class_agnostic_ap(plain_predictions, gts, threshold)["ap"],
                                 "stable_ap": ap_from_pairs(pairs, total_gt),
                                 "topk": {k: dict(v) for k, v in top.items()},
                                 "fixed_fp_per_scene": {k: dict(v) for k, v in fixed.items()},
                                 "all_outputs": dict(totals),
                                 "per_source": {k: dict(v) for k, v in by_source.items()}}
        curves[str(threshold)] = {"recall": recall, "precision": precision, "scores": pairs[:, 0]}
    return output, curves, threshold_rows


def nms_case(value, threshold, native_priority=False):
    rows, sources = value["rows"], np.asarray(value["sources"])
    if not rows:
        return case([], []), {"removed": 0, "removed_native": 0}
    boxes = np.stack([r[1] for r in rows]); scores = np.asarray([r[2] for r in rows])
    matrix = audit.aabb_iou(boxes, boxes)
    order = np.argsort(-scores, kind="stable")
    native = np.flatnonzero(sources == "native")
    keep = native.tolist() if native_priority else []
    for index in order:
        if native_priority and sources[index] == "native":
            continue
        if keep and np.any(matrix[index, keep] >= threshold):
            continue
        keep.append(int(index))
    # Preserve original assembly order; only the retained subset changes.
    keep = sorted(keep)
    return case([rows[i] for i in keep], sources[keep]), {
        "removed": len(rows)-len(keep),
        "removed_native": int((sources == "native").sum()-(sources[keep] == "native").sum()),
        "retained": len(keep)}


def check_small_metrics():
    def box(x):
        return np.asarray([[x+i, j, k] for i in (0., 1.) for j in (0., 1.) for k in (0., 1.)])
    g = np.stack([box(0), box(3)])
    b = np.stack([box(0), box(0), box(6), box(3)])
    score = np.asarray([.9, .8, .7, .6]); src = np.asarray(["native", "anchor", "proposal", "anchor"])
    _, _, f, within, cross = greedy_info(b, score, src, g, .5)
    assert f.tolist() == [1, 0, 0, 1] and within.sum() == 0 and cross.sum() == 1
    metrics, _, _ = workpoints({"s": (b, score, src)}, {"s": g}, (.5,), (1, 4), (1, 2))
    assert metrics["0.5"]["topk"]["1"]["recall_percent"] == 50
    assert metrics["0.5"]["fixed_fp_per_scene"]["1"]["recall_percent"] == 50
    assert metrics["0.5"]["fixed_fp_per_scene"]["2"]["recall_percent"] == 100
    # A whole-score threshold must not split an exact tie to evade its FP.
    tie, _, _ = workpoints({"s": (b[:2], np.array([.9, .9]), src[:2])}, {"s": g}, (.5,), (1,), (0,))
    assert tie["0.5"]["fixed_fp_per_scene"]["0"]["predictions"] == 0
    reduced, removed = nms_case(case([(0, x, y) for x, y in zip(b, score)], src), .5, True)
    assert len(reduced["rows"]) == 3 and removed["removed_native"] == 0


def cached_cases(out, scenes):
    protocol = json.loads((out/"protocol.json").read_text())
    parent_root = ROOT/protocol["parent_report"]
    parent = json.loads((parent_root/"results.json").read_text())
    all_cases = {}
    for fraction in (.25, .5, .75, 1.):
        tag = f"p{int(fraction*100)}"
        result = {arm: {} for arm in ("base", "hcsr", "plr", "proposal_memory", "calr", "full",
                  *[f"anchor_memory{a}" for a in (128, 256, 512, 640)],
                  *[f"memory_a{a}" for a in (128, 256, 512, 640)])}
        for scene in scenes:
            info = parent["per_scene"][scene]
            checkpoint = int(np.ceil(info["keyframes"]*fraction))-1
            counts = info["keyframe_trace"][checkpoint]["output_counts"]
            full = audit.read_rows(parent_root/"predictions"/f"full_{tag}"/f"{scene}_boxes.pkl")
            base = audit.read_rows(parent_root/"predictions"/f"base_{tag}"/f"{scene}_boxes.pkl")
            n, p = counts["full"]["native"], counts["full"]["proposal"]
            hcsr, plr, calr = full[:n], full[n:n+p], full[n+p:]
            result["base"][scene] = case(base, ["native"]*n)
            result["hcsr"][scene] = case(hcsr, ["native"]*n)
            result["plr"][scene] = case(hcsr+plr, ["native"]*n+["proposal"]*len(plr))
            result["calr"][scene] = case(hcsr+calr, ["native"]*n+["anchor"]*len(calr))
            result["full"][scene] = case(full, ["native"]*n+["proposal"]*p+["anchor"]*len(calr))
            for cap in (128, 256, 512, 640):
                arm = f"memory_a{cap}"
                rows = audit.read_rows(parent_root/"predictions"/f"{arm}_{tag}"/f"{scene}_boxes.pkl")
                mp = counts[arm]["proposal"]
                pp, aa = rows[n:n+mp], rows[n+mp:]
                result[arm][scene] = case(rows, ["native"]*n+["proposal"]*mp+["anchor"]*len(aa))
                result[f"anchor_memory{cap}"][scene] = case(hcsr+aa, ["native"]*n+["anchor"]*len(aa))
                if cap == 128:
                    result["proposal_memory"][scene] = case(hcsr+pp, ["native"]*n+["proposal"]*len(pp))
        all_cases[str(fraction)] = result
    return all_cases


def evaluate_cached(out, protocol, scenes, gts, alignments, transform):
    cases = cached_cases(out, scenes)
    result = {"protocol_sha256": audit.core.sha256(out/"protocol.json"),
              "prefixes": {}, "terminal_workpoints": {}, "count": len(scenes)}
    evaluated = {}
    for fraction, arms in cases.items():
        evaluated[fraction], result["prefixes"][fraction] = {}, {}
        for arm, values in arms.items():
            predictions = {s: as_prediction(v, s, alignments, transform) for s, v in values.items()}
            plain = {s: (v[0], v[1]) for s, v in predictions.items()}
            counts = Counter(s for v in values.values() for s in v["sources"])
            result["prefixes"][fraction][arm] = {
                "counts": dict(counts), "total": sum(counts.values()),
                "ap": {str(t): audit.core.class_agnostic_ap(plain, gts, t)["ap"] for t in TS}}
            if arm in ("plr", "proposal_memory", "calr", "anchor_memory128", "anchor_memory256", "anchor_memory512", "anchor_memory640"):
                evaluated[fraction][arm] = plain
        print(f"cached branch evaluation {fraction} complete", flush=True)
    for arm in protocol["finite_workpoints"]["methods"]:
        predictions = {s: as_prediction(v, s, alignments, transform) for s, v in cases["1.0"][arm].items()}
        values, curves, thresholds = workpoints(predictions, gts)
        result["terminal_workpoints"][arm] = values
        np.savez_compressed(out/f"pr_{arm}.npz", **{f"iou{t}_{k}": v for t, x in curves.items() for k, v in x.items()})
        with (out/f"oracle_fp_thresholds_{arm}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(thresholds[0])); writer.writeheader(); writer.writerows(thresholds)
    save_json(out/"cached_results.json", result)
    comparisons = {"plr-minus-proposal-memory": ("plr", "proposal_memory"), **{
        f"calr-minus-anchor-memory{a}": ("calr", f"anchor_memory{a}") for a in (128, 256, 512, 640)}}
    bootstrap = audit.bootstrap(evaluated["1.0"], gts, scenes, comparisons,
                                protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    for name, (a, b) in comparisons.items():
        for t in TS:
            bootstrap[name][str(t)]["point_ap"] = result["prefixes"]["1.0"][a]["ap"][str(t)]-result["prefixes"]["1.0"][b]["ap"][str(t)]
    result["bootstrap"] = bootstrap
    save_json(out/"cached_results.json", result)
    print("cached branch and finite-workpoint phase complete", flush=True)


def evaluate_sensitivity(out, protocol, scenes, gts, alignments, transform):
    manifest = json.loads((out/"replay_manifest.json").read_text())
    assert len(manifest["scenes"]) == len(scenes) == 100
    assert manifest["protocol_sha256"] == audit.core.sha256(out/"protocol.json")
    arms = ("full_current", "full_raw", "full_temperature2", "memory_current", "memory_raw",
            "memory_temperature2", "layered_common", "unified_common")
    prefix_cases = {str(f): {a: {} for a in arms} for f in (.25, .5, .75, 1.)}
    exact_removed = {a: 0 for a in ("layered_common", "unified_common")}
    for scene in scenes:
        with (out/"replay"/f"{scene}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        assert payload["protocol_sha256"] == manifest["protocol_sha256"]
        for f, values in payload["prefixes"].items():
            for arm in arms:
                prefix_cases[f][arm][scene] = values[arm]
        for arm in exact_removed:
            exact_removed[arm] += payload["prefixes"]["1.0"]["exact_readout_removed"][arm]
    result = {"protocol_sha256": manifest["protocol_sha256"], "prefixes": {}, "score_workpoints": {},
              "dedup": {}, "common_exact_readout_removed": exact_removed}
    terminal_predictions = {}
    for f, values in prefix_cases.items():
        result["prefixes"][f] = {}
        for arm, cases in values.items():
            predictions = {s: as_prediction(v, s, alignments, transform) for s, v in cases.items()}
            plain = {s: (v[0], v[1]) for s, v in predictions.items()}
            counts = Counter(s for v in cases.values() for s in v["sources"])
            result["prefixes"][f][arm] = {"counts": dict(counts), "total": sum(counts.values()),
                                         "ap": {str(t): audit.core.class_agnostic_ap(plain, gts, t)["ap"] for t in TS}}
            if f == "1.0":
                terminal_predictions[arm] = plain
                result["score_workpoints"][arm] = workpoints(predictions, gts)[0]
        print(f"scoring/unified evaluation {f} complete", flush=True)
    cached = json.loads((out/"cached_results.json").read_text())
    for f in result["prefixes"]:
        for current, old in (("full_current", "full"), ("memory_current", "memory_a256")):
            assert result["prefixes"][f][current]["ap"] == cached["prefixes"][f][old]["ap"]
    for scene in scenes:
        left = prefix_cases["1.0"]["layered_common"][scene]
        right = prefix_cases["1.0"]["unified_common"][scene]
        a = [x for x, s in zip(left["rows"], left["sources"]) if s == "native"]
        b = [x for x, s in zip(right["rows"], right["sources"]) if s == "native"]
        assert len(a) == len(b) and all(x[2] == y[2] and np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    result["shared_native_geometry_and_hcsr_verified"] = True
    # Preserve all point estimates before bootstrap or figure generation.
    save_json(out/"sensitivity_results.json", result)
    cached_values = cached_cases(out, scenes)["1.0"]
    for method in protocol["dedup_sensitivity"]["methods"]:
        result["dedup"][method] = {}
        for threshold in protocol["dedup_sensitivity"]["aabb_iou_thresholds"]:
            for policy in ("native_priority", "global_nms"):
                values, removal = {}, Counter()
                for scene, value in cached_values[method].items():
                    reduced, counts = nms_case(value, threshold, policy == "native_priority")
                    removal.update(counts)
                    values[scene] = as_prediction(reduced, scene, alignments, transform)
                result["dedup"][method][f"{policy}_{threshold}"] = {
                    "removed": dict(removal), "metrics": workpoints(values, gts, thresholds=TS)[0]}
        print(f"dedup evaluation {method} complete", flush=True)
    save_json(out/"sensitivity_results.json", result)
    bootstrap = audit.bootstrap(terminal_predictions, gts, scenes,
                                {"layered-minus-unified-common": ("layered_common", "unified_common")},
                                protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    for t in TS:
        bootstrap["layered-minus-unified-common"][str(t)]["point_ap"] = (
            result["prefixes"]["1.0"]["layered_common"]["ap"][str(t)]-
            result["prefixes"]["1.0"]["unified_common"]["ap"][str(t)])
    result["bootstrap"] = bootstrap
    save_json(out/"sensitivity_results.json", result)
    print("sensitivity evaluation complete", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render_report(out, protocol):
    cached = json.loads((out/"cached_results.json").read_text())
    sensitivity = json.loads((out/"sensitivity_results.json").read_text())
    manifest = json.loads((out/"replay_manifest.json").read_text())
    validation = json.loads((out/"validation_and_tie_audit.json").read_text())
    branch_rows, score_rows, fp_rows, k_rows, dedup_rows, state_rows, score_workpoint_rows = [], [], [], [], [], [], []
    for fraction, arms in cached["prefixes"].items():
        for arm in ("hcsr", "plr", "proposal_memory", "calr", "anchor_memory128", "anchor_memory256", "anchor_memory512", "anchor_memory640"):
            x = arms[arm]
            branch_rows.append({"prefix": fraction, "method": arm, "native": x["counts"].get("native", 0),
                                "proposal": x["counts"].get("proposal", 0), "anchor": x["counts"].get("anchor", 0),
                                "total": x["total"], **{f"AP{int(t*100)}": x["ap"][str(t)] for t in TS}})
    for fraction, arms in sensitivity["prefixes"].items():
        for arm, x in arms.items():
            score_rows.append({"prefix": fraction, "method": arm, "native": x["counts"].get("native", 0),
                               "proposal": x["counts"].get("proposal", 0), "anchor": x["counts"].get("anchor", 0),
                               "total": x["total"], **{f"AP{int(t*100)}": x["ap"][str(t)] for t in TS}})
    for method, thresholds in cached["terminal_workpoints"].items():
        for t, values in thresholds.items():
            for k, item in values["topk"].items():
                k_rows.append({"method": method, "iou": t, "K": k, **item})
            for b, item in values["fixed_fp_per_scene"].items():
                fp_rows.append({"method": method, "iou": t, "FP_budget": b, **item})
    for method, policies in sensitivity["dedup"].items():
        for policy, results in policies.items():
            for t, values in results["metrics"].items():
                dedup_rows.append({"method": method, "policy": policy, "iou": t,
                                   **results["removed"], "AP": values["legacy_ap"],
                                   **values["all_outputs"]})
    for method, thresholds in sensitivity["score_workpoints"].items():
        for t, values in thresholds.items():
            score_workpoint_rows.append({"method": method, "iou": t,
                                         "legacy_AP": values["legacy_ap"], "stable_AP": values["stable_ap"],
                                         "R_at_50": values["topk"]["50"]["recall_percent"],
                                         "R_at_5FP": values["fixed_fp_per_scene"]["5"]["recall_percent"],
                                         **values["all_outputs"]})
    for method in ("full", "memory", "layered_common", "unified_common"):
        rows = list(manifest["scenes"].values())
        peaks = [x["peaks"][f"{method}_reachable_bytes"] for x in rows]
        state_rows.append({"method": method, "peak_entries": max(x["peaks"][f"{method}_entries"] for x in rows),
                           "peak_reachable_bytes": max(peaks), "mean_scene_peak_reachable_bytes": float(np.mean(peaks)),
                           "p95_scene_peak_reachable_bytes": float(np.percentile(peaks, 95)),
                           "peak_serialized_bytes": max(x["peaks"][f"{method}_serialized_bytes"] for x in rows)})
    for name, rows in (("branch_prefix_results.csv", branch_rows), ("score_unified_prefix_results.csv", score_rows),
                       ("finite_topk_results.csv", k_rows), ("finite_fp_results.csv", fp_rows),
                       ("dedup_results.csv", dedup_rows), ("state_memory_results.csv", state_rows),
                       ("score_workpoint_results.csv", score_workpoint_rows)):
        write_csv(out/name, rows)
    lines = ["# ScanNet-100：分支机制、统一历史与输出协议实验", "",
             "100场景，1,433个GT，6,817个关键帧；原生输出1,787框。所有路线共享同一冻结原生缓存及辅助证据，"
             "不重新运行前端、不使用GPU。本组AP只在同源配对内比较，不替换旧主表的独立运行数字。", "",
             "## 1. 分支独立的严格因果简单记忆对照", "",
             "原生几何及三因素HCSR分数固定；评估proposal时关闭anchor分支，评估anchor时关闭proposal分支。"
             "读出上限预先固定：proposal=12，anchor=128/256/512/640。简单记忆无需跨关键帧确认。", "",
             "| 路线 | 实际新增框 | 总框 | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|---:|"]
    for r in branch_rows:
        if r["prefix"] == "1.0":
            lines.append(f"| {r['method']} | {r['proposal']+r['anchor']:,} | {r['total']:,} | {r['AP15']:.4f} | {r['AP25']:.4f} | {r['AP50']:.4f} |")
    lines += ["", "终态差值的场景级配对bootstrap：B=2,000，seed=20261029，百分位95%区间，未经多重比较校正。"
              "本表的点估计和区间均采用稳定排序破同；上表AP沿用旧评测器，因此小数末位相减可能与本表不同。", "",
              "| 对比 | AP15稳定破同差值 [95% CI] | AP25稳定破同差值 [95% CI] | AP50稳定破同差值 [95% CI] |", "|---|---|---|---|"]
    for name, rows in cached["bootstrap"].items():
        cells = [f"{rows[str(t)]['point_ap_stable_tie_protocol']:+.4f} [{rows[str(t)]['ci95_ap'][0]:+.4f}, {rows[str(t)]['ci95_ap'][1]:+.4f}]" for t in TS]
        lines.append("| "+name+" | "+" | ".join(cells)+" |")
    lines += ["", "四个前缀的全部AP和实际分支数量见branch_prefix_results.csv。前缀均使用完整场景GT集合，"
              "不是当前可见目标AP，也不是首次可见检测延迟。100%前缀等于该次终态。", "",
              "## 2. 分层历史与统一体素历史", "",
              "统一方案只有一套世界体素支持与淘汰状态，每体素每关键帧最多一次支持，三关键帧确认，"
              "支持计数饱和于3。来源标签仅用于保留各来源最高分代表和应用相同读出评分，不维护两套支持池。"
              "活动/确认容量为5,120/1,664，分别等于分层路径对应容量之和；这不是等字节内存约束。"
              "两路线均使用相同的逐源精确几何去重输入及相同的原生优先精确几何去重读出，"
              "不按终态数量倒推预算，也不填充未确认候选。", "",
              "| 前缀 | 路线 | 原生/P/A | 总框 | AP15 | AP25 | AP50 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in score_rows:
        if r["method"] in ("layered_common", "unified_common"):
            lines.append(f"| {float(r['prefix'])*100:.0f}% | {r['method']} | {r['native']}/{r['proposal']}/{r['anchor']} | {r['total']} | {r['AP15']:.4f} | {r['AP25']:.4f} | {r['AP50']:.4f} |")
    rows = sensitivity["bootstrap"]["layered-minus-unified-common"]
    lines += ["", "分层−统一的终态稳定破同差值及95%区间："+"；".join(
        f"AP{int(t*100)} {rows[str(t)]['point_ap_stable_tie_protocol']:+.4f} [{rows[str(t)]['ci95_ap'][0]:+.4f}, {rows[str(t)]['ci95_ap'][1]:+.4f}]" for t in TS)+"。", "",
        "统一方案是预先固定的一种体素组织实例，不是统一历史方法的最优上界；不能据此宣称所有统一组织均不如分层。", "",
        "## 3. 有限输出与固定误报工作点", "",
        "R@K采用逐场景分数排序和一对一贪心最大IoU匹配（严格IoU>阈值），不使用最大IoU覆盖率替代。"
        "固定FP工作点从每个测试场景的PR序列选择满足预算的完整分数阈值前缀，不拆分同分组；"
        "它是利用测试GT的逐场景oracle诊断，不是一个预先部署的阈值，也不是每100个GT的FP预算。", "",
        "| IoU | 路线 | R@20 | R@50 | R@100 | R@300 | R@1FP/场景 | R@5FP/场景 | R@10FP/场景 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in (.25, .5):
        for method, all_t in cached["terminal_workpoints"].items():
            x = all_t[str(t)]
            vals = [x["topk"][str(k)]["recall_percent"] for k in (20, 50, 100, 300)]
            vals += [x["fixed_fp_per_scene"][str(b)]["recall_percent"] for b in (1, 5, 10)]
            lines.append(f"| {t:.2f} | {method} | "+" | ".join(f"{v:.2f}%" for v in vals)+" |")
    lines += ["", "## 4. 固定几何与数量的评分敏感性", "",
              "仅改变恢复框的分数，原生HCSR分数不变。raw为PLR当前medoid观测原分数／简单记忆代表观测原分数，"
              "以及CALR代表anchor原分数。temperature2为辅助分支共同采用sigmoid(logit(raw)/2)，截断至[1e-6,1−1e-6]后计算。"
              "温度2在新增结果计算前固定，未用测试GT拟合，但也没有独立开发集校准；它只作为排序敏感性探针。", "",
              "| 路线/评分 | 总框 | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|"]
    for r in score_rows:
        if r["prefix"] == "1.0" and r["method"] not in ("layered_common", "unified_common"):
            lines.append(f"| {r['method']} | {r['total']:,} | {r['AP15']:.4f} | {r['AP25']:.4f} | {r['AP50']:.4f} |")
    lines += ["", "评分改变后的有限工作点见score_workpoint_results.csv；没有把仅改变分支内单调数值、"
              "却完全不改变跨分支排序的变换当作稳健性证据。", "",
              "## 5. 固定评分的去重敏感性", "",
              "对Ctrl、HCSR-only、两种简单记忆和Full执行完全相同的协议。原生优先保留全部原生框；"
              "global NMS允许原生框被高分恢复框删除。下表列出原始拼接及两种阈值，不按结果挑选最好阈值。", "",
              "NMS在缓存原始世界坐标系中计算AABB IoU，与现有PLR覆盖排除/分支内NMS一致；"
              "随后按公共ScanNet坐标变换执行检测评测。不是在GT匹配后删除重复预测。", "",
              "| 路线 | 协议 | 保留总框 | 删除原生框 | AP15 | AP25 | AP50 |", "|---|---|---:|---:|---:|---:|---:|"]
    for method in protocol["dedup_sensitivity"]["methods"]:
        original = cached["terminal_workpoints"][method]
        ap15 = cached["prefixes"]["1.0"][method]["ap"]["0.15"]
        lines.append(f"| {method} | concat | {original['0.25']['all_outputs']['predictions']} | 0 | {ap15:.4f} | {original['0.25']['legacy_ap']:.4f} | {original['0.5']['legacy_ap']:.4f} |")
        for policy, r in sensitivity["dedup"][method].items():
            lines.append(f"| {method} | {policy} | {r['removed']['retained']} | {r['removed']['removed_native']} | {r['metrics']['0.15']['legacy_ap']:.4f} | {r['metrics']['0.25']['legacy_ap']:.4f} | {r['metrics']['0.5']['legacy_ap']:.4f} |")
    lines += ["", "同协议下Full−Memory256的AP差值：", "",
              "| 协议 | AP15差值 | AP25差值 | AP50差值 |", "|---|---:|---:|---:|"]
    for policy in ("concat", "native_priority_0.25", "global_nms_0.25", "native_priority_0.5", "global_nms_0.5"):
        if policy == "concat":
            difference = [cached["prefixes"]["1.0"]["full"]["ap"][str(t)]-
                          cached["prefixes"]["1.0"]["memory_a256"]["ap"][str(t)] for t in TS]
        else:
            difference = [sensitivity["dedup"]["full"][policy]["metrics"][str(t)]["legacy_ap"]-
                          sensitivity["dedup"]["memory_a256"][policy]["metrics"][str(t)]["legacy_ap"] for t in TS]
        lines.append("| "+policy+" | "+" | ".join(f"{x:+.4f}" for x in difference)+" |")
    lines += ["", "重复预测是FP的子集。重复FP按其最佳GT首次TP的来源划分为同源重复和跨源重复，两者相加为重复FP；"
              "不能把TP、FP、重复三列当作互斥类别相加。dedup_results.csv另列出TP/FP、重复率、multiplicity及新增唯一GT覆盖。", "",
              "## 6. 实际状态占用", "",
              "每个关键帧测量恢复算法状态的Python可达对象字节，按对象身份消除别名重复并计入NumPy持有缓冲区；"
              "同时测量pickle协议5序列化字节。排除冻结输入缓存、GT、共享HCSR及分析用原分数溯源元数据。"
              "诊断事件日志不纳入算法状态，但另行记录其终态序列化大小。以下不是进程RSS或GPU显存，"
              "不同记录字段不等长，不能将同条目容量称为同内存。", "",
              "| 路线 | 实际条目峰值 | 可达对象峰值(MiB) | 序列化峰值(MiB) |", "|---|---:|---:|---:|"]
    for r in state_rows:
        lines.append(f"| {r['method']} | {r['peak_entries']} | {r['peak_reachable_bytes']/2**20:.4f} | {r['peak_serialized_bytes']/2**20:.4f} |")
    lines += ["", "当前CALR保留全部支持帧号集合，诊断日志也可继续增长；本实验不证明现有实现的历史字节存储对无限序列严格有界。"
              "统一方案使用饱和计数并不自动修复或修改当前Full实现。", "",
              "## 7. 复现与解释边界", "",
              "协议在新结果产生前写入protocol.json，逐场景缓存由协议SHA256绑定。重放逐场景核验四个前缀的Full和memory256框、"
              "分数、顺序与上一组预测完全相同；统一历史两路线的原生几何及HCSR分数也逐框核验。", "",
              "AP点估计沿用原评测器；Top-K、PR和bootstrap采用稳定排序破同。同分排序差异可能造成末位AP变化，"
              "JSON同时保留legacy_ap和stable_ap，不能将小数末位差异视为机制增益。"
              "本批是ScanNet-100探索性诊断，不包含测试集拟合的标定，不新增独立跨数据集确认主张。", "",
              f"新增分支/统一历史对比中，稳定与旧破同协议造成的差值最大变化为"
              f"{validation['maximum_contrast_delta_from_tie_protocol']:.6f}个AP点；"
              "统一历史AP25的极小差值可随破同改变符号，两种统计下均不能支持稳定的分层优势。", "",
              "ByteTrack风格关联、因果Seq-NMS、CP-Cluster及独立开发集标定不属于本批最小四组，未报告为已完成。", ""]
    lines += ["## 8. 可直接用于论文讨论的结论", "",
              "**候选保留与具体组织规则。** 在固定HCSR原生前缀、关闭另一恢复分支后，PLR相对proposal简单记忆的"
              "旧评测器AP15/AP25/AP50点估计高0.40/0.35/0.08，但三项配对区间均跨零；"
              "不能把此前终态同数量实验的稳定增益直接推广到本次更强的固定容量严格因果基线。"
              "CALR与256预算anchor记忆基本持平，512/640预算记忆在AP15上更高。"
              "这些结果表明持续保留辅助候选本身解释了相当部分覆盖收益，未证明三关键帧确认是更强筛选器。", "",
              "**层级状态的必要性。** 在相同候选流、公共精确去重及读出评分下，分层方案与预设统一体素方案终态AP接近，"
              "三项配对区间均跨零。因此当前结果支持分层设计作为一种实现选择，而不支持其在精度上的必要性或普遍优越性。", "",
              "**输出协议边界。** Full的评分从当前规则改为代表候选原分数后，AP从47.21/42.72/22.15降至"
              "42.35/38.85/21.31，表明跨分支评分影响绝对AP；两种恢复策略均受此影响。"
              "对所有方法施加相同去重后，低IoU AP可提高，但Full的AP50下降；Full相对Memory256的差距也缩小。"
              "因此应把评分与输出协议视为完整系统定义的一部分，不能声称恢复组织收益独立于这些协议。", "",
              "**可用工作点与存储。** 有限K结果仍显示完整方法相对Ctrl和HCSR-only的覆盖改善，"
              "但低FP收益必须逐工作点陈述，不能用完整PR尾部收益代替。"
              "完整方法恢复状态的可达对象峰值高于简单记忆，尽管条目数更少；本次数据不支持更省内存的主张。", "",
              "**建议的最终定位。** ReCaR-3D是组织独立辅助候选并重排原生记录的因果流式框架。"
              "系统总收益、候选记忆收益与层级组织的额外收益应分别陈述；"
              "CALR保留长尾覆盖定位，不再将其描述为已证明优于简单记忆的精筛机制。", ""]
    (out/"实验结果与论文补充内容.md").write_text("\n".join(lines), encoding="utf-8")
    print("CSV tables and paper-ready report written", flush=True)


def render_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.loads((out/"cached_results.json").read_text())
    methods = ["base", "hcsr", "memory_a128", "memory_a256", "full"]
    names = ["BoxFusion-Ctrl", "HCSR only", "Memory (128)", "Memory (256)", "ReCaR-3D"]
    colors = ["#8c8c8c", "#4e79a7", "#59a14f", "#af7aa1", "#e15759"]
    styles = ["-", "-.", "--", ":", "-"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5), sharey="row")
    for col, threshold in enumerate((.25, .5)):
        for method, name, color, style in zip(methods, names, colors, styles):
            vals = r["terminal_workpoints"][method][str(threshold)]
            axes[0, col].plot([20, 50, 100, 300], [vals["topk"][str(k)]["recall_percent"] for k in (20, 50, 100, 300)],
                              marker="o", markersize=3, lw=1.2, label=name, color=color, linestyle=style)
            axes[1, col].plot([1, 5, 10], [vals["fixed_fp_per_scene"][str(k)]["recall_percent"] for k in (1, 5, 10)],
                              marker="o", markersize=3, lw=1.2, label=name, color=color, linestyle=style)
        axes[0, col].set_title(f"3D AABB IoU > {threshold:.2f}")
        axes[0, col].set_xlabel("Per-scene output limit K")
        axes[1, col].set_xlabel("FP budget per scene (oracle threshold)")
        axes[0, col].set_xticks([20, 50, 100, 300]); axes[0, col].set_ylim(0, 100)
        axes[1, col].set_xticks([1, 5, 10]); axes[1, col].set_ylim(0, 65)
        for ax in axes[:, col]:
            ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Recall (%)"); axes[1, 0].set_ylabel("Recall (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out/f"finite_workpoints.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    for ax, threshold in zip(axes, (.25, .5)):
        for method, name, color, style in zip(methods, names, colors, styles):
            with np.load(out/f"pr_{method}.npz") as curve:
                ax.plot(curve[f"iou{threshold}_recall"], curve[f"iou{threshold}_precision"], label=name, color=color, lw=1.2, linestyle=style)
        ax.set_title(f"3D AABB IoU > {threshold:.2f}"); ax.set_xlabel("Recall")
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Precision")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .81))
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out/f"complete_pr_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--phase", choices=("cached", "sensitivity", "report"), default="cached")
    args = parser.parse_args()
    check_small_metrics()
    protocol = json.loads((args.output/"protocol.json").read_text())
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines() if s.strip() and not s.startswith("#")]
    gts, alignments, transform = audit.core.scannet_inputs()
    if args.phase == "cached":
        completed = args.output/"cached_results.json"
        if completed.exists() and "bootstrap" in json.loads(completed.read_text()):
            print("cached phase already complete; preserved existing results", flush=True)
            return
        evaluate_cached(args.output, protocol, scenes, gts, alignments, transform)
    elif args.phase == "sensitivity":
        completed = args.output/"sensitivity_results.json"
        if completed.exists() and "bootstrap" in json.loads(completed.read_text()):
            print("sensitivity phase already complete; preserved existing results", flush=True)
            return
        evaluate_sensitivity(args.output, protocol, scenes, gts, alignments, transform)
    else:
        render_report(args.output, protocol)
        render_figures(args.output)


if __name__ == "__main__":
    main()
