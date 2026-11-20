#!/usr/bin/env python3
"""Fixed-protocol branch-isolated MultiScan strong-memory comparison, CPU only."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev
from tools.bootstrap_final_recar_pairs import scene_pairs, ap_from_pairs
from tools.run_hcsr_current_zero_control import same_rows
from tools.replay_scannet_unified_history_controls import reachable_bytes
from boxfusion.online_candidate_map import OnlineProposalRecovery
from boxfusion.m1_anchor_online import OnlineAnchorRecovery

core = ev.audit.core
DEFAULT = ROOT/"reports/multiscan50_branch_strong_memory_20261031"
EMPTY = np.empty((0, 8, 3), dtype=np.float64)


def raw_rows(rows):
    return [(0, r["box"].copy(), float(r["score"])) for r in rows]


def check_reference(rows, reference):
    same_count = len(rows) == len(reference)
    if not same_count:
        return {"same_count": False, "new_count": len(rows), "reference_count": len(reference)}
    a = np.asarray([r[1] for r in rows]).reshape(-1, 8, 3)
    b = np.asarray([r[1] for r in reference]).reshape(-1, 8, 3)
    return {"same_count": True, "same_labels": [r[0] for r in rows] == [r[0] for r in reference],
            "scores_exact": [r[2] for r in rows] == [r[2] for r in reference],
            "geometry_exact": np.array_equal(a, b),
            "geometry_float32_exact": np.array_equal(a.astype(np.float32), b.astype(np.float32)),
            "max_corner_error_m": float(np.max(abs(a-b))) if len(a) else 0.}


def check_native_independence(config, experiment):
    path_root = ROOT/"development/final_controls/experiments/multiscan50_mvsr_temporal_patch_20261012/native_prefix"
    result = {}
    for path in sorted(path_root.glob("*.npz")):
        scene = path.stem
        native = dict(core.load_native(path))
        empty_state = OnlineProposalRecovery(**config["m1p"])
        native_state = OnlineProposalRecovery(**config["m1p"])
        checked, nonempty = 0, 0
        for ordinal, (frame, evidence) in enumerate(core.load_evidence(experiment/"evidence"/f"{scene}.npz")):
            boxes = native[frame].corners if frame in native else EMPTY
            kwargs = dict(proposal_ids=evidence.proposal_ids, proposal_corners=evidence.proposal_corners,
                          proposal_scores=evidence.proposal_scores)
            empty_state.update(ordinal, frame, **kwargs)
            native_state.update(ordinal, frame, native_corners=boxes, **kwargs)
            assert same_rows(raw_rows(empty_state.terminal_rows(boxes)), raw_rows(native_state.terminal_rows(boxes)))
            assert list(empty_state.tracks) == list(native_state.tracks)
            assert list(empty_state.births) == list(native_state.births)
            checked += 1; nonempty += int(len(boxes) > 0)
        result[scene] = {"all_keyframes_checked": checked, "nonempty_native_snapshots": nonempty,
                         "readout_and_track_identity_equal": True}
    assert len(result) == 5 and sum(r["nonempty_native_snapshots"] for r in result.values()) > 0
    return result


def replay(out, p, scenes):
    experiment = ROOT/p["experiment_root"]
    config = yaml.safe_load((experiment/"config.yaml").read_text())["online_candidate_map"]["state"]
    expected = dict(use_children=False, proposal_min_views=3, child_min_views=2, ttl_keyframes=10,
        match_iou=.1, match_center_m=.5, native_dedup_iou=.25, self_nms_iou=.5,
        max_tracks=1024, max_births=12, max_observations_per_track=12, max_observations_per_frame=192)
    assert config["m1p"] == expected
    digest = core.sha256(out/"protocol.json")
    manifest = {"protocol_sha256": digest, "source_sha256": {}, "input_sha256": {}, "scenes": {},
                "native_independence": check_native_independence(config, experiment),
                "prefix_AP_claimed": False, "models_executed": False, "GPU_used": False}
    for source in (Path(__file__), core.SOURCE/"boxfusion/online_candidate_map.py",
                   core.SOURCE/"boxfusion/m1_anchor_online.py", Path(core.__file__)):
        manifest["source_sha256"][str(source)] = core.sha256(source)
    (out/"scenes").mkdir(exist_ok=True)
    for index, scene in enumerate(scenes, 1):
        destination = out/"scenes"/f"{scene}.pkl"
        paths = {"evidence": experiment/"evidence"/f"{scene}.npz",
                 "native": experiment/"components/native_factor_no_visibility"/f"{scene}_boxes.pkl",
                 "plr": experiment/"components/births_plr_v1"/f"{scene}_boxes.pkl",
                 "calr": experiment/"components/births_calr_v1"/f"{scene}_boxes.pkl"}
        for path in paths.values():
            manifest["input_sha256"][str(path)] = core.sha256(path)
        if destination.exists():
            with destination.open("rb") as handle:
                saved = pickle.load(handle)
            assert saved["protocol_sha256"] == digest
            manifest["scenes"][scene] = saved["summary"]
            continue
        start = time.perf_counter()
        native = ev.audit.read_rows(paths["native"])
        native_boxes = np.asarray([r[1] for r in native]).reshape(-1, 8, 3)
        proposal = OnlineProposalRecovery(**config["m1p"])
        pmem = core.ProposalScoreMemory(active_capacity=1024, archive_capacity=1024,
            ttl_keyframes=10, output_capacity=12)
        anchor = OnlineAnchorRecovery(scene, voxel_m=.3, min_views=3, max_active=4096, max_births=640)
        amem = core.AnchorScoreMemory(capacity=p["anchor"]["baseline_cap"], output_capacity=640)
        peaks = {name: {"entries": 0, "bytes": 0} for name in ("plr", "p_memory", "calr", "a_memory")}
        times = {name: [] for name in peaks}
        evidence_frames = core.load_evidence(paths["evidence"])
        for ordinal, (frame, e) in enumerate(evidence_frames):
            actions = {
                "plr": lambda: proposal.update(ordinal, frame, proposal_ids=e.proposal_ids,
                    proposal_corners=e.proposal_corners, proposal_scores=e.proposal_scores),
                "p_memory": lambda: pmem.update(ordinal, frame, e.proposal_ids, e.proposal_corners, e.proposal_scores),
                "calr": lambda: anchor.update(ordinal, frame, e.anchor_ids, e.anchor_corners, e.anchor_scores),
                "a_memory": lambda: amem.update(ordinal, frame, e.anchor_ids, e.anchor_corners, e.anchor_scores),
            }
            names = list(actions); names = names[ordinal % 4:]+names[:ordinal % 4]
            for name in names:
                before = time.perf_counter(); actions[name](); times[name].append(time.perf_counter()-before)
            roots = {"plr": (proposal.tracks, proposal.births), "p_memory": (pmem.active, pmem.archive),
                     "calr": (anchor.active, anchor.births), "a_memory": (amem.state,)}
            for name, values in roots.items():
                peaks[name]["entries"] = max(peaks[name]["entries"], sum(map(len, values)))
                peaks[name]["bytes"] = max(peaks[name]["bytes"], reachable_bytes(values))
        before = time.perf_counter(); plr = raw_rows(proposal.terminal_rows(native_boxes)); ptime = time.perf_counter()-before
        before = time.perf_counter(); memory_p = pmem.rows(native); mptime = time.perf_counter()-before
        before = time.perf_counter(); calr = raw_rows(anchor.rows()); atime = time.perf_counter()-before
        before = time.perf_counter(); memory_a = amem.rows(scene); matime = time.perf_counter()-before
        readout_times = dict(plr=ptime, p_memory=mptime, calr=atime, a_memory=matime)
        parity = {"plr": check_reference(plr, ev.audit.read_rows(paths["plr"])),
                  "calr": check_reference(calr, ev.audit.read_rows(paths["calr"]))}
        # Preserve measured differences instead of overwriting replayed boxes
        # with reference geometry just to force agreement.
        outputs = {"hcsr": ev.case(native, ["native"]*len(native))}
        for cap in p["proposal"]["readout_caps"]:
            for name, rows in (("plr", plr), ("p_memory", memory_p)):
                outputs[f"{name}_k{cap}"] = ev.case(native+rows[:cap],
                    ["native"]*len(native)+["proposal"]*len(rows[:cap]))
        for cap in p["anchor"]["readout_caps"]:
            for name, rows in (("calr", calr), ("a_memory", memory_a)):
                outputs[f"{name}_k{cap}"] = ev.case(native+rows[:cap],
                    ["native"]*len(native)+["anchor"]*len(rows[:cap]))
        summary = {"keyframes": len(evidence_frames), "native": len(native), "reference_parity": parity,
            "auxiliary_outputs": {name: len(v["rows"])-len(native) for name, v in outputs.items()},
            "core_peak_state": peaks, "terminal_readout_seconds": readout_times,
            "core_update_seconds": {name: {"sum": float(sum(t)), "mean": float(np.mean(t)),
                "p95": float(np.percentile(t, 95))} for name, t in times.items()},
            "memory_anchor_capacity_drops": amem.capacity_drops,
            "seconds": time.perf_counter()-start}
        payload = {"protocol_sha256": digest, "summary": summary, "outputs": outputs}
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        temporary.replace(destination)
        for name, value in outputs.items():
            core.write_prediction(out/"predictions"/name/f"{scene}_boxes.pkl", value["rows"])
        manifest["scenes"][scene] = summary
        ev.save_json(out/"replay_manifest.partial.json", manifest)
        print(f"MultiScan branch replay {index}/{len(scenes)} {scene} {summary['seconds']:.1f}s", flush=True)
    assert len(manifest["scenes"]) == 50
    assert sum(v["native"] for v in manifest["scenes"].values()) == 621
    ev.save_json(out/"replay_manifest.json", manifest)
    return manifest


def bootstrap(predictions, gts, scenes, comparisons, p):
    thresholds = p["iou_thresholds"]
    needed = sorted({arm for pair in comparisons.values() for arm in pair})
    pairs = {arm: {t: {scene: scene_pairs(*predictions[arm][scene], gts[scene], t)
              for scene in scenes} for t in thresholds} for arm in needed}
    gt_sizes = np.asarray([len(gts[s]) for s in scenes])
    draws = np.random.default_rng(p["bootstrap_seed"]).integers(0, len(scenes),
        (p["bootstrap_replicates"], len(scenes)))
    samples = {name: {str(t): [] for t in thresholds} for name in comparisons}
    for index, draw in enumerate(draws):
        selected = [scenes[int(i)] for i in draw]; gt_total = int(gt_sizes[draw].sum())
        ap = {arm: {t: ap_from_pairs(np.concatenate([pairs[arm][t][s] for s in selected]), gt_total)
                    for t in thresholds} for arm in needed}
        for name, (left, right) in comparisons.items():
            for t in thresholds:
                samples[name][str(t)].append(ap[left][t]-ap[right][t])
        if (index+1) % 500 == 0:
            print(f"bootstrap {index+1}/{len(draws)}", flush=True)
    total_gt = int(gt_sizes.sum())
    points = {arm: {str(t): ap_from_pairs(np.concatenate([pairs[arm][t][s] for s in scenes]), total_gt)
                   for t in thresholds} for arm in needed}
    return {name: {str(t): {"point_ap": points[left][str(t)]-points[right][str(t)],
        "ci95_ap": np.percentile(samples[name][str(t)], [2.5, 97.5]).tolist(),
        "ci97_5_ap": np.percentile(samples[name][str(t)], [1.25, 98.75]).tolist(),
        "p_positive": float(np.mean(np.asarray(samples[name][str(t)]) > 0))}
        for t in thresholds} for name, (left, right) in comparisons.items()}


def union_csv(path, records):
    columns = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(records)


def evaluate(out, p, scenes, manifest):
    prepared = ROOT/p["prepared_root"]
    cases = {}
    for scene in scenes:
        with (out/"scenes"/f"{scene}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        assert payload["protocol_sha256"] == manifest["protocol_sha256"]
        for name, value in payload["outputs"].items():
            cases.setdefault(name, {})[scene] = value
    predictions = {name: {s: (np.asarray([r[1] for r in v["rows"]]).reshape(-1, 8, 3),
                                   np.asarray([r[2] for r in v["rows"]]), np.asarray(v["sources"]))
                             for s, v in values.items()} for name, values in cases.items()}
    comparisons = {**{f"PLR_K{k}": (f"plr_k{k}", f"p_memory_k{k}") for k in p["proposal"]["readout_caps"]},
                   **{f"CALR_K{k}": (f"calr_k{k}", f"a_memory_k{k}") for k in p["anchor"]["readout_caps"]}}
    result = {"protocol": p, "protocol_sha256": manifest["protocol_sha256"], "scene_count": len(scenes),
              "native_count": 621, "keyframes": sum(x["keyframes"] for x in manifest["scenes"].values()),
              "comparisons": comparisons, "protocols": {}}
    ap_rows, quality_rows, finite_rows, ci_rows = [], [], [], []
    for label, gt_file in (("nonstructural", "gt_aabb_nonstructural.npy"), ("scannet18", "gt_aabb_scannet18.npy")):
        gts = {s: np.load(prepared/s/gt_file).astype(np.float64) for s in scenes}
        point = {"GT_count": sum(map(len, gts.values())), "arms": {}}
        for arm, values in predictions.items():
            metrics, _, _ = ev.workpoints(values, gts, thresholds=tuple(p["iou_thresholds"]),
                topks=tuple(p["top_k_per_scene"]), fp_budgets=tuple(p["fp_per_scene"]))
            row = {"total_count": sum(len(v[0]) for v in values.values()),
                   "auxiliary_count": sum(sum(x != "native" for x in v[2]) for v in values.values()),
                   "metrics": metrics}
            point["arms"][arm] = row
            ap_rows.append({"GT_protocol": label, "arm": arm, "total_count": row["total_count"],
                "auxiliary_count": row["auxiliary_count"],
                **{f"AP{int(t*100)}": metrics[str(t)]["legacy_ap"] for t in p["iou_thresholds"]},
                **{f"stable_AP{int(t*100)}": metrics[str(t)]["stable_ap"] for t in p["iou_thresholds"]}})
            for t, m in metrics.items():
                quality_rows.append({"GT_protocol": label, "arm": arm, "iou": t, "source": "all", **m["all_outputs"]})
                for source, values in m["per_source"].items():
                    quality_rows.append({"GT_protocol": label, "arm": arm, "iou": t, "source": source, **values,
                        "precision_percent": 100*values["tp"]/max(values["predictions"], 1)})
                for budget, values in m["topk"].items():
                    finite_rows.append({"GT_protocol": label, "arm": arm, "iou": t,
                        "type": "TopK", "budget": budget, **values})
                for budget, values in m["fixed_fp_per_scene"].items():
                    finite_rows.append({"GT_protocol": label, "arm": arm, "iou": t,
                        "type": "oracle_FP_per_scene", "budget": budget, **values})
        result["protocols"][label] = point
        ev.save_json(out/"point_results.json", result)
        print(f"{label}: point estimates complete; computing intervals", flush=True)
        plain = {arm: {s: (x[0], x[1]) for s, x in values.items()} for arm, values in predictions.items()}
        ci = bootstrap(plain, gts, scenes, comparisons, p)
        point["paired_bootstrap"] = ci
        for name, rows in ci.items():
            for t, value in rows.items():
                ci_rows.append({"GT_protocol": label, "contrast": name, "iou": t,
                    "point_ap": value["point_ap"], "ci95_low": value["ci95_ap"][0], "ci95_high": value["ci95_ap"][1],
                    "ci97_5_low": value["ci97_5_ap"][0], "ci97_5_high": value["ci97_5_ap"][1]})
        point["predefined_checks"] = {}
        for branch, caps, main_cap in (("PLR", p["proposal"]["readout_caps"], 12),
                                      ("CALR", p["anchor"]["readout_caps"], 256)):
            point["predefined_checks"][branch] = {
                "primary_AP25_97_5_lower_gt0": ci[f"{branch}_K{main_cap}"]["0.25"]["ci97_5_ap"][0] > 0,
                "all_budget_AP25_95_lower_gt0": all(ci[f"{branch}_K{k}"]["0.25"]["ci95_ap"][0] > 0 for k in caps),
                "primary_cap_all_IoU_95_lower_gt0": all(ci[f"{branch}_K{main_cap}"][str(t)]["ci95_ap"][0] > 0 for t in p["iou_thresholds"]),
            }
        ev.save_json(out/"partial_results.json", result)
    for name, records in (("AP_results.csv", ap_rows), ("quality_counts.csv", quality_rows),
                          ("finite_workpoints.csv", finite_rows), ("paired_intervals.csv", ci_rows)):
        union_csv(out/name, records)
    resources = []
    for scene, s in manifest["scenes"].items():
        for name, peak in s["core_peak_state"].items():
            resources.append({"scene": scene, "method": name, "keyframes": s["keyframes"],
                **peak, "terminal_readout_seconds": s["terminal_readout_seconds"][name],
                **{f"core_update_seconds_{k}": v for k, v in s["core_update_seconds"][name].items()}})
    union_csv(out/"core_resources.csv", resources)
    result["core_resources"] = {name: {
        "peak_reachable_bytes": max(r["bytes"] for r in resources if r["method"] == name),
        "peak_entries": max(r["entries"] for r in resources if r["method"] == name),
        "mean_core_update_ms": 1000*sum(r["core_update_seconds_sum"] for r in resources if r["method"] == name)/result["keyframes"],
    } for name in ("plr", "p_memory", "calr", "a_memory")}
    ev.save_json(out/"results.json", result)
    return result


def report(out, r, manifest):
    lines = ["# MultiScan-50：PLR、CALR独立对强简单记忆的固定协议验证", "",
        "全部路线共享同一冻结候选流与同一组621个原生框及其HCSR分数，逐分支比较，另一恢复分支关闭。"
        "不运行模型或占用GPU，不根据结果修改参数、预算或评分。该集合以前已有评测，不能称为新的独立盲测。", "",
        "## 1. 协议与有效性", "",
        "proposal预设读出上限3/6/12，anchor为128/256/640。PLR与proposal记忆共享1024活动/1024归档条目上限、"
        "覆盖排除和分支内NMS及尺度分桶输出分数。CALR为4096活动+640确认状态，强anchor记忆给足4736条目，"
        "双方共享低分尾部映射。相同条目上限不是等字节内存，也不强制实际输出数量相同。", "",
        "主要端点预先固定为非结构GT协议的AP25：PLR K12和CALR K256。两项各用97.5%配对bootstrap区间"
        "（Bonferroni名义家族95%）；其余预算、AP阈值及ScanNet18映射均完整报告为探索性次要分析。B=2000，seed=20261031。", "",
        "有完整50场逐关键帧候选缓存，但原生前缀只有5场。本次不把终态原生图回填到过去：PLR确认与代表维护不依赖原生覆盖标志，"
        "该标志只影响当前读出；重建候选历史后，才在终态针对真实终态原生框读出。已在5场可用原生前缀的每个缓存关键帧验证"
        "带/不带历史原生覆盖的状态与当前读出等价。本组只报告终态分支精度，不报告完整系统的在线前缀AP。", ""]
    parity = [v["reference_parity"][branch] for v in manifest["scenes"].values() for branch in ("plr", "calr")]
    equal_f32 = sum(x.get("geometry_float32_exact", False) and x.get("scores_exact", False) for x in parity)
    max_error = max(x.get("max_corner_error_m", float("inf")) for x in parity)
    lines += [f"100份分支参考核对中，{equal_f32}份的分数及float32几何逐行一致；最大几何误差{max_error:.9g} m。"
              "新比较双方统一使用缓存几何，不以旧主表数字代替本次重放结果。", ""]
    for label, point in r["protocols"].items():
        lines += [f"## 2. {label}（{point['GT_count']}个GT）", "",
            "| 路线 | 新增框 | 总框 | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|---:|"]
        for arm, value in point["arms"].items():
            a = value["metrics"]
            lines.append(f"| {arm} | {value['auxiliary_count']:,} | {value['total_count']:,} | {a['0.15']['legacy_ap']:.4f} | {a['0.25']['legacy_ap']:.4f} | {a['0.5']['legacy_ap']:.4f} |")
        lines += ["", "以下差值及区间均采用稳定破同；上表旧评测器点估计的末位可能不同，不将两种排序协议混用。", "",
            "| 对照（提出方法−对应记忆） | ΔAP15 [95% CI] | ΔAP25 [95% CI] | ΔAP50 [95% CI] |", "|---|---|---|---|"]
        for contrast, values in point["paired_bootstrap"].items():
            cells = []
            for t in (.15, .25, .5):
                v = values[str(t)]; lo, hi = v["ci95_ap"]
                cells.append(f"{v['point_ap']:+.4f} [{lo:+.4f},{hi:+.4f}]")
            lines.append("| "+contrast+" | "+" | ".join(cells)+" |")
        lines += ["", "| 分支 | 主要预算AP25的97.5% CI | 所有预算AP25的95%下界均>0 | 主要预算三种IoU的95%下界均>0 |", "|---|---|---|---|"]
        for branch, cap in (("PLR", 12), ("CALR", 256)):
            lo, hi = point["paired_bootstrap"][f"{branch}_K{cap}"]["0.25"]["ci97_5_ap"]
            checks = point["predefined_checks"][branch]
            lines.append(f"| {branch} | [{lo:+.4f},{hi:+.4f}] | {checks['all_budget_AP25_95_lower_gt0']} | {checks['primary_cap_all_IoU_95_lower_gt0']} |")
    lines += ["", "## 3. 资源记录与解释边界", "",
        "以下仅为辅助候选历史核心状态，不含原生覆盖标志的在线维护、模型前端、调度与队列。CPU重放每帧轮换四方法执行次序。"
        "这些诊断不是端到端FPS或完整部署内存，不能据此声称系统更快。", "",
        "| 方法 | 核心状态条目峰值 | 可达对象峰值MiB | 核心更新均值ms/关键帧 |", "|---|---:|---:|---:|"]
    for name, v in r["core_resources"].items():
        lines.append(f"| {name} | {v['peak_entries']} | {v['peak_reachable_bytes']/2**20:.4f} | {v['mean_core_update_ms']:.4f} |")
    lines += ["", "## 4. 结果应如何使用", ""]
    for branch in ("PLR", "CALR"):
        c = r["protocols"]["nonstructural"]["predefined_checks"][branch]
        lines.append(f"{branch}：预设主要端点优势{'得到支持' if c['primary_AP25_97_5_lower_gt0'] else '未建立'}；"
                     f"预算稳健AP25优势{'得到支持' if c['all_budget_AP25_95_lower_gt0'] else '未建立'}。")
        lines.append("")
    lines += ["本组不覆盖所有可能的记忆方法；无正区间不等于统计等效。即使某个外部工作点更好，也不能覆盖此前ScanNet强记忆"
        "比较的非显著结果或据此声称跨数据集普遍优越。保留全部预设结果，不根据结果调整默认路线。", "",
        "AP_results.csv和paired_intervals.csv包含完整精度及区间；quality_counts.csv包含TP/FP、重复及multiplicity；"
        "finite_workpoints.csv包含Top-K及固定FP诊断（后者使用测试GT选阈值，不是部署工作点）；core_resources.csv是逐场景资源记录。"
        "预测及所有输入指纹均保留，未修改旧实验或主表。"]
    (out/"实验结果与结论.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    subprocess.run(["/home/admin1/miniconda3/envs/temp/bin/python", str(ROOT/"tools/build_supplement_docx.py"),
        str(out/"实验结果与结论.md"), str(out/"实验结果与结论.docx"), "--footer",
        "ReCaR-3D Branch Strong-Memory Replication — 2026-10-31"], check=True)


def validate(out, p, manifest, result):
    assert core.sha256(out/"protocol.json") == manifest["protocol_sha256"] == result["protocol_sha256"]
    for path, sha in manifest["input_sha256"].items():
        assert core.sha256(Path(path)) == sha
    count = 0
    for scene in manifest["scenes"]:
        with (out/"scenes"/f"{scene}.pkl").open("rb") as handle:
            data = pickle.load(handle)
        reference = data["outputs"]["hcsr"]["rows"]
        for arm, value in data["outputs"].items():
            assert same_rows(reference, value["rows"][:len(reference)])
            if arm != "hcsr":
                assert len(value["rows"])-len(reference) <= int(arm.split("_k")[1])
            count += 1
    assert count == 650
    assert result["protocols"]["nonstructural"]["GT_count"] == 1614
    assert result["protocols"]["scannet18"]["GT_count"] == 441
    for point in result["protocols"].values():
        for arm, values in point["arms"].items():
            for t, m in values["metrics"].items():
                for q in [m["all_outputs"], *m["per_source"].values()]:
                    assert q["predictions"] == q["tp"]+q["fp"]
                    assert q["within_duplicate_fp"]+q["cross_duplicate_fp"] <= q["fp"]
        for contrast, (left, right) in result["comparisons"].items():
            for t in p["iou_thresholds"]:
                delta = point["arms"][left]["metrics"][str(t)]["stable_ap"]-point["arms"][right]["metrics"][str(t)]["stable_ap"]
                assert abs(delta-point["paired_bootstrap"][contrast][str(t)]["point_ap"]) < 1e-9
    ev.save_json(out/"validation.json", {"status": "PASS", "scenes": 50, "shared_native_count": 621,
        "input_hashes_verified": len(manifest["input_sha256"]), "native_and_scores_identical_in_all650_cases": True,
        "all_readout_caps_verified": True, "TP_FP_duplicate_accounting": True,
        "paired_differences_match_stable_AP": True, "no_GPU_or_new_model_inference": True,
        "full_online_prefix_claimed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--phase", choices=("all", "replay", "evaluate", "report", "validate"), default="all")
    args = parser.parse_args()
    p = json.loads((args.output/"protocol.json").read_text())
    prepared = json.loads((ROOT/p["prepared_root"]/"manifest.json").read_text())
    scenes = [r["alias"] for r in prepared["scans"]]
    assert len(scenes) == len(set(scenes)) == 50
    if args.phase in ("all", "replay"):
        replay(args.output, p, scenes)
        if args.phase == "replay":
            return
    manifest = json.loads((args.output/"replay_manifest.json").read_text())
    if args.phase in ("all", "evaluate") and not (args.output/"results.json").exists():
        evaluate(args.output, p, scenes, manifest)
    r = json.loads((args.output/"results.json").read_text())
    if args.phase != "validate":
        report(args.output, r, manifest)
    validate(args.output, p, manifest, r)
    print(json.dumps({"status": "complete", "output": str(args.output),
        "primary_protocol_checks": r["protocols"]["nonstructural"]["predefined_checks"]}), flush=True)


if __name__ == "__main__":
    main()
