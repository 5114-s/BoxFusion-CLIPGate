#!/usr/bin/env python3
"""One-factor recovery controls on frozen ScanNet streams; never run a model."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
import io
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
from boxfusion.online_candidate_map import OnlineProposalRecovery
from boxfusion.m1_anchor_online import OnlineAnchorRecovery

core = ev.audit.core
DEFAULT = ROOT / "reports/scannet_recovery_factor_controls_20261031"


class ProposalControl(OnlineProposalRecovery):
    """Override a single PLR factor; inherit lifecycle, scores and readout."""
    def __init__(self, *, organizer, representative, **kwargs):
        super().__init__(**kwargs)
        self.organizer, self.representative = organizer, representative
        self.created, self.confirmed, self.first_readout = {}, {}, {}
        self.clock = (-1, -1)

    def _match(self, observation):
        if self.organizer == "track":
            return super()._match(observation)
        key = core.voxel_key(observation.corners)
        choices = [t for t in self.tracks.values()
                   if core.voxel_key(t.latest.corners) == key]
        return min(choices, key=lambda t: t.track_id) if choices else None

    def _medoid(self, track):
        if self.representative == "medoid":
            return OnlineProposalRecovery._medoid(track)
        return max(track.observations.values(),
                   key=lambda r: (r.score, -r.ordinal, -r.source_id))

    def _new_track(self):
        track = super()._new_track()
        if track is not None:
            self.created[track.track_id] = self.clock
        return track

    def _try_birth(self, track, frame_id, native_corners):
        birth = super()._try_birth(track, frame_id, native_corners)
        if birth is not None:
            self.confirmed[track.track_id] = self.clock
        return birth

    def update(self, ordinal, frame_id, **kwargs):
        self.clock = (int(ordinal), int(frame_id))
        return super().update(ordinal, frame_id, **kwargs)

    def readout(self, native):
        rows = self.terminal_rows(native)
        for row in rows:
            self.first_readout.setdefault(row["track_id"], self.clock)
        return rows


class AnchorControl(OnlineAnchorRecovery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created, self.confirmed, self.first_readout = {}, {}, {}
        self.clock = (-1, -1)

    def _key(self, box):
        key = super()._key(box)
        if key not in self.active and key not in self.births:
            self.created[key] = self.clock
        return key

    def update(self, ordinal, frame_id, *args):
        self.clock = (int(ordinal), int(frame_id))
        births = super().update(ordinal, frame_id, *args)
        for row in births:
            self.confirmed[tuple(row["voxel_key"])] = self.clock
        return births


def delay_rows(scene, arm, state, emitted=None):
    result = []
    readout = state.first_readout if emitted is None else emitted
    for identity, start in state.created.items():
        confirmation, first = state.confirmed.get(identity), readout.get(identity)
        result.append({
            "scene": scene, "arm": arm, "group": str(identity),
            "creation_ordinal": start[0], "creation_frame": start[1],
            "confirmed": confirmation is not None, "ever_read": first is not None,
            "confirmation_keyframe_intervals": None if confirmation is None else confirmation[0]-start[0],
            "confirmation_input_seconds": None if confirmation is None else (confirmation[1]-start[1])/30.,
            "readout_keyframe_intervals": None if first is None else first[0]-start[0],
            "readout_input_seconds": None if first is None else (first[1]-start[1])/30.,
        })
    return result


def self_test():
    def box(x):
        return np.array([[x+dx, y, z] for dx in (-.1, .1)
                         for y in (.4, .6) for z in (1., 1.2)])
    def update(state, ordinal, xs, scores):
        state.update(ordinal, ordinal*25, proposal_ids=np.arange(len(xs)),
                     proposal_corners=np.stack([box(x) for x in xs]),
                     proposal_scores=np.asarray(scores), native_corners=np.empty((0, 8, 3)))
    kwargs = dict(use_children=False, proposal_min_views=3)
    track = ProposalControl(organizer="track", representative="medoid", **kwargs)
    voxel = ProposalControl(organizer="voxel", representative="medoid", **kwargs)
    for ordinal, x in enumerate((.29, .31, .29)):
        update(track, ordinal, [x], [.7]); update(voxel, ordinal, [x], [.7])
    assert len(track.births) == 1 and len(voxel.births) == 0
    medoid = ProposalControl(organizer="track", representative="medoid", **kwargs)
    highest = ProposalControl(organizer="track", representative="highest", **kwargs)
    for ordinal, (x, score) in enumerate(zip((.01, .02, .09), (.6, .7, .9))):
        update(medoid, ordinal, [x], [score]); update(highest, ordinal, [x], [score])
        assert (len(medoid.births) > 0) == (ordinal == 2)
    assert np.allclose(medoid.births[0].corners, box(.02))
    assert np.allclose(highest.births[0].corners, box(.09))
    repeated = ProposalControl(organizer="track", representative="medoid", **kwargs)
    update(repeated, 0, [.01, .02, .03], [.6, .7, .8])
    assert len(repeated.births) == 0
    one = ProposalControl(organizer="track", representative="medoid",
                          use_children=False, proposal_min_views=1)
    update(one, 0, [.02], [.7]); assert len(one.births) == 1
    anchor = AnchorControl("test", min_views=3)
    for ordinal in range(3):
        anchor.update(ordinal, ordinal*25, [0, 1], np.stack([box(.02)]*2), [.6, .8])
        assert (len(anchor.births) > 0) == (ordinal == 2)
    assert len(anchor.confirmed) == 1
    print("PASS: association predicate, representatives, distinct-frame confirmation and one-frame control", flush=True)


def replay(out, protocol, scenes, stop_after=None):
    parent = ROOT / protocol["parent_report"]
    old = json.loads((parent/"results.json").read_text())
    config = yaml.safe_load((ROOT/protocol["state_config"]).read_text())["online_candidate_map"]["state"]
    protocol_hash = core.sha256(out/"protocol.json")
    manifest = {"protocol_sha256": protocol_hash, "scenes": {}, "input_sha256": {},
                "model_inference": False, "gpu_used": False,
                "production_source_modified": False}
    (out/"scenes").mkdir(exist_ok=True)
    for index, scene in enumerate(scenes, 1):
        paths = [ROOT/protocol[key]/f"{scene}.npz" for key in (
            "native_prefix_root", "evidence_root", "hcsr_readouts")]
        for path in paths:
            value = core.sha256(path)
            if str(path) in old["input_sha256"]:
                assert value == old["input_sha256"][str(path)]
            manifest["input_sha256"][str(path)] = value
        path = out/"scenes"/f"{scene}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                cached = pickle.load(handle)
            assert cached["protocol_sha256"] == protocol_hash
            manifest["scenes"][scene] = cached["summary"]
            continue
        started = time.perf_counter()
        native_frames, evidence_frames = core.load_native(paths[0]), core.load_evidence(paths[1])
        with np.load(paths[2], allow_pickle=False) as z:
            hcsr = {key: z[key] for key in ("frame_ids", "lengths", "ids", "scores_historical_max")}
        parts = core.slices(hcsr["lengths"])
        assert [f for f, _ in native_frames] == [f for f, _ in evidence_frames] == hcsr["frame_ids"].tolist()
        proposals = {}
        for name, spec in protocol["proposal_arms"].items():
            kwargs = {**config["m1p"], "proposal_min_views": spec["min_views"]}
            proposals[name] = ProposalControl(organizer=spec["organizer"],
                representative=spec["representative"], **kwargs)
        anchors = {n: AnchorControl(scene, voxel_m=.3, min_views=n,
                                   max_active=4096, max_births=640) for n in (1, 3)}
        emitted = {f"a{n}_k{k}": {} for n in (1, 3) for k in (128, 256, 640)}
        checkpoints = {int(np.ceil(len(native_frames)*f))-1: str(f)
                       for f in protocol["prefix_fractions"]}
        prefixes, trace = {}, []
        for ordinal, ((frame, native), (_, evidence)) in enumerate(zip(native_frames, evidence_frames)):
            valid = np.all(np.ptp(native.corners, axis=1) >= .3, axis=1)
            lo, hi = parts[ordinal]
            assert np.array_equal(native.ids[valid], hcsr["ids"][lo:hi])
            native_rows = [(0, b.copy(), float(s)) for b, s in zip(
                native.corners[valid], hcsr["scores_historical_max"][lo:hi])]
            outputs, counts = {}, {}
            for name, state in proposals.items():
                state.update(ordinal, frame, proposal_ids=evidence.proposal_ids,
                    proposal_corners=evidence.proposal_corners, proposal_scores=evidence.proposal_scores,
                    native_corners=native.corners)
                rows = state.readout(native.corners[valid])
                auxiliary = [(0, r["box"].copy(), float(r["score"])) for r in rows]
                outputs[name] = ev.case(native_rows+auxiliary,
                                        ["native"]*len(native_rows)+["proposal"]*len(auxiliary))
                counts[name] = {"output": len(auxiliary), "active": len(state.tracks),
                                "confirmed_archive": len(state.births)}
            for n, state in anchors.items():
                state.update(ordinal, frame, evidence.anchor_ids, evidence.anchor_corners, evidence.anchor_scores)
                rows = state.rows()
                for cap in protocol["anchor_common"]["readout_caps"]:
                    name = f"a{n}_k{cap}"
                    selected = rows[:cap]
                    for row in selected:
                        emitted[name].setdefault(tuple(row["voxel_key"]), (ordinal, frame))
                    auxiliary = [(0, r["box"].copy(), float(r["score"])) for r in selected]
                    outputs[name] = ev.case(native_rows+auxiliary,
                                            ["native"]*len(native_rows)+["anchor"]*len(auxiliary))
                    counts[name] = {"output": len(auxiliary), "active": len(state.active),
                                    "confirmed_archive": len(state.births)}
            trace.append({"frame": frame, "ordinal": ordinal, "native": len(native_rows), "arms": counts})
            if ordinal in checkpoints:
                fraction = checkpoints[ordinal]
                tag = f"p{int(float(fraction)*100)}"
                expected = ev.audit.read_rows(parent/"predictions"/f"full_{tag}"/f"{scene}_boxes.pkl")
                expected_count = old["per_scene"][scene]["keyframe_trace"][ordinal]["output_counts"]["full"]
                np_native, np_plr = expected_count["native"], expected_count["proposal"]
                assert same_rows(native_rows, expected[:np_native]), (scene, frame, "HCSR parity")
                assert same_rows(outputs["p_track3_medoid"]["rows"], expected[:np_native+np_plr]), (scene, frame, "PLR parity")
                assert same_rows(outputs["a3_k640"]["rows"], expected[:np_native]+expected[np_native+np_plr:]), (scene, frame, "CALR parity")
                prefixes[fraction] = copy.deepcopy(outputs)
        delays = []
        for name, state in proposals.items():
            delays.extend(delay_rows(scene, name, state))
        for name, first in emitted.items():
            delays.extend(delay_rows(scene, name, anchors[int(name[1])], first))
        summary = {"keyframes": len(native_frames), "terminal_native_count": len(native_rows),
                   "seconds": time.perf_counter()-started, "trace": trace,
                   "frozen_reference_parity_at_all_four_prefixes": True}
        payload = {"protocol_sha256": protocol_hash, "summary": summary,
                   "prefixes": prefixes, "delays": delays}
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        temporary.replace(path)
        manifest["scenes"][scene] = summary
        ev.save_json(out/"replay_manifest.partial.json", manifest)
        print(f"recovery-factor replay {index}/{len(scenes)} {scene}: {summary['seconds']:.1f}s", flush=True)
        if stop_after and index >= stop_after:
            return None
    assert sum(v["keyframes"] for v in manifest["scenes"].values()) == 6817
    assert sum(v["terminal_native_count"] for v in manifest["scenes"].values()) == 1787
    manifest["source_sha256"] = {str(p): core.sha256(p) for p in (
        Path(__file__), core.SOURCE/"boxfusion/online_candidate_map.py",
        core.SOURCE/"boxfusion/m1_anchor_online.py")}
    ev.save_json(out/"replay_manifest.json", manifest)
    return manifest


def summarize_delays(records):
    result = {}
    for arm in sorted({r["arm"] for r in records}):
        values = [r for r in records if r["arm"] == arm]
        current = {"created_groups": len(values),
                   "confirmed_groups": sum(r["confirmed"] for r in values),
                   "ever_read_groups": sum(r["ever_read"] for r in values)}
        for key in ("confirmation_keyframe_intervals", "confirmation_input_seconds",
                    "readout_keyframe_intervals", "readout_input_seconds"):
            observed = [r[key] for r in values if r[key] is not None]
            current[key] = {"n": len(observed), "mean": float(np.mean(observed)) if observed else None,
                            "median": float(np.median(observed)) if observed else None,
                            "p95": float(np.percentile(observed, 95)) if observed else None}
        result[arm] = current
    return result


def evaluate(out, protocol, scenes, manifest):
    with contextlib.redirect_stdout(io.StringIO()):
        gts, alignments, transform = core.scannet_inputs()
    cases = {str(f): {} for f in protocol["prefix_fractions"]}
    delays = []
    for scene in scenes:
        with (out/"scenes"/f"{scene}.pkl").open("rb") as handle:
            data = pickle.load(handle)
        assert data["protocol_sha256"] == manifest["protocol_sha256"]
        delays.extend(data["delays"])
        for fraction, arms in data["prefixes"].items():
            for arm, case in arms.items():
                cases[fraction].setdefault(arm, {})[scene] = case
    prefix_points, terminal_predictions, workpoints = {}, {}, {}
    ap_rows, count_rows, work_rows, quality_rows = [], [], [], []
    for fraction, arms in cases.items():
        prefix_points[fraction] = {}
        for arm, values in arms.items():
            transformed = {s: ev.as_prediction(v, s, alignments, transform) for s, v in values.items()}
            plain = {s: (v[0], v[1]) for s, v in transformed.items()}
            point = {"total_count": sum(len(v[0]) for v in transformed.values()),
                     "auxiliary_count": sum(sum(x != "native" for x in v[2]) for v in transformed.values()),
                     "legacy_ap": {}, "stable_ap": {}}
            for t in protocol["iou_thresholds"]:
                point["legacy_ap"][str(t)] = core.class_agnostic_ap(plain, gts, t)["ap"]
                point["stable_ap"][str(t)] = ap_from_pairs(np.concatenate([
                    scene_pairs(*plain[s], gts[s], t) for s in scenes]), sum(len(gts[s]) for s in scenes))
            prefix_points[fraction][arm] = point
            ap_rows.append({"prefix": fraction, "arm": arm, "total_count": point["total_count"],
                            "auxiliary_count": point["auxiliary_count"],
                            **{f"AP{int(t*100)}": point["legacy_ap"][str(t)] for t in protocol["iou_thresholds"]},
                            **{f"stable_AP{int(t*100)}": point["stable_ap"][str(t)] for t in protocol["iou_thresholds"]}})
            if fraction != "1.0":
                continue
            terminal_predictions[arm] = plain
            for scene, value in values.items():
                core.write_prediction(out/"predictions"/arm/f"{scene}_boxes.pkl", value["rows"])
                counts = Counter(value["sources"])
                count_rows.append({"scene": scene, "arm": arm, "total": len(value["rows"]),
                                   "native": counts["native"], "auxiliary": len(value["rows"])-counts["native"]})
            metrics, _, _ = ev.workpoints(transformed, gts, thresholds=tuple(protocol["iou_thresholds"]),
                topks=tuple(protocol["top_k_per_scene"]), fp_budgets=tuple(protocol["fp_per_scene"]))
            workpoints[arm] = metrics
            for t, m in metrics.items():
                quality_rows.append({"arm": arm, "iou": t, "source": "all", **m["all_outputs"]})
                source = "proposal" if arm.startswith("p_") else "anchor"
                source_metrics = m["per_source"][source]
                quality_rows.append({"arm": arm, "iou": t, "source": source, **source_metrics,
                    "precision_percent": 100*source_metrics["tp"]/max(source_metrics["predictions"], 1)})
                for k, r in m["topk"].items():
                    work_rows.append({"arm": arm, "iou": t, "type": "TopK", "budget": k, **r})
                for k, r in m["fixed_fp_per_scene"].items():
                    work_rows.append({"arm": arm, "iou": t, "type": "oracle_FP_per_scene", "budget": k, **r})
            print(f"evaluated {arm}", flush=True)
    ev.write_csv(out/"prefix_ap.csv", ap_rows)
    ev.write_csv(out/"output_counts.csv", count_rows)
    ev.write_csv(out/"confirmation_delays.csv", delays)
    # Union columns because source-level and all-output summaries differ.
    def union_csv(path, records):
        import csv
        fields = list(dict.fromkeys(k for row in records for k in row))
        with path.open("w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=fields); w.writeheader(); w.writerows(records)
    union_csv(out/"quality_counts.csv", quality_rows)
    union_csv(out/"finite_workpoints.csv", work_rows)
    result = {"protocol": protocol, "scene_count": len(scenes), "gt_count": sum(map(len, gts.values())),
              "native_count": 1787, "keyframes": 6817, "prefix_ap": prefix_points,
              "workpoints": workpoints, "delay_summary": summarize_delays(delays),
              "validation": {"reference_PLR_CALR_HCSR_all_prefixes_exact": True,
                 "only_prespecified_factor_overrides": True, "model_inference": False,
                 "input_sha256_verified_against_parent": True},
              "protocol_sha256": manifest["protocol_sha256"]}
    ev.save_json(out/"point_results.json", result)
    comparisons = protocol["bootstrap_comparisons"]
    ci = ev.audit.bootstrap(terminal_predictions, gts, scenes, comparisons,
                            protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    diagnostics = []
    for name, (left, right) in comparisons.items():
        for t in protocol["iou_thresholds"]:
            row = ci[name][str(t)]
            row["point_ap"] = prefix_points["1.0"][left]["stable_ap"][str(t)]-prefix_points["1.0"][right]["stable_ap"][str(t)]
            row["legacy_point_ap"] = prefix_points["1.0"][left]["legacy_ap"][str(t)]-prefix_points["1.0"][right]["legacy_ap"][str(t)]
            row["two_sided_bootstrap_sign_tail"] = min(1., 2*min(row["p_positive"], 1-row["p_positive"])+2/(protocol["bootstrap_replicates"]+1))
            diagnostics.append((row["two_sided_bootstrap_sign_tail"], name, str(t)))
    running = 0.
    for rank, (p, name, t) in enumerate(sorted(diagnostics)):
        running = max(running, min(1., (len(diagnostics)-rank)*p))
        ci[name][t]["holm_adjusted_bootstrap_sign_tail_diagnostic"] = running
    result["paired_bootstrap"] = ci
    ev.save_json(out/"results.json", result)
    return result


def report(out, result):
    p = result["protocol"]
    lines = ["# ScanNet-100：恢复组织、确认与代表框的单因素验证", "",
        "本组使用同一冻结候选流、逐关键帧原生状态及三因素历史最大HCSR分数。仅CPU重放，没有重跑模型或使用GPU。"
        "每条路线只启用一种恢复分支；不同实验运行的主表数字不能直接拼接比较。", "",
        "## 1. 预先固定的协议", "",
        "PLR默认轨迹/三帧/medoid分别只替换为体素组织、一帧确认或最高分代表。四条路线共享原始PLR的生命周期、"
        "1024活动与1024归档容量、12条当前读出上限、平均观测分数排序、尺度分桶输出评分、原生覆盖排除及分支内NMS。"
        "体素替代仍保留每组最多12个不同帧观测，因此它不是此前只保存最高分代表的简单记忆，应视为受控组织替代。", "",
        "CALR两条状态仅改变确认帧数1或3，共享4096活动体素、640确认记录容量；在128/256/640预设读出上限下全部报告。"
        "相同容量和读出上限不表示实际输出数量或字节内存相同。这里的一帧CALR保留640确认归档的原入池策略，"
        "不等同于此前4096体素最高分简单记忆；本组不能替代与该强基线的比较。预算未根据逐场景终态数量设置；"
        "本组是同一ScanNet测试集上的探索性后续诊断。", "",
        "## 2. 终态AP与实际输出", "",
        "| 路线 | 新增框 | 总框 | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, row in result["prefix_ap"]["1.0"].items():
        ap = row["legacy_ap"]
        lines.append(f"| {arm} | {row['auxiliary_count']:,} | {row['total_count']:,} | {ap['0.15']:.4f} | {ap['0.25']:.4f} | {ap['0.5']:.4f} |")
    lines += ["", "## 3. 场景级配对区间", "",
        "B=2000，seed=20261031；差值和区间均采用确定性稳定破同。表2点估计采用既有评测器，末位差异单独保留，"
        "不混用两种排序协议。以下为未经多重比较校正的探索性百分位95%区间。results.json同时提供18项比较的"
        "Holm调整bootstrap符号尾部诊断值；该诊断不是独立盲测证据，也不应代替效应量和区间。", "",
        "| 对照（左减右） | ΔAP15 [95% CI] | ΔAP25 [95% CI] | ΔAP50 [95% CI] |",
        "|---|---|---|---|"]
    for name, by_t in result["paired_bootstrap"].items():
        cells = []
        for t in p["iou_thresholds"]:
            row = by_t[str(t)]; lo, hi = row["ci95_ap"]
            cells.append(f"{row['point_ap']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append("| "+name+" | "+" | ".join(cells)+" |")
    lines += ["", "## 4. 新增分支质量（IoU阈值0.15）", "",
        "TP/FP在与共享原生记录合并后的排序与一对一匹配中计算，再按来源统计。重复预测属于FP，不可把TP、FP、重复相加。"
        "本组关闭另一恢复分支，因此新增TP不能直接与此前固定PLR+HCSR前缀的CALR统计混用。", "",
        "| 路线 | 新增TP | 新增FP | 分支内重复FP | 跨分支重复FP | 分支精度 |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, rows in result["workpoints"].items():
        source = "proposal" if arm.startswith("p_") else "anchor"
        r = rows["0.15"]["per_source"][source]
        precision = 100*r['tp']/max(r['predictions'],1)
        lines.append(f"| {arm} | {r['tp']} | {r['fp']} | {r['within_duplicate_fp']} | {r['cross_duplicate_fp']} | {precision:.4f}% |")
    lines += ["", "## 5. 确认等待与读出等待", "",
        "等待从算法轨迹/体素建立开始，以关键帧间隔和输入帧差/30秒计，不是墙钟处理耗时、首次可见延迟或首次正确检测延迟。"
        "确认统计仅针对已确认状态；从未确认和从未读出的状态另计，不能只报告成功状态的等待而忽略未完成状态。"
        "PLR组按轨迹身份、CALR组按不同体素键统计。CALR未确认体素淘汰后重建时，以最近一次建立时间计算等待，"
        "其建立组数不是所有瞬时创建事件的总数。确认等待针对全部已确认组，读出等待仅针对曾读出子集，"
        "两列的中位数不是同一对象集合；单个组的首次读出不会早于确认。", "",
        "| 路线 | 建立/确认/曾读出组数 | 确认等待中位数/P95（关键帧间隔） | 首次读出等待中位数/P95（关键帧间隔） |",
        "|---|---:|---:|---:|"]
    for arm, r in result["delay_summary"].items():
        c, q = r["confirmation_keyframe_intervals"], r["readout_keyframe_intervals"]
        def fmt(x):
            return "NA" if x is None else f"{x:.2f}"
        lines.append(f"| {arm} | {r['created_groups']}/{r['confirmed_groups']}/{r['ever_read_groups']} | {fmt(c['median'])}/{fmt(c['p95'])} | {fmt(q['median'])}/{fmt(q['p95'])} |")
    lines += ["", "## 6. 结论边界与完整数据", "",
        "单因素差值回答相应规则在本次固定协议下是否有额外收益，不证明任意轨迹、体素或记忆方法的普遍优劣。"
        "若区间跨零，结论为增益证据不足，而非证明算法等效或规则毫无作用；若输出数不同，精度变化需结合召回、误检与等待同时解读。"
        "本组未测端到端FPS，CPU重放速度不能代替在线系统效率。", "",
        "prefix_ap.csv：四个前缀AP；output_counts.csv：逐场景实际输出；quality_counts.csv：三种IoU下TP/FP、重复与multiplicity；"
        "finite_workpoints.csv：全部预设Top-K与逐场景oracle FP预算；confirmation_delays.csv：组级确认/读出等待；"
        "source和输入指纹见replay_manifest.json。100%前缀与本组终态一致。", "",
        "## 7. 本次结果的直接解释", ""]
    titles = {"track-minus-voxel": "轨迹组织相对体素组织",
              "proposal-three-minus-one": "PLR三帧相对一帧确认",
              "medoid-minus-highest": "medoid相对最高分代表"}
    for name, title in titles.items():
        rows = result["paired_bootstrap"][name]
        effects = "/".join(f"{rows[str(t)]['point_ap']:+.2f}" for t in p["iou_thresholds"])
        positive = [f"AP{int(t*100)}" for t in p["iou_thresholds"] if rows[str(t)]["ci95_ap"][0] > 0]
        lines.append(f"{title}：稳定破同差值为{effects}个AP点；未经多重比较校正的95%区间高于0的项目为"
                     f"{'、'.join(positive) if positive else '无'}。")
        lines.append("")
    lines += [
        "CALR三帧相对一帧确认在三个预设读出预算、三个IoU阈值下的九个95%区间全部跨零，"
        "本批没有建立其AP优势。减少输出与误检的取舍仍然存在，应与损失TP和确认等待共同解释。", "",
        "综合而言，PLR三帧确认获得了比此前整体记忆对照更直接的单因素证据，但它不意味着整套PLR已稳定优于最强简单记忆。"
        "轨迹组织和medoid的边际证据只出现在部分IoU阈值。18项比较的Holm调整bootstrap符号尾部诊断中，"
        "仅PLR三帧确认的AP25低于0.05；这只是探索性诊断，不应表述为独立确认性检验。"
        "保持现有默认配置，不根据本批结果追选更有利的参数或修改旧主表。"
    ]
    text = "\n".join(lines)+"\n"
    (out/"实验结果与解释.md").write_text(text, encoding="utf-8")
    try:
        from tools.build_supplement_docx import build
    except ModuleNotFoundError as error:
        if error.name != "docx":
            raise
        subprocess.run([
            "/home/admin1/miniconda3/envs/temp/bin/python",
            str(ROOT/"tools/build_supplement_docx.py"),
            str(out/"实验结果与解释.md"), str(out/"实验结果与解释.docx"),
            "--footer", "ReCaR-3D Recovery Factor Controls — 2026-10-31",
        ], check=True)
    else:
        build(out/"实验结果与解释.md", out/"实验结果与解释.docx",
              footer_text="ReCaR-3D Recovery Factor Controls — 2026-10-31")
    print(text, flush=True)


def validate(out):
    import csv
    import zipfile
    p = json.loads((out/"protocol.json").read_text())
    m = json.loads((out/"replay_manifest.json").read_text())
    r = json.loads((out/"results.json").read_text())
    assert core.sha256(out/"protocol.json") == m["protocol_sha256"] == r["protocol_sha256"]
    assert len(m["scenes"]) == 100
    assert sum(v["keyframes"] for v in m["scenes"].values()) == 6817
    parent = json.loads((ROOT/p["parent_report"]/"results.json").read_text())
    parent_verified = 0
    for path, sha in m["input_sha256"].items():
        assert core.sha256(Path(path)) == sha
        if path in parent["input_sha256"]:
            assert parent["input_sha256"][path] == sha
            parent_verified += 1
    assert len(m["input_sha256"]) == 300 and parent_verified == 200
    counts = {str(f): 0 for f in p["prefix_fractions"]}
    delay_count = 0
    for scene in m["scenes"]:
        with (out/"scenes"/f"{scene}.pkl").open("rb") as handle:
            d = pickle.load(handle)
        assert d["protocol_sha256"] == m["protocol_sha256"]
        for f, arms in d["prefixes"].items():
            reference = arms["p_track3_medoid"]
            native = [row for row, s in zip(reference["rows"], reference["sources"]) if s == "native"]
            counts[f] += len(native)
            for arm, value in arms.items():
                current = [row for row, s in zip(value["rows"], value["sources"]) if s == "native"]
                assert same_rows(native, current)
                cap = 12 if arm.startswith("p_") else int(arm.split("_k")[1])
                assert len(value["rows"])-len(native) <= cap
        for row in d["delays"]:
            delay_count += 1
            c, q = row["confirmation_keyframe_intervals"], row["readout_keyframe_intervals"]
            if c is not None:
                assert c >= 0
                if "track3" in row["arm"] or "voxel3" in row["arm"] or row["arm"].startswith("a3_"):
                    assert c >= 2
            if q is not None:
                assert c is not None and q >= c
    assert counts["1.0"] == 1787
    for arm, values in r["workpoints"].items():
        for t, value in values.items():
            assert abs(value["stable_ap"]-r["prefix_ap"]["1.0"][arm]["stable_ap"][t]) < 1e-9
            for x in [value["all_outputs"], *value["per_source"].values()]:
                assert x["predictions"] == x["tp"]+x["fp"]
                assert x["within_duplicate_fp"]+x["cross_duplicate_fp"] <= x["fp"]
    expected = {"prefix_ap.csv": 40, "quality_counts.csv": 60, "finite_workpoints.csv": 210}
    for name, number in expected.items():
        with (out/name).open() as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == number
    with zipfile.ZipFile(out/"实验结果与解释.docx") as z:
        xml = z.read("word/document.xml").decode()
        assert "0.7983" in xml and "0.3823" in xml and "一帧CALR保留640确认归档" in xml
    check = {"status": "PASS", "scene_count": 100, "keyframes": 6817,
             "input_files_hash_verified": 300, "native_and_evidence_hashes_matching_parent": 200,
             "prefix_native_counts": counts, "group_delay_records": delay_count,
             "shared_native_geometry_scores_order_all_arms_and_prefixes": True,
             "all_readouts_at_or_after_confirmation": True, "TP_FP_duplicate_accounting": True,
             "csv_rows": expected, "DOCX_contents_verified": True,
             "max_legacy_stable_AP_difference": max(abs(v["legacy_ap"][t]-v["stable_ap"][t])
                for arms in r["prefix_ap"].values() for v in arms.values() for t in v["legacy_ap"]),
             "final_report_renderer_sha256": core.sha256(Path(__file__))}
    ev.save_json(out/"validation.json", check)
    print(json.dumps(check, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--phase", choices=("all", "replay", "evaluate", "report", "validate"), default="all")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test_only:
        return
    if args.phase == "validate":
        validate(args.output)
        return
    protocol = json.loads((args.output/"protocol.json").read_text())
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines()
              if s.strip() and not s.startswith("#")]
    assert len(scenes) == len(set(scenes)) == 100
    if args.phase in ("all", "replay"):
        manifest = replay(args.output, protocol, scenes, args.stop_after)
        if manifest is None or args.phase == "replay":
            return
    if args.phase in ("all", "evaluate"):
        manifest = json.loads((args.output/"replay_manifest.json").read_text())
        if (args.output/"results.json").exists():
            result = json.loads((args.output/"results.json").read_text())
        else:
            result = evaluate(args.output, protocol, scenes, manifest)
    else:
        result = json.loads((args.output/"results.json").read_text())
    report(args.output, result)
    validate(args.output)


if __name__ == "__main__":
    main()
