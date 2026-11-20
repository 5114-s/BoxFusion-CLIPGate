#!/usr/bin/env python3
"""Shared-match current-zero versus historical-max HCSR control, CPU only."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev
from tools.bootstrap_final_recar_pairs import scene_pairs, ap_from_pairs

core = ev.audit.core
from boxfusion.native_reliability_reranker import OnlineNativeReliabilityReranker, _project_with_visibility

DEFAULT = ROOT/"reports/scannet_hcsr_current_zero_vs_max_20261029"
ARMS = ("native", "current_zero", "last_matched", "historical_max")


def make_evidence(box=None, rectangle=None):
    return core.EvidenceFrame(
        proposal_ids=np.asarray([] if box is None else [1], dtype=np.int64),
        proposal_boxes_2d=np.empty((0, 4)) if box is None else rectangle[None],
        proposal_corners=np.empty((0, 8, 3)) if box is None else box[None],
        proposal_scores=np.empty(0) if box is None else np.asarray([.9]),
        anchor_ids=np.empty(0, dtype=np.int64), anchor_corners=np.empty((0, 8, 3)), anchor_scores=np.empty(0))


def test_current_zero():
    box = np.asarray([[x, y, z] for x in (-.2, .2) for y in (-.2, .2) for z in (2., 2.4)])
    intrinsic = np.asarray([[200., 0, 320.], [0, 200., 240.], [0, 0, 1.]])
    camera = np.eye(4)
    native = core.NativeFrame(ids=np.asarray([7]), corners=box[None], scores=np.asarray([.6]),
                              camera_to_world=camera, intrinsic=intrinsic, width=640, height=480)
    state = OnlineNativeReliabilityReranker(support_mode="max", strength_variant="no_visibility")
    rectangle, _ = _project_with_visibility(box, camera, intrinsic, 640, 480)
    state.update(0, 0, native, make_evidence(box, rectangle))
    first = state.last_frame_strength.copy()
    assert first[0] > .5 and state.last_frame_iou2d[0] >= state.match_iou
    state.update(1, 25, native, make_evidence())
    assert state.last_frame_strength[0] == 0
    assert state._support(7, "latest") == state._support(7, "max") == first[0]
    current = state._score(.6, 0., .5, 2.)
    maximum = state._score(.6, first[0], .5, 2.)
    assert abs(current-.6) < 1e-12 and maximum > current
    camera_behind = camera.copy(); camera_behind[2, 3] = 5.
    behind = core.NativeFrame(ids=native.ids, corners=native.corners, scores=native.scores,
                              camera_to_world=camera_behind, intrinsic=intrinsic, width=640, height=480)
    state.update(2, 50, behind, make_evidence())
    assert state.last_frame_strength[0] == 0 and state._support(7, "max") == first[0]
    shifted = core.NativeFrame(ids=native.ids, corners=native.corners+np.array([20., 0., 0.]), scores=native.scores,
                               camera_to_world=camera, intrinsic=intrinsic, width=640, height=480)
    state.update(3, 75, shifted, make_evidence())
    assert state._support(7, "max") == state._support(7, "latest") == state.last_frame_strength[0] == 0


def same_rows(a, b):
    return len(a) == len(b) and all(x[0] == y[0] and x[2] == y[2] and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def replay(out, protocol, scenes):
    config = yaml.safe_load((ROOT/protocol["state_config"]).read_text())["online_candidate_map"]["state"]["m2"]
    parameters = {**config, "strength_variant": "no_visibility", "support_mode": "max"}
    for key, expected in protocol["shared_parameters"].items():
        if key != "score_clip":
            assert parameters[key] == expected, (key, parameters[key], expected)
    parent = ROOT/protocol["parent_report"]
    old = json.loads((parent/"results.json").read_text())
    protocol_sha = core.sha256(out/"protocol.json")
    output = out/"scenes"; output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol_sha256": protocol_sha, "shared_parameters": parameters, "scenes": {},
                "input_sha256": {}, "new_model_inference": False, "one_matching_call_per_keyframe": True}
    for index, scene in enumerate(scenes, 1):
        destination = output/f"{scene}.pkl"
        npath = ROOT/protocol["native_prefix_root"]/f"{scene}.npz"
        epath = ROOT/protocol["evidence_root"]/f"{scene}.npz"
        for path in (npath, epath):
            value = core.sha256(path)
            assert value == old["input_sha256"][str(path)]
            manifest["input_sha256"][str(path)] = value
        if destination.exists():
            with destination.open("rb") as handle:
                payload = pickle.load(handle)
            assert payload["protocol_sha256"] == protocol_sha
            manifest["scenes"][scene] = payload["summary"]
            continue
        natives, evidence = core.load_native(npath), core.load_evidence(epath)
        assert [f for f, _ in natives] == [f for f, _ in evidence]
        state = OnlineNativeReliabilityReranker(**parameters)
        checkpoints = {int(np.ceil(len(natives)*f))-1: str(f) for f in protocol["prefix_fractions"]}
        prefixes, trace, frame_vectors = {}, [], []
        for ordinal, ((frame, native), (_, auxiliary)) in enumerate(zip(natives, evidence)):
            maximum_scores = state.update(ordinal, frame, native, auxiliary)
            assert np.array_equal(state.last_ids, native.ids)
            current_support = state.last_frame_strength.copy()
            maximum_support = state.last_support.copy()
            latest_support = np.asarray([state._support(int(identity), "latest") for identity in native.ids])
            matched = state.last_frame_iou2d >= state.match_iou
            assert np.all(current_support[~matched] == 0)
            assert np.all(maximum_support+1e-12 >= latest_support)
            assert np.all(latest_support+1e-12 >= current_support)
            current_scores = np.asarray([state._score(s, q, state.support_threshold, state.logit_weight)
                                         for s, q in zip(native.scores, current_support)])
            latest_scores = state.materialize_mode(native.ids, native.scores, "latest")[0]
            assert np.all(maximum_scores+1e-12 >= latest_scores)
            assert np.all(latest_scores+1e-12 >= current_scores)
            keep = np.all(np.ptp(native.corners, axis=1) >= protocol["native_min_axis_m"], axis=1)
            values = {"native": native.scores[keep], "current_zero": current_scores[keep],
                      "last_matched": latest_scores[keep], "historical_max": maximum_scores[keep]}
            trace.append({"ordinal": ordinal, "frame": frame, "native_output_count": int(keep.sum()),
                          "matched": int(matched[keep].sum()),
                          "current_positive": int((current_support[keep] > 0).sum()),
                          "current_boosted": int((current_support[keep] > state.support_threshold).sum()),
                          "max_boosted": int((maximum_support[keep] > state.support_threshold).sum()),
                          "unmatched_but_historical_boosted": int(((~matched[keep]) & (maximum_support[keep] > state.support_threshold)).sum())})
            frame_vectors.append({"ids": native.ids[keep], "current_support": current_support[keep],
                                  "last_matched_support": latest_support[keep], "max_support": maximum_support[keep],
                                  "matched": matched[keep], "iou2d": state.last_frame_iou2d[keep],
                                  **{f"scores_{a}": value for a, value in values.items()}})
            if ordinal not in checkpoints:
                continue
            f = checkpoints[ordinal]; tag = f"p{int(float(f)*100)}"
            cases = {arm: ev.case([(0, b.copy(), float(s)) for b, s in zip(native.corners[keep], scores)],
                                  ["native"]*int(keep.sum())) for arm, scores in values.items()}
            expected_native = ev.audit.read_rows(parent/"predictions"/f"base_{tag}"/f"{scene}_boxes.pkl")
            expected_full = ev.audit.read_rows(parent/"predictions"/f"full_{tag}"/f"{scene}_boxes.pkl")
            assert same_rows(cases["native"]["rows"], expected_native)
            assert same_rows(cases["historical_max"]["rows"], expected_full[:len(expected_native)])
            for arm in ARMS:
                assert len(cases[arm]["rows"]) == len(expected_native)
                assert all(np.array_equal(x[1], y[1]) for x, y in zip(cases[arm]["rows"], expected_native))
                core.write_prediction(out/"predictions"/f"{arm}_{tag}"/f"{scene}_boxes.pkl", cases[arm]["rows"])
            prefixes[f] = cases
        fields = frame_vectors[0].keys()
        vector_path = out/"frame_readouts"/f"{scene}.npz"
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(vector_path, frame_ids=np.asarray([f for f, _ in natives]),
                            lengths=np.asarray([x["native_output_count"] for x in trace]),
                            **{key: np.concatenate([x[key] for x in frame_vectors]) for key in fields})
        summary = {"keyframes": len(natives), "trace": trace, "geometry_resets": state.geometry_resets,
                   "last_native_count": trace[-1]["native_output_count"]}
        payload = {"protocol_sha256": protocol_sha, "summary": summary, "prefixes": prefixes}
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        temporary.replace(destination)
        manifest["scenes"][scene] = summary
        if index % 10 == 0:
            print(f"shared-match HCSR replay {index}/{len(scenes)}", flush=True)
    assert sum(x["keyframes"] for x in manifest["scenes"].values()) == 6817
    assert sum(x["last_native_count"] for x in manifest["scenes"].values()) == 1787
    manifest["source_sha256"] = {str(path.relative_to(ROOT)): core.sha256(path) for path in (
        Path(__file__), core.SOURCE/"boxfusion/native_reliability_reranker.py",
        core.SOURCE/"boxfusion/causal_reliability.py")}
    ev.save_json(out/"replay_manifest.json", manifest)
    return manifest


def evaluate(out, protocol, scenes, manifest):
    with contextlib.redirect_stdout(io.StringIO()):
        gts, alignments, transform = core.scannet_inputs()
    cases = {str(f): {a: {} for a in ARMS} for f in protocol["prefix_fractions"]}
    for scene in scenes:
        with (out/"scenes"/f"{scene}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        assert payload["protocol_sha256"] == manifest["protocol_sha256"]
        for f, arms in payload["prefixes"].items():
            for arm, value in arms.items():
                cases[f][arm][scene] = value
    points, predictions = {}, {}
    for f, arms in cases.items():
        points[f], predictions[f] = {}, {}
        for arm, values in arms.items():
            full = {s: ev.as_prediction(v, s, alignments, transform) for s, v in values.items()}
            plain = {s: (v[0], v[1]) for s, v in full.items()}
            points[f][arm] = {"count": sum(len(v[0]) for v in full.values()),
                              "legacy_ap": {str(t): core.class_agnostic_ap(plain, gts, t)["ap"] for t in ev.TS},
                              "stable_ap": {str(t): ap_from_pairs(np.concatenate([scene_pairs(*plain[s], gts[s], t) for s in scenes]), 1433) for t in ev.TS}}
            predictions[f][arm] = plain
    workpoints = {}
    top_rows, fp_rows, ap_rows = [], [], []
    for arm in ARMS:
        values = {s: ev.as_prediction(v, s, alignments, transform) for s, v in cases["1.0"][arm].items()}
        result, curves, thresholds = ev.workpoints(values, gts, topks=tuple(protocol["top_k_per_scene"]))
        workpoints[arm] = result
        np.savez_compressed(out/f"pr_{arm}.npz", **{f"iou{t}_{k}": v for t, x in curves.items() for k, v in x.items()})
        ev.write_csv(out/f"oracle_fp_thresholds_{arm}.csv", thresholds)
        for t, x in result.items():
            assert abs(x["stable_ap"]-points["1.0"][arm]["stable_ap"][t]) < 1e-9
            for k, row in x["topk"].items():
                top_rows.append({"method": arm, "iou": t, "K": k, **row})
            for b, row in x["fixed_fp_per_scene"].items():
                fp_rows.append({"method": arm, "iou": t, "FP_per_scene": b, **row})
    for f, arms in points.items():
        for arm, values in arms.items():
            ap_rows.append({"prefix": f, "method": arm, "native_count": values["count"],
                            **{f"AP{int(t*100)}": values["legacy_ap"][str(t)] for t in ev.TS},
                            **{f"stable_AP{int(t*100)}": values["stable_ap"][str(t)] for t in ev.TS}})
    ev.write_csv(out/"prefix_ap.csv", ap_rows)
    ev.write_csv(out/"unified_topk.csv", top_rows)
    ev.write_csv(out/"fixed_fp.csv", fp_rows)
    sizes = np.asarray([manifest["scenes"][s]["last_native_count"] for s in scenes])
    result = {"protocol": protocol, "protocol_sha256": manifest["protocol_sha256"], "prefix_ap": points,
              "terminal_workpoints": workpoints, "native_count": int(sizes.sum()), "gt_count": 1433,
              "native_scene_count_statistics": {"mean": float(sizes.mean()), "median": float(np.median(sizes)),
                                                "maximum": int(sizes.max()), "scenes_over20": int((sizes > 20).sum()),
                                                "scenes_over50": int((sizes > 50).sum()), "scenes_over100": int((sizes > 100).sum())},
              "terminal_matched": sum(x["trace"][-1]["matched"] for x in manifest["scenes"].values()),
              "terminal_current_boosted": sum(x["trace"][-1]["current_boosted"] for x in manifest["scenes"].values()),
              "terminal_max_boosted": sum(x["trace"][-1]["max_boosted"] for x in manifest["scenes"].values()),
              "terminal_unmatched_but_historical_boosted": sum(x["trace"][-1]["unmatched_but_historical_boosted"] for x in manifest["scenes"].values())}
    ev.save_json(out/"point_results.json", result)
    ci = ev.audit.bootstrap(predictions["1.0"], gts, scenes, protocol["bootstrap_comparisons"],
                             protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    for name, (left, right) in protocol["bootstrap_comparisons"].items():
        for t in ev.TS:
            ci[name][str(t)]["point_ap"] = points["1.0"][left]["stable_ap"][str(t)]-points["1.0"][right]["stable_ap"][str(t)]
            ci[name][str(t)]["legacy_point_ap"] = points["1.0"][left]["legacy_ap"][str(t)]-points["1.0"][right]["legacy_ap"][str(t)]
    result["paired_bootstrap"] = ci
    result["validation"] = {"same_matching_execution": True, "all_unmatched_current_support_exact_zero": True,
                             "native_geometry_count_order_fixed": True, "max_matches_prior_frozen_predictions": True,
                             "all200_input_files_hash_verified": True, "current_to_max_monotonicity_verified_every_frame": True,
                             "new_model_inference": False}
    ev.save_json(out/"results.json", result)
    return result


def add_topk_bootstrap(out, result):
    """Paired intervals at every prespecified K, with no new parameter choice."""
    protocol = result["protocol"]
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines() if s.strip() and not s.startswith("#")]
    with contextlib.redirect_stdout(io.StringIO()):
        gts, alignments, transform = core.scannet_inputs()
    counts = {str(t): {str(k): {a: [] for a in ("current_zero", "historical_max")}
                      for k in protocol["top_k_per_scene"]} for t in (.25, .5)}
    for scene in scenes:
        for arm in ("current_zero", "historical_max"):
            rows = ev.audit.read_rows(out/"predictions"/f"{arm}_p100"/f"{scene}_boxes.pkl")
            boxes, scores = core.prediction(rows, transform, alignments[scene])
            for t in (.25, .5):
                _, _, flags, _, _ = ev.greedy_info(boxes, scores, np.asarray(["native"]*len(rows)), gts[scene], t)
                for k in protocol["top_k_per_scene"]:
                    counts[str(t)][str(k)][arm].append(int(flags[:k].sum()))
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(
        0, len(scenes), (protocol["bootstrap_replicates"], len(scenes)))
    gt_sizes = np.asarray([len(gts[s]) for s in scenes])
    denominator = gt_sizes[draws].sum(1)
    summary, csv_rows = {}, []
    for t, by_k in counts.items():
        summary[t] = {}
        for k, methods in by_k.items():
            difference = np.asarray(methods["historical_max"])-np.asarray(methods["current_zero"])
            distribution = 100*difference[draws].sum(1)/denominator
            row = {"delta_recall_pp": 100*float(difference.sum())/1433,
                   "ci95_pp": [float(np.percentile(distribution, 2.5)), float(np.percentile(distribution, 97.5))]}
            summary[t][k] = row
            csv_rows.append({"iou": t, "K": int(k), "delta_recall_pp": row["delta_recall_pp"],
                             "ci95_low_pp": row["ci95_pp"][0], "ci95_high_pp": row["ci95_pp"][1]})
    result["paired_topk_bootstrap_max_minus_current_zero"] = summary
    result["topk_interval_note"] = "Additional paired summaries over every prespecified K; no K or parameter selected after evaluation. Same 2000 draws and seed as AP."
    ev.write_csv(out/"paired_topk_bootstrap.csv", csv_rows)
    ev.save_json(out/"results.json", result)


def validate_saved(out, result):
    manifest = json.loads((out/"replay_manifest.json").read_text())
    assert manifest["protocol_sha256"] == result["protocol_sha256"] == core.sha256(out/"protocol.json")
    for filename, expected in manifest["input_sha256"].items():
        assert core.sha256(Path(filename)) == expected
    for filename, expected in manifest["source_sha256"].items():
        if filename.startswith("development/"):
            assert core.sha256(ROOT/filename) == expected
    frames = records = native_terminal = 0
    for scene, summary in manifest["scenes"].items():
        with np.load(out/"frame_readouts"/f"{scene}.npz") as v:
            assert len(v["frame_ids"]) == len(v["lengths"]) == summary["keyframes"]
            assert int(v["lengths"].sum()) == len(v["ids"])
            assert np.all(v["current_support"][~v["matched"]] == 0)
            assert np.all(v["max_support"]+1e-12 >= v["last_matched_support"])
            assert np.all(v["last_matched_support"]+1e-12 >= v["current_support"])
            for arm, field in (("current_zero", "current_support"), ("last_matched", "last_matched_support"),
                               ("historical_max", "max_support")):
                computed = np.asarray([OnlineNativeReliabilityReranker._score(s, q, .5, 2.)
                                       for s, q in zip(v["scores_native"], v[field])])
                assert np.array_equal(computed, v[f"scores_{arm}"])
            frames += len(v["frame_ids"]); records += len(v["ids"])
            native_terminal += int(v["lengths"][-1])
        for prefix in (25, 50, 75, 100):
            reference = ev.audit.read_rows(out/"predictions"/f"native_p{prefix}"/f"{scene}_boxes.pkl")
            for arm in ARMS:
                rows = ev.audit.read_rows(out/"predictions"/f"{arm}_p{prefix}"/f"{scene}_boxes.pkl")
                assert len(rows) == len(reference)
                assert all(np.array_equal(x[1], y[1]) for x, y in zip(rows, reference))
    assert frames == 6817 and native_terminal == 1787 and len(manifest["input_sha256"]) == 200
    checks = {"passed": True, "keyframes": frames, "native_output_observations_checked": records,
              "terminal_native_boxes_each_arm": native_terminal, "input_sha256_files_verified": 200,
              "unmatched_current_support_exact_zero": True, "same_score_function_all_frames_verified": True,
              "fixed_geometry_count_row_order_all_prefixes_verified": True,
              "historical_max_reproduces_existing_frozen_run": True,
              "source_for_reproduction_sha256": core.sha256(Path(__file__)), "GPU_inference": False}
    ev.save_json(out/"validation.json", checks)


def render(out, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"native": "Native score", "current_zero": "Current view (unmatched = 0)",
              "last_matched": "Last matched view", "historical_max": "Historical maximum"}
    colors = {"native": "#999999", "current_zero": "#4E79A7", "last_matched": "#59A14F", "historical_max": "#E15759"}
    styles = {"native": ":", "current_zero": "--", "last_matched": "-.", "historical_max": "-"}
    current_text = "/".join(f"{result['prefix_ap']['1.0']['current_zero']['legacy_ap'][str(t)]:.2f}" for t in ev.TS)
    max_text = "/".join(f"{result['prefix_ap']['1.0']['historical_max']['legacy_ap'][str(t)]:.2f}" for t in ev.TS)
    delta_text = "/".join(f"{result['paired_bootstrap']['max-minus-current-zero'][str(t)]['point_ap']:.2f}" for t in ev.TS)
    interval_text = "、".join(f"[{result['paired_bootstrap']['max-minus-current-zero'][str(t)]['ci95_ap'][0]:.2f},"
                             f"{result['paired_bootstrap']['max-minus-current-zero'][str(t)]['ci95_ap'][1]:.2f}]" for t in ev.TS)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3), sharey=True)
    for col, threshold in enumerate((.25, .5)):
        for row, ks in enumerate(((1, 5, 10, 20), (20, 50, 100, 300))):
            ax = axes[row, col]
            for arm in ARMS:
                values = result["terminal_workpoints"][arm][str(threshold)]["topk"]
                ax.plot(ks, [values[str(k)]["recall_percent"] for k in ks], marker="o", markersize=3,
                        color=colors[arm], ls=styles[arm], lw=1.2, label=labels[arm])
            ax.set_xticks(ks); ax.set_xlabel("Per-scene native-box limit K")
            ax.set_ylim(0, 65); ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
        axes[0, col].set_title(f"3D AABB IoU > {threshold:.2f}")
    axes[0, 0].set_ylabel("Recall (%)"); axes[1, 0].set_ylabel("Recall (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out/f"hcsr_current_zero_vs_max_topk.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    lines = ["# HCSR严格共享匹配：当前帧零填充与历史最大支持", "",
             "ScanNet-100，100场景、1,433个GT、6,817个关键帧。四条路线在每个前缀共享原生身份、框几何、原始分数、数量和行顺序；"
             "终态均为1,787框。每个关键帧只执行一次二维投影和贪心一对一匹配，随后从同一支持结果读取不同聚合值。"
             "不启用PLR/CALR、不增加NMS，不运行学习模型，不占用GPU。", "",
             "## 1. 控制定义与公共参数", "",
             "- Native：原生分数不变。",
             "- Current-zero：只使用当前关键帧last_frame_strength；无匹配或无有效投影时严格为0，不保留上次支持。",
             "- Last-matched：旧Latest的真实含义，保留最近一次成功匹配的支持；无匹配不清零。",
             "- Historical-max：相同身份自上次几何重置以来的最大匹配支持。", "",
             "公共三因素视图支持为(proposal分数×2D IoU×3D几何一致性)的立方根，不含visibility。"
             "匹配阈值0.10，支持激活阈值0.50，native-logit系数2.0，分数裁剪[1e-4,1−1e-4]，"
             "几何重置阈值0.20，状态失效10个关键帧，容量4,096。原生逐轴0.30 m过滤在共享匹配之后执行。"
             "四条路线只改变支持聚合读出，不改变匹配或参数。", "",
             "## 2. 终态类别无关AP", "",
             "| 支持读出 | 原生框数 | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        x = result["prefix_ap"]["1.0"][arm]
        lines.append(f"| {labels[arm]} | {x['count']} | "+" | ".join(f"{x['legacy_ap'][str(t)]:.4f}" for t in ev.TS)+" |")
    lines += ["", "上述点估计沿用原评测器破同方式；以下配对bootstrap点估计与区间均使用确定性稳定破同。"
              "B=2,000，seed=20261029，场景级百分位95%区间，未经多重比较校正。", "",
              "| 对比 | AP15差值 [95% CI] | AP25差值 [95% CI] | AP50差值 [95% CI] |", "|---|---|---|---|"]
    for name, values in result["paired_bootstrap"].items():
        lines.append("| "+name+" | "+" | ".join(
            f"{values[str(t)]['point_ap']:+.4f} [{values[str(t)]['ci95_ap'][0]:+.4f}, {values[str(t)]['ci95_ap'][1]:+.4f}]" for t in ev.TS)+" |")
    lines += ["", "## 3. 同一原生集合的统一Top-K", "",
              "每个场景独立截断Top-K，再按公共一对一贪心匹配统计召回率，不使用最大IoU覆盖率替代检测召回。"
              "所有方法共享K=1/5/10/20/50/100/300；小K在新增结果产生前即已固定。", "",
              "| IoU | 路线 | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | R@300 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in (.25, .5):
        for arm in ARMS:
            values = result["terminal_workpoints"][arm][str(t)]["topk"]
            lines.append(f"| {t:.2f} | {labels[arm]} | "+" | ".join(f"{values[str(k)]['recall_percent']:.2f}%" for k in result["protocol"]["top_k_per_scene"])+" |")
    lines += ["", "历史最大值−Current-zero的配对Top-K召回差值（百分点，B=2,000；所有预设K均报告）：", "",
              "| IoU | K | 召回差值 | 95% CI |", "|---|---:|---:|---|"]
    for t, by_k in result["paired_topk_bootstrap_max_minus_current_zero"].items():
        for k, row in by_k.items():
            lines.append(f"| {t} | {k} | {row['delta_recall_pp']:+.4f} | [{row['ci95_pp'][0]:+.4f}, {row['ci95_pp'][1]:+.4f}] |")
    size = result["native_scene_count_statistics"]
    lines += ["", f"原生框每场均值{size['mean']:.2f}、中位数{size['median']:.1f}、最大{size['maximum']}；"
              f"超过20/50/100框的场景数为{size['scenes_over20']}/{size['scenes_over50']}/{size['scenes_over100']}。"
              "因此较大K接近全部原生集合，曲线重合/饱和是固定框评测的预期性质，不代表支持重排序完全无效。", "",
              "## 4. 当前支持与历史支持的可核验差异", "",
              f"终态共有{result['terminal_matched']}个原生记录在最后关键帧匹配到proposal；"
              f"Current-zero使{result['terminal_current_boosted']}条记录获得正向修正，Historical-max为{result['terminal_max_boosted']}条。"
              f"其中{result['terminal_unmatched_but_historical_boosted']}条当前无匹配但仍有历史正向修正。"
              "逐关键帧支持、分数、匹配掩码和身份保存在frame_readouts/*.npz，可直接验证所有未匹配Current-zero支持为0。", "",
              "## 5. 逐前缀AP", "",
              "各前缀使用同一完整场景GT集合，不是当前可见目标AP。100%前缀直接读取最后已提交状态，与本次终态一致。", "",
              "| 前缀 | 路线 | 原生框数 | AP15 | AP25 | AP50 |", "|---|---|---:|---:|---:|---:|"]
    for f, arms in result["prefix_ap"].items():
        for arm, x in arms.items():
            lines.append(f"| {100*float(f):.0f}% | {labels[arm]} | {x['count']} | "+" | ".join(f"{x['legacy_ap'][str(t)]:.4f}" for t in ev.TS)+" |")
    lines += ["", "## 6. 解释与论文写法", "",
              "本实验只检验固定持续原生集合上保留历史支持相对于当前帧支持的排序作用。"
              "历史最大值仍可由单个历史视图决定，结果不能表述为至少两个视图共同确认的证据。"
              "Current-zero在当前无匹配时不产生正向logit修正，而不是把原生检测分数本身改成0。", "",
              "旧表中的Latest应明确改称Last matched support（最近成功匹配支持），不能继续标作Current frame。"
              "本次数据来自与前缀实验相同的冻结原生缓存，不能与旧canonical运行的36.17/32.78/17.13等数值混用。", "",
              "**正文可用段落。** 为检验历史候选支持维护的作用，我们固定同一批1,787个原生框及其原始分数，"
              "每个关键帧仅执行一次投影与一对一匹配，并共享三因素视图支持和native-logit修正参数。"
              "当前帧对照在无匹配或投影无效时将支持置0，历史方案维护相同身份自几何重置以来的最大支持。"
              f"两者AP15/AP25/AP50分别为{current_text}和{max_text}，历史最大值差值为{delta_text}个AP点；"
              f"2,000次场景级配对bootstrap的95%区间分别为{interval_text}。"
              "结果支持在持续原生地图上保留历史候选支持的排序价值，但不构成多个视图必须共同确认的证据。", "",
              "## 7. 验证与文件", "",
              "原生与历史最大值四个前缀的全部框、分数和行顺序均与上一冻结实验逐项核对；"
              "200个输入缓存SHA256一致。所有关键帧验证Current-zero在未匹配时精确为0，且历史最大支持不小于当前支持。"
              "prefix_ap.csv、unified_topk.csv和fixed_fp.csv提供可编辑表格，结果与区间见results.json。"
              "固定FP统计仍为逐测试场景oracle阈值诊断，不是经开发集选择的部署阈值。", ""]
    (out/"HCSR当前帧与历史最大值严格对照.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    test_current_zero()
    protocol = json.loads((args.output/"protocol.json").read_text())
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines() if s.strip() and not s.startswith("#")]
    assert len(scenes) == 100
    if (args.output/"results.json").exists():
        result = json.loads((args.output/"results.json").read_text())
        assert result["protocol_sha256"] == core.sha256(args.output/"protocol.json")
    else:
        manifest = replay(args.output, protocol, scenes)
        result = evaluate(args.output, protocol, scenes, manifest)
    if "paired_topk_bootstrap_max_minus_current_zero" not in result:
        add_topk_bootstrap(args.output, result)
    validate_saved(args.output, result)
    render(args.output, result)
    print(json.dumps({"terminal_AP": result["prefix_ap"]["1.0"], "paired_bootstrap": result["paired_bootstrap"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
