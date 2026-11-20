#!/usr/bin/env python3
"""Quantify PTR/VCR dependence on recovery-score rules using frozen replay caches.

The experiment never changes a box, source label, native HSR score, or output count.
It only substitutes the scores assigned to PTR proposals and/or VCR dense pre-NMS
candidates.  The historical internal names ``proposal`` and ``anchor`` in the
cache correspond to PTR and VCR, respectively.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev
from tools import run_recovery_origin_memory_budget_audit as audit

SOURCE = ROOT / "reports/scannet_mechanism_protocol_controls_20261029"
OUT = ROOT / "reports/scannet_ptr_vcr_scoring_dependency_20261115"
THRESHOLDS = (.15, .25, .5)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def same_geometry(a, b) -> bool:
    return a[0] == b[0] and np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def mix(current, alternative, changed_sources):
    """Use alternative scores for selected sources while retaining current rows."""
    assert tuple(current["sources"]) == tuple(alternative["sources"])
    rows = []
    for now, alt, source in zip(current["rows"], alternative["rows"], current["sources"]):
        assert same_geometry(now, alt)
        score = alt[2] if source in changed_sources else now[2]
        rows.append((now[0], now[1], float(score)))
    return {"rows": rows, "sources": tuple(current["sources"])}


def retain(value, sources):
    rows, labels = [], []
    for row, source in zip(value["rows"], value["sources"]):
        if source in sources:
            rows.append(row)
            labels.append(source)
    return {"rows": rows, "sources": tuple(labels)}


def build_cases(scenes):
    arms = (
        "hsr_only",
        "ptr_current", "ptr_raw", "ptr_temperature2",
        "vcr_current", "vcr_raw", "vcr_temperature2",
        "full_current", "full_ptr_raw", "full_vcr_raw", "full_both_raw",
        "full_ptr_temperature2", "full_vcr_temperature2", "full_both_temperature2",
    )
    result = {arm: {} for arm in arms}
    checks = {
        "scenes": 0, "geometry_and_sources_exact": 0,
        "native_scores_exact_for_all_rules": 0,
    }
    for scene in scenes:
        path = SOURCE / "replay" / f"{scene}.pkl"
        payload = pickle.loads(path.read_bytes())["prefixes"]["1.0"]
        current = payload["full_current"]
        raw = payload["full_raw"]
        temp = payload["full_temperature2"]
        assert tuple(current["sources"]) == tuple(raw["sources"]) == tuple(temp["sources"])
        assert len(current["rows"]) == len(raw["rows"]) == len(temp["rows"])
        assert all(same_geometry(a, b) and same_geometry(a, c)
                   for a, b, c in zip(current["rows"], raw["rows"], temp["rows"]))
        native = [i for i, source in enumerate(current["sources"]) if source == "native"]
        assert all(current["rows"][i][2] == raw["rows"][i][2] == temp["rows"][i][2]
                   for i in native)

        ptr_raw = mix(current, raw, {"proposal"})
        vcr_raw = mix(current, raw, {"anchor"})
        ptr_temp = mix(current, temp, {"proposal"})
        vcr_temp = mix(current, temp, {"anchor"})
        both_raw = mix(current, raw, {"proposal", "anchor"})
        both_temp = mix(current, temp, {"proposal", "anchor"})

        result["hsr_only"][scene] = retain(current, {"native"})
        result["ptr_current"][scene] = retain(current, {"native", "proposal"})
        result["ptr_raw"][scene] = retain(ptr_raw, {"native", "proposal"})
        result["ptr_temperature2"][scene] = retain(ptr_temp, {"native", "proposal"})
        result["vcr_current"][scene] = retain(current, {"native", "anchor"})
        result["vcr_raw"][scene] = retain(vcr_raw, {"native", "anchor"})
        result["vcr_temperature2"][scene] = retain(vcr_temp, {"native", "anchor"})
        result["full_current"][scene] = current
        result["full_ptr_raw"][scene] = ptr_raw
        result["full_vcr_raw"][scene] = vcr_raw
        result["full_both_raw"][scene] = both_raw
        result["full_ptr_temperature2"][scene] = ptr_temp
        result["full_vcr_temperature2"][scene] = vcr_temp
        result["full_both_temperature2"][scene] = both_temp
        checks["scenes"] += 1
        checks["geometry_and_sources_exact"] += 1
        checks["native_scores_exact_for_all_rules"] += 1
    return result, checks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gts, alignments, transform = audit.core.scannet_inputs()
    scenes = [x.strip() for x in (ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt").read_text().splitlines()
              if x.strip()][:100]
    cases, checks = build_cases(scenes)

    predictions, plain, results = {}, {}, {}
    for arm, by_scene in cases.items():
        predictions[arm] = {scene: ev.as_prediction(value, scene, alignments, transform)
                            for scene, value in by_scene.items()}
        plain[arm] = {scene: (value[0], value[1]) for scene, value in predictions[arm].items()}
        counts = {source: sum(value["sources"].count(source) for value in by_scene.values())
                  for source in ("native", "proposal", "anchor")}
        results[arm] = {
            "counts": counts,
            "total": sum(counts.values()),
            "ap": {str(t): audit.core.class_agnostic_ap(plain[arm], gts, t)["ap"]
                   for t in THRESHOLDS},
        }
        print(f"evaluated {arm}", flush=True)

    comparisons = {
        "ptr_current_minus_raw": ("ptr_current", "ptr_raw"),
        "vcr_current_minus_raw": ("vcr_current", "vcr_raw"),
        "full_current_minus_ptr_raw": ("full_current", "full_ptr_raw"),
        "full_current_minus_vcr_raw": ("full_current", "full_vcr_raw"),
        "full_current_minus_both_raw": ("full_current", "full_both_raw"),
    }
    bootstrap = audit.bootstrap(plain, gts, scenes, comparisons, 2000, 20261115)
    for name, (left, right) in comparisons.items():
        for threshold in THRESHOLDS:
            bootstrap[name][str(threshold)]["point_ap_legacy"] = (
                results[left]["ap"][str(threshold)] - results[right]["ap"][str(threshold)])

    # Ranking-sensitive operational points. Geometry/counts are identical within
    # each current/raw pair, so any difference here is purely score ordering.
    workpoint_arms = ("ptr_current", "ptr_raw", "vcr_current", "vcr_raw",
                      "full_current", "full_ptr_raw", "full_vcr_raw", "full_both_raw")
    workpoints = {}
    for arm in workpoint_arms:
        workpoints[arm], _, _ = ev.workpoints(predictions[arm], gts, (.25, .5), (50,), (5,))

    # Diagnostic dependence ratios: how much of the gain over HSR-only disappears
    # when the branch's current score rule is replaced by its raw candidate score.
    dependence = {}
    for branch in ("ptr", "vcr"):
        dependence[branch] = {}
        for threshold in THRESHOLDS:
            key = str(threshold)
            base = results["hsr_only"]["ap"][key]
            current_gain = results[f"{branch}_current"]["ap"][key] - base
            raw_gain = results[f"{branch}_raw"]["ap"][key] - base
            temp_gain = results[f"{branch}_temperature2"]["ap"][key] - base
            dependence[branch][key] = {
                "current_gain_over_hsr": current_gain,
                "raw_gain_over_hsr": raw_gain,
                "temperature2_gain_over_hsr": temp_gain,
                "gain_lost_with_raw_percent": (100 * (current_gain - raw_gain) / current_gain
                                                if current_gain != 0 else None),
                "gain_lost_with_temperature2_percent": (100 * (current_gain - temp_gain) / current_gain
                                                         if current_gain != 0 else None),
            }

    output = {
        "protocol": {
            "dataset": "ScanNet-100",
            "source_report": str(SOURCE.relative_to(ROOT)),
            "terminal_prefix": "1.0",
            "native_outputs": 1787,
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20261115,
            "invariants": [
                "same frozen native records and HSR scores",
                "same recovery boxes, source labels, and output counts",
                "only PTR and/or VCR output scores are substituted",
                "no model inference and no GPU use",
            ],
            "score_rules": {
                "current": "PTR size-bucket output mapping; VCR deterministic low-tail mapping",
                "raw": "PTR medoid observation score; VCR representative dense pre-NMS candidate score",
                "temperature2": "sigmoid(logit(raw)/2), used only as a sensitivity probe",
            },
        },
        "checks": checks,
        "terminal": results,
        "dependence": dependence,
        "bootstrap": bootstrap,
        "workpoints": workpoints,
        "interpretation_limit": (
            "The gain-lost percentage is a diagnostic ratio relative to HSR-only, not an additive causal "
            "decomposition of AP. AP is nonlinear and PTR/VCR predictions interact in global ranking."
        ),
    }
    save_json(OUT / "results.json", output)

    with (OUT / "terminal_ap.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "native", "PTR", "VCR", "total", "AP15", "AP25", "AP50"])
        for arm, value in results.items():
            c = value["counts"]
            writer.writerow([arm, c["native"], c["proposal"], c["anchor"], value["total"],
                             value["ap"]["0.15"], value["ap"]["0.25"], value["ap"]["0.5"]])

    lines = [
        "# ScanNet-100：PTR/VCR收益对评分规则的依赖", "",
        "本实验使用冻结的终态重放缓存，不重新运行模型。恢复框几何、来源、数量、原生HSR分数均固定；"
        "只替换PTR和/或VCR的输出分数。因此，AP与有限工作点差异只来自跨分支排序变化。", "",
        "## 结果", "",
        "完整数值见 `terminal_ap.csv` 与 `results.json`。", "",
        "## 解释边界", "",
        "`gain_lost_with_raw_percent` 衡量分支相对HSR-only的表观AP增益中，有多少在改用原始候选分数后消失。"
        "它是诊断比例，不是可加和的因果贡献分解；AP对全局排序是非线性的。", "",
    ]
    # Keep the hand-reviewed REPORT.md intact on reruns; this short file is the
    # automatically regenerated pointer to the machine-readable results.
    (OUT / "AUTO_SUMMARY.md").write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
