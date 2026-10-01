#!/usr/bin/env python3
"""Audit and select qualitative cases from the sealed strict-online v2 run.

The audit mode is deliberately independent of model inference.  Selection is
allowed only after the audit passes and remains provisional until a selected-
scene evidence export supplies the histories that were not persisted by the
original official100 run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
from typing import Any

import numpy as np


ROOT = Path("/data/ZhaoX/BoxFusion")
REPORT = ROOT / "reports/strict_causal_nochild_official100_v2_20260922"
FACTORIAL = ROOT / "results/scannet_strict_causal_nochild_official100_v2_factorial"
DIAGNOSTICS = ROOT / "diagnostics/strict_causal_nochild_official100_v2"
RAW_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
GT_ROOT = ROOT / "evaluation/data_util/scannet_train_detection_data"
SCENE_LIST = ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
DEFAULT_OUTPUT = ROOT / "reports/qualitative_strict_online_nochild_original"
ARMS = ("base", "p", "a", "p_a", "m2", "p_m2", "a_m2", "p_a_m2")


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def scenes() -> list[str]:
    rows = [line.strip() for line in SCENE_LIST.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 100 or len(set(rows)) != 100:
        raise RuntimeError("the locked official100 scene list is invalid")
    return rows


def read_rows(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    output = []
    for row in payload[0]:
        if len(row) != 3:
            raise RuntimeError(f"invalid prediction row: {path}")
        output.append((int(row[0]), np.asarray(row[1], dtype=np.float64).reshape(8, 3),
                       float(row[2])))
    return output


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def current_fingerprint() -> str:
    paths = [
        ROOT / "config/scannet_t05_boxer_strict_causal_nochild_official100_v2.yaml",
        ROOT / "demo.py", ROOT / "boxfusion/online_candidate_map.py",
        ROOT / "boxfusion/online_candidate_runtime.py",
        ROOT / "boxfusion/box_manager.py", ROOT / "boxfusion/instances.py",
        ROOT / "models/cutr_rgbd.pth", ROOT / "models/open_clip_pytorch_model.bin",
        ROOT / "data/panoptic_categories_nomerge.txt",
    ]
    first = "".join(f"{sha256(path)}  {path}\n" for path in paths).encode()
    return hashlib.sha256(first).hexdigest()


def _arm_path(arm: str, scene: str) -> Path:
    return FACTORIAL / arm / f"{scene}_boxes.pkl"


def audit_assets(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    locked_scenes = scenes()
    manifest_path = FACTORIAL / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    warnings: list[str] = []
    verified_hashes: dict[str, str] = {}

    expected = {
        "scene_count": 100,
        "paired_from_one_run": True,
        "children_enabled": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(f"factorial manifest {key}={manifest.get(key)!r}, expected {value!r}")

    for raw_path, expected_hash in manifest.get("input_sha256", {}).items():
        path = Path(raw_path)
        if not path.is_file():
            errors.append(f"missing sealed input: {path}")
            continue
        actual = sha256(path)
        if actual != expected_hash:
            errors.append(f"sealed input hash mismatch: {path}")
        else:
            verified_hashes[str(path)] = actual

    arm_counts: dict[str, int] = {}
    structural_scenes = 0
    for scene in locked_scenes:
        missing = [arm for arm in ARMS if not _arm_path(arm, scene).is_file()]
        if missing:
            errors.append(f"{scene}: missing factorial arms {missing}")
            continue
        rows = {arm: read_rows(_arm_path(arm, scene)) for arm in ARMS}
        for arm in ARMS:
            arm_counts[arm] = arm_counts.get(arm, 0) + len(rows[arm])
        n = len(rows["base"])
        p = len(rows["p"]) - n
        a = len(rows["a"]) - n
        expected_counts = manifest["per_scene"].get(scene, {})
        if expected_counts != {"native": n, "m1p": p, "m1a": a}:
            errors.append(f"{scene}: component counts disagree with manifest")
        if not (len(rows["m2"]) == n and len(rows["p_a_m2"]) == n + p + a):
            errors.append(f"{scene}: factorial lengths are inconsistent")
        elif n:
            base_boxes = np.asarray([row[1] for row in rows["base"]])
            m2_boxes = np.asarray([row[1] for row in rows["m2"]])
            if not np.array_equal(base_boxes, m2_boxes):
                errors.append(f"{scene}: M2 changed native geometry/order")
        structural_scenes += 1

    raw_summary: dict[str, dict[str, int | bool]] = {}
    raw_complete = 0
    gt_complete = 0
    for scene in locked_scenes:
        frame_root = RAW_ROOT / scene
        counts = {
            "rgb": len(list((frame_root / "color").glob("*.jpg"))),
            "depth": len(list((frame_root / "depth").glob("*.png"))),
            "pose": len(list((frame_root / "pose").glob("*.txt"))),
            "intrinsics": len(list((frame_root / "intrinsic").glob("*.txt"))),
        }
        gt = GT_ROOT / f"{scene}_bbox.npy"
        meta = frame_root / f"{scene}.txt"
        counts["gt"] = gt.is_file()
        counts["scene_metadata"] = meta.is_file()
        raw_summary[scene] = counts
        if counts["rgb"] and counts["rgb"] == counts["depth"] == counts["pose"] \
                and counts["intrinsics"] >= 2:
            raw_complete += 1
        if gt.is_file() and meta.is_file():
            gt_complete += 1

    saved_fingerprint_path = ROOT / "logs/scannet_strict_causal_nochild_official100_v2/source.fingerprint"
    saved_fingerprint = saved_fingerprint_path.read_text().strip()
    live_fingerprint = current_fingerprint()
    if live_fingerprint != saved_fingerprint:
        warnings.append(
            "the current source/config aggregate does not match the run fingerprint; "
            "a selected-scene exporter must run from an isolated source snapshot and "
            "prove terminal prediction parity before its histories are accepted"
        )
    runtime_commit = None
    warnings.append("the original run did not record a Git commit or per-source-file hashes")

    field_audit = {
        "base_and_full_predictions": {
            "status": "present",
            "evidence": "eight factorial arms, 100 scenes, sealed input hashes",
        },
        "native_plr_calr_sources": {
            "status": "derivable",
            "evidence": "paired one-run component counts and fixed concatenation order",
            "limitation": "prediction rows themselves carry no source label",
        },
        "mvsr_scores_and_rank": {
            "status": "partially_present",
            "evidence": "base and m2 have identical native geometry/order; scores permit ranks",
            "missing": "explicit per-frame support history and matched proposal identity",
        },
        "plr_proposal_tracks": {
            "status": "missing",
            "present": "terminal PLR boxes and per-scene counts",
            "missing": "track observations, proposal boxes, keyframe IDs and 2D regions",
        },
        "calr_anchor_voxels": {
            "status": "missing",
            "present": "terminal CALR boxes and aggregate active/birth counts",
            "missing": "anchor observations, voxel keys and supporting keyframe IDs",
        },
        "scannet_inputs": {
            "status": "present" if raw_complete == 100 and gt_complete == 100 else "incomplete",
            "complete_rgbd_pose_intrinsics_scenes": raw_complete,
            "complete_gt_and_metadata_scenes": gt_complete,
        },
        "run_provenance": {
            "status": "partial",
            "run_id": saved_fingerprint,
            "config_path": str(ROOT / "config/scannet_t05_boxer_strict_causal_nochild_official100_v2.yaml"),
            "runtime_git_commit": runtime_commit,
            "current_git_head": command_output(["git", "rev-parse", "HEAD"]),
            "sealed_prediction_and_diagnostic_hashes": len(verified_hashes),
            "current_source_matches_run_fingerprint": live_fingerprint == saved_fingerprint,
        },
    }
    ready_for_selection = not errors
    ready_for_final_render = ready_for_selection and all(
        field_audit[name]["status"] == "present"
        for name in ("plr_proposal_tracks", "calr_anchor_voxels")
    ) and field_audit["mvsr_scores_and_rank"].get("missing") is None

    audit = {
        "schema": "boxfusion.qualitative_asset_audit.v1",
        "locked_run": "strict_causal_nochild_official100_v2_20260922",
        "expected_ap": {"base": [34.89, 31.46, 15.74],
                        "full": [44.23, 39.66, 20.00]},
        "scene_count": len(locked_scenes),
        "structurally_verified_scenes": structural_scenes,
        "manifest_components": manifest.get("components"),
        "manifest_arm_counts": manifest.get("arm_counts"),
        "observed_arm_counts": arm_counts,
        "field_audit": field_audit,
        "errors": errors,
        "warnings": warnings,
        "ready_for_case_selection": ready_for_selection,
        "ready_for_final_render": ready_for_final_render,
        "minimal_export_required": not ready_for_final_render,
        "raw_asset_summary": raw_summary,
    }
    (output / "asset_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    current_source_paths = [
        ROOT / "config/scannet_t05_boxer_strict_causal_nochild_official100_v2.yaml",
        ROOT / "scripts/run_scannet_strict_causal_nochild_official100_v2.sh",
        ROOT / "demo.py", ROOT / "boxfusion/online_candidate_map.py",
        ROOT / "boxfusion/online_candidate_runtime.py",
        ROOT / "boxfusion/box_manager.py", ROOT / "boxfusion/instances.py",
        ROOT / "models/cutr_rgbd.pth", ROOT / "models/open_clip_pytorch_model.bin",
        ROOT / "data/panoptic_categories_nomerge.txt",
    ]
    current_source_hashes = {str(path.resolve()): sha256(path)
                             for path in current_source_paths}
    source_files = {
        str(manifest_path.resolve()): sha256(manifest_path),
        str((REPORT / "REPORT.md").resolve()): sha256(REPORT / "REPORT.md"),
        str((REPORT / "summary.json").resolve()): sha256(REPORT / "summary.json"),
        str(SCENE_LIST.resolve()): sha256(SCENE_LIST),
        str(saved_fingerprint_path.resolve()): sha256(saved_fingerprint_path),
    }
    source_files.update(verified_hashes)
    source_files.update(current_source_hashes)
    source_manifest = {
        "schema": "boxfusion.qualitative_source_manifest.v1",
        "locked_run_id": saved_fingerprint,
        "runtime_git_commit": runtime_commit,
        "current_git_head": command_output(["git", "rev-parse", "HEAD"]),
        "saved_aggregate_fingerprint": saved_fingerprint,
        "current_aggregate_fingerprint": live_fingerprint,
        "current_config_path": str(current_source_paths[0]),
        "current_config_sha256": current_source_hashes[str(current_source_paths[0].resolve())],
        "current_source_hashes": current_source_hashes,
        "roots": {
            "report": str(REPORT), "factorial": str(FACTORIAL),
            "diagnostics": str(DIAGNOSTICS), "raw_scannet": str(RAW_ROOT),
            "ground_truth": str(GT_ROOT),
        },
        "source_file_count": len(source_files),
        "current_source_matches_run_fingerprint": live_fingerprint == saved_fingerprint,
        "final_render_authorized_by_audit": ready_for_final_render,
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "source_hashes.sha256").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(source_files.items())),
        encoding="utf-8",
    )

    lines = [
        "# 原始严格在线无 child 版本：定性图资产审计", "",
        f"- 锁定运行：`{audit['locked_run']}`",
        f"- 运行指纹：`{saved_fingerprint}`",
        f"- official100结构核验：{structural_scenes}/100",
        f"- 已验证的历史预测/诊断哈希：{len(verified_hashes)}",
        f"- RGB-D/位姿/内参完整场景：{raw_complete}/100",
        f"- GT与axis-alignment元数据完整场景：{gt_complete}/100", "",
        "| 资产 | 状态 | 结论 |", "|---|---|---|",
        "| Base/Full逐场预测 | 已存在 | 八臂由同一次运行配对拆分，哈希通过 |",
        "| Native/PLR/CALR来源 | 可推导 | 根据manifest计数和拼接顺序拆分；pkl没有显式source字段 |",
        "| MVSR分数与rank | 部分存在 | base/m2几何顺序相同，可计算分数与rank；缺逐帧support历史 |",
        "| PLR轨迹 | 缺失 | 仅有终态框和计数，缺proposal观测与关键帧编号 |",
        "| CALR体素状态 | 缺失 | 仅有终态框和计数，缺anchor、体素key和支持帧 |",
        "| ScanNet RGB-D/位姿/内参/GT | 已存在 | 可用于选定场景的证据导出和绘图 |",
        "| 运行来源 | 部分存在 | 有聚合run fingerprint；未记录运行时Git commit和逐源码哈希 |",
        "", "## 决策", "",
        "现有资产足以进行终态候选筛选，但不足以直接绘制满足论文要求的证据链。",
        "必须只对选定场景运行最小导出器，并同时满足：使用隔离源码；不写入历史目录；",
        "导出后Base/Full终态框与历史文件在几何、分数和顺序上完全一致；否则拒绝使用。",
        "在证据导出完成前，渲染器必须保持fail-closed。", "",
    ]
    if errors:
        lines += ["## 错误", ""] + [f"- {row}" for row in errors] + [""]
    if warnings:
        lines += ["## 限制", ""] + [f"- {row}" for row in warnings] + [""]
    (output / "ASSET_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def axis_alignment(scene: str) -> np.ndarray:
    path = RAW_ROOT / scene / f"{scene}.txt"
    for line in path.read_text().splitlines():
        if line.startswith("axisAlignment ="):
            return np.asarray([float(x) for x in line.split("=", 1)[1].split()],
                              dtype=np.float64).reshape(4, 4)
    raise RuntimeError(f"missing axisAlignment in {path}")


def aligned_boxes(rows: list[tuple[int, np.ndarray, float]], align: np.ndarray) -> np.ndarray:
    if not rows:
        return np.empty((0, 8, 3), dtype=np.float64)
    boxes = np.asarray([row[1] for row in rows], dtype=np.float64)
    return np.einsum("ij,nkj->nki", align[:3, :3], boxes) + align[:3, 3]


def aabb_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    l0, l1 = left.min(1), left.max(1)
    r0, r1 = right.min(1), right.max(1)
    lo = np.maximum(l0[:, None], r0[None])
    hi = np.minimum(l1[:, None], r1[None])
    inter = np.prod(np.maximum(hi - lo, 0.0), axis=2)
    lv = np.prod(np.maximum(l1 - l0, 0.0), axis=1)[:, None]
    rv = np.prod(np.maximum(r1 - r0, 0.0), axis=1)[None]
    return inter / np.maximum(lv + rv - inter, 1e-12)


def gt_corners(scene: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.load(GT_ROOT / f"{scene}_bbox.npy")
    centers, sizes = rows[:, :3], rows[:, 3:6]
    signs = np.asarray([[x, y, z] for z in (-1, 1) for y in (-1, 1)
                        for x in (-1, 1)], dtype=np.float64)
    boxes = centers[:, None] + signs[None] * sizes[:, None] / 2.0
    labels = rows[:, 6].astype(int) if rows.shape[1] > 6 else np.full(len(rows), -1)
    return boxes, labels


def rank(scores: list[float], index: int) -> int:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order.index(index) + 1


def logit(value: float) -> float:
    value = min(max(value, 1e-9), 1.0 - 1e-9)
    return math.log(value / (1.0 - value))


def select_cases(output: Path) -> dict[str, Any]:
    audit_path = output / "asset_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("run --mode audit before selecting cases")
    audit = json.loads(audit_path.read_text())
    if not audit.get("ready_for_case_selection"):
        raise RuntimeError("asset audit did not authorize case selection")

    candidates: dict[str, list[dict[str, Any]]] = {
        "plr_recovery": [], "calr_recovery": [], "mvsr_reranking": [],
        "failure_plr_localization": [], "failure_calr_background": [],
        "failure_mvsr_fp_promotion": [],
    }
    for scene in scenes():
        manifest = json.loads((FACTORIAL / "manifest.json").read_text())
        counts = manifest["per_scene"][scene]
        n, p, a = counts["native"], counts["m1p"], counts["m1a"]
        base = read_rows(_arm_path("base", scene))
        m2 = read_rows(_arm_path("m2", scene))
        p_arm = read_rows(_arm_path("p", scene))
        a_arm = read_rows(_arm_path("a", scene))
        plr, calr = p_arm[n:n + p], a_arm[n:n + a]
        align = axis_alignment(scene)
        boxes = {
            "native": aligned_boxes(base, align),
            "plr": aligned_boxes(plr, align),
            "calr": aligned_boxes(calr, align),
        }
        gt, labels = gt_corners(scene)
        ious = {name: aabb_iou(value, gt) for name, value in boxes.items()}
        native_gt = ious["native"].max(0) if n else np.zeros(len(gt))
        plr_gt = ious["plr"].max(0) if p else np.zeros(len(gt))
        calr_gt = ious["calr"].max(0) if a else np.zeros(len(gt))

        for j in range(p):
            g = int(np.argmax(ious["plr"][j])) if len(gt) else -1
            value = float(ious["plr"][j, g]) if g >= 0 else 0.0
            native_value = float(native_gt[g]) if g >= 0 else 0.0
            calr_value = float(calr_gt[g]) if g >= 0 else 0.0
            duplicate = float(aabb_iou(boxes["plr"][j:j+1], boxes["calr"]).max()) \
                if a else 0.0
            row = {"scene": scene, "source_index": j, "gt_index": g,
                   "gt_semantic_id": int(labels[g]) if g >= 0 else -1,
                   "iou": value, "native_gt_iou": native_value,
                   "calr_gt_iou": calr_value, "calr_duplicate_iou": duplicate,
                   "score": float(plr[j][2]),
                   "box_aligned": boxes["plr"][j].round(6).tolist(),
                   "gt_box_aligned": gt[g].round(6).tolist() if g >= 0 else None}
            # CALR is considered a duplicate recovery only when it reaches the
            # same 0.25 coverage threshold or overlaps this PLR box by 0.25.
            if native_value < 0.15 and value >= 0.25 and calr_value < 0.25 and duplicate < 0.25:
                candidates["plr_recovery"].append(row)
            if native_value < 0.15 and 0.25 <= value < 0.50:
                candidates["failure_plr_localization"].append(row)

        for j in range(a):
            values = ious["calr"][j]
            g = int(np.argmax(values)) if len(gt) else -1
            value = float(values[g]) if g >= 0 else 0.0
            native_value = float(native_gt[g]) if g >= 0 else 0.0
            plr_value = float(plr_gt[g]) if g >= 0 else 0.0
            row = {"scene": scene, "source_index": j, "gt_index": g,
                   "gt_semantic_id": int(labels[g]) if g >= 0 else -1,
                   "iou": value, "native_gt_iou": native_value,
                   "plr_gt_iou": plr_value, "score": float(calr[j][2]),
                   "box_aligned": boxes["calr"][j].round(6).tolist(),
                   "gt_box_aligned": gt[g].round(6).tolist() if g >= 0 else None}
            if native_value < 0.15 and plr_value < 0.15 and value >= 0.25:
                candidates["calr_recovery"].append(row)
            if value < 0.05:
                row = dict(row)
                row["failure_reason"] = "no GT overlap at IoU 0.05"
                candidates["failure_calr_background"].append(row)

        original_scores = [row[2] for row in base]
        reranked_scores = [row[2] for row in m2]
        for j, (before, after) in enumerate(zip(original_scores, reranked_scores)):
            gt_value = float(ious["native"][j].max()) if len(gt) else 0.0
            before_rank, after_rank = rank(original_scores, j), rank(reranked_scores, j)
            delta = logit(after) - logit(before)
            inferred = 0.5 + delta / 2.0 if delta > 1e-8 else None
            row = {"scene": scene, "source_index": j, "native_gt_iou": gt_value,
                   "original_score": before, "reranked_score": after,
                   "original_rank": before_rank, "reranked_rank": after_rank,
                   "rank_gain": before_rank - after_rank,
                   "max_support_inferred_from_scores": inferred,
                   "support_inference_requires_rerun_validation": True,
                   "box_aligned": boxes["native"][j].round(6).tolist()}
            if gt_value >= 0.25 and before_rank - after_rank >= 1 and delta > 1e-8:
                candidates["mvsr_reranking"].append(row)
            if gt_value < 0.15 and before_rank - after_rank >= 1 and delta > 1e-8:
                row = dict(row)
                row["failure_reason"] = "native false positive promoted by MVSR"
                candidates["failure_mvsr_fp_promotion"].append(row)

    candidates["plr_recovery"].sort(key=lambda x: (-x["iou"], x["scene"], x["source_index"]))
    candidates["calr_recovery"].sort(key=lambda x: (-x["iou"], x["scene"], x["source_index"]))
    candidates["mvsr_reranking"].sort(key=lambda x: (-x["rank_gain"], -x["native_gt_iou"], x["scene"]))
    candidates["failure_plr_localization"].sort(key=lambda x: (-x["iou"], x["scene"]))
    candidates["failure_calr_background"].sort(key=lambda x: (-x["score"], x["scene"]))
    candidates["failure_mvsr_fp_promotion"].sort(key=lambda x: (-x["rank_gain"], x["scene"]))

    selected: dict[str, Any] = {}
    for key in ("plr_recovery", "calr_recovery", "mvsr_reranking"):
        if not candidates[key]:
            raise RuntimeError(f"no terminal candidate found for {key}")
        selected[key] = candidates[key][0]
    failure_priority = ("failure_mvsr_fp_promotion", "failure_plr_localization",
                        "failure_calr_background")
    failure_key = next((key for key in failure_priority if candidates[key]), None)
    if failure_key is None:
        raise RuntimeError("no real failure candidate found")
    selected["failure"] = dict(candidates[failure_key][0], failure_type=failure_key)

    for panel, row in selected.items():
        row.update({
            "panel": panel,
            "selection_status": "terminal-output candidate; evidence export pending",
            "evidence_verified": False,
            "evidence_export": None,
            "minimum_required_distinct_keyframes": 3 if panel in ("plr_recovery", "calr_recovery") else None,
        })
    payload = {
        "schema": "boxfusion.qualitative_cases.v1",
        "locked_run": "strict_causal_nochild_official100_v2_20260922",
        "provisional": True,
        "render_blocked_until_evidence_export": True,
        "selection_thresholds": {"native_miss_iou": 0.15, "recovery_match_iou": 0.25,
                                 "cross_branch_duplicate_iou": 0.25,
                                 "strict_localization_iou": 0.50},
        "cases": selected,
        "candidate_counts": {key: len(value) for key, value in candidates.items()},
    }
    (output / "selected_cases.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = sorted({key for row in selected.values() for key in row if key not in
                      ("box_aligned", "gt_box_aligned")})
    with (output / "case_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in selected.values():
            writer.writerow({key: row.get(key) for key in columns})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "select"), default="audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.mode == "audit":
        result = audit_assets(args.output)
        print(json.dumps({key: result[key] for key in
                          ("ready_for_case_selection", "ready_for_final_render",
                           "minimal_export_required")}, indent=2))
    else:
        result = select_cases(args.output)
        print(json.dumps({key: value["scene"] for key, value in result["cases"].items()},
                         indent=2))


if __name__ == "__main__":
    main()
