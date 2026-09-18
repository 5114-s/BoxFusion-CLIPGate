#!/usr/bin/env python3
"""Read-only GT coverage of native NMS children; this does not estimate AP.

Use the final predictions from the *same observer run*. CA1MDetectionDataset
loads after_filter_boxes.npy directly in world coordinates. The current anchor
evaluator's box3d_iou_v2 uses axis-aligned corner bounds, despite its OBB name.
We verify the vectorized equivalent before use and match its strict IoU > t.
Any-child coverage permits one candidate to cover multiple GT and ignores score
ordering. It is a candidate-set diagnostic, not one-to-one detection recall/AP.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

THRESHOLDS = (0.15, 0.25, 0.5)
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_boxes(value, name):
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 8, 3), dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        raise ValueError(f"{name}: expected [N,8,3], got {boxes.shape}")
    if not np.isfinite(boxes).all() or (np.ptp(boxes, axis=1) <= 0).any():
        raise ValueError(f"{name}: nonfinite or degenerate box")
    return boxes


def pairwise_iou(boxes, gt):
    """Vectorized equivalent of anchor box3d_iou_v2's actual AABB metric."""
    if not len(boxes) or not len(gt):
        return np.zeros((len(boxes), len(gt)), dtype=np.float64)
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = gt.min(1), gt.max(1)
    inter = np.maximum(0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)).prod(2)
    union = (hi - lo).prod(1)[:, None] + (gh - gl).prod(1) - inter
    return inter / union


def verify_anchor(root):
    root = Path(root).resolve()
    path = root / "utils/box_util.py"
    spec = importlib.util.spec_from_file_location("ca1m_anchor_box_util", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(20260907)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    centers = rng.uniform(-2, 2, (32, 3))
    sizes = rng.uniform(0.1, 3, (32, 3))
    boxes = centers[:, None] + signs[None] * sizes[:, None] / 2
    # Rotated corners also must reduce to AABB bounds under this evaluator.
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    other = boxes @ rotation.T + rng.normal(0, 0.1, boxes.shape)
    matrix = pairwise_iou(boxes, other)
    pairs = [(i, i) for i in range(32)] + [(i, (i + 9) % 32) for i in range(32)]
    error = max(abs(matrix[i, j] - float(module.box3d_iou_v2(boxes[i], other[j])[0])) for i, j in pairs)
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError(f"Anchor metric differs from vectorized AABB IoU: max error={error}")
    eval_source = (root / "utils/eval_det.py").read_text()
    if "if ovmax > ovthresh:" not in eval_source:
        raise ValueError("Anchor threshold comparison changed; inspect evaluator before running")
    return {"function": str(path) + ":box3d_iou_v2", "actual_metric": "AABB IoU from world corner bounds", "comparison": "strictly greater than threshold", "self_check_pairs": len(pairs), "self_check_max_error": error, "box_util_sha256": sha256(path), "eval_det_sha256": sha256(root / "utils/eval_det.py")}


def read_scenes(path, expected):
    scenes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(scenes) != expected or len(set(scenes)) != len(scenes) or not all(s.isdecimal() for s in scenes):
        raise ValueError(f"Expected exactly {expected} unique numeric CA-1M scene IDs")
    return scenes


def preflight(args):
    scenes = read_scenes(args.scenes, args.expected_scenes)
    cfg = yaml.safe_load(Path(args.config).read_text())
    reference = yaml.safe_load(Path(args.reference_config).read_text())
    pvq = cfg["association"]["pvq_ar"]
    if not (pvq["enabled"] is True and pvq["nms_observer"] is True and pvq["mode"] == "shadow" and not (pvq.get("nms_stage") or {}).get("enabled", False)):
        raise ValueError("Require enabled NMS observer, shadow mode, and disabled NMS adjudication")
    if cfg["detection"]["score_thresh"] != 0.15:
        raise ValueError("Require threshold=0.15")
    if Path(cfg["data"]["output_dir"]).resolve() != Path(args.baseline_root).resolve() or Path(pvq["diagnostics_dir"]).resolve() != Path(args.diagnostics_root).resolve():
        raise ValueError("Audit paths must match this observer configuration")
    def comparable(data):
        data = copy.deepcopy(data)
        data["data"].pop("output_dir")
        data["lifting"]["boxer"].pop("diagnostics_dir")
        for key in ("enabled", "diagnostics_dir", "nms_stage"):
            data["association"]["pvq_ar"].pop(key, None)
        return data
    if comparable(cfg) != comparable(reference):
        raise ValueError("Observer config changed native settings beyond logging and artifact paths")
    sys.path.insert(0, str(REPO))
    from boxfusion.pvq_ar import PVQAR
    summary = PVQAR(cfg).finalize()  # No scene opened: no diagnostic files written.
    for key in ("nms_observer", "nms_records", "nms_record_cap", "nms_record_cap_hit"):
        if key not in summary:
            raise ValueError(f"PVQ summary cannot validate NMS completeness: missing {key}")
    root = Path(args.data_root)
    if set(p.name for p in root.iterdir()) != set(scenes):
        raise ValueError("GT root must contain exactly the requested scene set; anchor evaluator lists the whole root")
    for scene in scenes:
        path = root / scene / "after_filter_boxes.npy"
        valid_boxes(np.load(path, allow_pickle=False), str(path))
    return scenes, verify_anchor(args.evaluator_root)


def load_summary(path, scene):
    summary = json.loads(Path(path).read_text())
    if summary.get("scene_id") != scene or summary.get("mode") != "shadow" or summary.get("nms_observer") is not True:
        raise ValueError(f"{path}: wrong scene/mode or NMS observer disabled")
    n, cap = summary.get("nms_records"), summary.get("nms_record_cap")
    if type(n) is not int or type(cap) is not int or not 0 <= n < cap:
        raise ValueError(f"{path}: missing/invalid NMS count or reached logging cap")
    if summary.get("nms_record_cap_hit") is not False:
        raise ValueError(f"{path}: missing/true NMS cap-hit marker")
    return n


def event_batches(path, scene, expected, batch_size=512):
    path = Path(path)
    if not path.exists():
        if expected == 0:
            return
        raise ValueError(f"{path}: missing ledger with expected {expected} events")
    count, batch = 0, []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n") or not raw.strip():
                raise ValueError(f"{path}:{line_number}: truncated or blank JSONL record")
            record = json.loads(raw)
            if record.get("type") != "nms_merge" or record.get("scene_id") != scene:
                raise ValueError(f"{path}:{line_number}: wrong event type or scene")
            for key in ("keyframe_id", "parent_frame_id", "child_frame_id", "parent_init_id", "child_init_id"):
                if type(record.get(key)) is not int or record[key] < 0:
                    raise ValueError(f"{path}:{line_number}: invalid {key}")
            if max(record["parent_frame_id"], record["child_frame_id"]) > record["keyframe_id"]:
                raise ValueError(f"{path}:{line_number}: future observation in NMS event")
            valid_boxes([record["parent_corners_world"], record["child_corners_world"]], f"{path}:{line_number}")
            record["event_line"] = line_number
            batch.append(record)
            count += 1
            if count > expected:
                raise ValueError(f"{path}: ledger longer than finalized NMS count")
            if len(batch) == batch_size:
                yield batch
                batch = []
    if count != expected:
        raise ValueError(f"{path}: {count} records, summary declares {expected}")
    if batch:
        yield batch


def empty_support():
    return {"events": 0, "child_observation_ids": set(), "child_frame_ids": set(), "emit_keyframe_ids": set()}


def add_support(support, record):
    support["events"] += 1
    support["child_observation_ids"].add((record["child_init_id"], record["child_frame_id"]))
    support["child_frame_ids"].add(record["child_frame_id"])
    support["emit_keyframe_ids"].add(record["keyframe_id"])


def serialize_support(support):
    return {"events": support["events"], "unique_child_observations": len(support["child_observation_ids"]), "distinct_child_frames": len(support["child_frame_ids"]), "distinct_emit_keyframes": len(support["emit_keyframe_ids"]), "child_observation_ids": [list(v) for v in sorted(support["child_observation_ids"])], "child_frame_ids": sorted(support["child_frame_ids"]), "emit_keyframe_ids": sorted(support["emit_keyframe_ids"])}


def audit_scene(scene, args):
    diag, baseline, root = Path(args.diagnostics_root), Path(args.baseline_root), Path(args.data_root)
    summary_path = diag / f"{scene}_pvq_ar_summary.json"
    n = load_summary(summary_path, scene)
    pred_path = baseline / f"{scene}_boxes.pkl"
    with pred_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(f"{pred_path}: expected one scene batch in evaluator prediction format")
    rows = payload[0]
    for row in rows:
        if len(row) != 3 or int(row[0]) != 0 or not np.isfinite(float(row[2])):
            raise ValueError(f"{pred_path}: expected class-agnostic [0,corners,finite_score] rows")
    gt_path = root / scene / "after_filter_boxes.npy"
    gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
    pred = valid_boxes([row[1] for row in rows], str(pred_path))
    base_iou = pairwise_iou(pred, gt).max(0) if len(pred) else np.zeros(len(gt))
    state = {t: [{"child": empty_support(), "different_parent_missing": empty_support(), "best_child_iou": 0.0, "best_event": None} for _ in gt] for t in THRESHOLDS}
    partitions = {t: {"child_uncovered": 0, "same_best_gt": 0, "different_best_gt": 0, "parent_uncovered": 0} for t in THRESHOLDS}
    ledger = diag / f"{scene}_pvq_nms.jsonl"
    for batch in event_batches(ledger, scene, n):
        ciou = pairwise_iou(np.array([r["child_corners_world"] for r in batch]), gt)
        piou = pairwise_iou(np.array([r["parent_corners_world"] for r in batch]), gt)
        for t in THRESHOLDS:
            for index, record in enumerate(batch):
                if not len(gt) or ciou[index].max() <= t:
                    partitions[t]["child_uncovered"] += 1
                    continue
                cb, pb = int(ciou[index].argmax()), int(piou[index].argmax())
                category = "parent_uncovered" if piou[index, pb] <= t else "same_best_gt" if pb == cb else "different_best_gt"
                partitions[t][category] += 1
                for g in np.flatnonzero(ciou[index] > t):
                    info = state[t][g]
                    add_support(info["child"], record)
                    # Parent must fail to cover this particular GT, even when
                    # its best GT differs. This avoids overlap-based false claims.
                    if piou[index, g] <= t and base_iou[g] <= t:
                        add_support(info["different_parent_missing"], record)
                    if float(ciou[index, g]) > info["best_child_iou"]:
                        info["best_child_iou"] = float(ciou[index, g])
                        info["best_event"] = {"event_id": f"{scene}:{record['event_line']}", **{key: record[key] for key in ("event_line", "child_init_id", "child_frame_id", "parent_init_id", "parent_frame_id", "keyframe_id")}, "parent_iou_for_gt": float(piou[index, g])}
    thresholds = {}
    for t, infos in state.items():
        per_gt = []
        for g, info in enumerate(infos):
            per_gt.append({"gt_id": f"{scene}:{g}", "baseline_best_iou": float(base_iou[g]), "baseline_covered": bool(base_iou[g] > t), "child_covered": bool(info["child"]["events"]), "best_child_iou": info["best_child_iou"], "best_child_event": info["best_event"], "child_support": serialize_support(info["child"]), "different_parent_missing_support": serialize_support(info["different_parent_missing"])})
        thresholds[f"{t:.2f}"] = {"event_partition_best_gt": partitions[t], "gt": per_gt}
    return {"scene": scene, "predictions": len(pred), "gt_count": len(gt), "nms_records": n, "summary_sha256": sha256(summary_path), "nms_ledger_sha256": sha256(ledger) if ledger.exists() else None, "prediction_sha256": sha256(pred_path), "gt_sha256": sha256(gt_path), "thresholds": thresholds}


def aggregate(scenes):
    totals = {}
    for threshold in (f"{t:.2f}" for t in THRESHOLDS):
        rows = [g for s in scenes for g in s["thresholds"][threshold]["gt"]]
        missing = [g for g in rows if not g["baseline_covered"]]
        recovered = [g for g in missing if g["child_covered"]]
        different = [g for g in missing if g["different_parent_missing_support"]["events"]]
        totals[threshold] = {"gt_total": len(rows), "baseline_any_prediction_covered_gt": len(rows) - len(missing), "baseline_uncovered_gt": len(missing), "any_child_covered_gt": sum(g["child_covered"] for g in rows), "additional_any_child_covered_gt": len(recovered), "additional_coverage_percentage_points": 100 * len(recovered) / max(len(rows), 1), "different_parent_uncovered_gt": len(different), "different_parent_uncovered_gt_two_child_frames": sum(g["different_parent_missing_support"]["distinct_child_frames"] >= 2 for g in different), "different_parent_uncovered_gt_three_child_frames": sum(g["different_parent_missing_support"]["distinct_child_frames"] >= 3 for g in different), "different_parent_uncovered_events": sum(g["different_parent_missing_support"]["events"] for g in different), "additional_gt_ids": [g["gt_id"] for g in recovered]}
    return totals


def markdown_report(result):
    lines = ["# CA-1M NMS-child 候选覆盖诊断", "", "本结果是 GT 辅助的离线候选集合覆盖上限，不是 AP、实际可实现增益或一对一召回率。一个候选可能覆盖多个 GT；未建模排序、误检、birth 确认及跨帧实例可实现性。", "", "基线为同一次 observer 运行的最终 native 输出。GT 使用原坐标 after_filter_boxes.npy；复用锚点评测器实际 AABB IoU，匹配条件为严格 IoU > 阈值。", "", f"完整场景：{len(result['scenes'])}；NMS 事件：{sum(s['nms_records'] for s in result['scenes'])}。每场已核对 summary、完整 JSONL 行数及 cap 状态。", "", "| IoU | GT总数 | native已覆盖 | native未覆盖 | child额外覆盖 | parent不覆盖该GT的额外覆盖 | ≥2原始观测帧 | ≥3原始观测帧 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t, stats in result["totals"].items():
        keys = ("gt_total", "baseline_any_prediction_covered_gt", "baseline_uncovered_gt", "additional_any_child_covered_gt", "different_parent_uncovered_gt", "different_parent_uncovered_gt_two_child_frames", "different_parent_uncovered_gt_three_child_frames")
        lines.append("| " + t + " | " + " | ".join(str(stats[k]) for k in keys) + " |")
    lines.extend(["", "支持帧按 child_frame_id 去重；keyframe_id 是事件产生时刻，不能当作新观测。同一个 child 在多个 keyframe 被反复吞并不构成多视角支持。原始 child ID、帧 ID、事件行号及最佳 child 见 JSON。", "", "所有计数均允许 GT 独立覆盖；多个对象共用同一候选时无法同时作为正确检测输出。本表不提供 AP 预测。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference-config", "scenes", "baseline-root", "diagnostics-root", "data-root", "evaluator-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-scenes", type=int, default=107)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--markdown")
    args = parser.parse_args()
    try:
        scenes, anchor = preflight(args)
        if args.preflight_only:
            print(json.dumps({"preflight": "passed", "scenes": len(scenes), "anchor": anchor}, indent=2))
            return 0
        if not args.output or not args.markdown:
            parser.error("--output and --markdown are required for an audit")
        if Path(args.output).exists() or Path(args.markdown).exists():
            raise ValueError("Refusing to overwrite an existing audit report")
        prediction_names = {p.name for p in Path(args.baseline_root).glob("*_boxes.pkl")}
        if prediction_names != {f"{s}_boxes.pkl" for s in scenes}:
            raise ValueError("Prediction scene coverage does not exactly match scene list")
        summaries = {p.name for p in Path(args.diagnostics_root).glob("*_pvq_ar_summary.json")}
        if summaries != {f"{s}_pvq_ar_summary.json" for s in scenes}:
            raise ValueError("Finalized summary scene coverage does not exactly match scene list")
        scene_results = []
        for i, scene in enumerate(scenes, 1):
            scene_results.append(audit_scene(scene, args))
            print(f"[{i}/{len(scenes)}] audited {scene}: {scene_results[-1]['nms_records']} NMS records", flush=True)
        result = {"schema": "boxfusion.ca1m_nms_child_coverage.v1", "interpretation": "Independent GT coverage only; not AP, one-to-one recall, or implementable birth gain", "baseline": "Final native predictions from this same observer run", "config": str(Path(args.config).resolve()), "config_sha256": sha256(args.config), "scene_list_sha256": sha256(args.scenes), "baseline_root": str(Path(args.baseline_root).resolve()), "diagnostics_root": str(Path(args.diagnostics_root).resolve()), "gt_loader": "CA1MDetectionDataset semantics: np.load(after_filter_boxes.npy), no coordinate transform", "anchor": anchor, "thresholds": list(THRESHOLDS), "totals": aggregate(scene_results), "scenes": scene_results}
        for target in (args.output, args.markdown):
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        with Path(args.markdown).open("x") as handle:
            handle.write(markdown_report(result))
        print(json.dumps(result["totals"], indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, pickle.UnpicklingError) as exc:
        print(f"NMS_CHILD_AUDIT_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
