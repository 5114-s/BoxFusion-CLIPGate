#!/usr/bin/env python3
"""Build the verified data bundle for CaRR-3D supplementary Sections A--E.

The script only reads frozen source code, replay caches, and completed reports.
It does not run detector inference or alter any existing experiment artifact.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/carr3d_supplement_data_A_E_20261120"
ZIP = ROOT / "reports/carr3d_supplement_data_A_E_20261120.zip"
SOURCE = ROOT / "development/official_github_exact_paired_efficiency_20261009/recar_source/boxfusion"
REPLAY = ROOT / "reports/scannet_mechanism_protocol_controls_20261029/replay"
DUP = ROOT / "reports/scannet_cross_branch_duplicates_20261115"
SCORE = ROOT / "reports/scannet_ptr_vcr_scoring_dependency_20261115"
BASE_NATIVE = ROOT / "reports/scannet_online_prefix_terminal_readout_20261014/predictions/base_p100"
GT_NPZ = DUP / "evaluation_gt_and_alignment.npz"
SCAN_LIST = ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
CA1M_LIST = ROOT / (
    "development/reliability_v2/tools/boxfusion_tr3d_pipeline/evaluation/"
    "data_util/meta_data/ca1m_val_full107.txt"
)
BOXFUSION_OFFICIAL = ROOT / "development/official_github_exact_paired_efficiency_20261009/official_source_b2e0219"
BOXFUSION_SCAN_LIST = BOXFUSION_OFFICIAL / "evaluation/data_util/meta_data/scannetv2_val.txt"
BOXFUSION_README = BOXFUSION_OFFICIAL / "README.md"
EFFECTIVE_CONFIG = ROOT / "development/final_controls/experiments/mvsr_multiview_audit_20261012/config.yaml"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aabb_iou(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 8, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 8, 3)
    if not len(boxes) or not len(gt):
        return np.zeros((len(boxes), len(gt)), dtype=np.float64)
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = gt.min(1), gt.max(1)
    inter = np.maximum(0.0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)).prod(2)
    union = (hi - lo).prod(1)[:, None] + (gh - gl).prod(1) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def aligned_aabb(corners: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    """Match the ScanNet evaluator's axisAlignment -> camera-axis -> AABB path."""
    value = np.transpose(
        alignment[None, :3, :3] @ np.transpose(corners, (0, 2, 1)),
        (0, 2, 1),
    )
    value += alignment[None, :3, 3]
    value = value[..., [0, 2, 1]]
    value[..., 1] *= -1.0
    low, high = value.min(1), value.max(1)
    corners_out = np.empty((len(value), 8, 3), dtype=np.float64)
    for index, (lo, hi) in enumerate(zip(low, high)):
        corners_out[index] = np.asarray([
            [lo[0], hi[1], lo[2]], [hi[0], hi[1], lo[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]],
        ])
    return corners_out


def build_a() -> None:
    section = {
        "name": "HSR Geometric Consistency",
        "symbols": {
            "B_N": "native-record 3D corner box (8x3)",
            "B_P": "matched proposal 3D corner box (8x3)",
            "mu(B)": "AABB center, (min(B)+max(B))/2",
            "e(B)": "axis extents, max(B)-min(B)",
        },
        "constants": {"epsilon_geometry": 1e-6, "projection_near_z_m": 0.1, "min_clipped_area_px2": 4.0},
        "center_consistency": "exp(-||mu(B_N)-mu(B_P)||_2 / max(||e(B_N)||_2, 1e-6))",
        "scale_consistency": "exp(-mean_d |log((e_d(B_P)+1e-6)/(e_d(B_N)+1e-6))|)",
        "aabb_iou": "prod(max(0,min(u_N,u_P)-max(l_N,l_P))) / (V_N+V_P-intersection); 0 if union<=0",
        "geometry_combination": "g=(center_consistency+scale_consistency+AABB_IoU)/3; components and result clipped to [0,1]",
        "final_support_strength_frozen_no_visibility": "s=(IoU_2D * g * proposal_quality)^(1/3), clipped to [0,1]",
        "matching": "greedy exclusive 2D projected-box matching with IoU_2D >= 0.10",
        "history": "H_i,t=max(H_i,t-1,s_i,t) for the frozen support_mode=max",
        "native_score_update": "sigmoid(logit(clip(c_i,1e-4,1-1e-4)) + 2*max(0,H_i,t-0.5))",
        "geometry_reset": "reset accumulated evidence when the same native identity has AABB_IoU(previous,current) < 0.20",
        "source": str((SOURCE / "native_reliability_reranker.py").relative_to(ROOT)),
    }
    save_json(OUT / "A_HSR_geometric_consistency/formulas_and_parameters.json", section)
    latex = r"""% Exact formulas recovered from the frozen executed code.
\newcommand{\epsg}{10^{-6}}
\begin{align}
\mathbf e(B)&=\max(B)-\min(B),\\
\mu(B)&=\tfrac12\bigl(\min(B)+\max(B)\bigr),\\
g^{\mathrm{ctr}}_{i,j,t}&=\exp\!\left(-\frac{\lVert\mu(B^N_{i,t})-\mu(B^P_{j,t})\rVert_2}{\max(\lVert\mathbf e(B^N_{i,t})\rVert_2,\epsg)}\right),\\
g^{\mathrm{scl}}_{i,j,t}&=\exp\!\left(-\frac13\sum_{d=1}^{3}\left|\log\frac{e_d(B^P_{j,t})+\epsg}{e_d(B^N_{i,t})+\epsg}\right|\right),\\
g^{\mathrm{iou}}_{i,j,t}&=\operatorname{IoU}_{\mathrm{AABB}}(B^N_{i,t},B^P_{j,t}),\\
g_{i,j,t}&=\frac{g^{\mathrm{ctr}}_{i,j,t}+g^{\mathrm{scl}}_{i,j,t}+g^{\mathrm{iou}}_{i,j,t}}{3},\\
s_{i,t}&=\left(q_{j,t}\,\operatorname{IoU}_{2D}(i,j,t)\,g_{i,j,t}\right)^{1/3},\\
H_{i,t}&=\max(H_{i,t-1},s_{i,t}),\\
\widehat c^N_{i,t}&=\sigma\!\left(\operatorname{logit}(\operatorname{clip}(c^N_{i,t},10^{-4},1-10^{-4}))+2[H_{i,t}-0.5]_+\right).
\end{align}
"""
    (OUT / "A_HSR_geometric_consistency/formulas.tex").write_text(latex, encoding="utf-8")


def build_b() -> None:
    mapping = {
        "PTR": {
            "input": "maximum axis extent of the selected IoU medoid AABB in meters",
            "bands": [
                {"condition": "x < 0.30", "score": 0.05},
                {"condition": "0.30 <= x < 0.50", "score": 0.10},
                {"condition": "0.50 <= x < 0.70", "score": 0.25},
                {"condition": "0.70 <= x < 1.00", "score": 0.40},
                {"condition": "x >= 1.00", "score": 0.50},
            ],
            "medoid_tie": "larger sum of pairwise AABB IoUs; then higher observation score; then earlier ordinal; then smaller source_id",
            "branch_selection_and_self_nms_order": "descending raw_mean_score, then earlier confirmation_frame_id, then smaller track_id",
            "self_nms": "suppress when AABB IoU >= 0.50",
        },
        "VCR": {
            "combined_raw_value": "clip(0.8*reliability_lower_bound + 0.2*representative_anchor_score,0,1)",
            "mapping": "0.040001 + (0.049999-0.040001) * composite/maximum",
            "composite": "(IEEE754_float32_bits(combined_raw_value) << 20) | blake2s(scene_id\\0frame_id\\0anchor_id)[low 20 bits]",
            "maximum": "(0x3F800000 << 20) | (2^20-1)",
            "properties": "strictly below 0.05; preserves positive float32 raw order; deterministic 20-bit identity digest breaks exact raw ties",
            "representative_tie": "IoU medoid total; then higher anchor score; then earlier frame_id; then smaller anchor_id",
            "output_order": "higher reliability lower bound; more positive views; higher representative score; smaller track_id",
        },
        "global_exact_ties": {
            "assembly_order": ["Native", "PTR", "VCR"],
            "evaluation_sort": "stable descending final_ranking_score",
            "effect": "an exact cross-branch tie follows Native -> PTR -> VCR assembly order; VCR scores are designed to be unique and <0.05",
        },
        "sources": [
            str((SOURCE / "online_candidate_map.py").relative_to(ROOT)),
            str((SOURCE / "m1_anchor_online.py").relative_to(ROOT)),
            str((SOURCE / "m1_anchor_reliability_online.py").relative_to(ROOT)),
            str((SOURCE / "reliability_candidate_map.py").relative_to(ROOT)),
        ],
    }
    save_json(OUT / "B_score_mapping/score_mapping_and_ties.json", mapping)
    shutil.copy2(SCORE / "terminal_ap.csv", OUT / "B_score_mapping/fixed_box_score_ablation_all.csv")
    score_results = json.loads((SCORE / "results.json").read_text(encoding="utf-8"))
    save_json(OUT / "B_score_mapping/fixed_box_score_ablation_protocol_and_CI.json", {
        "protocol": score_results["protocol"],
        "checks": score_results["checks"],
        "bootstrap": score_results["bootstrap"],
        "interpretation_limit": score_results["interpretation_limit"],
    })
    full = []
    for arm in (
        "full_current", "full_ptr_raw", "full_vcr_raw", "full_both_raw",
        "full_ptr_temperature2", "full_vcr_temperature2", "full_both_temperature2",
    ):
        value = score_results["terminal"][arm]
        full.append({
            "variant": arm,
            "AP15": value["ap"]["0.15"],
            "AP25": value["ap"]["0.25"],
            "AP50": value["ap"]["0.5"],
            "predictions": value["total"],
        })
    write_csv(OUT / "B_score_mapping/fixed_box_score_ablation_full_system.csv",
              ["variant", "AP15", "AP25", "AP50", "predictions"], full)


def build_c() -> None:
    scan = [x.strip() for x in SCAN_LIST.read_text().splitlines() if x.strip()][:100]
    official_scan = [x.strip() for x in BOXFUSION_SCAN_LIST.read_text().splitlines() if x.strip()]
    ca1m = [x.strip() for x in CA1M_LIST.read_text().splitlines() if x.strip()]
    assert len(scan) == 100 and scan == official_scan and len(ca1m) == 107
    target = OUT / "C_dataset_evaluation"
    target.mkdir(parents=True, exist_ok=True)
    (target / "scannet100_scene_ids.txt").write_text("\n".join(scan) + "\n", encoding="utf-8")
    (target / "ca1m107_scene_ids.txt").write_text("\n".join(ca1m) + "\n", encoding="utf-8")
    prefix = json.loads((ROOT / "reports/scannet_online_prefix_terminal_readout_20261014/report/results.json").read_text())
    checkpoint_rows = []
    for fraction, value in prefix["replay"]["checkpoints"].items():
        checkpoint_rows.append({"fraction": fraction, **value})
    write_csv(target / "prefix_checkpoint_counts.csv",
              ["fraction", "keyframes", "native_records", "recar_records", "scenes"], checkpoint_rows)
    protocol = {
        "ScanNet100": {
            "selection": "the exact predefined 100-scene ScanNetV2 validation subset released by BoxFusion; the BoxFusion README states that these scenes were uniformly selected, and our list is byte-identical to the released BoxFusion scannetv2_val.txt",
            "sampling_seed": "not reported by the released BoxFusion repository; CaRR-3D performs no resampling and reuses the fixed released subset",
            "official_boxfusion_list_sha256": "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5",
            "comparison_to_full_scannet_val": "this is not the first 100 entries of the standard 312-scene ScanNetV2 validation list",
            "keyframe_rule": "every 25th input frame (gap=25)",
            "rgb_size": [640, 480],
            "depth_scale": 1000.0,
            "rgb_depth_preprocessing": "read RGB and 16-bit PNG depth, divide depth by 1000 to obtain meters, resize RGB to the depth resolution, then provide aligned 640x480 RGB-D tensors with the scene depth intrinsics",
            "invalid_depth": "the mapping unprojection valid mask is depth>0 and depth<10 m; invalid pixels do not become valid world points",
            "pose_preprocessing": "poses are loaded in numeric frame order as 4x4 camera-to-world matrices; an infinite pose reuses the most recent valid pose; the stream length is the minimum aligned RGB/depth/pose prefix",
            "coordinate_rule": "apply ScanNet axisAlignment in world coordinates, flip X-right/Y-forward/Z-up to X-right/Y-down/Z-forward, then replace each OBB by its enclosing AABB",
            "gt_boxes": "ScanNet validation GT parsed by the repository evaluation dataset and converted to nondegenerate AABBs",
            "native_terminal_size_filter": "at every terminal or prefix readout, retain a Native box only when all three AABB extents are >=0.30 m",
            "auxiliary_box_validation": "PTR and VCR reject nonfinite corner boxes and boxes with any nonpositive axis extent; auxiliary scores must lie in [0,1]",
            "empty_prediction_scenes": "allowed; the scene contributes all of its GT to the AP denominator and zero predictions/TP/FP",
            "matching": "stable descending score; best-IoU GT; at most one TP per GT",
            "iou_comparison": "strictly greater than the threshold (>)",
            "AP": "VOC-style precision-envelope integration, reported as percentage for IoU 0.15, 0.25, 0.50",
            "prefix_rounding": "checkpoint_index=min(T-1,ceil(T*fraction)-1), so the prefix contains ceil(T*fraction) keyframes",
            "prefix_causality": "only observations processed by the checkpoint are used; no history replay or future observations",
            "scene_count": 100,
            "gt_count": 1433,
        },
        "CA1M107": {
            "selection": "all 107 IDs in ca1m_val_full107.txt; no random sampling",
            "sampling_seed": "not applicable",
            "scene_count": 107,
        },
        "code_references": [
            "tools/audit_seedless_ablation_matrix.py:build_scannet_eval",
            "tools/true_fusion_audit_core.py:aabb_iou,class_agnostic_ap",
            "tools/utils.py:unproject",
            "development/official_github_exact_paired_efficiency_20261009/recar_source/boxfusion/capture_stream.py",
            "tools/replay_scannet_online_prefix.py",
            "tools/run_original_boxfusion_online_prefix.py",
            "development/official_github_exact_paired_efficiency_20261009/official_source_b2e0219/README.md:157",
            "development/official_github_exact_paired_efficiency_20261009/official_source_b2e0219/evaluation/data_util/meta_data/scannetv2_val.txt",
        ],
    }
    save_json(target / "evaluation_protocol.json", protocol)


def build_d() -> None:
    state = {
        "HSR": {
            "max_identity_states": 4096,
            "max_direction_slots_per_identity": 8,
            "minimum_direction_separation_deg": 30.0,
            "state_ttl_keyframes": 10,
            "creation": "created on the first reliability update for a native identity",
            "update": "matched support updates running max and direction slots; visible unmatched records add negative evidence because use_negative_evidence=true",
            "reset": "clear accumulated evidence if AABB_IoU(previous geometry,current geometry) < 0.20 for the same identity",
            "deletion": "an identity absent from the current Native Map is pruned when ordinal-last_seen > 10",
            "capacity": "when creating a state at capacity, evict the oldest last_seen_ordinal, tie by repr(identity); a current Native Map with >4096 identities raises an error",
            "hidden_native_records": "records retained in the current Native Map remain protected from TTL pruning even if not matched in the current view; nonprojectable records receive no current evidence update",
        },
        "PTR": {
            "max_active_tracks": 1024,
            "track_ttl_keyframes": 10,
            "max_observations_per_track": 12,
            "max_observations_per_frame": 192,
            "confirmed_candidate_capacity": 1024,
            "current_output_budget": 12,
            "confirmation": "at least 3 retained observations from distinct keyframes because each track stores at most one observation per frame; use_children=false",
            "observation_eviction": "replace same-frame entry only if higher score/tie order wins; if >12 frames, remove oldest ordinal/frame",
            "track_expiry": "drop active track when ordinal-last_ordinal >10; already confirmed birth stays in the bounded birth dictionary",
            "capacity_policy": "when 1024 active tracks are present, a new unmatched observation is dropped; when confirmed dictionary is full, the new candidate and its active track are dropped",
            "native_filter": "exclude current output when AABB IoU with a Native record is >=0.25; this is reversible if the native row disappears",
            "within_branch_filter": "greedy self-NMS at AABB IoU >=0.50 in raw_mean_score order",
        },
        "VCR": {
            "voxel_hash_m": 0.3,
            "min_views": 3,
            "min_reliability_lower_bound": 0.35,
            "max_active_tracks": 4096,
            "max_observations_per_track": 12,
            "max_anchors_per_frame": 300,
            "track_ttl_keyframes": 12,
            "max_direction_slots": 8,
            "minimum_direction_separation_deg": 20.0,
            "current_output_budget": 640,
            "capacity_eviction": "evict unconfirmed before confirmed, then lower reliability lower bound, older last ordinal, and larger track_id",
            "observation_eviction": "retain a bounded temporal sample while preserving the strongest observation; then recompute the IoU medoid",
            "native_filter": "exclude output when representative AABB IoU with Native is >=0.25",
            "PTR_filter": "exclude output when representative AABB IoU with PTR recovery is >=0.25",
            "within_branch_filter": "greedy VCR suppression at AABB IoU >=0.25",
            "archive": "no separate unbounded frame-ID set or permanent recovery archive; confirmed outputs are views of current active tracks and disappear when tracks expire",
        },
        "asynchronous_runtime": {
            "enabled": True,
            "queue_capacity": 2,
            "keyframe_deadline_ms": 833.333333,
            "ordering": "strictly increasing frame_id; worker commits queue order after the matching native keyframe state becomes ready",
            "close": "reject unfinished begin/finish submissions; enqueue sentinel after submitted tasks and join worker, thereby draining only already submitted tasks",
            "future_observations": "none",
            "terminal_readout": "requires a drained/closed runtime and performs no new proposal inference",
        },
        "source_config": str(EFFECTIVE_CONFIG.relative_to(ROOT)),
    }
    save_json(OUT / "D_state_capacity/state_capacity_and_management.json", state)
    (OUT / "D_state_capacity").mkdir(parents=True, exist_ok=True)
    shutil.copy2(EFFECTIVE_CONFIG, OUT / "D_state_capacity/frozen_effective_config.yaml")


def load_scene_payload(scene: str):
    with (REPLAY / (scene + ".pkl")).open("rb") as handle:
        payload = pickle.load(handle)["prefixes"]["1.0"]
    current = payload["full_current"]
    raw = payload["full_raw"]
    assert tuple(current["sources"]) == tuple(raw["sources"])
    assert len(current["rows"]) == len(raw["rows"])
    return current, raw


def export_predictions_and_gt(scenes):
    target = OUT / "E_cross_branch_duplication"
    archive = np.load(GT_NPZ, allow_pickle=False)
    pred_path = target / "predictions_scannet100.csv.gz"
    gt_path = target / "ground_truth_scannet100.csv.gz"
    all_for_conflict = {}
    with gzip.open(str(pred_path), "wt", encoding="utf-8-sig", newline="") as handle:
        fields = ["scene_id", "branch", "record_id", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "raw_score", "final_ranking_score"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            current, raw = load_scene_payload(scene)
            with (BASE_NATIVE / (scene + "_boxes.pkl")).open("rb") as base_handle:
                native_original = pickle.load(base_handle)[0]
            sources = np.asarray(current["sources"])
            native_rows = np.flatnonzero(sources == "native")
            assert len(native_rows) == len(native_original)
            assert all(np.array_equal(current["rows"][row][1], native_original[i][1]) for i, row in enumerate(native_rows))
            raw_scores = [float(row[2]) for row in raw["rows"]]
            for i, row in enumerate(native_rows):
                raw_scores[int(row)] = float(native_original[i][2])
            corners_world = np.stack([row[1] for row in current["rows"]]) if current["rows"] else np.empty((0, 8, 3))
            boxes = aligned_aabb(corners_world, archive[scene + "_alignment"])
            branch_counts = {"native": 0, "proposal": 0, "anchor": 0}
            labels = []
            for index, (source, box, raw_score, current_row) in enumerate(zip(sources, boxes, raw_scores, current["rows"])):
                branch = {"native": "Native", "proposal": "PTR", "anchor": "VCR"}[str(source)]
                local = branch_counts[str(source)]
                branch_counts[str(source)] += 1
                lo, hi = box.min(0), box.max(0)
                writer.writerow({
                    "scene_id": scene, "branch": branch,
                    "record_id": "%s:%s:%06d" % (scene, branch, local),
                    "xmin": lo[0], "ymin": lo[1], "zmin": lo[2],
                    "xmax": hi[0], "ymax": hi[1], "zmax": hi[2],
                    "raw_score": raw_score, "final_ranking_score": float(current_row[2]),
                })
                labels.append(branch)
            all_for_conflict[scene] = (boxes, np.asarray(labels, dtype=object), archive[scene + "_gt"])
    with gzip.open(str(gt_path), "wt", encoding="utf-8-sig", newline="") as handle:
        fields = ["scene_id", "gt_id", "xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            for index, box in enumerate(archive[scene + "_gt"]):
                lo, hi = box.min(0), box.max(0)
                writer.writerow({"scene_id": scene, "gt_id": "%s:GT:%04d" % (scene, index),
                                 "xmin": lo[0], "ymin": lo[1], "zmin": lo[2],
                                 "xmax": hi[0], "ymax": hi[1], "zmax": hi[2]})
    return all_for_conflict


def recall_bootstrap(per_scene, scenes, gt_counts):
    rng = np.random.RandomState(20261115)
    policies = ["cross_only_0.25", "cross_only_0.5", "global_nms_0.25", "global_nms_0.5", "native_priority_0.25", "native_priority_0.5"]
    rows = []
    index = {scene: i for i, scene in enumerate(scenes)}
    gt = np.asarray([gt_counts[s] for s in scenes], dtype=np.float64)
    for policy in policies:
        for threshold in (0.15, 0.25, 0.5):
            base = np.asarray([per_scene[("concat", s, threshold)] for s in scenes], dtype=np.float64)
            test = np.asarray([per_scene[(policy, s, threshold)] for s in scenes], dtype=np.float64)
            point = 100.0 * (test.sum() - base.sum()) / gt.sum()
            samples = np.empty(2000, dtype=np.float64)
            for b in range(2000):
                chosen = rng.randint(0, len(scenes), len(scenes))
                samples[b] = 100.0 * (test[chosen].sum() - base[chosen].sum()) / gt[chosen].sum()
            rows.append({
                "policy": policy, "iou": threshold,
                "delta_recall_percentage_points": point,
                "ci95_low": float(np.percentile(samples, 2.5)),
                "ci95_high": float(np.percentile(samples, 97.5)),
                "fraction_positive": float(np.mean(samples > 0.0)),
                "bootstrap_replicates": 2000, "seed": 20261115,
            })
    return rows


def build_e() -> None:
    target = OUT / "E_cross_branch_duplication"
    target.mkdir(parents=True, exist_ok=True)
    scenes = [x.strip() for x in SCAN_LIST.read_text().splitlines() if x.strip()][:100]
    archive = np.load(GT_NPZ, allow_pickle=False)
    gt_counts = {s: len(archive[s + "_gt"]) for s in scenes}
    all_for_conflict = export_predictions_and_gt(scenes)

    metrics = read_csv(DUP / "metrics.csv")
    full = [row for row in metrics if row["method"] == "full"]
    write_csv(target / "nms_AP_recall_counts.csv", list(full[0].keys()), full)

    summary = []
    for row in full:
        if float(row["iou"]) != 0.25:
            continue
        summary.append({
            "policy": row["policy"], "retained_total": row["predictions"],
            "retained_Native": row["native"], "retained_PTR": row["ptr"], "retained_VCR": row["vcr"],
            "removed_total": row["removed"],
            "removed_Native": 1787 - int(row["native"]),
            "removed_PTR": 1101 - int(row["ptr"]),
            "removed_VCR": 23172 - int(row["vcr"]),
            "remaining_cross_duplicate_FP_at_IoU25": row["cross_duplicate_fp"],
            "remaining_within_duplicate_FP_at_IoU25": row["within_duplicate_fp"],
        })
    write_csv(target / "nms_removed_by_branch.csv", list(summary[0].keys()), summary)
    shutil.copy2(DUP / "paired_bootstrap.csv", target / "nms_AP_paired_bootstrap_CI.csv")

    per_scene_rows = read_csv(DUP / "per_scene_metrics.csv")
    per_scene = {}
    for row in per_scene_rows:
        if row["method"] == "full":
            per_scene[(row["policy"], row["scene"], float(row["iou"]))] = int(row["tp"])
    recall_ci = recall_bootstrap(per_scene, scenes, gt_counts)
    write_csv(target / "nms_recall_paired_bootstrap_CI.csv", list(recall_ci[0].keys()), recall_ci)

    duplicate_rows = read_csv(DUP / "full_duplicate_records.csv")
    pair_counts = {}
    totals = {float(row["iou"]): int(row["fp"]) for row in full if row["policy"] == "concat"}
    for row in duplicate_rows:
        if row["duplicate_type"] != "cross":
            continue
        threshold = float(row["evaluation_iou"])
        pair = "--".join(sorted([row["branch"], row["first_tp_branch"]]))
        pair_counts[(threshold, pair)] = pair_counts.get((threshold, pair), 0) + 1
    pair_out = []
    for (threshold, pair), count in sorted(pair_counts.items()):
        pair_out.append({"iou": threshold, "branch_pair": pair, "duplicate_FP": count,
                         "percent_of_all_predictions": 100.0 * count / 26060.0,
                         "percent_of_all_FP": 100.0 * count / totals[threshold]})
    write_csv(target / "branch_pair_duplicate_FP.csv", list(pair_out[0].keys()), pair_out)

    coverage_rows = []
    triple_rows = []
    for threshold in (0.15, 0.25, 0.5):
        combination_counts = {}
        triple_gt = triple_predictions = triple_duplicates = 0
        branch_predictions = {"Native": 0, "PTR": 0, "VCR": 0}
        for scene in scenes:
            boxes, labels, gt = all_for_conflict[scene]
            hits = aabb_iou(boxes, gt) > threshold
            for gt_index in range(len(gt)):
                branch_set = tuple(sorted(set(labels[hits[:, gt_index]].tolist())))
                key = "+".join(branch_set) if branch_set else "none"
                combination_counts[key] = combination_counts.get(key, 0) + 1
                if set(branch_set) == {"Native", "PTR", "VCR"}:
                    triple_gt += 1
                    count = int(hits[:, gt_index].sum())
                    triple_predictions += count
                    triple_duplicates += max(0, count - 1)
                    for branch in branch_predictions:
                        branch_predictions[branch] += int(hits[labels == branch, gt_index].sum())
        for combination, count in sorted(combination_counts.items()):
            coverage_rows.append({"iou": threshold, "covering_branch_set": combination, "GT_count": count,
                                  "percent_of_all_GT": 100.0 * count / 1433.0})
        triple_rows.append({"iou": threshold, "GT_covered_by_all_three_branches": triple_gt,
                            "predictions_covering_those_GT": triple_predictions,
                            "duplicate_predictions_on_those_GT": triple_duplicates,
                            "Native_predictions": branch_predictions["Native"],
                            "PTR_predictions": branch_predictions["PTR"],
                            "VCR_predictions": branch_predictions["VCR"]})
    write_csv(target / "GT_branch_coverage_sets.csv", list(coverage_rows[0].keys()), coverage_rows)
    write_csv(target / "three_branch_common_conflicts.csv", list(triple_rows[0].keys()), triple_rows)

    for name in ("geometric_overlap_per_scene.csv", "removed_prediction_accounting.csv", "oracle_duplicate_only_AP.csv"):
        shutil.copy2(DUP / name, target / name)
    save_json(target / "duplication_protocol.json", json.loads((DUP / "protocol.json").read_text()))


def readme() -> None:
    text = """# CaRR-3D supplementary data package (A--E)

This package was generated from the frozen executed code and completed ScanNet-100 reports. No detector inference was rerun.

## Contents

- `A_HSR_geometric_consistency`: exact center/scale/AABB-IoU definitions, support combination, reset rule, and LaTeX.
- `B_score_mapping`: exact PTR and VCR score maps, tie rules, fixed-box AP ablations, and paired confidence intervals.
- `C_dataset_evaluation`: exact ScanNet-100 and CA-1M-107 scene IDs, evaluation rules, and prefix checkpoint counts.
- `D_state_capacity`: HSR/PTR/VCR capacities, TTLs, eviction rules, asynchronous drain behavior, and the frozen YAML config.
- `E_cross_branch_duplication`: NMS AP/recall/count tables, deletion counts per branch, AP and recall paired bootstrap CIs, pairwise and three-way conflict statistics, plus per-prediction and per-GT exports.

## Important interpretation notes

1. `record_id` in the prediction CSV is a deterministic export ID because the frozen replay cache does not retain every runtime-internal ID.
2. Prediction and GT coordinates in the CSV are the aligned evaluator AABBs, so the files can be used directly for ScanNet AP and overlap recomputation.
3. Native `raw_score` is recovered from the geometry-identical `base_p100` output before HSR. PTR/VCR raw scores come from the frozen `full_raw` replay. `final_ranking_score` is the exact current output score.
4. The fixed-box score ablation uses the legacy evaluator values archived by the completed experiment. Rounded to two decimals, its full-current result is 47.21/42.72/22.15; the stable evaluator used by the cross-branch report gives 47.22/42.74/22.16. Do not mix unrounded values across these two evaluator tables.
5. `three_branch_common_conflicts.csv` counts GT objects covered by all three branches at a threshold. Pairwise duplicate-FP attribution remains score-order dependent and counts each duplicate prediction once.
6. The Recall bootstrap table was newly computed from the archived per-scene TP counts with 2,000 scene-level paired percentile replicates and seed 20261115; no model output was changed.
7. ScanNet-100 is the exact fixed subset released by BoxFusion. The BoxFusion README describes it as a uniform selection; CaRR-3D does not draw a new random sample.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def manifest() -> None:
    source_files = [
        SOURCE / "native_reliability_reranker.py", SOURCE / "causal_reliability.py",
        SOURCE / "online_candidate_map.py", SOURCE / "m1_anchor_online.py",
        SOURCE / "m1_anchor_reliability_online.py", SOURCE / "reliability_candidate_map.py",
        ROOT / "tools/true_fusion_audit_core.py", ROOT / "tools/analyze_cross_branch_duplicates_20261115.py",
        SCORE / "results.json", DUP / "results.json", GT_NPZ, SCAN_LIST, CA1M_LIST, EFFECTIVE_CONFIG,
        BOXFUSION_README, BOXFUSION_SCAN_LIST,
    ]
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
    outputs = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.json":
            outputs[str(path.relative_to(OUT))] = sha256(path)
    save_json(OUT / "manifest_sha256.json", {"source_sha256": sources, "output_sha256": outputs})


def pack() -> None:
    if OUT.exists():
        raise RuntimeError("Refusing to overwrite existing output: %s" % OUT)
    if ZIP.exists():
        raise RuntimeError("Refusing to overwrite existing archive: %s" % ZIP)
    build_a()
    build_b()
    build_c()
    build_d()
    build_e()
    readme()
    manifest()
    with zipfile.ZipFile(str(ZIP), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(str(path), str(Path(OUT.name) / path.relative_to(OUT)))
    print(OUT)
    print(ZIP)


if __name__ == "__main__":
    pack()
