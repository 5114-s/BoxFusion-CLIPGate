#!/usr/bin/env python3
"""Native-preserving real 3D comparisons, causal evidence and HCSR boundaries.

All geometry, scores, rankings and RGB are archived measurements. This is a
visualization-only workflow: no model inference, GPU work or prediction edits.
"""
from __future__ import annotations

import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qualitative import render_boxfusion_ctrl_recar_3d as old

INPUT = ROOT/"figure_assets/qualitative_3d_three_scenes_individual_20261101"
HCSR = ROOT/"figure_assets/hcsr_qualitative_pair_20261031"
CASES = ROOT/"development/final_pipeline/reports/qualitative_recar3d_final/selected_cases.json"
OUT = ROOT/"figure_assets/qualitative_optimized_ptr_var_hcsr_20261102"
COLORS = {**old.COLORS, "hcsr": "#8052B8", "failure": "#D73335"}
INK, MUTED = "#172A40", "#566477"
WIDTH = 183/25.4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig, stem, dpi=400):
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = OUT/f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        paths.append(path.name)
    fig.savefig(OUT/f"{stem}_preview.png", dpi=170, facecolor="white")
    plt.close(fig)
    return paths


def read_rows(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)[0]


def sorted_corners(box):
    return np.asarray(sorted(np.asarray(box).tolist()))


def draw_box(ax, box, renderer, args, color, width=.9, dashed=False, alpha=.95):
    uv = renderer.project(box, *args)
    assert np.all(uv[:, 2] > 0)
    edges = renderer.box_edges(box)
    assert edges.shape == (12, 2)
    ax.add_collection(LineCollection(uv[:, :2][edges], colors=color,
        linewidths=width, linestyles="--" if dashed else "-", alpha=alpha, zorder=5))


def pointer(ax, box, renderer, args, label, color, location=(.70, .9), size=6.5):
    target = renderer.project(np.asarray(box).mean(0)[None], *args)[0, :2]
    ax.annotate(label, xy=target, xytext=location, textcoords="axes fraction",
        fontsize=size, color=color, ha="center", va="center", zorder=10,
        bbox={"boxstyle": "round,pad=.17", "facecolor": "white", "edgecolor": "none", "alpha": .94},
        arrowprops={"arrowstyle": "-|>", "lw": .75, "color": color, "shrinkA": 2, "shrinkB": 2})


def base_axis(ax, payload):
    ax.imshow(payload["background"])
    bounds = payload["camera"]["display_limits"]
    ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:])
    ax.set_aspect("equal"); ax.axis("off")


def load_payloads(renderer, manifest, input_hashes):
    payloads, audits = [], []
    for entry in manifest["scenes"]:
        scene = entry["scene"]
        path = INPUT/f"{scene}_measured_boxes.npz"
        input_hashes[str(path)] = digest(path)
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        background_path = INPUT/f"{scene}_mesh.png"
        input_hashes[str(background_path)] = digest(background_path)
        for method, source in entry["methods"].items():
            assert digest(source["path"]) == source["sha256"]
            input_hashes[source["path"]] = source["sha256"]
        assert digest(entry["gt_path"]) == entry["gt_sha256"]
        assert np.array_equal(arrays["ctrl_boxes"], arrays["recar_boxes"][:len(arrays["ctrl_boxes"])])
        gt = arrays["gt_boxes"]
        targets = entry["selected_gt_indices"]
        native_count = len(arrays["ctrl_boxes"])
        native_overlap = old.iou_matrix(arrays["ctrl_boxes"], gt)
        full_overlap = old.iou_matrix(arrays["recar_boxes"], gt)
        chosen = arrays["ctrl_selected_indices"].tolist()
        recovered, unresolved = [], []
        for target in targets:
            if native_overlap[:, target].max() > .15:
                continue
            if full_overlap[native_count:, target].max(initial=0.) > .15:
                index = native_count+int(full_overlap[native_count:, target].argmax())
                chosen.append(index)
                recovered.append({"gt_index": int(target), "prediction_index": index,
                    "branch": "PTR" if arrays["recar_scores"][index] >= .05 else "VAR",
                    "native_best_iou": float(native_overlap[:, target].max()),
                    "recovered_iou": float(full_overlap[index, target])})
            else:
                unresolved.append({"gt_index": int(target),
                    "full_best_iou": float(full_overlap[:, target].max())})
        predictions = {}
        for method, key in (("BoxFusion", "boxfusion"), ("BoxFusion-Ctrl", "ctrl"), ("ReCaR-3D", "recar")):
            boxes, scores = arrays[key+"_boxes"], arrays[key+"_scores"]
            indices = chosen if key == "recar" else arrays[key+"_selected_indices"].tolist()
            branches = ["native" if i < native_count else "calr" if s < .05 else "plr"
                        for i, s in enumerate(scores)] if key == "recar" else ["native"]*len(boxes)
            predictions[method] = {"boxes": boxes, "scores": scores, "selected": indices, "branches": branches}
        camera = entry["camera"]
        args = (np.asarray(camera["position"]), np.asarray(camera["look_at"]),
                camera["focal_pixels"], np.asarray(camera["principal_point_pixels"]))
        before, after = arrays["ctrl_scores"], arrays["recar_scores"][:native_count]
        ranks = []
        for scores in (before, after):
            rank = np.empty(native_count, dtype=int)
            rank[np.argsort(-scores, kind="stable")] = np.arange(1, native_count+1)
            ranks.append(rank)
        hcsr = None
        if scene == "scene0432_01":
            index = 4
            hcsr = {"native_index": index, "before_score": float(before[index]),
                "after_score": float(after[index]), "before_rank": int(ranks[0][index]),
                "after_rank": int(ranks[1][index]), "best_gt_iou": float(native_overlap[index].max())}
            assert hcsr["before_rank"] == 4 and hcsr["after_rank"] == 3
            assert index in chosen
        payload = {"spec": {"scene": scene, "gt_indices": targets}, "gt": gt,
            "predictions": predictions, "background": np.array(Image.open(background_path)),
            "camera": camera, "projection_args": args, "recoveries": recovered,
            "unresolved": unresolved, "hcsr": hcsr}
        payloads.append(payload)
        audits.append({"scene": scene, "selected_gt_indices": targets,
            "method_selected_indices": {k: v["selected"] for k, v in predictions.items()},
            "same_selected_native_boxes_in_ctrl_and_full": True,
            "recoveries": recovered, "unresolved": unresolved, "hcsr": hcsr,
            "camera": camera})
    return payloads, audits


def render_comparison(payloads, renderer):
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH, 6.85))
    fig.subplots_adjust(left=.025, right=.99, bottom=.12, top=.925, wspace=.035, hspace=.25)
    for col, name in enumerate(("GT (selected)", *old.METHODS[1:])):
        position = axes[0, col].get_position()
        fig.text(position.x0+position.width/2, .970, name, ha="center", fontsize=8.1,
                 fontweight="bold", color=INK)
    for row, p in enumerate(payloads):
        for col, name in enumerate(old.METHODS):
            ax = axes[row, col]; base_axis(ax, p)
            if col == 0:
                for i in p["spec"]["gt_indices"]:
                    draw_box(ax, p["gt"][i], renderer, p["projection_args"], COLORS["gt"], .9, True)
                count = f"{len(p['spec']['gt_indices'])} selected GT"
            else:
                pred = p["predictions"][name]
                for i in pred["selected"]:
                    branch = pred["branches"][i]
                    draw_box(ax, pred["boxes"][i], renderer, p["projection_args"], COLORS[branch],
                             .8 if branch == "native" else 1.35)
                count = f"{len(pred['selected'])} shown / {len(pred['boxes'])} total"
            ax.text(.5, -.035, count, transform=ax.transAxes, ha="center", va="top",
                    color=MUTED, fontsize=5.8)
            if row < 2:
                item = p["recoveries"][0]
                target = p["gt"][item["gt_index"]]
                if col == 0:
                    pointer(ax, target, renderer, p["projection_args"], "Target", COLORS["gt"], (.77, .86))
                elif col < 3:
                    pointer(ax, target, renderer, p["projection_args"], "No match", MUTED, (.77, .86))
                else:
                    branch = "plr" if item["branch"] == "PTR" else "calr"
                    pointer(ax, pred["boxes"][item["prediction_index"]], renderer, p["projection_args"],
                        f"{item['branch']}\nIoU {item['recovered_iou']:.2f}", COLORS[branch], (.77, .86))
            elif col in (0, 3):
                missed = p["unresolved"][0]
                pointer(ax, p["gt"][missed["gt_index"]], renderer, p["projection_args"],
                        "Unrecovered GT" if col else "Missed target", COLORS["failure"], (.75, .92), 6.2)
                if col == 3:
                    item = p["hcsr"]
                    box = p["predictions"]["ReCaR-3D"]["boxes"][item["native_index"]]
                    pointer(ax, box, renderer, p["projection_args"], "HCSR\nrank 4 → 3",
                            COLORS["hcsr"], (.24, .24), 6.2)
        position = axes[row, 0].get_position()
        suffix = ("PTR recovery", "VAR recovery", "Reranking and an unresolved target")[row]
        fig.text(.025, position.y1+.012,
            f"({chr(97+row)}) {p['spec']['scene']} · {suffix}", fontsize=6.9,
            fontweight="bold", color=INK)
    handles = [Line2D([], [], color=COLORS[key], lw=1.6,
        ls="--" if key == "gt" else "-", label=label) for key, label in (
        ("gt", "GT"), ("native", "Native box"), ("plr", "PTR recovery"),
        ("calr", "VAR recovery"), ("hcsr", "HCSR callout"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .057),
        ncol=5, frameon=False, fontsize=6.5, handlelength=1.7, columnspacing=1.1)
    fig.text(.5, .037, "GT-assisted selected targets; not all predictions. Red callouts mark an unrecovered GT.",
             ha="center", color=MUTED, fontsize=6.1)
    fig.text(.5, .019, "Same native geometry in Ctrl and ReCaR-3D. Scene mesh is used only for visualization.",
             ha="center", color=MUTED, fontsize=6.1)
    return export(fig, "qualitative_3d_comparison_optimized")


def evidence_module():
    path = ROOT/"development/final_pipeline/render_final_qualitative.py"
    spec = importlib.util.spec_from_file_location("qualitative_source", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify_temporal_memberships(evidences, input_hashes):
    """Recover actual track/voxel observations, not nearby high-IoU examples."""
    from tools import run_scannet_fixed_capacity_prefix_memory as cache
    from boxfusion.online_candidate_map import OnlineProposalRecovery
    from boxfusion.m1_anchor_online import OnlineAnchorRecovery
    config_path = ROOT/"development/final_controls/experiments/mvsr_multiview_audit_20261012/config.yaml"
    import yaml
    config = yaml.safe_load(config_path.read_text())["online_candidate_map"]["state"]["m1p"]
    input_hashes[str(config_path)] = digest(config_path)
    audits = {}
    for key in ("plr_recovery", "calr_recovery"):
        evidence = evidences[key]
        diagnostic_path = Path(evidence["diagnostics_path"])
        input_hashes[str(diagnostic_path)] = digest(diagnostic_path)
        diagnostics = json.loads(diagnostic_path.read_text())
        cache_path = Path(diagnostics["provider"]["evidence_cache"])
        assert digest(cache_path) == evidence["evidence_cache_sha256"]
        input_hashes[str(cache_path)] = digest(cache_path)
        frames = cache.load_evidence(cache_path)
        shown = {r["frame_id"]: r for r in evidence["frames"]}
        trace = evidence["terminal_trace"]
        state = OnlineProposalRecovery(**config) if key == "plr_recovery" else OnlineAnchorRecovery(evidence["scene"])
        selection_audit = []
        for ordinal, (frame_id, frame) in enumerate(frames):
            if key == "plr_recovery":
                state.update(ordinal, frame_id, proposal_ids=frame.proposal_ids,
                    proposal_corners=frame.proposal_corners, proposal_scores=frame.proposal_scores,
                    native_corners=np.empty((0, 8, 3)))
            else:
                state.update(ordinal, frame_id, frame.anchor_ids, frame.anchor_corners, frame.anchor_scores)
            if frame_id not in shown:
                continue
            old_row = copy.deepcopy(shown[frame_id])
            if key == "plr_recovery":
                observed = state.tracks[trace["track_id"]].observations[frame_id]
                indices = np.flatnonzero(frame.proposal_ids == observed.source_id)
                assert len(indices) == 1
                index = int(indices[0]); box = frame.proposal_corners[index]
                score = float(frame.proposal_scores[index]); source_id = int(frame.proposal_ids[index])
                assert np.array_equal(box, observed.corners)
                shown[frame_id]["candidate_xyxy"] = frame.proposal_boxes_2d[index].tolist()
                support_count = len(state.tracks[trace["track_id"]].observations)
            else:
                voxel = tuple(trace["voxel_key"])
                membership = np.floor(frame.anchor_corners.mean(1)/.3).astype(int)
                indices = np.flatnonzero((membership == np.asarray(voxel)).all(1))
                assert len(indices) > 0
                index = min(indices, key=lambda i: (-float(frame.anchor_scores[i]), int(frame.anchor_ids[i])))
                box = frame.anchor_corners[index]
                score = float(frame.anchor_scores[index]); source_id = int(frame.anchor_ids[index])
                shown[frame_id].pop("candidate_xyxy", None)
                group = state.births.get(voxel, state.active.get(voxel))
                support_count = len(group.frames)
                assert frame_id in group.frames
            shown[frame_id]["candidate_box"] = box.tolist()
            shown[frame_id]["candidate_score"] = score
            shown[frame_id]["source_id"] = source_id
            selection_audit.append({"frame": frame_id, "source_id": source_id,
                "support_count_after_current_keyframe": support_count,
                "original_candidate_replaced": not np.array_equal(np.asarray(old_row["candidate_box"]), box),
                "candidate_world_voxel": np.floor(box.mean(0)/.3).astype(int).tolist(),
                "raw_score": score})
        assert [r["support_count_after_current_keyframe"] for r in selection_audit] == [1, 2, 3]
        if key == "plr_recovery":
            terminal = state.births[trace["track_id"]]
            box = terminal.corners
            assert terminal.confirmation_frame_id == trace["confirmation_frame_id"] == 550
        else:
            terminal = state.births[tuple(trace["voxel_key"])]; box = terminal.box
            assert terminal.frame_id == trace["exemplar_frame_id"]
        assert np.allclose(sorted_corners(box), sorted_corners(trace["box"]), atol=1e-7)
        audits[key] = {"observations": selection_audit,
            "terminal_representative_matches_original_trace": True,
            "state_replay_uses_only_current_and_past_candidates": True,
            "native_coverage_omitted_only_for_history_membership_audit": True}
    return audits


def rgb_crop(ax, case, frame, source, *, native=False, failure=False):
    rgb = Image.open(frame["rgb_path"]).convert("RGB").resize((640, 480), Image.Resampling.LANCZOS)
    candidate = frame.get("candidate_xyxy")
    if candidate is None:
        candidate = source.project_box(np.asarray(frame["candidate_box"]), old.RAW, case["scene"], frame["frame_id"])
    predicted = source.project_box(np.asarray(case["box"]), old.RAW, case["scene"], frame["frame_id"]) if native else None
    focus = candidate if candidate is not None else predicted
    rgb, coordinates = source.crop_evidence_frame(rgb, focus, {"candidate": candidate, "native": predicted}, scale=1.65)
    ax.imshow(rgb)
    color = COLORS["failure"] if failure else COLORS["native"] if native else COLORS[case["source"]]
    source.draw_rect(ax, coordinates["candidate"], COLORS["hcsr"] if native else color,
                     dashed=native, width=1.1, outlined=not native)
    if native:
        source.draw_rect(ax, coordinates["native"], color, width=1.5)
    ax.set_xlim(0, 640); ax.set_ylim(480, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D8DEE7"); spine.set_linewidth(.5)
    ax.set_title(f"t = {frame['frame_id']}", loc="left", fontsize=7.0, pad=3)


def terminal_zoom(ax, payload, item, renderer):
    ax.imshow(payload["background"])
    box = payload["predictions"]["ReCaR-3D"]["boxes"][item["prediction_index"]]
    target = payload["gt"][item["gt_index"]]
    draw_box(ax, target, renderer, payload["projection_args"], COLORS["gt"], 1.0, True)
    draw_box(ax, box, renderer, payload["projection_args"],
             COLORS["plr"] if item["branch"] == "PTR" else COLORS["calr"], 1.35)
    uv = renderer.project(np.concatenate((box, target)), *payload["projection_args"])[:, :2]
    center = (uv.min(0)+uv.max(0))/2
    span = max(float(np.ptp(uv, axis=0).max())*1.9, 420)
    ax.set_xlim(center[0]-span/2, center[0]+span/2)
    ax.set_ylim(center[1]+span*.40, center[1]-span*.40)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Terminal representative", loc="left", fontsize=6.6, pad=3)


def render_temporal(payloads, cases, evidences, renderer, source):
    fig = plt.figure(figsize=(WIDTH, 3.65))
    for row, key in enumerate(("plr_recovery", "calr_recovery")):
        y = .565 if row == 0 else .12
        evidence, case, payload = evidences[key], cases[key], payloads[row]
        item = payload["recoveries"][0]
        trace = evidence["terminal_trace"]
        support_frames = trace["evidence_frame_ids"] if row == 0 else trace["support_frame_ids"]
        assert [r["frame_id"] for r in evidence["frames"]] == sorted(support_frames)[:3]
        measured = np.asarray(trace["box"]) @ old.alignment(case["scene"])[:3, :3].T + old.alignment(case["scene"])[:3, 3]
        final_box = payload["predictions"]["ReCaR-3D"]["boxes"][item["prediction_index"]]
        assert old.iou_matrix(measured[None], final_box[None])[0, 0] > .99999
        if row == 0:
            assert trace["confirmation_frame_id"] == 550
        label = "PTR" if row == 0 else "VAR"
        color = COLORS["plr"] if row == 0 else COLORS["calr"]
        fig.text(.025, y+.345, f"({chr(97+row)}) {label} · {case['scene']}", fontsize=8.2,
                 color=INK, fontweight="bold")
        for col, frame in enumerate(evidence["frames"]):
            left = .025+col*.242
            ax = fig.add_axes([left, y+.04, .218, .26])
            rgb_crop(ax, case, frame, source)
            status = ("1 supporting keyframe", "2 supporting keyframes", "Confirmed: 3 keyframes")[col]
            if row == 1 and col == 2:
                status = "3-frame condition reached"
            fig.text(left+.109, y+.008, status, ha="center", fontsize=6.3,
                     color=color if col == 2 else MUTED)
        ax = fig.add_axes([.752, y+.04, .218, .26])
        terminal_zoom(ax, payload, item, renderer)
        fig.text(.861, y+.008, f"3D AABB IoU = {item['recovered_iou']:.2f}",
                 ha="center", fontsize=6.3, color=color)
        for x in (.247, .489):
            fig.text(x, y+.17, "→", fontsize=10, color=MUTED)
        fig.text(.725, y+.17, "⋯→", fontsize=9, color=MUTED)
    fig.text(.5, .065, "Left to right: sampled keyframes in chronological order; dashed gap includes later observations.",
             ha="center", fontsize=6.3, color=MUTED)
    fig.text(.5, .025, "The rightmost box is the terminal representative, not the geometry available at the third shown keyframe.",
             ha="center", fontsize=6.1, color=MUTED)
    return export(fig, "temporal_candidate_evidence")


def perspective_geometry(ax, scene, box, gt, renderer, input_hashes, *, failure=False):
    path = ROOT/"evaluation/data_util/scannet_train_detection_data"/f"{scene}_vert.npy"
    input_hashes[str(path)] = digest(path)
    values = np.load(path)
    center = box.mean(0)
    radius = max(float(np.ptp(box, axis=0).max())*1.8, 1.25)
    mask = (np.abs(values[:, :3]-center) <= [radius, radius, radius*.9]).all(1)
    points = values[mask]
    assert len(points) > 50
    if len(points) > 12000:
        points = points[::int(np.ceil(len(points)/12000))]
    az, elevation = np.deg2rad([-58, 25])
    direction = np.array([np.cos(elevation)*np.cos(az), np.cos(elevation)*np.sin(az), np.sin(elevation)])
    camera = center+6*radius*direction
    args = (camera, center, 1300., np.array([600., 450.]))
    uv = renderer.project(points[:, :3], *args)
    order = np.argsort(-uv[:, 2], kind="stable")
    colors = np.clip((points[:, 3:6]/255.)**.8*.85+.08, 0, 1)
    ax.scatter(uv[order, 0], uv[order, 1], c=colors[order], s=1.1,
               linewidths=0, rasterized=True, alpha=.95)
    color = COLORS["failure"] if failure else COLORS["native"]
    draw_box(ax, box, renderer, args, color, 1.3)
    if gt is not None and not failure:
        draw_box(ax, gt, renderer, args, COLORS["gt"], 1.0, True)
    box_uv = renderer.project(box, *args)[:, :2]
    midpoint = (box_uv.min(0)+box_uv.max(0))/2
    extent = max(float(np.ptp(box_uv, axis=0).max())*1.65, 280.)
    ax.set_xlim(midpoint[0]-extent*.63, midpoint[0]+extent*.63)
    ax.set_ylim(midpoint[1]+extent*.5, midpoint[1]-extent*.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#FAFBFC")
    for spine in ax.spines.values(): spine.set_color("#E0E5ED"); spine.set_linewidth(.5)
    return {"aligned_native_box": box.tolist(), "points_in_roi": int(mask.sum()),
            "roi_radius_m": radius, "camera_position": camera.tolist(),
            "coordinate_system": "ScanNet axisAlignment applied to original raw-world prediction; GT and evaluation vertices already aligned"}


def render_hcsr(cases, evidences, source, renderer, input_hashes):
    reference = json.loads((HCSR/"source_manifest.json").read_text())
    fig = plt.figure(figsize=(WIDTH, 2.85))
    audits = {}
    for panel, key in enumerate(("mvsr_reranking", "failure")):
        case, info = cases[key], reference["cases"][key]
        left = .025+panel*.505
        failure = panel == 1
        fig.text(left, .925, "(a) HCSR native-record reranking" if not failure else "(b) HCSR false-positive promotion",
                 fontweight="bold", fontsize=7.9, color=INK)
        rgb = fig.add_axes([left, .345, .236, .49])
        rgb_crop(rgb, case, evidences[key]["frames"][0], source, native=True, failure=failure)
        for name, hash_key in (("native_component", "native_sha256"), ("reranked_component", "reranked_sha256")):
            path = Path(info[name]); assert digest(path) == info[hash_key]
            input_hashes[str(path)] = info[hash_key]
        native, scored = read_rows(info["native_component"]), read_rows(info["reranked_component"])
        index = info["native_index"]
        assert len(native) == len(scored)
        assert all(np.array_equal(a[1], b[1]) for a, b in zip(native, scored))
        assert native[index][2] == info["original_score"] and scored[index][2] == info["reranked_score"]
        for rows, field in ((native, "original_rank"), (scored, "reranked_rank")):
            order = np.argsort(-np.asarray([row[2] for row in rows]), kind="stable")
            assert int(np.flatnonzero(order == index)[0]+1) == info[field]
        matrix = old.alignment(case["scene"])
        box = np.asarray(native[index][1]) @ matrix[:3, :3].T + matrix[:3, 3]
        gt_path = ROOT/"evaluation/data_util/scannet_train_detection_data"/f"{case['scene']}_bbox.npy"
        input_hashes[str(gt_path)] = digest(gt_path)
        values = np.load(gt_path)
        gt_boxes = values[:, None, :3]+(old.CORNER_CODES[None]-.5)*values[:, None, 3:6]
        overlaps = old.iou_matrix(box[None], gt_boxes)[0]
        maximum = float(overlaps.max())
        assert maximum < .15 if failure else maximum > .5
        geometry = fig.add_axes([left+.251, .345, .206, .49])
        audit = perspective_geometry(geometry, case["scene"], box,
            None if failure else gt_boxes[int(overlaps.argmax())], renderer, input_hashes, failure=failure)
        geometry.set_title("Shared 3D geometry", loc="left", fontsize=6.8, pad=3)
        fig.text(left, .285, "No GT match at IoU threshold 0.15" if failure else "Same native box; geometry unchanged",
                 fontsize=6.7, color=COLORS["failure"] if failure else MUTED)
        card = fig.add_axes([left, .075, .457, .165]); card.axis("off")
        card.add_patch(Rectangle((0, 0), 1, 1, color="#F2F5FA"))
        card.text(.035, .73, f"Score  {info['original_score']:.3f} → {info['reranked_score']:.3f}", fontsize=7.6, color=INK)
        card.text(.035, .20, f"Native-record rank  {info['original_rank']} → {info['reranked_rank']}", fontsize=7.6, color=INK)
        audits[key] = {**audit, "source_metrics": info, "max_gt_iou": maximum,
                       "native_geometry_before_after_identical": True}
    fig.text(.5, .018, "RGB: solid native box / dashed matched proposal. Ranks are within each scene's fixed native records.",
             ha="center", fontsize=6.2, color=MUTED)
    return export(fig, "hcsr_3d_success_failure"), audits


def captions(audits):
    value = """定性结果优化版（2026-11-02）

命名：本版按用户提供的文字使用PTR和VAR，分别对应既有代码和结果中的PLR与CALR。没有修改算法、预测或统计数值。

主图：qualitative_3d_comparison_optimized
图X｜选定目标的三维定性比较。从左至右为标注、BoxFusion、BoxFusion-Ctrl与ReCaR-3D。每行共享相同透视相机、网格、剖切与坐标系。绿色虚线为选定GT，蓝色为原生预测，橙色和金色分别为PTR和VAR恢复框；紫色标注指向HCSR重排序的原生框，并不表示新的几何。(a) PTR补出受控原生输出未覆盖的目标，3D AABB IoU为0.58；(b) VAR补出另一遗漏目标，IoU为0.86；(c) 原生框的场景内原生排名由4升至3，分数由0.825升至0.909，但另一个GT仍未在IoU阈值0.15下被匹配，说明恢复并不完整。展示目标及代表框使用GT辅助选择：完整方法优先保留Ctrl中已显示的相同原生框，只为未被原生集合覆盖的选定GT添加恢复框。该图不是全部输出，也不是Top-K预测，未显示部分误检和重复框，不能据此推断总体精度。框坐标均来自真实预测，网格只用于可视化。

补充时序图：temporal_candidate_evidence
图SY｜候选证据的时序维护。(a) 同一PTR轨迹在关键帧425、450和550累积支持，缓存记录的确认时刻为550；(b) 同一VAR体素在1400、1575和1600获得前三个不同关键帧支持。图中从左至右按时间排列，RGB中的框是相应候选的实际二维区域或三维投影。右端三维图为后续观测处理完成后的终态代表，其几何不能回填为第三个展示关键帧的输出。虚线间隔明确省略后续观测。时序候选与三维终态代表的坐标一致性已核对；该选定案例用于展示因果更新规则，不能代替完整在线前缀AP或首次可见到正确检测的延迟统计。

HCSR成功与失败图：hcsr_3d_success_failure
图SZ｜HCSR的排序作用及适用边界。(a) 在保持原生几何不变的条件下，分数由0.506升至0.627，场景内原生排名由51升至32；(b) 一个在三维AABB IoU阈值0.15下未匹配任何GT的原生预测也被提升，分数由0.586升至0.693，排名由45升至35。每个案例展示一张真实RGB和一张同一原生框的局部透视三维点云视图；绿色虚线为GT，蓝色为原生框，红色为失败预测。RGB紫色虚线表示匹配proposal。三维坐标由原始预测重新应用ScanNet axisAlignment获得，与GT和可视化点云使用相同坐标系。HCSR只改变置信度与排名，不修正框几何；单个误检被提升不能单独证明候选匹配错误。

可用于4.3节的精简正文：
图X展示了三个场景中的选定目标。PTR和VAR分别利用历史proposal和anchor证据补出原生输出未覆盖的目标；HCSR在不改变原生框几何的条件下调整其排名。第三个场景仍存在未恢复目标，表明历史候选不能保证完整覆盖。补充时序图给出支持积累与确认过程，HCSR失败案例则表明较高历史支持并非检测正确性的充分条件。所有可视化均基于真实预测与标注，选定目标使用GT辅助匹配，整体性能仍以定量评测为准。

数据与范围说明：三场景来自20261009公开CLI效率复现实验存档；HCSR成功/失败来自已核验的20261005无visibility组件；两者不是同一执行运行，不跨面板拼接计算AP增益。前三场景保留既有GT集合，新增的恢复框须满足原生所有框均不覆盖、辅助框覆盖该GT的严格IoU>0.15判据。第三场景不是额外的恢复成功证据。所有预测坐标未修改，没有重跑模型或占用GPU。独立三维PNG仍为3000×2400、400 DPI、透明背景；主图PDF/SVG含可编辑框线与标注。旧图及源文件均保留。

审计：source_manifest.json给出全部路径、哈希、选择索引、世界坐标系、支持帧和实际排名；measurement_audit.json提供目标覆盖及失败检查。既有全输出/Top-50审计图仍保留于原始素材目录，不能用清晰选定目标图替代它们。

时序素材修正：原VAR示例在1600帧使用了相邻体素[8,16,1]的高IoU候选。本版从真实候选缓存选取属于确认体素[8,16,2]的anchor，并重放原状态验证三个显示时刻的支持数确为1、2、3。PTR也逐帧核对所示proposal实际属于轨迹39。两条终态代表均与原缓存追踪几何一致。该审计只重放候选状态，没有重新运行感知模型。
"""
    (OUT/"定性结果段落与图注.txt").write_text(value, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                        "svg.fonttype": "none", "text.color": INK})
    old_manifest_path = INPUT/"source_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text())
    input_hashes = {str(old_manifest_path): digest(old_manifest_path), str(CASES): digest(CASES)}
    renderer = old.load_renderer()
    payloads, audits = load_payloads(renderer, old_manifest, input_hashes)
    assert [p["recoveries"][0]["branch"] for p in payloads[:2]] == ["PTR", "VAR"]
    assert len(payloads[2]["unresolved"]) == 1 and payloads[2]["unresolved"][0]["full_best_iou"] < .15
    artifacts = render_comparison(payloads, renderer)
    individual_exports = []
    for payload in payloads:
        individual_exports.extend(old.export_individuals(payload, renderer, OUT, "selected"))
        print(f"3D panels complete: {payload['spec']['scene']}", flush=True)
    cases = json.loads(CASES.read_text())["cases"]
    evidences = {}
    for name, case in cases.items():
        path = Path(case["evidence_export"])
        assert digest(path) == case["evidence_sha256"] and case["evidence_verified"]
        input_hashes[str(path)] = digest(path)
        evidences[name] = json.loads(path.read_text())
        assert evidences[name]["future_frames_used"] is False
        for frame in evidences[name]["frames"]:
            input_hashes[frame["rgb_path"]] = digest(frame["rgb_path"])
    source = evidence_module()
    temporal_audit = verify_temporal_memberships(evidences, input_hashes)
    artifacts.extend(render_temporal(payloads, cases, evidences, renderer, source))
    hcsr_files, hcsr_audit = render_hcsr(cases, evidences, source, renderer, input_hashes)
    artifacts.extend(hcsr_files)
    captions(audits)
    measured = {"three_scene_cases": audits, "hcsr_cases": hcsr_audit, "temporal_membership_audit": temporal_audit,
        "temporal_terminal_geometry_matches_archived_prediction": True,
        "all_native_geometry_unchanged": True}
    (OUT/"measurement_audit.json").write_text(json.dumps(measured, indent=2, ensure_ascii=False)+"\n")
    manifest = {"schema": "recar3d.qualitative.native_preserving.v2",
        "display_name_mapping": {"PTR": "PLR", "VAR": "CALR"},
        "input_sha256": input_hashes, "main_cases": audits,
        "individual_exports": individual_exports, "hcsr_cases": hcsr_audit,
        "temporal_membership_audit": temporal_audit,
        "source_run_note": "Main 3D comparisons and HCSR diagnostic examples have separately disclosed archived runs",
        "prediction_coordinates_edited": False, "new_model_inference": False, "gpu_used": False,
        "GT_assisted_selection": True, "all_output_predictions_shown": False,
        "generated_scene_content": False, "script_sha256": digest(Path(__file__)),
        "artifacts_sha256": {name: digest(OUT/name) for name in artifacts}}
    (OUT/"source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    with zipfile.ZipFile(OUT.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file(): archive.write(path, path.relative_to(OUT))
    print(json.dumps({"output": str(OUT), "figures": 3, "individual_panels": len(individual_exports),
                      "new_model_inference": False, "gpu_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
